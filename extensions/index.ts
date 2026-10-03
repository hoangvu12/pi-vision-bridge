/**
 * pi-vision-bridge — give non-vision pi models the ability to understand
 * images AND video.
 *
 * Architecture (synthesized from research of the opencode vision plugins,
 * z.ai's vision-mcp-server, pi's ecosystem, and video-sampling write-ups):
 *
 *  IMAGES
 *  1. Eager in-context swap (`context` event): when the active model cannot
 *     accept images, every ImageContent in the outgoing request is replaced
 *     with a thorough text description produced by a vision model, BEFORE
 *     the request leaves. Zero extra model round-trips. Pi itself would
 *     otherwise strip the image to "(image omitted: model does not support
 *     images)".
 *  2. Non-destructive: the session keeps the original images. Switch to a
 *     vision model later and you get native vision back.
 *  3. Content-hash LRU cache: each image is analyzed exactly once per
 *     process, across turns, retries, and cache warming.
 *  4. describe_image tool (lazy, targeted): the cached description is
 *     deliberately generic so it stays cache-coherent; the tool provides
 *     the focused re-look ("quote the exact error text") — optionally with
 *     a task mode (ocr/error/ui/diagram/chart: curated readings) or a
 *     region [x, y, w, h] (the crop alone is analyzed; dimensions are
 *     published in the swap text so the model can construct boxes).
 *  4b. compare_images tool: "what changed between these two?" in ONE
 *     vision call with a diff-oriented prompt; visible to every model —
 *     even a vision model cannot diff two images it saw in different
 *     turns without re-attaching them.
 *
 *  VIDEO
 *  5. Videos cannot enter pi messages at all (no video content type), so
 *     they are file references in text. The describe_video tool extracts
 *     evenly-sampled frames with ffmpeg (scaled, capped, timestamped) and
 *     sends them as one multi-image vision call — works with every vision
 *     model through pi's provider stack, no native video API needed.
 *  6. before_agent_start injects a prompt-section hint when the user's
 *     message references a video file, so the model knows to call the tool.
 *     The video tool stays visible for vision models too — nothing in pi
 *     can watch video natively.
 *
 *  BOTH
 *  7. Vision-model selection as an ORDERED candidate list with health
 *     tracking: failed models are benched for a cooldown and the analysis
 *     rotates to the next entry; the winner sticks (a relay listing a model
 *     says nothing about its upstream actually serving it). No name-based
 *     scoring — a model name says nothing a test verified, so auto mode
 *     uses catalog order and the user can set an explicit order via
 *     /visionbridge model, the config file, the visionBridge section in
 *     pi's own settings (global or project), or an env var.
 *  8. Passthrough with zero overhead when the active model supports images.
 *  9. The helper vision call is a plain no-tools model call: the media and
 *     a fixed prompt only — never tools, files, or bash, so hostile media
 *     cannot turn the helper into an execution path.
 */

import type { AgentToolResult, ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent, Model, ToolResultMessage } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { statSync, openSync, readSync, closeSync } from "node:fs";
import { extname, isAbsolute, resolve } from "node:path";
import {
	type BridgeConfig,
	CONFIG_PATH,
	debug,
	isEnvDisabled,
	loadConfig,
	saveConfig,
} from "../src/config.ts";
import {
	analyzeImage,
	analyzeImages,
	fingerprint,
	resolveVisionModel,
	HEALTH_COOLDOWN_MS,
	isRetryableFailure,
	MAX_ATTEMPTS,
	rankVisionModels,
	type VisionAnalysis,
	type VisionCandidate,
} from "../src/vision.ts";
import {
	ANALYSIS_MODES,
	buildCompareUserText,
	buildSwapText,
	COMPARE_SYSTEM_PROMPT,
	composeCompareText,
	composeRegionText,
	IMAGE_TOOL,
	modesGuidanceList,
	resolveModePrompt,
} from "../src/prompts.ts";
import { clampRegion, cropImage, parseImageDimensions, type ImageDimensions, type Region } from "../src/image.ts";
import { analyzeVideo, probeVideo, VIDEO_EXTENSIONS, videoFingerprint } from "../src/video.ts";

const VIDEO_TOOL = "describe_video";
const COMPARE_TOOL = "compare_images";
const VIDEO_HINT_SECTION = "vision_bridge_video";

/** Shared error text for "every candidate is down" across the three tools. */
const NO_VISION_MODEL_MESSAGE =
	"No vision-capable model is currently available (every connected candidate failed recently, likely an upstream outage). " +
	"Tell the user to retry later or connect another vision model.";

interface CacheEntry {
	description: string;
	model: string;
}

type ToolResult = AgentToolResult<ToolDetails>;

interface ToolDetails {
	fingerprint: string;
	model: string;
	question?: string;
	mode?: string;
	region?: number[];
	cached: boolean;
	media?: "image" | "video";
	frames?: number;
}

interface AnalysisOutcome {
	description: string;
	model: string;
	cached?: boolean;
}

interface ImageRef {
	image: ImageContent;
	fingerprint: string;
	messageRole: "user" | "toolResult";
	toolName?: string;
}

/** Video file references in user prompts, for the hint injection. */
const VIDEO_GLOB_RE = /(?:[A-Za-z]:)?[\w\-./\\@+()\[\]]+\.(?:mp4|mov|mkv|m4v|webm|avi)/gi;
const VIDEO_QUOTED_RE = /["']([^"']+\.(?:mp4|mov|mkv|m4v|webm|avi))["']/gi;

export default function visionBridge(pi: ExtensionAPI) {
	/** Env var per config field: env always wins over the pi-settings section. */
	const FIELD_ENV: Partial<Record<keyof BridgeConfig, string>> = {
		enabled: "PI_VISION_BRIDGE_OFF",
		visionModels: "PI_VISION_BRIDGE_MODEL",
		maxTokens: "PI_VISION_BRIDGE_MAX_TOKENS",
		cacheMax: "PI_VISION_BRIDGE_CACHE_MAX",
		videoFrames: "PI_VISION_BRIDGE_VIDEO_FRAMES",
	};

	/**
	 * pi's settings.json has no official per-extension section, but unknown
	 * keys survive its loader and deep merge, and getSettings() exposes the
	 * merged result. So an optional "visionBridge" block (global AND project
	 * scope; project wins via pi's own merge) is honored as a config layer.
	 * Unknown keys inside the block are ignored, never rejected.
	 */
	type SettingsLike = { getSettings?: () => unknown };
	function readPiSection(source: unknown): Partial<BridgeConfig> | undefined {
		const getter = (source as SettingsLike | undefined)?.getSettings;
		if (typeof getter !== "function") return undefined;
		try {
			const settings = (getter as () => unknown)() as { visionBridge?: Record<string, unknown> } | undefined;
			const raw = settings?.visionBridge;
			if (!raw || typeof raw !== "object") return undefined;
			const out: Partial<BridgeConfig> = {};
			if (typeof raw.enabled === "boolean") out.enabled = raw.enabled;
			if (Array.isArray(raw.visionModels)) {
				const list = raw.visionModels
					.filter((s): s is string => typeof s === "string")
					.map((s) => s.trim())
					.filter((s) => s.length > 0);
				if (list.length > 0) out.visionModels = list;
			}
			if (typeof raw.maxTokens === "number" && raw.maxTokens > 0) out.maxTokens = raw.maxTokens;
			if (typeof raw.temperature === "number" && raw.temperature >= 0) out.temperature = raw.temperature;
			if (typeof raw.cacheMax === "number" && raw.cacheMax > 0) out.cacheMax = raw.cacheMax;
			if (typeof raw.notify === "boolean") out.notify = raw.notify;
			if (typeof raw.videoFrames === "number" && raw.videoFrames > 0) out.videoFrames = raw.videoFrames;
			return out;
		} catch (err) {
			debug("pi settings read failed:", err instanceof Error ? err.message : err);
			return undefined;
		}
	}

	/** Effective config: defaults <- file <- pi-settings section <- env. */
	function buildConfig(source: unknown): BridgeConfig {
		const base = isEnvDisabled() ? { ...loadConfig(), enabled: false } : loadConfig();
		const section = readPiSection(source);
		piSettingsActive = section !== undefined && Object.keys(section).length > 0;
		if (!section) return base;
		const next = { ...base };
		for (const [key, value] of Object.entries(section) as Array<[keyof BridgeConfig, unknown]>) {
			const envVar = FIELD_ENV[key];
			if (envVar && process.env[envVar] !== undefined) continue; // env wins
			(next as Record<string, unknown>)[key] = value;
		}
		return next;
	}

	let piSettingsActive = false; // set by buildConfig; true when a visionBridge section applied
	let config = buildConfig(pi);

	// LRU cache: key -> description. Map preserves insertion order.
	const cache = new Map<string, CacheEntry>();
	// In-flight analysis dedup: key -> promise.
	const pending = new Map<string, Promise<AnalysisOutcome>>();

	// Memoized ranked candidates (registry can change; re-resolve on model_select / TTL).
	let visionMemo: { candidates: VisionCandidate[]; at: number; key: string } | undefined;
	// Health tracking: "provider/model-id" -> unix ms until which it is benched.
	const benched = new Map<string, number>();
	// The candidate that last succeeded — kept at the front while healthy.
	let stickyKey: { key: string; until: number } | undefined;
	let notifiedNoVision = false;
	let notifiedAutoPick = false;
	let lastNotifyKey = "";

	function refreshConfig(ctx: ExtensionContext): void {
		config = buildConfig(ctx);
	}

	function cacheGet(hash: string): CacheEntry | undefined {
		const hit = cache.get(hash);
		if (hit) {
			// LRU touch: re-insert at the end.
			cache.delete(hash);
			cache.set(hash, hit);
		}
		return hit;
	}

	function cacheSet(hash: string, entry: CacheEntry): void {
		cache.delete(hash);
		cache.set(hash, entry);
		while (cache.size > config.cacheMax) {
			const oldest = cache.keys().next().value;
			if (oldest === undefined) break;
			cache.delete(oldest);
		}
	}

	/** Success toast: names the answering vision model and the elapsed time,
	 *  so the user sees what the bridge spent. Silent without UI or notify. */
	function notifyAnalysisDone(ctx: ExtensionContext, body: string, startedAt: number): void {
		if (!ctx.hasUI || !config.notify) return;
		const seconds = ((Date.now() - startedAt) / 1000).toFixed(1);
		ctx.ui.notify(`pi-vision-bridge: ${body} in ${seconds}s`, "info");
	}

	function candidateKey(c: VisionCandidate): string {
		return `${c.model.provider}/${c.model.id}`;
	}

	/** Ranked, healthy candidates with the sticky winner first. */
	function applyHealth(candidates: VisionCandidate[]): VisionCandidate[] {
		const now = Date.now();
		for (const [key, until] of benched) {
			if (until <= now) benched.delete(key);
		}
		const healthy = candidates.filter((c) => !benched.has(candidateKey(c)));
		if (stickyKey && stickyKey.until > now) {
			const sticky = healthy.find((c) => candidateKey(c) === stickyKey!.key);
			if (sticky) return [sticky, ...healthy.filter((c) => c !== sticky)];
		}
		return healthy;
	}

	function getVisionCandidates(ctx: ExtensionContext): VisionCandidate[] {
		if (!config.enabled) return [];
		const now = Date.now();
		const modelKey = config.visionModels?.join(",") ?? "";
		if (visionMemo && now - visionMemo.at < 30_000 && visionMemo.key === modelKey) {
			return applyHealth(visionMemo.candidates);
		}
		const candidates = rankVisionModels(ctx, config.visionModels);
		visionMemo = { candidates, at: now, key: modelKey };

		if (candidates.length === 0) {
			if (!notifiedNoVision && ctx.hasUI && config.notify) {
				notifiedNoVision = true;
				ctx.ui.notify(
					"pi-vision-bridge: no vision-capable model is connected, so media cannot be analyzed for this text-only model. Connect a vision model (e.g. via /login) or run /visionbridge.",
					"warning",
				);
			}
			return [];
		}
		const first = applyHealth(candidates)[0];
		if (first && first.source === "auto" && !notifiedAutoPick && ctx.hasUI && config.notify) {
			notifiedAutoPick = true;
			ctx.ui.notify(`pi-vision-bridge: vision model auto-selected ${candidateKey(first)}`, "info");
		}
		return applyHealth(candidates);
	}

	/**
	 * Run one analysis through the ranked candidates with fallback.
	 * Benches failed models for a cooldown and rotates to the next; remembers
	 * the winner as sticky so subsequent calls start there.
	 */
	async function runWithFallback<T>(
		ctx: ExtensionContext,
		analyze: (model: Model<any>) => Promise<T>,
	): Promise<{ result: T; modelKey: string }> {
		const candidates = getVisionCandidates(ctx);
		if (candidates.length === 0) throw new Error("no vision model available");

		let lastError: unknown;
		for (let attempt = 0; attempt < Math.min(MAX_ATTEMPTS, candidates.length); attempt++) {
			const candidate = candidates[attempt];
			const key = candidateKey(candidate);
			try {
				const result = await analyze(candidate.model);
				stickyKey = { key, until: Date.now() + HEALTH_COOLDOWN_MS };
				if (attempt > 0) debug(`fallback succeeded after ${attempt} failure(s):`, key);
				return { result, modelKey: key };
			} catch (err) {
				lastError = err;
				if (!isRetryableFailure(err)) throw err; // aborts and 4xx: rotating won't help
				benched.set(key, Date.now() + HEALTH_COOLDOWN_MS);
				debug("candidate failed, rotating:", key, err instanceof Error ? err.message : err);
			}
		}
		throw lastError instanceof Error ? lastError : new Error(String(lastError));
	}

	/** Analyze one image (generic description), with cache + in-flight dedup. */
	function describeCached(ctx: ExtensionContext, image: ImageContent): Promise<AnalysisOutcome> {
		const hash = fingerprint(image);
		const hit = cacheGet(hash);
		if (hit) return Promise.resolve({ description: hit.description, model: hit.model, cached: true });

		const inflight = pending.get(hash);
		if (inflight) return inflight;

		const job = (async () => {
			const started = Date.now();
			const { result } = await runWithFallback<VisionAnalysis>(ctx, (model) =>
				analyzeImage(ctx, model, image, {
					maxTokens: config.maxTokens,
					temperature: config.temperature,
				}),
			);
			cacheSet(hash, { description: result.description, model: result.model });
			debug("described", hash, `${Date.now() - started}ms`, `${result.description.length} chars`);
			return { description: result.description, model: result.model, cached: false };
		})().finally(() => {
			pending.delete(hash);
		});

		pending.set(hash, job);
		return job;
	}

	/** Collect every image present in the persisted session branch, in order. */
	function collectSessionImages(ctx: ExtensionContext): ImageRef[] {
		const messages: AgentMessage[] = [];
		for (const entry of ctx.sessionManager.getBranch()) {
			if (entry.type !== "message") continue;
			const msg = (entry as { message?: AgentMessage }).message;
			if (msg) messages.push(msg);
		}
		return imagesInMessages(messages);
	}

	/** Match a model-supplied fingerprint: exact, or a unique-ish prefix. */
	function findByFingerprint(images: ImageRef[], wanted: string): ImageRef | undefined {
		const key = wanted.trim().toLowerCase();
		return images.find((ref) => ref.fingerprint === key || ref.fingerprint.startsWith(key));
	}

	/** List the session's images for error messages (retry affordance). */
	function listImages(images: ImageRef[]): string {
		return images
			.map((ref) => `  ${ref.fingerprint} (${ref.messageRole}${ref.toolName ? ` from ${ref.toolName}` : ""})`)
			.join("\n");
	}

	/** Region as a plain [x, y, w, h] array (details, cache keys). */
	function regionToArray(region: Region): number[] {
		return [region.x, region.y, region.w, region.h];
	}

	/** Shared "unknown fingerprint" error with the retry affordance. */
	function noFingerprintError(wanted: string, images: ImageRef[]): string {
		return `No image with fingerprint "${wanted}". Images in this conversation:\n${listImages(images)}\nRetry with one of these exact fingerprints.`;
	}

	/**
	 * Resolve the (first, second) pair for compare_images. Omitted sides
	 * default to the most recent image distinct from the resolved other
	 * side; with both omitted, that is the two most recent distinct images.
	 */
	function resolveComparePair(
		images: ImageRef[],
		firstSpec: string | undefined,
		secondSpec: string | undefined,
	): { ok: true; first: ImageRef; second: ImageRef } | { ok: false; error: string } {
		// Distinct images, most recent first.
		const recentDistinct: ImageRef[] = [];
		const seen = new Set<string>();
		for (let i = images.length - 1; i >= 0; i--) {
			const ref = images[i];
			if (seen.has(ref.fingerprint)) continue;
			seen.add(ref.fingerprint);
			recentDistinct.push(ref);
		}

		const explicitFirst = firstSpec ? findByFingerprint(images, firstSpec) : undefined;
		const explicitSecond = secondSpec ? findByFingerprint(images, secondSpec) : undefined;
		if ((firstSpec && !explicitFirst) || (secondSpec && !explicitSecond)) {
			const wanted = firstSpec && !explicitFirst ? firstSpec : secondSpec;
			return { ok: false, error: noFingerprintError(wanted ?? "", images) };
		}

		// The SECOND slot prefers the most recent distinct image (the
		// "newer" version); the FIRST slot then resolves to the most recent
		// remaining one — so omitted pair = (earlier, later) of the two most
		// recent distinct images.
		const secondRef =
			explicitSecond ??
			(explicitFirst
				? recentDistinct.find((r) => r.fingerprint !== explicitFirst.fingerprint)
				: recentDistinct[0]);
		const firstRef =
			explicitFirst ??
			(explicitSecond
				? recentDistinct.find((r) => r.fingerprint !== explicitSecond.fingerprint)
				: recentDistinct.find((r) => r !== secondRef));
		if (!firstRef || !secondRef) {
			return {
				ok: false,
				error:
					recentDistinct.length === 1
						? `Only one distinct image (${recentDistinct[0].fingerprint}) is attached to this conversation; two are needed to compare. Have the user attach or paste another image first.`
						: "No images are attached to this conversation, so there is nothing to compare.",
			};
		}
		if (firstRef.fingerprint === secondRef.fingerprint) {
			return {
				ok: false,
				error: `Both fingerprints resolve to the same image (${firstRef.fingerprint}); comparing an image with itself is not useful. Choose two different images.`,
			};
		}
		return { ok: true, first: firstRef, second: secondRef };
	}

	/**
	 * Cache key for a targeted image analysis. Every analysis-shaping input
	 * (fingerprint, mode, region, question) is encoded explicitly, so no two
	 * different analyses can ever share a key and omitting all of them keeps
	 * today's per-(image, question) semantics.
	 */
	function toolCacheKey(args: {
		fingerprint: string;
		mode?: string;
		region?: number[];
		question?: string;
	}): string {
		return `q:${JSON.stringify([args.fingerprint, args.mode ?? null, args.region ?? null, args.question ?? null])}`;
	}

	/**
	 * Pixel dimensions per image fingerprint, parsed once from the image
	 * header (no model call, no re-encode). Unknown formats stay undefined.
	 */
	const dimsCache = new Map<string, ImageDimensions | null>();
	function imageDimensions(image: ImageContent, fp: string): ImageDimensions | undefined {
		if (!dimsCache.has(fp)) {
			dimsCache.set(fp, parseImageDimensions(Buffer.from(image.data, "base64")) ?? null);
		}
		return dimsCache.get(fp) ?? undefined;
	}

	/** Every image block in a message list, in order, with its attribution. */
	function imagesInMessages(messages: AgentMessage[]): ImageRef[] {
		const images: ImageRef[] = [];
		for (const msg of messages) {
			if (msg.role !== "user" && msg.role !== "toolResult") continue;
			if (!Array.isArray(msg.content)) continue;
			for (const block of msg.content) {
				if (block.type !== "image") continue;
				images.push({
					image: block,
					fingerprint: fingerprint(block),
					messageRole: msg.role,
					toolName: msg.role === "toolResult" ? (msg as ToolResultMessage).toolName : undefined,
				});
			}
		}
		return images;
	}

	/** Collect every image present in request messages, deduped by hash, in order. */
	function collectImages(messages: AgentMessage[]): { refs: Map<string, ImageRef>; order: string[] } {
		const refs = new Map<string, ImageRef>();
		const order: string[] = [];
		for (const ref of imagesInMessages(messages)) {
			if (!refs.has(ref.fingerprint)) {
				refs.set(ref.fingerprint, ref);
				order.push(ref.fingerprint);
			}
		}
		return { refs, order };
	}

	/**
	 * The core: swap images for descriptions in the outgoing request.
	 *
	 * `event.messages` is a structured clone of the session state, so in-place
	 * edits never touch the persisted history.
	 */
	pi.on("context", async (event, ctx) => {
		if (!config.enabled) return;
		const model = ctx.model;
		if (!model || model.input.includes("image")) return; // native vision: passthrough

		const { refs, order } = collectImages(event.messages);
		if (order.length === 0) return;

		const candidates = getVisionCandidates(ctx);
		if (candidates.length === 0) return undefined; // nothing we can do; pi strips the image itself

		// Analyze every distinct image; parallel, deduped, cached.
		const fresh = order.filter((fp) => !cache.has(fp));
		const notifyKey = fresh.join(",");
		if (fresh.length > 0 && ctx.hasUI && config.notify && notifyKey !== lastNotifyKey) {
			lastNotifyKey = notifyKey;
			ctx.ui.notify(
				`pi-vision-bridge: analyzing ${fresh.length} image${fresh.length > 1 ? "s" : ""} with ${candidateKey(candidates[0])}…`,
				"info",
			);
		}

		const descriptions = new Map<string, string>();
		const models = new Map<string, string>();
		const batchStarted = Date.now();
		await Promise.all(
			order.map(async (fp) => {
				const ref = refs.get(fp)!;
				try {
					const outcome = await describeCached(ctx, ref.image);
					descriptions.set(fp, outcome.description);
					models.set(fp, outcome.model);
				} catch (err) {
					const message = err instanceof Error ? err.message : String(err);
					debug("analysis failed", fp, message);
					const failure = `[Image analysis failed: ${message}. The image was attached but could not be described. Call ${IMAGE_TOOL} with fingerprint "${fp}" to retry the analysis.]`;
					descriptions.set(fp, failure);
					models.set(fp, "unavailable");
					if (ctx.hasUI && config.notify) {
						ctx.ui.notify(`pi-vision-bridge: image analysis failed — ${message}`, "error");
					}
				}
			}),
		);

		// Completion toast: what the bridge spent once the batch settles.
		const freshModels = [
			...new Set(fresh.map((fp) => models.get(fp)).filter((m): m is string => !!m && m !== "unavailable")),
		];
		if (fresh.length > 0 && freshModels.length > 0) {
			const modelText = freshModels.length > 1 ? `${freshModels[0]} (+${freshModels.length - 1} more)` : freshModels[0];
			notifyAnalysisDone(ctx, `described ${fresh.length} image${fresh.length > 1 ? "s" : ""} with ${modelText}`, batchStarted);
		}

		// Swap every image block for its description, in place.
		for (const msg of event.messages) {
			if (msg.role !== "user" && msg.role !== "toolResult") continue;
			if (!Array.isArray(msg.content)) continue;
			for (let i = 0; i < msg.content.length; i++) {
				const block = msg.content[i];
				if (block.type !== "image") continue;
				const fp = fingerprint(block);
				const description = descriptions.get(fp);
				if (description === undefined) continue;
				const model = models.get(fp);
				const ref = refs.get(fp);
				msg.content[i] = {
					type: "text",
					text: buildSwapText({
						fingerprint: fp,
						description,
						origin: msg.role === "toolResult" ? ` (from ${msg.toolName ?? "tool"} output)` : undefined,
						describedBy: model && model !== "unavailable" ? model : undefined,
						dimensions: ref ? imageDimensions(ref.image, fp) : undefined,
					}),
				};
			}
		}

		return { messages: event.messages };
	});

	/**
	 * Video discoverability: when the user's message references a video file,
	 * tell the model (via a prompt section) to call describe_video. The tool
	 * stays out of the way otherwise.
	 */
	pi.on("before_agent_start", (event, ctx) => {
		if (!config.enabled) return;
		const refs = findVideoReferences(event.prompt, ctx.cwd);
		if (refs.length === 0) {
			delete event.systemPromptOptions.sections[VIDEO_HINT_SECTION];
			return;
		}
		const list = refs.map((p) => `"${p}"`).join(", ");
		event.systemPromptOptions.sections[VIDEO_HINT_SECTION] =
			`The user's message references a video file (${list}). You cannot watch videos directly. ` +
			`Call the ${VIDEO_TOOL} tool with the file path to have it analyzed (frames are sampled and described by a vision model); ` +
			`pass a focused question for specifics. Audio is not analyzed.`;
	});

	/** Extract video file references that actually exist from prompt text. */
	function findVideoReferences(prompt: string, cwd: string): string[] {
		const found: string[] = [];
		const seen = new Set<string>();
		const candidates: string[] = [];
		for (const match of prompt.matchAll(VIDEO_QUOTED_RE)) candidates.push(match[1]);
		for (const match of prompt.matchAll(VIDEO_GLOB_RE)) candidates.push(match[0]);
		for (const raw of candidates) {
			const trimmed = raw.trim();
			if (!trimmed || seen.has(trimmed)) continue;
			seen.add(trimmed);
			if (!VIDEO_EXTENSIONS.has(extname(trimmed).toLowerCase())) continue;
			const resolved = isAbsolute(trimmed) ? trimmed : resolve(cwd, trimmed);
			try {
				if (statSync(resolved).isFile()) found.push(resolved);
			} catch {
				// not a real file — ignore
			}
		}
		return found;
	}

	/**
	 * The lazy image path: targeted re-examination of an image the model
	 * already saw described. Images are pulled from persisted session
	 * history, so the tool works even after the request copy is gone.
	 */
	pi.registerTool({
		name: IMAGE_TOOL,
		label: "Describe Image",
		description:
			"Re-examine an image from this conversation with a focused question, using a vision model. " +
			"Use it when an inline [Image <fingerprint>] description lacks a detail you need — the exact text of a " +
			"specific line, a color, a small region, or a comparison. Pass the fingerprint from the image's tag; " +
			"omit it to inspect the most recent image. Pass `question` describing exactly what to look for; " +
			"omit it for a full fresh description. " +
			"Pass `region` as [x, y, w, h] in image pixels (the image's dimensions are published in its " +
			"[Image …] description) to zoom into part of it — the crop alone is analyzed. " +
			"Optional `mode` tunes how the image is read — use when: " +
			`${modesGuidanceList()}. ` +
			"Without a mode you get a thorough generic description.",
		promptSnippet:
			"Re-inspect an image with a focused question or a task mode (ocr, error, ui, diagram, chart) when its inline description is not enough.",
		parameters: Type.Object({
			question: Type.Optional(
				Type.String({
					description:
						"What to look for in the image, e.g. \"quote the exact error text at the top\" or " +
						"\"what is the hex color of the header bar\". Omit for a full description.",
				}),
			),
			fingerprint: Type.Optional(
				Type.String({
					description:
						"The image id from an [Image <fingerprint>] tag in the conversation. Omit to use the most recent image.",
				}),
			),
			mode: Type.Optional(
				Type.Union([...ANALYSIS_MODES.map((m) => Type.Literal(m))], {
					description:
						"Reading mode: ocr (verbatim transcription — use when you need the exact words), " +
						"error (error messages and stack traces — use on failure screenshots and logs), " +
						"ui (component and layout inventory — use on interface screenshots), " +
						"diagram (nodes, arrows, relationships — use on flowcharts and architecture), " +
						"chart (axes, series, values — use on plots and graphs). Omit for a generic thorough description.",
				}),
			),
			region: Type.Optional(
				Type.Tuple([Type.Number(), Type.Number(), Type.Number(), Type.Number()], {
					description:
						"[x, y, w, h] in image pixel coordinates, against the dimensions published in the image's " +
						"[Image …] description. That region is cropped out and the crop alone is analyzed — " +
						"use it to zoom into a part of the image. Out-of-bounds boxes are clamped; zero-area boxes error.",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<ToolResult> {
			refreshConfig(ctx);
			if (!config.enabled) {
				return {
					content: [{ type: "text", text: "pi-vision-bridge is disabled. Images cannot be examined." }],
					details: { fingerprint: params.fingerprint ?? "", model: "", cached: false },
				};
			}

			// Gather images from the session branch (persisted history).
			const images = collectSessionImages(ctx);

			if (images.length === 0) {
				return {
					content: [
						{ type: "text", text: "No images are attached to this conversation. There is nothing to examine." },
					],
					details: { fingerprint: params.fingerprint ?? "", model: "", cached: false },
				};
			}

			// Resolve the target image: explicit fingerprint, or the most recent.
			let target: ImageRef | undefined;
			if (params.fingerprint) {
				target = findByFingerprint(images, params.fingerprint);
				if (!target) {
					return {
						content: [
							{
								type: "text",
								text: noFingerprintError(params.fingerprint, images),
							},
						],
						details: { fingerprint: params.fingerprint, model: "", cached: false },
					};
				}
			} else {
				target = images[images.length - 1];
			}

			const candidates = getVisionCandidates(ctx);
			if (candidates.length === 0) {
				return {
					content: [
						{
							type: "text",
							text: NO_VISION_MODEL_MESSAGE,
						},
					],
					details: { fingerprint: target.fingerprint, model: "", cached: false },
				};
			}

			const mode: string | undefined = params.mode;
			if (mode !== undefined && !ANALYSIS_MODES.includes(mode as never)) {
				// Schema validation rejects this before execute; guard direct calls.
				return {
					content: [
						{
							type: "text",
							text: `Unknown mode "${mode}". Valid modes: ${ANALYSIS_MODES.join(", ")}. Omit mode for a generic description.`,
						},
					],
					details: { fingerprint: target.fingerprint, model: "", mode, cached: false },
					isError: true,
				};
			}

			// Region: [x, y, w, h] pixels against the published dimensions.
			// Invalid boxes are clamped (noted in the result); zero/negative
			// area is a clear error; unknown dimensions cannot be clamped.
			let region: Region | undefined;
			let regionOriginal: number[] | undefined;
			let regionClamped = false;
			let regionDims: ImageDimensions | undefined;
			if (params.region !== undefined) {
				const raw = params.region;
				if (!Array.isArray(raw) || raw.length !== 4 || !raw.every((n) => typeof n === "number" && Number.isFinite(n))) {
					return {
						content: [
							{
								type: "text",
								text: `Invalid region ${JSON.stringify(raw)}: region must be [x, y, w, h] with four numbers in image pixels. Retry with a proper box.`,
							},
						],
						details: { fingerprint: target.fingerprint, model: "", question: params.question, mode, cached: false },
						isError: true,
					};
				}
				if (raw[2] <= 0 || raw[3] <= 0) {
					return {
						content: [
							{
								type: "text",
								text: `Region [${raw.join(", ")}] has zero or negative area. Provide a region with positive width and height.`,
							},
						],
						details: { fingerprint: target.fingerprint, model: "", question: params.question, mode, cached: false },
						isError: true,
					};
				}
				regionDims = imageDimensions(target.image, target.fingerprint);
				if (!regionDims) {
					return {
						content: [
							{
								type: "text",
								text: `Cannot analyze a region of image ${target.fingerprint}: its pixel dimensions could not be determined from the image data ` +
									`(unsupported or unrecognized format). Describe the whole image instead, without a region.`,
							},
						],
						details: { fingerprint: target.fingerprint, model: "", question: params.question, mode, cached: false },
						isError: true,
					};
				}
				const clamped = clampRegion(raw, regionDims);
				if (clamped.region.w <= 0 || clamped.region.h <= 0) {
					return {
						content: [
							{
								type: "text",
								text: `Region [${raw.join(", ")}] lies entirely outside the image (${regionDims.width}x${regionDims.height} px). Provide a region that overlaps the image.`,
							},
						],
						details: { fingerprint: target.fingerprint, model: "", question: params.question, mode, cached: false },
						isError: true,
					};
				}
				region = clamped.region;
				regionOriginal = [...raw];
				regionClamped = clamped.clamped;
			}

			// Targeted answers are cached per (image, mode, question) — every
			// analysis-shaping input is in the key, so two modes over the same
			// image are two analyses, and the common retry loop (model re-asking
			// the same thing) stays free.
			const cacheKey = toolCacheKey({
				fingerprint: target.fingerprint,
				mode,
				region: region ? regionToArray(region) : undefined,
				question: params.question,
			});
			const hit = cacheGet(cacheKey);
			if (hit) {
				// The region wrapper is recomposed per call so a hit always
				// reports THIS call's raw box in any clamping notice.
				return {
					content: [
						{ type: "text", text: region ? composeRegionResult(hit.description) : hit.description },
					],
					details: {
						fingerprint: target.fingerprint,
						model: hit.model,
						question: params.question,
						mode,
						region: region ? regionToArray(region) : undefined,
						cached: true,
					},
				};
			}

			// On a cache miss only: cut the crop and analyze the crop alone.
			let analysisImage = target.image;
			if (region) {
				try {
					analysisImage = await cropImage(target.image, region, { signal: ctx.signal });
				} catch (err) {
					const message = err instanceof Error ? err.message : String(err);
					return {
						content: [
							{
								type: "text",
								text: `Could not crop region [${region.x}, ${region.y}, ${region.w}, ${region.h}] from image ${target.fingerprint}: ${message}. ` +
									`The image format may need ffmpeg on PATH for region analysis; PNG is cropped natively.`,
							},
						],
						details: { fingerprint: target.fingerprint, model: "", question: params.question, mode, cached: false },
						isError: true,
					};
				}
			}

			const prompt = mode ? resolveModePrompt(mode, params.question) : undefined;
			const started = Date.now();
			const { result } = await runWithFallback(ctx, (model) =>
				analyzeImage(ctx, model, analysisImage, {
					prompt,
					question: params.question,
					maxTokens: config.maxTokens,
					temperature: config.temperature,
				}),
			);
			// The description alone is cached; the region wrapper is
			// recomposed per call (see the cache-hit path).
			cacheSet(cacheKey, { description: result.description, model: result.model });
			notifyAnalysisDone(ctx, `image described by ${result.model}`, started);
			debug(
				"tool described",
				target.fingerprint,
				region ? `region [${region.x},${region.y},${region.w},${region.h}]` : "",
				`${Date.now() - started}ms`,
			);

			return {
				content: [
					{ type: "text", text: region ? composeRegionResult(result.description) : result.description },
				],
				// usage from the nested call keeps session statistics accurate.
				usage: result.usage,
				details: {
					fingerprint: target.fingerprint,
					model: result.model,
					question: params.question,
					mode,
					region: region ? regionToArray(region) : undefined,
					cached: false,
				},
			};

			/** The result names the region (post-clamping) so the model can cite it. */
			function composeRegionResult(description: string): string {
				return composeRegionText({
					fingerprint: target!.fingerprint,
					dimensions: regionDims!,
					region: region!,
					original: regionOriginal,
					clamped: regionClamped,
					description,
				});
			}
		},
	});

	/**
	 * The comparison path: two images, ONE vision call, a diff-oriented
	 * reading. Visible to every model (like describe_video): even a vision
	 * model cannot diff two images it was shown in different turns without
	 * re-attaching them.
	 */
	pi.registerTool({
		name: COMPARE_TOOL,
		label: "Compare Images",
		description:
			"Compare two images from this conversation with a vision model and report what changed between them — " +
			"text, values, layout, state, color — plus what stayed the same, attributing every difference to the " +
			"right version. Pass `first` and `second` as the images' fingerprints (from their [Image <fingerprint>] " +
			"tags); omit both to compare the two most recent distinct images, or omit one to compare the most recent " +
			"other image. Pass `question` to focus the comparison. Use it after an edit, re-run, or re-render to " +
			"verify exactly what changed.",
		promptSnippet: "Compare two images from the conversation — find what changed between them.",
		parameters: Type.Object({
			first: Type.Optional(
				Type.String({
					description:
						"The fingerprint of the first image, from its [Image <fingerprint>] tag. Omit to use the most recent distinct image.",
				}),
			),
			second: Type.Optional(
				Type.String({
					description:
						"The fingerprint of the second image, from its [Image <fingerprint>] tag. Omit to use the most recent distinct image other than the first.",
				}),
			),
			question: Type.Optional(
				Type.String({
					description:
						'Focused question about the difference, e.g. "did the error banner disappear" or "which fields changed".',
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<ToolResult> {
			refreshConfig(ctx);
			if (!config.enabled) {
				return {
					content: [{ type: "text", text: "pi-vision-bridge is disabled. Images cannot be compared." }],
					details: { fingerprint: "", model: "", cached: false },
				};
			}

			const images = collectSessionImages(ctx);
			if (images.length === 0) {
				return {
					content: [
						{ type: "text", text: "No images are attached to this conversation. There is nothing to compare." },
					],
					details: { fingerprint: "", model: "", cached: false },
				};
			}

			const pair = resolveComparePair(images, params.first, params.second);
			if (!pair.ok) {
				return {
					content: [{ type: "text", text: pair.error }],
					details: { fingerprint: `${params.first ?? "?"}+${params.second ?? "?"}`, model: "", cached: false },
					isError: true,
				};
			}

			const candidates = getVisionCandidates(ctx);
			if (candidates.length === 0) {
				return {
					content: [
						{
							type: "text",
							text: NO_VISION_MODEL_MESSAGE,
						},
					],
					details: { fingerprint: pairKey(pair), model: "", cached: false },
				};
			}

			// Cache is keyed on the order-insensitive pair + question: the
			// same two images asked in either order is one analysis. Only the
			// model's description is cached; the pair mapping is recomposed
			// per call so a reversed hit still labels first/second correctly.
			const cacheKey = `cmp:${JSON.stringify([pair.first.fingerprint, pair.second.fingerprint].sort().concat(params.question ? [params.question] : []))}`;
			const hit = cacheGet(cacheKey);
			if (hit) {
				return {
					content: [{ type: "text", text: composeCompareText(pair.first.fingerprint, pair.second.fingerprint, hit.description) }],
					details: { fingerprint: pairKey(pair), model: hit.model, question: params.question, cached: true },
				};
			}

			const started = Date.now();
			const { result } = await runWithFallback(ctx, (model) =>
				analyzeImages(ctx, model, [pair.first.image, pair.second.image], {
					prompt: {
						systemPrompt: COMPARE_SYSTEM_PROMPT,
						userText: buildCompareUserText(pair.first.fingerprint, pair.second.fingerprint, params.question),
					},
					maxTokens: config.maxTokens,
					temperature: config.temperature,
				}),
			);
			cacheSet(cacheKey, { description: result.description, model: result.model });
			notifyAnalysisDone(ctx, `images compared by ${result.model}`, started);
			debug("compared", pairKey(pair), `${Date.now() - started}ms`);

			return {
				content: [{ type: "text", text: composeCompareText(pair.first.fingerprint, pair.second.fingerprint, result.description) }],
				// usage from the nested call keeps session statistics accurate.
				usage: result.usage,
				details: { fingerprint: pairKey(pair), model: result.model, question: params.question, cached: false },
			};

			function pairKey(p: { first: ImageRef; second: ImageRef }): string {
				return [p.first.fingerprint, p.second.fingerprint].sort().join("+");
			}
		},
	});


	/**
	 * The video path: sample frames with ffmpeg, describe them through the
	 * vision model. Useful for ALL models — pi has no native video input
	 * anywhere.
	 */
	pi.registerTool({
		name: VIDEO_TOOL,
		label: "Describe Video",
		description:
			"Watch a video file (screen recording, demo, clip) by sampling frames and describing them with a vision " +
			"model. Use it whenever the user references a video file — nothing in this session can view video natively. " +
			"Supported: mp4, mov, mkv, m4v, webm, avi. Pass `path` (absolute or relative to the working directory); " +
			"optionally `question` for a focused analysis, and `max_frames` to control sampling density (4-16). " +
			"Audio is not analyzed.",
		promptSnippet: "Analyze a video file the user references — you cannot watch video any other way.",
		parameters: Type.Object({
			path: Type.String({
				description: "Path to the video file (absolute, or relative to the working directory).",
			}),
			question: Type.Optional(
				Type.String({
					description:
						'Focused question about the video, e.g. "at what timestamp does the error dialog appear" or ' +
						'"describe the workflow being demonstrated step by step". Omit for a full description.',
				}),
			),
			max_frames: Type.Optional(
				Type.Number({
					description: "Frames to sample across the video (4-16). More frames = finer detail, more cost. Default 10.",
				}),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<ToolResult> {
			refreshConfig(ctx);
			if (!config.enabled) {
				return {
					content: [{ type: "text", text: "pi-vision-bridge is disabled. Videos cannot be examined." }],
					details: { fingerprint: "", model: "", cached: false, media: "video" },
				};
			}

			// Resolve and validate the file.
			const rawPath = params.path.trim().replace(/^["']|["']$/g, "");
			const filePath = isAbsolute(rawPath) ? rawPath : resolve(ctx.cwd, rawPath);
			const candidates0 = getVisionCandidates(ctx);
			if (candidates0.length === 0) {
				return {
					content: [
						{
							type: "text",
							text: NO_VISION_MODEL_MESSAGE,
						},
					],
					details: { fingerprint: "", model: "", cached: false, media: "video" },
				};
			}

			let stats;
			try {
				stats = statSync(filePath);
				if (!stats.isFile()) throw new Error("not a file");
			} catch {
				return {
					content: [
						{
							type: "text",
							text: `Video file not found: "${filePath}" (from "${params.path}"). Check the path and retry; the file must exist on disk.`,
						},
					],
					details: { fingerprint: "", model: "", cached: false, media: "video" },
					isError: true,
				};
			}

			// Probe duration / dimensions.
			let info;
			try {
				info = await probeVideo(filePath);
			} catch (err) {
				return {
					content: [
						{
							type: "text",
							text: `Could not read video: ${err instanceof Error ? err.message : String(err)}. ` +
								`Supported formats: mp4, mov, mkv, m4v, webm, avi. Verify the file is a valid video.`,
						},
					],
					details: { fingerprint: "", model: "", cached: false, media: "video" },
					isError: true,
				};
			}

			// Fingerprint (size + head/tail megabytes) for caching.
			const fp = readHeadTailFingerprint(stats.size, filePath);
			const cacheKey = `v:${fp}:${params.question ?? ""}:${Math.floor(params.max_frames ?? config.videoFrames)}`;
			const hit = cacheGet(cacheKey);
			if (hit) {
				return {
					content: [{ type: "text", text: hit.description }],
					details: { fingerprint: fp, model: hit.model, question: params.question, cached: true, media: "video" },
				};
			}

			// Analyze with frame extraction + candidate fallback.
			const started = Date.now();
			const { result } = await runWithFallback(ctx, (model) =>
				analyzeVideo(ctx, model, filePath, info, {
					maxFrames: params.max_frames ?? config.videoFrames,
					question: params.question,
					maxTokens: config.maxTokens,
					temperature: config.temperature,
				}),
			);
			cacheSet(cacheKey, { description: result.description, model: result.model });
			notifyAnalysisDone(ctx, `video analyzed by ${result.model} (${result.frames} frames)`, started);
			debug("video described", fp, `${Date.now() - started}ms`, `${result.frames} frames`);

			return {
				content: [{ type: "text", text: result.description }],
				usage: result.usage,
				details: {
					fingerprint: fp,
					model: result.model,
					question: params.question,
					cached: false,
					media: "video",
					frames: result.frames,
				},
			};

			/** Read head+tail of the file for the fingerprint. */
			function readHeadTailFingerprint(size: number, path: string): string {
				const CHUNK = 1024 * 1024;
				const head = Buffer.alloc(Math.min(CHUNK, size));
				const tail = Buffer.alloc(Math.min(CHUNK, Math.max(0, size - CHUNK)));
				const fd = openSync(path, "r");
				try {
					readSync(fd, head, 0, head.length, 0);
					if (tail.length > 0) readSync(fd, tail, 0, tail.length, Math.max(0, size - tail.length));
				} finally {
					closeSync(fd);
				}
				return videoFingerprint(size, head, tail);
			}
		},
	});

	/**
	 * Tool visibility: describe_image hides when the active model sees
	 * natively; describe_video and compare_images stay visible for every
	 * model (pi has no native video, and a vision model cannot diff two
	 * images it saw in different turns without re-attaching them).
	 */
	function syncToolVisibility(ctx: ExtensionContext): void {
		const model = ctx.model;
		// describe_image only makes sense when the active model cannot see;
		// the video and compare tools serve every model.
		const wants: Array<[string, boolean]> = [
			[IMAGE_TOOL, config.enabled && !!model && !model.input.includes("image")],
			[VIDEO_TOOL, config.enabled],
			[COMPARE_TOOL, config.enabled],
		];
		const active = pi.getActiveTools();
		if (wants.every(([name, want]) => active.includes(name) === want)) return;
		const next = [...active];
		for (const [name, want] of wants) {
			const has = next.includes(name);
			if (want && !has) next.push(name);
			else if (!want && has) next.splice(next.indexOf(name), 1);
		}
		debug("tool visibility", ...wants.map(([name, want]) => `${name}:${want ? "show" : "hide"}`));
		pi.setActiveTools(next);
	}

	pi.on("session_start", (_event, ctx) => {
		refreshConfig(ctx);
		syncToolVisibility(ctx);
	});

	pi.on("model_select", (_event, ctx) => {
		// Registry / active model may have changed; drop the memoized pick.
		visionMemo = undefined;
		notifiedNoVision = false;
		syncToolVisibility(ctx);
	});

	pi.registerCommand("visionbridge", {
		description: "Configure pi-vision-bridge (vision bridge for text-only models: images + video)",
		getArgumentCompletions: (prefix) => {
			const subs = ["status", "model", "auto", "on", "off", "cache"];
			const filtered = subs.filter((s) => s.startsWith(prefix));
			return filtered.length > 0 ? filtered.map((s) => ({ value: s, label: s })) : null;
		},
		handler: async (args, ctx) => {
			refreshConfig(ctx); // status reflects the current file + pi-settings section
			const parts = args.trim().split(/\s+/).filter(Boolean);
			const [sub, ...rest] = parts;

			switch (sub) {
				case undefined:
				case "status": {
					const model = ctx.model;
					const candidates = getVisionCandidates(ctx);
					const lines = [
						"pi-vision-bridge status",
						`enabled: ${config.enabled}`,
						`vision models (ordered, healthy): ${
							candidates.length > 0
								? candidates.slice(0, 3).map(candidateKey).join(", ")
								: config.visionModels
									? "configured but unavailable"
									: "none available"
						}`,
						`vision models config: ${
							config.visionModels ? config.visionModels.join(", ") : "auto (catalog order)"
						}${piSettingsActive ? " [from pi settings visionBridge section]" : ""}`,
						`active model: ${model ? `${model.provider}/${model.id} (${model.input.includes("image") ? "vision — images pass through, video still needs the tool" : "text-only — bridge active"})` : "none"}`,
						`tools: image ${pi.getActiveTools().includes(IMAGE_TOOL) ? "visible" : "hidden"}, video ${pi.getActiveTools().includes(VIDEO_TOOL) ? "visible" : "hidden"}, compare ${pi.getActiveTools().includes(COMPARE_TOOL) ? "visible" : "hidden"}`,
						`cache: ${cache.size} entr${cache.size === 1 ? "y" : "ies"} (cap ${config.cacheMax})`,
						`video frames: ${config.videoFrames}`,
						`config: ${CONFIG_PATH}`,
						"subcommands: model <provider/id> | auto | on | off | cache clear",
					];
					ctx.ui.notify(lines.join("\n"), "info");
					return;
				}

				case "model": {
					// One spec pins a single model (no fallback); several specs
					// (space or comma separated) give an ordered fallback list.
					const specs = rest
						.join(" ")
						.split(/[,\s]+/)
						.map((s) => s.trim())
						.filter(Boolean);
					if (specs.length === 0 || specs.some((s) => !s.includes("/") || s.indexOf("/") === 0 || s.endsWith("/"))) {
						ctx.ui.notify(
							"Usage: /visionbridge model provider/model-id [provider/model-id …] — one model pins it, several set an ordered fallback list.",
							"warning",
						);
						return;
					}
					const bad = specs.filter((spec) => !resolveVisionModel(ctx, spec));
					if (bad.length > 0) {
						ctx.ui.notify(
							`${bad.map((s) => `"${s}"`).join(", ")} ${bad.length === 1 ? "is" : "are"} not connected vision-capable model(s). Check /models or use provider/model-id values you can authenticate.`,
							"error",
						);
						return;
					}
					config = { ...config, visionModels: specs };
					const saved = saveConfig(config);
					visionMemo = undefined;
					syncToolVisibility(ctx);
					ctx.ui.notify(
						saved.ok
							? `Vision model${specs.length > 1 ? "s" : ""} set to ${specs.join(", ")}${specs.length > 1 ? " (ordered, with fallback)" : " (pinned, no fallback)"}.`
							: `Vision model${specs.length > 1 ? "s" : ""} set for this session (config write failed: ${saved.error}).` +
									(piSettingsActive ? " NOTE: a visionBridge section in pi settings overrides this file." : ""),
						saved.ok && !piSettingsActive ? "info" : "warning",
					);
					return;
				}

				case "auto": {
					config = { ...config, visionModels: null };
					const saved = saveConfig(config);
					visionMemo = undefined;
					const candidates = getVisionCandidates(ctx);
					ctx.ui.notify(
						saved.ok && candidates.length > 0
							? `Auto-selecting vision models. Current pick: ${candidateKey(candidates[0])}.`
							: "Auto-select enabled, but no connected vision model was found.",
						"info",
					);
					return;
				}

				case "on":
				case "off": {
					const enabled = sub === "on";
					config = { ...config, enabled };
					const saved = saveConfig(config);
					syncToolVisibility(ctx);
					ctx.ui.notify(
						saved.ok
							? `pi-vision-bridge ${enabled ? "enabled" : "disabled"}.`
							: `Set for this session (config write failed: ${saved.error}).`,
						"info",
					);
					return;
				}

				case "cache": {
					if (rest[0] === "clear") {
						const n = cache.size;
						cache.clear();
						pending.clear();
						ctx.ui.notify(`Cleared ${n} cache entr${n === 1 ? "y" : "ies"}.`, "info");
					} else {
						ctx.ui.notify("Usage: /visionbridge cache clear", "warning");
					}
					return;
				}

				default:
					ctx.ui.notify(`Unknown subcommand "${sub}". Use: status | model | auto | on | off | cache clear`, "warning");
			}
		},
	});
}
