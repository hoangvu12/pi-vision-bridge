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
 *     the focused re-look ("quote the exact error text").
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
 *  7. Vision model auto-selection from the session's connected models
 *     (flash/nano/haiku tier preferred), as a RANKED candidate list with
 *     health tracking: failed models are benched for a cooldown and the
 *     analysis rotates to the next candidate (a relay listing a model says
 *     nothing about its upstream actually serving it). Override with
 *     /visionbridge model, config file, or env var.
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
	describeConfig,
	isEnvDisabled,
	loadConfig,
	saveConfig,
} from "../src/config.ts";
import {
	analyzeImage,
	fingerprint,
	HEALTH_COOLDOWN_MS,
	isRetryableFailure,
	MAX_ATTEMPTS,
	rankVisionModels,
	type VisionAnalysis,
	type VisionCandidate,
} from "../src/vision.ts";
import { buildSwapText, IMAGE_TOOL } from "../src/prompts.ts";
import { analyzeVideo, probeVideo, VIDEO_EXTENSIONS, videoFingerprint } from "../src/video.ts";

const VIDEO_TOOL = "describe_video";
const VIDEO_HINT_SECTION = "vision_bridge_video";

interface CacheEntry {
	description: string;
	model: string;
}

type ToolResult = AgentToolResult<ToolDetails>;

interface ToolDetails {
	fingerprint: string;
	model: string;
	question?: string;
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
	let config = isEnvDisabled() ? { ...loadConfig(), enabled: false } : loadConfig();

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

	function refreshConfig(): void {
		config = isEnvDisabled() ? { ...loadConfig(), enabled: false } : loadConfig();
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
		const modelKey = config.visionModel ?? "";
		if (visionMemo && now - visionMemo.at < 30_000 && visionMemo.key === modelKey) {
			return applyHealth(visionMemo.candidates);
		}
		const candidates = rankVisionModels(ctx, config.visionModel);
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

	/** Collect every image present in request messages, in order, deduped by hash. */
	function collectImages(messages: AgentMessage[]): { refs: Map<string, ImageRef>; order: string[] } {
		const refs = new Map<string, ImageRef>();
		const order: string[] = [];
		for (const msg of messages) {
			if (msg.role !== "user" && msg.role !== "toolResult") continue;
			if (!Array.isArray(msg.content)) continue;
			for (const block of msg.content) {
				if (block.type !== "image") continue;
				const fp = fingerprint(block);
				if (!refs.has(fp)) {
					refs.set(fp, {
						image: block,
						fingerprint: fp,
						messageRole: msg.role,
						toolName: msg.role === "toolResult" ? (msg as ToolResultMessage).toolName : undefined,
					});
					order.push(fp);
				}
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
				msg.content[i] = {
					type: "text",
					text: buildSwapText({
						fingerprint: fp,
						description,
						origin: msg.role === "toolResult" ? ` (from ${msg.toolName ?? "tool"} output)` : undefined,
						describedBy: model && model !== "unavailable" ? model : undefined,
						dimensions: undefined, // filled in by the dimensions ticket
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
			"omit it for a full fresh description.",
		promptSnippet: "Re-inspect an image with a focused question when its inline description is not enough.",
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
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx): Promise<ToolResult> {
			refreshConfig();
			if (!config.enabled) {
				return {
					content: [{ type: "text", text: "pi-vision-bridge is disabled. Images cannot be examined." }],
					details: { fingerprint: params.fingerprint ?? "", model: "", cached: false },
				};
			}

			// Gather images from the session branch (persisted history).
			const images: ImageRef[] = [];
			for (const entry of ctx.sessionManager.getBranch()) {
				if (entry.type !== "message") continue;
				const msg = (entry as { message?: AgentMessage }).message;
				if (!msg || (msg.role !== "user" && msg.role !== "toolResult")) continue;
				if (!Array.isArray(msg.content)) continue;
				for (const block of msg.content) {
					if (block.type === "image") {
						images.push({
							image: block,
							fingerprint: fingerprint(block),
							messageRole: msg.role,
							toolName: msg.role === "toolResult" ? (msg as ToolResultMessage).toolName : undefined,
						});
					}
				}
			}

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
				const wanted = params.fingerprint.trim().toLowerCase();
				target = images.find((ref) => ref.fingerprint === wanted || ref.fingerprint.startsWith(wanted));
				if (!target) {
					const available = images
						.map((ref) => `  ${ref.fingerprint} (${ref.messageRole}${ref.toolName ? ` from ${ref.toolName}` : ""})`)
						.join("\n");
					return {
						content: [
							{
								type: "text",
								text: `No image with fingerprint "${params.fingerprint}". Images in this conversation:\n${available}\nRetry with one of these exact fingerprints.`,
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
							text: "No vision-capable model is currently available (every connected candidate failed recently, likely an upstream outage). Tell the user to retry later or connect another vision model.",
						},
					],
					details: { fingerprint: target.fingerprint, model: "", cached: false },
				};
			}

			// Targeted answers are cached per (image, question) — the common
			// retry loop (model re-asking the same thing) stays free.
			const cacheKey = `q:${target.fingerprint}:${params.question ?? ""}`;
			const hit = cacheGet(cacheKey);
			if (hit) {
				return {
					content: [{ type: "text", text: hit.description }],
					details: { fingerprint: target.fingerprint, model: hit.model, question: params.question, cached: true },
				};
			}

			const started = Date.now();
			const { result } = await runWithFallback(ctx, (model) =>
				analyzeImage(ctx, model, target!.image, {
					question: params.question,
					maxTokens: config.maxTokens,
					temperature: config.temperature,
				}),
			);
			cacheSet(cacheKey, { description: result.description, model: result.model });
			debug("tool described", target.fingerprint, `${Date.now() - started}ms`);

			return {
				content: [{ type: "text", text: result.description }],
				// usage from the nested call keeps session statistics accurate.
				usage: result.usage,
				details: { fingerprint: target.fingerprint, model: result.model, question: params.question, cached: false },
			};
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
			refreshConfig();
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
							text: "No vision-capable model is currently available (every connected candidate failed recently). Tell the user to retry later or connect another vision model.",
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
	 * natively; describe_video stays visible for every model (pi has no
	 * native video anywhere).
	 */
	function syncToolVisibility(ctx: ExtensionContext): void {
		const model = ctx.model;
		const wantImage = config.enabled && !!model && !model.input.includes("image");
		const wantVideo = config.enabled;
		const active = pi.getActiveTools();
		const hasImage = active.includes(IMAGE_TOOL);
		const hasVideo = active.includes(VIDEO_TOOL);
		if (wantImage === hasImage && wantVideo === hasVideo) return;
		let next = [...active];
		next = wantImage ? (hasImage ? next : [...next, IMAGE_TOOL]) : next.filter((n) => n !== IMAGE_TOOL);
		next = wantVideo ? (hasVideo ? next : [...next, VIDEO_TOOL]) : next.filter((n) => n !== VIDEO_TOOL);
		debug("tool visibility", `image:${wantImage ? "show" : "hide"}`, `video:${wantVideo ? "show" : "hide"}`);
		pi.setActiveTools(next);
	}

	pi.on("session_start", (_event, ctx) => {
		refreshConfig();
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
						`vision models (ranked, healthy): ${
							candidates.length > 0
								? candidates.slice(0, 3).map(candidateKey).join(", ")
								: config.visionModel
									? "configured but unavailable"
									: "none available"
						}`,
						`active model: ${model ? `${model.provider}/${model.id} (${model.input.includes("image") ? "vision — images pass through, video still needs the tool" : "text-only — bridge active"})` : "none"}`,
						`tools: image ${pi.getActiveTools().includes(IMAGE_TOOL) ? "visible" : "hidden"}, video ${pi.getActiveTools().includes(VIDEO_TOOL) ? "visible" : "hidden"}`,
						`cache: ${cache.size} entr${cache.size === 1 ? "y" : "ies"} (cap ${config.cacheMax})`,
						`video frames: ${config.videoFrames}`,
						`config: ${CONFIG_PATH}`,
						"subcommands: model <provider/id> | auto | on | off | cache clear",
					];
					ctx.ui.notify(lines.join("\n"), "info");
					return;
				}

				case "model": {
					const spec = rest.join(" ").trim();
					if (!spec || !spec.includes("/") || spec.indexOf("/") === 0 || spec.endsWith("/")) {
						ctx.ui.notify("Usage: /visionbridge model provider/model-id (e.g. google/gemini-2.5-flash)", "warning");
						return;
					}
					const probe = rankVisionModels(ctx, spec);
					if (probe.length === 0) {
						ctx.ui.notify(
							`"${spec}" is not a connected vision-capable model. Check /models or use a provider/model-id you can authenticate.`,
							"error",
						);
						return;
					}
					config = { ...config, visionModel: spec };
					const saved = saveConfig(config);
					visionMemo = undefined;
					syncToolVisibility(ctx);
					ctx.ui.notify(
						saved.ok
							? `Vision model set to ${spec}.`
							: `Vision model set to ${spec} for this session (config write failed: ${saved.error}).`,
						saved.ok ? "info" : "warning",
					);
					return;
				}

				case "auto": {
					config = { ...config, visionModel: null };
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
