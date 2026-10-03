/**
 * Vision-model resolution and image analysis for pi-image-bridge.
 *
 * Design notes (from researching opencode-image-vision, opencode-image-proxy,
 * opencode-see-image, pi-vlm-proxy, pi-vision-watcher and vision-prompting
 * write-ups):
 *
 * - Nested calls go through ctx.modelRegistry.complete(): provider-neutral,
 *   reuses pi's auth and streaming stack, and returns usage.
 * - The helper call receives ONLY the image and a fixed prompt. It never gets
 *   tools, files, or bash — the vision model processes attacker-influenceable
 *   content, so it must not be able to execute anything (the pi-native
 *   equivalent of opencode-see-image issue #6).
 * - The generic description is factual, OCR-heavy, and standalone so it stays
 *   cache-coherent across turns. Targeted questions go through the
 *   describe_image tool instead.
 * - Model selection returns a RANKED candidate list, not a single pick: relay
 *   availability (the /models listing) says nothing about upstream health, so
 *   failures rotate to the next candidate and mark the failed one unhealthy
 *   for a cooldown window (opencode-see-image's route-fallback pattern).
 */

import { createHash } from "node:crypto";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ImageContent, Message, Model, Usage } from "@earendil-works/pi-ai";
import { debug } from "./config.ts";

export interface VisionCandidate {
	model: Model<any>;
	source: "configured" | "env" | "auto";
	rank: number;
}

export interface VisionAnalysis {
	description: string;
	model: string;
	usage?: Usage;
}

/** How long a failed model stays out of rotation. */
export const HEALTH_COOLDOWN_MS = 5 * 60_000;
/** How many candidates one analysis may try before giving up. */
export const MAX_ATTEMPTS = 3;

/** Short stable id for an image, shown to the model and matched by the tool. */
export function fingerprint(image: ImageContent): string {
	return createHash("sha256").update(image.mimeType).update(":").update(image.data).digest("hex").slice(0, 10);
}

const SYSTEM_PROMPT = `You are the vision stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

Produce one complete, standalone text description of the image. Priorities, in order:

1. TEXT FIRST: transcribe ALL visible text verbatim — error messages, code, terminal output, labels, buttons, headings, table cells, axis labels. Preserve line breaks, indentation, and layout order. Text accuracy matters more than anything else. Mark any character you are unsure about with "‹?›" instead of guessing.
2. STRUCTURE: if it is a screenshot or UI, describe the window/application, regions (header, sidebar, main content, dialogs), the key components, their states (selected, disabled, loading, error), and the overall layout top-to-bottom.
3. DATA: if it is a diagram or chart, describe nodes, labeled arrows and their direction, axes, series, units, and the important values.
4. VISUAL FACTS: salient colors, counts, sizes, and positions of notable elements.

Rules:
- Describe only what is visible. Explicitly mark inferences as uncertain; never invent text or details.
- Plain text/markdown only. No preamble like "This image shows", no advice, no questions, no concluding remarks.
- Be thorough but do not pad: every sentence must carry information about the image.`;

const GENERIC_USER_PROMPT = "Describe this image.";

function buildQuestionPrompt(question: string): string {
	return `Answer this question about the image, with the transcription precision of an OCR pass for anything textual it touches:\n\n${question}`;
}

/**
 * Resolve the RANKED vision-model candidates.
 *
 * Priority: env/config "provider/model-id" (single candidate, no fallback —
 * the user asked for that model) → auto-ranked list among the session's
 * available, authenticated, image-capable models.
 *
 * Auto scoring (lower = better): each cheap-tier name match (flash, mini,
 * nano, haiku, lite, turbo, …) subtracts; vision families with strong OCR
 * track records subtract; reasoning models and expensive models add.
 *
 * The ranked pool is diversified by id-prefix (the first id segment — for
 * relay-style providers that segment is the upstream route): at most two
 * candidates per prefix, so one dead route cannot occupy the whole pool.
 */
export function rankVisionModels(ctx: ExtensionContext, configured: string | null): VisionCandidate[] {
	const registry = ctx.modelRegistry;

	if (configured) {
		const slash = configured.indexOf("/");
		if (slash <= 0 || slash === configured.length - 1) return [];
		const model = registry.find(configured.slice(0, slash), configured.slice(slash + 1));
		if (!model || !model.input.includes("image") || !registry.hasConfiguredAuth(model)) return [];
		return [{ model, source: process.env.PI_IMAGE_BRIDGE_MODEL ? "env" : "configured", rank: 0 }];
	}

	const candidates = registry
		.getAvailable()
		.filter((m) => m.input.includes("image") && registry.hasConfiguredAuth(m));

	const scored = candidates
		.map((model) => ({ model, score: autoPickScore(model) }))
		.sort((a, b) => (a.score - b.score) || a.model.id.localeCompare(b.model.id));

	// Diversify: at most two per id prefix (upstream route), then fill up.
	const pool: typeof scored = [];
	const perPrefix = new Map<string, number>();
	const MAX_PER_PREFIX = 2;
	const POOL_SIZE = 10;
	for (const entry of scored) {
		const prefix = entry.model.id.split("/")[0] ?? entry.model.id;
		const count = perPrefix.get(prefix) ?? 0;
		if (count >= MAX_PER_PREFIX) continue;
		perPrefix.set(prefix, count + 1);
		pool.push(entry);
		if (pool.length >= POOL_SIZE) break;
	}
	// Top-up if diversity cut the pool short.
	for (const entry of scored) {
		if (pool.length >= POOL_SIZE) break;
		if (!pool.includes(entry)) pool.push(entry);
	}

	if (process.env.PI_IMAGE_BRIDGE_DEBUG === "1" && pool.length > 0) {
		debug("vision candidates:", pool.map((s) => `${s.model.id}(${s.score})`).join(", "));
	}

	return pool.map((s, i) => ({ model: s.model, source: "auto" as const, rank: i }));
}

const CHEAP_TIERS = /(^|[^a-z])(flash|mini|nano|haiku|lite|instant|turbo|fast|spark|air)([^a-z]|$)/g;
const VISION_FAMILIES = /(gemini|gpt-4\.1|gpt-5|qwen|glm|vl|pixtral|minimax|moondream|llama|internvl|step)/i;

function autoPickScore(m: Model<any>): number {
	let score = 0;
	const name = `${m.provider}/${m.id} ${m.name ?? ""}`.toLowerCase();

	// Reasoning models burn time and tokens on thinking for a caption task.
	if (m.reasoning) score += 2;

	// Every cheap/fast tier word in the name is a signal (flash-lite beats flash).
	const tiers = name.match(CHEAP_TIERS);
	if (tiers) score -= 3 * Math.min(tiers.length, 2);

	// Vision families with good OCR track records.
	if (VISION_FAMILIES.test(name)) score -= 1;

	// Price heuristics ($ per million input tokens).
	const input = m.cost?.input ?? 1;
	if (input === 0) score -= 2;
	else if (input <= 1) score -= 1;
	else if (input >= 10) score += 2;

	return score;
}

/** Should a failed analysis be retried against a different candidate? */
export function isRetryableFailure(error: unknown): boolean {
	if (!(error instanceof Error)) return true;
	const message = error.message;
	if (message === "aborted") return false;
	const statusMatch = /^(\d{3})\b/.exec(message);
	if (statusMatch) {
		const status = Number(statusMatch[1]);
		return status >= 500 || status === 429 || status === 408;
	}
	// Network-ish failures: connection, timeout, fetch, socket, stream.
	return /(fetch|network|timeout|timed?\s?out|econn|socket|hang|stream|closed|reset|unavailable)/i.test(message);
}

/**
 * Run one image (optionally with a focused question) through a vision model.
 * Uses a plain no-tools model call: the image and the fixed system prompt only.
 */
export async function analyzeImage(
	ctx: ExtensionContext,
	model: Model<any>,
	image: ImageContent,
	options?: { question?: string; maxTokens?: number; temperature?: number },
): Promise<VisionAnalysis> {
	const messages: Message[] = [
		{
			role: "user",
			content: [
				{ type: "image", data: image.data, mimeType: image.mimeType },
				{
					type: "text",
					text: options?.question ? buildQuestionPrompt(options.question) : GENERIC_USER_PROMPT,
				},
			],
			timestamp: Date.now(),
		},
	];

	debug("vision call", model.provider, model.id, `question=${options?.question ? "yes" : "generic"}`);

	const response = await ctx.modelRegistry.complete(
		model,
		{ systemPrompt: SYSTEM_PROMPT, messages },
		{
			signal: ctx.signal,
			maxTokens: options?.maxTokens,
			temperature: options?.temperature,
			cacheRetention: "none",
			// Caption calls do not benefit from thinking; skip it when possible.
			...(model.reasoning ? { reasoning: "minimal" as const } : {}),
		},
	);

	if (response.stopReason === "aborted") {
		throw new Error("aborted");
	}
	if (response.stopReason === "error") {
		throw new Error(response.errorMessage || "vision model call failed");
	}

	const description = response.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text")
		.map((c) => c.text)
		.join("\n")
		.trim();

	if (!description) {
		throw new Error("vision model returned no text");
	}

	return { description, model: `${model.provider}/${model.id}`, usage: response.usage };
}
