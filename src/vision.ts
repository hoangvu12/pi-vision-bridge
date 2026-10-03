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
import { buildQuestionPrompt, GENERIC_SYSTEM_PROMPT, GENERIC_USER_PROMPT } from "./prompts.ts";
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

/** Prompt spec for a nested analysis call. Unspecified fields fall back to the generic image-analysis prompt. */
export interface AnalysisPromptSpec {
	/** System prompt override. */
	systemPrompt?: string;
	/** User text override. Wins over `question`. */
	userText?: string;
}

export interface AnalyzeOptions {
	/** Injectable prompt spec; reaches the model call verbatim. */
	prompt?: AnalysisPromptSpec;
	/** Focused question; used when no prompt.userText override is given. */
	question?: string;
	maxTokens?: number;
	temperature?: number;
}

/**
 * Resolve one "provider/model-id" spec to a connected, image-capable model.
 * Returns undefined when the spec is malformed, unknown, or not vision-capable.
 */
export function resolveVisionModel(ctx: ExtensionContext, spec: string): Model<any> | undefined {
	const slash = spec.indexOf("/");
	if (slash <= 0 || slash === spec.length - 1) return undefined;
	const model = ctx.modelRegistry.find(spec.slice(0, slash), spec.slice(slash + 1));
	if (!model || !model.input.includes("image") || !ctx.modelRegistry.hasConfiguredAuth(model)) return undefined;
	return model;
}

/**
 * Resolve the ordered vision-model candidates.
 *
 * - Configured list: the user's order is law. Candidates are returned in
 *   exactly that order (one entry = a pinned model with no fallback);
 *   unresolvable entries are skipped with a debug line, not an error.
 * - Auto (null): the registry's image-capable, authenticated models in
 *   catalog order. No quality guessing \u2014 a name says nothing a test has
 *   verified, so none is scored. The pool is only diversified by id-prefix
 *   (the first id segment is the upstream route for relay-style providers):
 *   at most two candidates per prefix so one dead route cannot occupy the
 *   whole pool. That is availability insurance, not a ranking.
 *
 * Either way, health rotation (benching, sticky winner) applies on top.
 */
export function rankVisionModels(ctx: ExtensionContext, configured: string[] | null): VisionCandidate[] {
	const registry = ctx.modelRegistry;

	if (configured && configured.length > 0) {
		const fromEnv = !!process.env.PI_VISION_BRIDGE_MODEL;
		const candidates: VisionCandidate[] = [];
		for (const spec of configured) {
			const model = resolveVisionModel(ctx, spec);
			if (model) {
				candidates.push({ model, source: fromEnv ? "env" : "configured", rank: candidates.length });
			} else {
				debug("configured vision model not found or not image-capable, skipping:", spec);
			}
		}
		return candidates;
	}

	const models = registry
		.getAvailable()
		.filter((m) => m.input.includes("image") && registry.hasConfiguredAuth(m));

	// Diversify: at most two per id prefix (upstream route), then fill up.
	const pool: typeof models = [];
	const perPrefix = new Map<string, number>();
	const MAX_PER_PREFIX = 2;
	const POOL_SIZE = 10;
	for (const model of models) {
		const prefix = model.id.split("/")[0] ?? model.id;
		const count = perPrefix.get(prefix) ?? 0;
		if (count >= MAX_PER_PREFIX) continue;
		perPrefix.set(prefix, count + 1);
		pool.push(model);
		if (pool.length >= POOL_SIZE) break;
	}
	// Top-up if diversity cut the pool short.
	for (const model of models) {
		if (pool.length >= POOL_SIZE) break;
		if (!pool.includes(model)) pool.push(model);
	}

	if (process.env.PI_IMAGE_BRIDGE_DEBUG === "1" && pool.length > 0) {
		debug("vision candidates:", pool.map((m) => m.id).join(", "));
	}

	return pool.map((model, i) => ({ model, source: "auto" as const, rank: i }));
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
 * Run one or many images through a vision model in a single model request,
 * with an injectable prompt spec (system prompt + user text). Without an
 * override this is today's generic image analysis (question or generic).
 * Uses a plain no-tools model call: the images and a fixed prompt only.
 */
export async function analyzeImages(
	ctx: ExtensionContext,
	model: Model<any>,
	images: ImageContent[],
	options?: AnalyzeOptions,
): Promise<VisionAnalysis> {
	const userText =
		options?.prompt?.userText ?? (options?.question ? buildQuestionPrompt(options.question) : GENERIC_USER_PROMPT);

	const content: Array<ImageContent | { type: "text"; text: string }> = images.map((image) => ({
		type: "image",
		data: image.data,
		mimeType: image.mimeType,
	}));
	content.push({ type: "text", text: userText });
	const messages: Message[] = [
		{
			role: "user",
			content,
			timestamp: Date.now(),
		},
	];

	debug("vision call", model.provider, model.id, `${images.length} image${images.length === 1 ? "" : "s"}`, options?.prompt ? "prompt=custom" : options?.question ? "question=yes" : "generic");

	const response = await ctx.modelRegistry.complete(
		model,
		{ systemPrompt: options?.prompt?.systemPrompt ?? GENERIC_SYSTEM_PROMPT, messages },
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

/** Analyze one image — the one-image convenience form of analyzeImages. */
export function analyzeImage(
	ctx: ExtensionContext,
	model: Model<any>,
	image: ImageContent,
	options?: AnalyzeOptions,
): Promise<VisionAnalysis> {
	return analyzeImages(ctx, model, [image], options);
}
