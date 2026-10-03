/**
 * Video support for pi-vision-bridge: sample frames with ffmpeg and describe
 * them through any vision model.
 *
 * Why frames instead of native video input:
 *
 * - pi's model layer has no video content type (images only), so native
 *   video would require provider-specific raw HTTP calls with their own auth
 *   handling (z.ai's vision-mcp-server does this for GLM-4.6V's video_url
 *   API — one provider, one format). Frame extraction works with EVERY
 *   vision model through pi's own provider stack, auth included.
 * - Sampling research (multigrid.ai, ffmpeg-cookbook): uniform sampling with
 *   a frame cap and 768px scaling keeps vision tokens low while preserving
 *   chronological structure; timestamps paired with frames let the model
 *   cite moments (z.ai's video_analysis describes "scenes, moments, and
 *   entities" the same way).
 * - Audio is not analyzed (pi has no STT path); the prompt says so.
 */

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, rmSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ImageContent, Model } from "@earendil-works/pi-ai";
import { debug } from "./config.ts";

export const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi"]);
const FRAME_LONG_EDGE = 768;
const JPEG_QUALITY = 4;
const MIN_FRAMES = 4;
const MAX_FRAMES_HARD = 16;

export interface VideoInfo {
	durationSec: number;
	width: number;
	height: number;
}

export interface VideoAnalysis {
	description: string;
	model: string;
	frames: number;
	durationSec: number;
	usage?: import("@earendil-works/pi-ai").Usage;
}

const VIDEO_SYSTEM_PROMPT = `You are the vision stage of a coding assistant that cannot watch videos directly. Another language model — not a human — will read your output as its only view of the video.

You are given a chronological sequence of frames sampled from the video, each labeled with its timestamp. Produce one complete, standalone text description of the video. Priorities, in order:

1. ON-SCREEN TEXT: transcribe visible text verbatim in each frame where it changes — titles, captions, terminal output, code, UI labels, dialog text.
2. TIMELINE: describe the content scene-by-scene in chronological order, citing timestamps like (0:12). Note what changes between consecutive frames — actions, motion, UI state transitions, edits, dialog progress.
3. STRUCTURE: for screen recordings, describe the application, the workflow being demonstrated, and the visible UI. For camera footage, describe subjects, setting, and events.
4. VISUAL FACTS: salient colors, counts, notable elements, and anything unusual.

Rules:
- The frames are stills sampled at intervals — motion between frames is inferred, so mark inferred actions as such; never invent events.
- Audio is NOT available; do not speculate about sound or speech.
- Describe only what is visible. Plain text/markdown only, no preamble, no advice, no questions.`;

function runCommand(bin: string, args: string[], timeoutMs: number): Promise<{ code: number; stdout: string; stderr: string }> {
	return new Promise((resolve, reject) => {
		const child = spawn(bin, args, { windowsHide: true });
		let stdout = "";
		let stderr = "";
		const timer = setTimeout(() => {
			child.kill();
			reject(new Error(`${basename(bin)} timed out after ${timeoutMs / 1000}s`));
		}, timeoutMs);
		child.stdout.on("data", (d) => (stdout += d.toString()));
		child.stderr.on("data", (d) => (stderr += d.toString()));
		child.on("error", (err) => {
			clearTimeout(timer);
			reject(err);
		});
		child.on("close", (code) => {
			clearTimeout(timer);
			resolve({ code: code ?? -1, stdout, stderr });
		});
	});
}

/** Probe a video file with ffprobe. */
export async function probeVideo(filePath: string): Promise<VideoInfo> {
	const res = await runCommand(
		"ffprobe",
		[
			"-v", "error",
			"-select_streams", "v:0",
			"-show_entries", "stream=width,height:format=duration",
			"-of", "json",
			filePath,
		],
		30_000,
	);
	if (res.code !== 0) {
		throw new Error(`not a readable video file (ffprobe): ${res.stderr.trim().split("\n").slice(-1)[0] ?? ""}`);
	}
	const parsed = JSON.parse(res.stdout) as {
		streams?: Array<{ width?: number; height?: number }>;
		format?: { duration?: string };
	};
	const stream = parsed.streams?.[0];
	const duration = Number(parsed.format?.duration ?? 0);
	if (!stream?.width || !stream?.height || !Number.isFinite(duration) || duration <= 0) {
		throw new Error("no video stream found in the file");
	}
	return { durationSec: duration, width: stream.width, height: stream.height };
}

/**
 * Frame-count policy, shared by extraction and tests: bounded by the
 * caller's request, the hard cap, the model's per-message image limit, and
 * the duration (one frame per ~2s at most — short videos get fewer frames).
 */
export function computeFrameCount(desiredFrames: number, durationSec: number, perMessageLimit?: number): number {
	return Math.max(
		MIN_FRAMES,
		Math.min(
			perMessageLimit ?? MAX_FRAMES_HARD,
			Math.min(desiredFrames, MAX_FRAMES_HARD, Math.ceil(durationSec / 2)),
		),
	);
}

/**
 * Sample frames evenly across the video. Uniform sampling with a frame cap:
 * cheap, deterministic, and preserves chronology.
 */
async function extractFrames(
	filePath: string,
	info: VideoInfo,
	desiredFrames: number,
	perMessageLimit?: number,
): Promise<{ files: string[]; timestamps: number[] }> {
	const frames = computeFrameCount(desiredFrames, info.durationSec, perMessageLimit);
	const fps = frames / info.durationSec;

	const dir = mkdtempSync(join(tmpdir(), "pi-vision-bridge-frames-"));
	const outPattern = join(dir, "frame_%03d.jpg");
	try {
		const res = await runCommand(
			"ffmpeg",
			[
				"-hide_banner", "-loglevel", "error",
				"-y",
				"-i", filePath,
				"-vf", `fps=${fps.toFixed(6)},scale=${FRAME_LONG_EDGE}:-2`,
				"-frames:v", String(frames),
				"-q:v", String(JPEG_QUALITY),
				outPattern,
			],
			180_000,
		);
		if (res.code !== 0) {
			throw new Error(`frame extraction failed (ffmpeg): ${res.stderr.trim().split("\n").slice(-1)[0] ?? ""}`);
		}
	} catch (err) {
		rmSync(dir, { recursive: true, force: true });
		throw err;
	}

	const files = readdirSync(dir)
		.filter((f) => f.endsWith(".jpg"))
		.sort()
		.map((f) => join(dir, f));
	if (files.length === 0) {
		rmSync(dir, { recursive: true, force: true });
		throw new Error("no frames could be extracted from the video");
	}

	// Evenly spaced midpoints for the frames actually produced.
	const interval = info.durationSec / files.length;
	const timestamps = files.map((_, i) => Math.min(info.durationSec, (i + 0.5) * interval));
	return { files, timestamps };
}

function formatTimestamp(seconds: number): string {
	const m = Math.floor(seconds / 60);
	const s = Math.floor(seconds % 60);
	return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Fingerprint a video file: size + first/last megabyte (full-hash would be
 * slow for large files; this is stable and collision-resistant enough for
 * cache identity).
 */
export function videoFingerprint(size: number, head: Buffer, tail: Buffer): string {
	const hash = createHash("sha256");
	hash.update(String(size)).update(":");
	hash.update(head.subarray(0, 1024 * 1024));
	hash.update(":");
	hash.update(tail.subarray(0, 1024 * 1024));
	return hash.digest("hex").slice(0, 10);
}

/**
 * Describe a video: extract frames, send them as one multi-image vision call
 * (interleaved with timestamp labels), return the description.
 */
export async function analyzeVideo(
	ctx: ExtensionContext,
	model: Model<any>,
	filePath: string,
	info: VideoInfo,
	options: { maxFrames: number; question?: string; maxTokens?: number; temperature?: number },
): Promise<VideoAnalysis> {
	const perMessageLimit = model.inputLimits?.images?.maxPerMessage;
	const { files, timestamps } = await extractFrames(filePath, info, options.maxFrames, perMessageLimit);
	const frameDir = dirname(files[0]);
	try {
		const content: Array<ImageContent | { type: "text"; text: string }> = [];
		for (let i = 0; i < files.length; i++) {
			if (ctx.signal?.aborted) throw new Error("aborted");
			const data = await readFile(files[i]);
			content.push({ type: "image", data: data.toString("base64"), mimeType: "image/jpeg" });
			content.push({ type: "text", text: `[frame ${i + 1}/${files.length} at ${formatTimestamp(timestamps[i])}]` });
		}
		content.push({
			type: "text",
			text: options.question
				? `These frames are from a video (${info.durationSec.toFixed(1)}s, sampled evenly). Answer this question about it, citing frame timestamps:\n\n${options.question}`
				: `These frames are from a video (${info.durationSec.toFixed(1)}s, sampled evenly). Describe the video.`,
		});

		debug("vision call", model.provider, model.id, `video ${files.length} frames${perMessageLimit ? ` (limit ${perMessageLimit})` : ""}`);

		const response = await ctx.modelRegistry.complete(
			model,
			{ systemPrompt: VIDEO_SYSTEM_PROMPT, messages: [{ role: "user", content, timestamp: Date.now() }] },
			{
				signal: ctx.signal,
				maxTokens: options.maxTokens,
				temperature: options.temperature,
				cacheRetention: "none",
				...(model.reasoning ? { reasoning: "minimal" as const } : {}),
			},
		);

		if (response.stopReason === "aborted") throw new Error("aborted");
		if (response.stopReason === "error") {
			throw new Error(response.errorMessage || "vision model call failed");
		}
		const description = response.content
			.filter((c): c is { type: "text"; text: string } => c.type === "text")
			.map((c) => c.text)
			.join("\n")
			.trim();
		if (!description) throw new Error("vision model returned no text");

		return { description, model: `${model.provider}/${model.id}`, frames: files.length, durationSec: info.durationSec, usage: response.usage };
	} finally {
		rmSync(frameDir, { recursive: true, force: true });
	}
}
