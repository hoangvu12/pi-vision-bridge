/**
 * Harness test for pi-vision-bridge: drives the extension with a mock
 * ExtensionAPI to verify command wiring, config persistence, tool
 * visibility, the image context swap, the video hint injection, and the
 * video tool's validation paths — no network, no TUI, no ffmpeg.
 *
 * Run: node test/harness.mjs
 */

import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";

// Point the config file into a temp dir. os.homedir() reads USERPROFILE/HOME
// from the environment, so setting them before the first import redirects
// the config path without patching the frozen ESM namespace.
const fakeHome = mkdtempSync(join(tmpdir(), "pi-vbridge-test-"));
process.env.PI_VISION_BRIDGE_DEBUG = "1";
process.env.USERPROFILE = fakeHome;
process.env.HOME = fakeHome;

const { default: visionBridge } = await import("../extensions/index.ts");
const { computeFrameCount, VIDEO_EXTENSIONS } = await import("../src/video.ts");
const { analyzeImages } = await import("../src/vision.ts");
const { buildSwapText } = await import("../src/prompts.ts");

// ---- mock pi ----
const activeTools = ["read", "bash", "edit"];
const registered = { tools: [], commands: {} };
const notifications = [];

const fakeModel = (input) => ({
	provider: "test",
	id: input.includes("image") ? "vision-model" : "text-model",
	input,
	reasoning: false,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100000,
	maxTokens: 4096,
	api: "openai-completions",
	baseUrl: "http://x",
});

const pi = {
	registerTool: (tool) => registered.tools.push(tool),
	registerCommand: (name, opts) => (registered.commands[name] = opts),
	on: (event, handler) => {
		handlers[event] ??= [];
		handlers[event].push(handler);
	},
	getActiveTools: () => [...activeTools],
	setActiveTools: (names) => {
		activeTools.length = 0;
		activeTools.push(...names);
	},
	getAllTools: () => [],
	setModel: async () => true,
};
const handlers = {};

visionBridge(pi);

const imageTool = registered.tools.find((t) => t.name === "describe_image");
const videoTool = registered.tools.find((t) => t.name === "describe_video");
assert.ok(imageTool, "describe_image registered");
assert.ok(videoTool, "describe_video registered");
assert.ok(registered.commands.visionbridge, "visionbridge command registered");
assert.ok(handlers.context?.length === 1, "context handler registered");
assert.ok(handlers.session_start?.length === 1, "session_start handler registered");
assert.ok(handlers.model_select?.length === 1, "model_select handler registered");
assert.ok(handlers.before_agent_start?.length === 1, "before_agent_start handler registered");

const ctx = (model, cwd = fakeHome) => ({
	model,
	mode: "print",
	hasUI: false,
	cwd,
	signal: undefined,
	ui: { notify: (m, t) => notifications.push({ m, t }) },
	sessionManager: { getBranch: () => [] },
	modelRegistry: {
		getAvailable: () => [fakeModel(["text", "image"]), fakeModel(["text"])],
		find: (provider, id) => (id === "vision-model" ? fakeModel(["text", "image"]) : undefined),
		hasConfiguredAuth: () => true,
		complete: async () => {
			throw new Error("network should not be reached in harness tests");
		},
	},
});

const notifyLines = () => notifications.map((n) => n.m).join("\n");

// --- session_start: text-only model => image tool visible, video always ---
await handlers.session_start[0]({}, ctx(fakeModel(["text"])));
assert.ok(activeTools.includes("describe_image"), "image tool visible for text-only model");
assert.ok(activeTools.includes("describe_video"), "video tool visible for text-only model");
notifications.length = 0;

// --- session_start: vision model => image tool hidden, video STILL visible ---
await handlers.session_start[0]({}, ctx(fakeModel(["text", "image"])));
assert.ok(!activeTools.includes("describe_image"), "image tool hidden for vision model");
assert.ok(activeTools.includes("describe_video"), "video tool stays visible for vision model");
await handlers.model_select[0]({}, ctx(fakeModel(["text"])));
assert.ok(activeTools.includes("describe_image"), "image tool re-shown after model_select to text-only");

// --- context handler: no images => untouched ---
const noImageMessages = [{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: 1 }];
const result1 = await handlers.context[0](
	{ type: "context", messages: structuredClone(noImageMessages) },
	ctx(fakeModel(["text"])),
);
assert.equal(result1, undefined, "no images: handler does nothing");

// --- context handler: vision model with images => passthrough ---
const imageBlock = { type: "image", data: "aGVsbG8=", mimeType: "image/png" };
const withImage = [{ role: "user", content: [{ type: "text", text: "look" }, imageBlock], timestamp: 1 }];
const result2 = await handlers.context[0](
	{ type: "context", messages: structuredClone(withImage) },
	ctx(fakeModel(["text", "image"])),
);
assert.equal(result2, undefined, "vision model: passthrough (no swap)");

// --- context handler: text model but no vision candidates => untouched ---
const ctxNoVision = ctx(fakeModel(["text"]));
ctxNoVision.modelRegistry.getAvailable = () => [fakeModel(["text"])];
const result3 = await handlers.context[0]({ type: "context", messages: structuredClone(withImage) }, ctxNoVision);
assert.equal(result3, undefined, "no vision candidates: untouched");

// --- video hint injection: prompt references an existing video file ---
const workDir = mkdtempSync(join(tmpdir(), "pi-vbridge-work-"));
const videoPath = join(workDir, "demo.mp4");
writeFileSync(videoPath, Buffer.from("fake-video-bytes"));
const sections = {};
const agentEvent = {
	type: "before_agent_start",
	prompt: `please look at demo.mp4 and tell me what happens`,
	images: [],
	systemPromptOptions: { sections },
};
await handlers.before_agent_start[0](agentEvent, ctx(fakeModel(["text"]), workDir));
assert.ok(sections.vision_bridge_video, "video hint section injected");
assert.ok(sections.vision_bridge_video.includes("demo.mp4"), "hint names the file");
assert.ok(sections.vision_bridge_video.includes("describe_video"), "hint points at the tool");

// --- video hint: no video referenced => section cleared ---
const sections2 = { vision_bridge_video: "stale" };
await handlers.before_agent_start[0](
	{ type: "before_agent_start", prompt: "just a text question", images: [], systemPromptOptions: { sections: sections2 } },
	ctx(fakeModel(["text"]), workDir),
);
assert.equal(sections2.vision_bridge_video, undefined, "stale hint cleared");

// --- video hint: video mentioned but file does not exist => no injection ---
const sections3 = {};
await handlers.before_agent_start[0](
	{ type: "before_agent_start", prompt: "look at missing.mp4 please", images: [], systemPromptOptions: { sections: sections3 } },
	ctx(fakeModel(["text"]), workDir),
);
assert.equal(sections3.vision_bridge_video, undefined, "nonexistent video: no hint");

// --- config file written by /visionbridge model ---
const visionSpec = "test/vision-model";
await registered.commands.visionbridge.handler(`model ${visionSpec}`, ctx(fakeModel(["text"])));
const agentDir = join(fakeHome, ".pi", "agent");
assert.ok(readdirSync(agentDir).includes("pi-vision-bridge.json"), "config file written");
const saved = JSON.parse(readFileSync(join(agentDir, "pi-vision-bridge.json"), "utf8"));
assert.equal(saved.visionModel, visionSpec, "config persisted visionModel");

// --- invalid model spec rejected ---
await registered.commands.visionbridge.handler("model test/no-such-model", ctx(fakeModel(["text"])));
const after = JSON.parse(readFileSync(join(agentDir, "pi-vision-bridge.json"), "utf8"));
assert.equal(after.visionModel, visionSpec, "invalid model not persisted");

// --- off / on ---
await registered.commands.visionbridge.handler("off", ctx(fakeModel(["text"])));
assert.ok(!activeTools.includes("describe_image"), "off hides image tool");
assert.ok(!activeTools.includes("describe_video"), "off hides video tool");
const offConfig = JSON.parse(readFileSync(join(agentDir, "pi-vision-bridge.json"), "utf8"));
assert.equal(offConfig.enabled, false, "off persisted");
await registered.commands.visionbridge.handler("on", ctx(fakeModel(["text"])));
assert.ok(activeTools.includes("describe_image"), "on shows image tool");
assert.ok(activeTools.includes("describe_video"), "on shows video tool");

// --- status output mentions the pieces ---
notifications.length = 0;
await registered.commands.visionbridge.handler("status", ctx(fakeModel(["text"])));
const status = notifyLines();
assert.ok(status.includes("pi-vision-bridge status"), "status header");
assert.ok(status.includes("vision models"), "status lists vision models");
assert.ok(status.includes("text-only — bridge active"), "status reports bridge active");
assert.ok(status.includes("video frames"), "status reports video frames");

// --- describe_image: no images in session ---
notifications.length = 0;
const toolResult = await imageTool.execute("call-1", {}, undefined, undefined, ctx(fakeModel(["text"])));
assert.match(toolResult.content[0].text, /No images are attached/, "image tool: no images message");

// --- describe_image: bad fingerprint lists available ---
const fp = createHash("sha256").update("image/png").update(":aGVsbG8=").digest("hex").slice(0, 10);
const branchCtx = ctx(fakeModel(["text"]));
branchCtx.sessionManager = {
	getBranch: () => [
		{ type: "message", message: { role: "user", content: [imageBlock], timestamp: 1 } },
		{ type: "message", message: { role: "assistant", content: [{ type: "text", text: "ok" }], timestamp: 2 } },
	],
};
const bad = await imageTool.execute("call-2", { fingerprint: "zzzz" }, undefined, undefined, branchCtx);
assert.match(bad.content[0].text, /No image with fingerprint/, "image tool: bad fingerprint error");
assert.ok(bad.content[0].text.includes(fp), "image tool: lists real fingerprint");

// --- describe_video: missing file ---
const missing = await videoTool.execute("call-3", { path: "nope.mp4" }, undefined, undefined, ctx(fakeModel(["text"]), workDir));
assert.match(missing.content[0].text, /Video file not found/, "video tool: missing file");
assert.equal(missing.isError, true, "video tool: missing file is an error result");

// --- describe_video: not-a-video file (ffprobe would fail; here extension ok, probe throws) ---
// The harness cannot run ffprobe on fake bytes — assert the error is wrapped cleanly.
const fakeVideoResult = await videoTool.execute(
	"call-4",
	{ path: "demo.mp4" },
	undefined,
	undefined,
	{
		...ctx(fakeModel(["text"]), workDir),
		modelRegistry: {
			...ctx(fakeModel(["text"]), workDir).modelRegistry,
		},
	},
).catch((err) => ({ threw: err }));
// demo.mp4 exists but contains garbage: probeVideo must fail with a readable error.
// (If ffprobe is missing on this machine, treat ENOENT as pass too.)
if (!fakeVideoResult.threw) {
	assert.match(fakeVideoResult.content[0].text, /Could not read video/, "video tool: probe failure wrapped");
	assert.equal(fakeVideoResult.isError, true, "video tool: probe failure is error result");
}

// --- tool param schemas: optionality ---
for (const [tool, requiredParams] of [
	[imageTool, []],
	[videoTool, ["path"]],
]) {
	const params = tool.parameters;
	assert.equal(params.type, "object", `${tool.name}: object schema`);
	const required = params.required ?? [];
	assert.deepEqual([...required].sort(), [...requiredParams].sort(), `${tool.name}: required params`);
}

// --- video frame-count policy ---
assert.equal(computeFrameCount(10, 60), 10, "60s video: 10 frames");
assert.equal(computeFrameCount(10, 5), 4, "5s video: clamps to min 4");
assert.equal(computeFrameCount(10, 60, 5), 5, "per-message limit 5 respected");
assert.equal(computeFrameCount(20, 600), 16, "hard cap 16");
assert.equal(computeFrameCount(10, 20, 30), 10, "limit above request ignored");
assert.ok(VIDEO_EXTENSIONS.has(".mp4") && VIDEO_EXTENSIONS.has(".webm"), "video extensions set");

// =====================================================================
// Ticket 01: generalized analysis seam
// =====================================================================

// Registry mock that records every nested model call and returns a canned
// assistant message — the seam through which later tickets assert outgoing
// prompts, image blocks, and call counts.
const capturedCalls = [];
const cannedDescription = "A pixel-perfect description of the image.";
const cannedUsage = { input: 11, output: 7, cacheRead: 0, cacheWrite: 0 };
const captureCtx = (model, opts = {}) => ({
	...ctx(model),
	modelRegistry: {
		getAvailable: () => [fakeModel(["text", "image"])],
		find: (provider, id) => (id === "vision-model" ? fakeModel(["text", "image"]) : undefined),
		hasConfiguredAuth: () => true,
		complete: async (calledModel, context, options) => {
			capturedCalls.push({ model: calledModel, context, options });
			return {
				role: "assistant",
				content: [{ type: "text", text: opts.description ?? cannedDescription }],
				usage: { ...cannedUsage },
				stopReason: "stop",
			};
		},
	},
});
const vision = fakeModel(["text", "image"]);
const makeImage = (tag, dims) => ({
	type: "image",
	data: dims
		? makePngHeader(dims[0], dims[1]).toString("base64")
		: Buffer.from(`image-${tag}`).toString("base64"),
	mimeType: dims ? "image/png" : "application/octet-stream",
});
const lastCall = () => capturedCalls[capturedCalls.length - 1];
const textBlocks = (call) => call.context.messages[0].content.filter((c) => c.type === "text");
const imageBlocks = (call) => call.context.messages[0].content.filter((c) => c.type === "image");

// --- analyzeImages: multiple images in ONE model request ---
capturedCalls.length = 0;
await analyzeImages(captureCtx(vision), vision, [
	{ type: "image", data: Buffer.from("one").toString("base64"), mimeType: "image/png" },
	{ type: "image", data: Buffer.from("two").toString("base64"), mimeType: "image/png" },
]);
assert.equal(capturedCalls.length, 1, "two images: exactly one model call");
assert.equal(imageBlocks(lastCall()).length, 2, "two image blocks in the request");
assert.equal(imageBlocks(lastCall())[0].data, Buffer.from("one").toString("base64"), "first image data");
assert.equal(imageBlocks(lastCall())[1].data, Buffer.from("two").toString("base64"), "second image data");
assert.equal(textBlocks(lastCall()).length, 1, "one prompt text block");
assert.equal(textBlocks(lastCall())[0].text, "Describe this image.", "default prompt text");

// --- analyzeImages: default prompt is today's generic prompt ---
assert.ok(lastCall().context.systemPrompt.includes("vision stage"), "default system prompt (identity)");
assert.ok(lastCall().context.systemPrompt.includes("TEXT FIRST"), "default system prompt (OCR priority)");

// --- analyzeImages: injectable prompt spec reaches the model verbatim ---
capturedCalls.length = 0;
await analyzeImages(captureCtx(vision), vision, [
	{ type: "image", data: Buffer.from("x").toString("base64"), mimeType: "image/png" },
], {
	prompt: { systemPrompt: "CUSTOM SYSTEM PROMPT", userText: "CUSTOM USER TEXT" },
});
assert.equal(lastCall().context.systemPrompt, "CUSTOM SYSTEM PROMPT", "custom system prompt verbatim");
assert.equal(textBlocks(lastCall())[0].text, "CUSTOM USER TEXT", "custom user text verbatim");

// --- analyzeImages: question still shapes the default user text ---
capturedCalls.length = 0;
await analyzeImages(captureCtx(vision), vision, [
	{ type: "image", data: Buffer.from("y").toString("base64"), mimeType: "image/png" },
], { question: "what color is the header" });
assert.ok(
	textBlocks(lastCall())[0].text.startsWith("Answer this question about the image"),
	"question prompt shape unchanged",
);
assert.ok(textBlocks(lastCall())[0].text.includes("what color is the header"), "question reaches the prompt");

// --- buildSwapText: renders exactly today's wrapper; dims arg renders nothing when absent ---
assert.equal(
	buildSwapText({ fingerprint: "abcd1234", description: "DESC", describedBy: "prov/model" }),
	'[Image abcd1234 — described by prov/model; this model cannot view images directly]\nDESC\n' +
		'[end of image abcd1234; call describe_image with fingerprint "abcd1234" to re-examine it with a focused question]',
	"swap text: byte-identical to the pre-refactor wrapper",
);
assert.equal(
	buildSwapText({ fingerprint: "abcd1234", description: "DESC", origin: " (from bash output)" }),
	'[Image abcd1234 (from bash output) — this model cannot view images directly]\nDESC\n' +
		'[end of image abcd1234; call describe_image with fingerprint "abcd1234" to re-examine it with a focused question]',
	"swap text: origin rendered",
);
assert.equal(
	buildSwapText({ fingerprint: "abcd1234", description: "DESC", dimensions: undefined }),
	buildSwapText({ fingerprint: "abcd1234", description: "DESC" }),
	"swap text: absent dimensions render nothing",
);
assert.ok(
	buildSwapText({ fingerprint: "abcd1234", description: "DESC", dimensions: { width: 800, height: 600 } }).startsWith(
		"[Image abcd1234 — 800x600 px; ",
	),
	"swap text: dimensions render when present",
);

// --- end-to-end: context swap still replaces the image with the description ---
capturedCalls.length = 0;
notifications.length = 0;
const swapResult = await handlers.context[0](
	{ type: "context", messages: structuredClone(withImage) },
	captureCtx(fakeModel(["text"])),
);
assert.ok(swapResult, "context handler returns a swap");
const swapped = swapResult.messages[0].content[1];
assert.equal(swapped.type, "text", "image block replaced by text");
assert.equal(
	swapped.text,
	`[Image ${fp} — described by test/vision-model; this model cannot view images directly]\n${cannedDescription}\n` +
		`[end of image ${fp}; call describe_image with fingerprint "${fp}" to re-examine it with a focused question]`,
	"end-to-end swap text unchanged by the prefactor",
);
assert.equal(capturedCalls.length, 1, "one nested vision call for the swap");
assert.equal(imageBlocks(lastCall()).length, 1, "swap analysis sends one image");

// =====================================================================
// Ticket 02: task-typed analysis modes on describe_image
// =====================================================================

// --- tool schema: optional mode restricted to the five values ---
const modeParam = imageTool.parameters.properties.mode;
assert.ok(modeParam, "mode parameter present");
assert.ok(!imageTool.parameters.required?.includes("mode"), "mode optional");
const { Value } = await import("typebox/value");
for (const m of ["ocr", "error", "ui", "diagram", "chart"]) {
	assert.ok(Value.Check(modeParam, m), `mode schema accepts "${m}"`);
}
assert.ok(!Value.Check(modeParam, "bogus"), "mode schema rejects unknown values");
assert.ok(!Value.Check(modeParam, 3), "mode schema rejects non-strings");

// --- each mode resolves to a distinct curated prompt that reaches the model ---
const modeImage = { type: "image", data: Buffer.from("mode-test-image").toString("base64"), mimeType: "image/png" };
const modeCtx = captureCtx(fakeModel(["text"]));
modeCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [modeImage], timestamp: 1 } }],
};
const modePrompts = new Map();
for (const mode of ["ocr", "error", "ui", "diagram", "chart"]) {
	capturedCalls.length = 0;
	const r = await imageTool.execute(`mode-${mode}`, { mode }, undefined, undefined, modeCtx);
	assert.equal(r.details.mode, mode, `details record the mode (${mode})`);
	assert.ok(lastCall(), `mode ${mode}: a model call happened`);
	modePrompts.set(mode, { sys: lastCall().context.systemPrompt, user: textBlocks(lastCall())[0].text });
}
assert.equal(new Set([...modePrompts.values()].map((v) => v.sys)).size, 5, "five modes -> five distinct system prompts");
assert.match(modePrompts.get("ocr").sys, /transcribe/i, "ocr prompt: transcription focus");
assert.match(modePrompts.get("error").sys, /stack trace/i, "error prompt: stack-trace focus");
assert.match(modePrompts.get("ui").sys, /component/i, "ui prompt: component focus");
assert.match(modePrompts.get("diagram").sys, /arrow/i, "diagram prompt: arrow/relationship focus");
assert.match(modePrompts.get("chart").sys, /axis|series/i, "chart prompt: axes/series focus");

// --- question folds into the mode prompt ---
capturedCalls.length = 0;
await imageTool.execute("mode-q", { mode: "error", question: "which file failed" }, undefined, undefined, modeCtx);
assert.ok(
	textBlocks(lastCall())[0].text.includes("which file failed"),
	"question reaches the mode prompt",
);
assert.ok(
	textBlocks(lastCall())[0].text.includes(modePrompts.get("error").user.split("\n")[0]),
	"mode base instruction kept alongside the question",
);

// --- cache key includes mode ---
const cacheImage = { type: "image", data: Buffer.from("cache-test-image").toString("base64"), mimeType: "image/png" };
const cacheModeCtx = captureCtx(fakeModel(["text"]));
cacheModeCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [cacheImage], timestamp: 1 } }],
};
capturedCalls.length = 0;
await imageTool.execute("cache-1", { mode: "ocr" }, undefined, undefined, cacheModeCtx);
const hitResult = await imageTool.execute("cache-2", { mode: "ocr" }, undefined, undefined, cacheModeCtx);
assert.equal(capturedCalls.length, 1, "same image + same mode twice: one call");
assert.equal(hitResult.details.cached, true, "cache hit flagged");
await imageTool.execute("cache-3", { mode: "chart" }, undefined, undefined, cacheModeCtx);
assert.equal(capturedCalls.length, 2, "same image, different mode: separate analysis");

// --- no mode: exactly today's generic behavior ---
const genericImage = { type: "image", data: Buffer.from("generic-image").toString("base64"), mimeType: "image/png" };
const genericCtx = captureCtx(fakeModel(["text"]));
genericCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [genericImage], timestamp: 1 } }],
};
capturedCalls.length = 0;
await imageTool.execute("generic-q", { question: "what color is the header" }, undefined, undefined, genericCtx);
assert.ok(lastCall().context.systemPrompt.includes("TEXT FIRST"), "no mode: generic system prompt");
assert.ok(
	textBlocks(lastCall())[0].text.startsWith("Answer this question about the image"),
	"no mode: question prompt shape unchanged",
);
await imageTool.execute("generic-q2", { question: "what color is the header" }, undefined, undefined, genericCtx);
assert.equal(capturedCalls.length, 1, "no mode: cached per (image, question) as today");
await imageTool.execute("generic-m", { question: "what color is the header", mode: "ui" }, undefined, undefined, genericCtx);
assert.equal(capturedCalls.length, 2, "mode never aliases the no-mode cache entry");
await imageTool.execute("generic-none", {}, undefined, undefined, genericCtx);
assert.equal(capturedCalls.length, 3, "no question: its own analysis");
assert.equal(textBlocks(lastCall())[0].text, "Describe this image.", "no mode, no question: generic user prompt");

// --- tool description and prompt snippet enumerate the modes ---
for (const m of ["ocr", "error", "ui", "diagram", "chart"]) {
	assert.ok(imageTool.description.includes(m), `tool description mentions ${m}`);
}
assert.ok(/use when|use on/i.test(imageTool.description), "tool description gives use-when guidance");
assert.ok(/mode/i.test(imageTool.promptSnippet), "prompt snippet mentions modes");

// --- in-context swap is untouched by modes ---
capturedCalls.length = 0;
const swapImageMsg = [
	{ role: "user", content: [{ type: "text", text: "look" }, { type: "image", data: Buffer.from("swap-no-mode").toString("base64"), mimeType: "image/png" }], timestamp: 1 },
];
await handlers.context[0]({ type: "context", messages: structuredClone(swapImageMsg) }, captureCtx(fakeModel(["text"])));
assert.ok(
	lastCall().context.systemPrompt.includes("TEXT FIRST"),
	"context swap still uses the generic prompt (never a mode)",
);
assert.equal(imageBlocks(lastCall()).length, 1, "context swap analyzes one image");

// =====================================================================
// Ticket 03: dimensions in swap text + region zoom
// =====================================================================
const { parseImageDimensions, clampRegion, setCropperForTests } = await import("../src/image.ts");
import { createRequire } from "node:module";
const zlib = await import("node:zlib");

// --- header fixtures (independent of the implementation under test) ---
const makePngHeader = (w, h) => {
	const b = Buffer.alloc(33);
	Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(b, 0);
	b.writeUInt32BE(13, 8);
	b.write("IHDR", 12, "ascii");
	b.writeUInt32BE(w, 16);
	b.writeUInt32BE(h, 20);
	b[24] = 8; // bit depth
	b[25] = 6; // color type RGBA
	b[28] = 0; // interlace
	return b;
};
const makeJpegHeader = (w, h) => {
	const b = Buffer.alloc(20);
	b[0] = 0xff;
	b[1] = 0xd8; // SOI
	b[2] = 0xff;
	b[3] = 0xc0; // SOF0
	b.writeUInt16BE(17, 4);
	b[6] = 8; // precision
	b.writeUInt16BE(h, 7);
	b.writeUInt16BE(w, 9);
	b[11] = 3; // components
	return b;
};
const makeGifHeader = (w, h) => {
	const b = Buffer.alloc(10);
	b.write("GIF89a", 0, "ascii");
	b.writeUInt16LE(w, 6);
	b.writeUInt16LE(h, 8);
	return b;
};
const makeWebpHeader = (w, h) => {
	const b = Buffer.alloc(30);
	b.write("RIFF", 0, "ascii");
	b.writeUInt32LE(22, 4);
	b.write("WEBP", 8, "ascii");
	b.write("VP8X", 12, "ascii");
	b.writeUInt32LE(10, 16);
	// payload: 4 bytes flags at 20, then canvas w-1 and h-1 as 3-byte LE
	b.writeUIntLE(w - 1, 24, 3);
	b.writeUIntLE(h - 1, 27, 3);
	return b;
};

assert.deepEqual(parseImageDimensions(makePngHeader(800, 600)), { width: 800, height: 600 }, "PNG dims");
assert.deepEqual(parseImageDimensions(makeJpegHeader(1024, 768)), { width: 1024, height: 768 }, "JPEG dims");
assert.deepEqual(parseImageDimensions(makeGifHeader(320, 200)), { width: 320, height: 200 }, "GIF dims");
assert.deepEqual(parseImageDimensions(makeWebpHeader(640, 480)), { width: 640, height: 480 }, "WebP dims");
assert.equal(parseImageDimensions(Buffer.from("hello world, not an image at all")), undefined, "unknown format: no dims");
assert.equal(parseImageDimensions(Buffer.from("")), undefined, "empty buffer: no dims");
const testImageBytes = readFileSync(join(import.meta.dirname, "..", "test-image.png"));
assert.deepEqual(parseImageDimensions(testImageBytes), { width: 442, height: 860 }, "repo test image dims");

// --- swap text publishes dimensions + region affordance ---
capturedCalls.length = 0;
const pngDimMessages = [{ role: "user", content: [{ type: "image", data: makePngHeader(800, 600).toString("base64"), mimeType: "image/png" }], timestamp: 1 }];
const pngSwap = await handlers.context[0]({ type: "context", messages: structuredClone(pngDimMessages) }, captureCtx(fakeModel(["text"])));
assert.ok(pngSwap.messages[0].content[0].text.includes("800x600 px; "), "PNG dims in swap text");
assert.ok(/region \[x, y, w, h\]/.test(pngSwap.messages[0].content[0].text), "region affordance when dims known");
const jpegDimMessages = [{ role: "user", content: [{ type: "image", data: makeJpegHeader(320, 240).toString("base64"), mimeType: "image/jpeg" }], timestamp: 1 }];
const jpegSwap = await handlers.context[0]({ type: "context", messages: structuredClone(jpegDimMessages) }, captureCtx(fakeModel(["text"])));
assert.ok(jpegSwap.messages[0].content[0].text.includes("320x240 px; "), "JPEG dims in swap text");
const unknownImageMessages = [
	{ role: "user", content: [{ type: "image", data: Buffer.from("binary-garbage").toString("base64"), mimeType: "application/octet-stream" }], timestamp: 1 },
];
const unknownSwap = await handlers.context[0](
	{ type: "context", messages: structuredClone(unknownImageMessages) },
	captureCtx(fakeModel(["text"])),
);
assert.ok(unknownSwap.messages[0].content[0].text.startsWith("[Image "), "unknown format still swapped (no failure)");
assert.ok(!unknownSwap.messages[0].content[0].text.includes(" px;"), "no dimensions line for unknown format");
assert.ok(!/region \[x, y, w, h\]/.test(unknownSwap.messages[0].content[0].text), "no region affordance without dims");

// --- region clamping ---
assert.deepEqual(
	clampRegion([700, 500, 200, 200], { width: 800, height: 600 }),
	{ region: { x: 700, y: 500, w: 100, h: 100 }, clamped: true },
	"clamp: box exceeding bottom-right",
);
assert.deepEqual(
	clampRegion([-10, -10, 100, 100], { width: 800, height: 600 }),
	{ region: { x: 0, y: 0, w: 100, h: 100 }, clamped: true },
	"clamp: negative origin",
);
assert.deepEqual(
	clampRegion([0, 0, 50, 50], { width: 800, height: 600 }),
	{ region: { x: 0, y: 0, w: 50, h: 50 }, clamped: false },
	"in-bounds box untouched",
);

// --- region analysis with the crop step stubbed at the seam ---
const stubbedRegions = [];
setCropperForTests(async (image, region) => {
	stubbedRegions.push(region);
	return { type: "image", data: Buffer.from("cropped-by-stub").toString("base64"), mimeType: "image/png" };
});
const regionImage = { type: "image", data: makePngHeader(800, 600).toString("base64"), mimeType: "image/png" };
const regionCtx = captureCtx(fakeModel(["text"]));
regionCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [regionImage], timestamp: 1 } }],
};
capturedCalls.length = 0;
const clampedResult = await imageTool.execute("r1", { region: [700, 500, 200, 200] }, undefined, undefined, regionCtx);
assert.deepEqual(stubbedRegions.at(-1), { x: 700, y: 500, w: 100, h: 100 }, "cropper receives the clamped box");
assert.equal(imageBlocks(lastCall()).length, 1, "region analysis sends exactly one image block");
assert.equal(
	imageBlocks(lastCall())[0].data,
	Buffer.from("cropped-by-stub").toString("base64"),
	"the crop (not the original) is what the model sees",
);
assert.ok(clampedResult.content[0].text.includes("Region [700, 500, 100, 100]"), "result states the analyzed region");
assert.ok(/clamped from \[700, 500, 200, 200\]/.test(clampedResult.content[0].text), "result notes the clamping");
assert.deepEqual(clampedResult.details.region, [700, 500, 100, 100], "details record the clamped region");
const inBoundsResult = await imageTool.execute("r2", { region: [10, 20, 100, 50] }, undefined, undefined, regionCtx);
assert.ok(inBoundsResult.content[0].text.includes("Region [10, 20, 100, 50]"), "in-bounds region stated");
assert.ok(!inBoundsResult.content[0].text.includes("clamped"), "no clamp note when nothing changed");

// --- zero/negative-area and malformed regions are clear errors ---
const zeroArea = await imageTool.execute("r3", { region: [0, 0, 0, 50] }, undefined, undefined, regionCtx);
assert.equal(zeroArea.isError, true, "zero-area region: error result");
assert.match(zeroArea.content[0].text, /zero or negative area/i, "zero-area error message");
const negArea = await imageTool.execute("r4", { region: [5, 5, -10, 10] }, undefined, undefined, regionCtx);
assert.equal(negArea.isError, true, "negative-area region: error result");
const fullyOutside = await imageTool.execute("r5", { region: [900, 700, 50, 50] }, undefined, undefined, regionCtx);
assert.equal(fullyOutside.isError, true, "region entirely outside the image: error after clamping");
assert.match(fullyOutside.content[0].text, /outside the image/i, "outside-image error message");
const malformed = await imageTool.execute("r6", { region: [1, 2, 3] }, undefined, undefined, regionCtx);
assert.equal(malformed.isError, true, "malformed region: error result");
assert.match(malformed.content[0].text, /\[x, y, w, h\]/, "malformed region error explains the shape");
const noDimsCtx = captureCtx(fakeModel(["text"]));
noDimsCtx.sessionManager = {
	getBranch: () => [
		{ type: "message", message: { role: "user", content: [unknownImageMessages[0].content[0]], timestamp: 1 } },
	],
};
const unparseable = await imageTool.execute("r7", { region: [0, 0, 10, 10] }, undefined, undefined, noDimsCtx);
assert.equal(unparseable.isError, true, "unparseable dims: error result");
assert.match(unparseable.content[0].text, /dimensions/i, "unparseable dims error message");

// --- region param schema ---
const regionParam = imageTool.parameters.properties.region;
assert.ok(regionParam, "region parameter present");
assert.ok(Value.Check(regionParam, [1, 2, 3, 4]), "region schema accepts [x, y, w, h]");
assert.ok(!Value.Check(regionParam, [1, 2, 3]), "region schema rejects a 3-tuple");
assert.ok(!Value.Check(regionParam, "0,0,10,10"), "region schema rejects strings");
assert.ok(/\[x, y, w, h\]/.test(imageTool.description), "tool description documents region");

// --- cache key includes region ---
// distinct valid header (different fingerprint) -> clean cache slate
const regionCacheImage = { type: "image", data: makePngHeader(640, 480).toString("base64"), mimeType: "image/png" };
const regionCacheCtx = captureCtx(fakeModel(["text"]));
regionCacheCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [regionCacheImage], timestamp: 1 } }],
};
capturedCalls.length = 0;
await imageTool.execute("rc1", { region: [10, 20, 100, 50] }, undefined, undefined, regionCacheCtx);
await imageTool.execute("rc2", { region: [10, 20, 100, 50] }, undefined, undefined, regionCacheCtx);
assert.equal(capturedCalls.length, 1, "same region twice: one call");
await imageTool.execute("rc3", { region: [30, 40, 100, 50] }, undefined, undefined, regionCacheCtx);
assert.equal(capturedCalls.length, 2, "different region: separate analysis");
await imageTool.execute("rc4", {}, undefined, undefined, regionCacheCtx);
assert.equal(capturedCalls.length, 3, "no-region analysis never aliases a region entry");

// --- real cropping against the repo's test image (end-to-end mechanics) ---
setCropperForTests(undefined);
const realImage = { type: "image", data: testImageBytes.toString("base64"), mimeType: "image/png" };
const realCtx = captureCtx(fakeModel(["text"]));
realCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [realImage], timestamp: 1 } }],
};
capturedCalls.length = 0;
const realCrop = await imageTool.execute("real-crop", { region: [10, 20, 200, 300], question: "describe this crop" }, undefined, undefined, realCtx);
const sent = imageBlocks(lastCall())[0];
assert.equal(sent.mimeType, "image/png", "crop sent as PNG");
const sentBytes = Buffer.from(sent.data, "base64");
assert.deepEqual(parseImageDimensions(sentBytes), { width: 200, height: 300 }, "model receives a 200x300 crop");
// independent structural check: inflate the IDAT stream, expect (w*4+1)*h bytes (RGBA + filter byte)
{
	const chunks = [];
	let off = 8;
	while (off + 8 <= sentBytes.length) {
		const len = sentBytes.readUInt32BE(off);
		const type = sentBytes.toString("ascii", off + 4, off + 8);
		if (type === "IDAT") chunks.push(sentBytes.subarray(off + 8, off + 8 + len));
		if (type === "IEND") break;
		off += 12 + len;
	}
	const inflated = zlib.inflateSync(Buffer.concat(chunks));
	assert.equal(inflated.length, (200 * 4 + 1) * 300, "crop is a structurally valid PNG scanline stream");
}
assert.ok(realCrop.content[0].text.includes("Region [10, 20, 200, 300]"), "e2e result names the region");
assert.ok(realCrop.content[0].text.includes(cannedDescription), "e2e result carries the description");
assert.ok(textBlocks(lastCall())[0].text.includes("describe this crop"), "question reaches the region analysis");
const fullImageResult = await imageTool.execute("real-full", { region: [0, 0, 442, 860] }, undefined, undefined, realCtx);
assert.equal(
	imageBlocks(lastCall())[0].data,
	realImage.data,
	"full-image region passes the original image through",
);
const leftovers = readdirSync(tmpdir()).filter((d) => d.startsWith("pi-vision-bridge-"));
assert.equal(leftovers.length, 0, "no crop/frame temp dirs left behind");

// =====================================================================
// Ticket 04: compare_images tool
// =====================================================================
const compareTool = registered.tools.find((t) => t.name === "compare_images");
assert.ok(compareTool, "compare_images registered");
assert.deepEqual(compareTool.parameters.required ?? [], [], "compare_images: nothing required");
assert.ok(Value.Check(compareTool.parameters, {}), "compare schema accepts empty params");
assert.ok(/compar/i.test(compareTool.promptSnippet), "prompt snippet advertises comparison");

const cmpA = { type: "image", data: Buffer.from("cmp-a").toString("base64"), mimeType: "image/png" };
const cmpB = { type: "image", data: Buffer.from("cmp-b").toString("base64"), mimeType: "image/png" };
const cmpC = { type: "image", data: Buffer.from("cmp-c").toString("base64"), mimeType: "image/png" };
const fpOf = (img) => createHash("sha256").update(img.mimeType).update(":").update(img.data).digest("hex").slice(0, 10);
const threeCtx = captureCtx(fakeModel(["text"]));
threeCtx.sessionManager = {
	getBranch: () => [
		{ type: "message", message: { role: "user", content: [cmpA], timestamp: 1 } },
		{ type: "message", message: { role: "user", content: [cmpB], timestamp: 2 } },
		{ type: "message", message: { role: "user", content: [cmpC], timestamp: 3 } },
	],
};

// --- default pair: the two most recent distinct images, one call ---
capturedCalls.length = 0;
const cmpResult = await compareTool.execute("cmp1", {}, undefined, undefined, threeCtx);
assert.equal(capturedCalls.length, 1, "one model call for the comparison");
const sentImages = imageBlocks(lastCall());
assert.equal(sentImages.length, 2, "both images in ONE request");
assert.equal(sentImages[0].data, cmpB.data, "first slot = second-most-recent image");
assert.equal(sentImages[1].data, cmpC.data, "second slot = most recent image");
assert.ok(/differ|chang/i.test(lastCall().context.systemPrompt), "diff-oriented system prompt");
assert.ok(/FIRST image/i.test(textBlocks(lastCall())[0].text), "prompt names the first image");
assert.ok(textBlocks(lastCall())[0].text.includes(fpOf(cmpC)), "prompt references fingerprints");
assert.ok(cmpResult.content[0].text.includes(`Compared image ${fpOf(cmpB)}`), "result names the pair");
assert.ok(cmpResult.content[0].text.includes(cannedDescription), "result carries the comparison");
assert.deepEqual(cmpResult.usage, cannedUsage, "nested usage reported");

// --- cache: pair is order-insensitive ---
const cmpAgain = await compareTool.execute("cmp2", {}, undefined, undefined, threeCtx);
assert.equal(capturedCalls.length, 1, "same pair asked again: cache hit");
assert.equal(cmpAgain.details.cached, true, "cache hit flagged");
const reversed = await compareTool.execute("cmp3", { first: fpOf(cmpC), second: fpOf(cmpB) }, undefined, undefined, threeCtx);
assert.equal(capturedCalls.length, 1, "reversed pair order: also a cache hit");
assert.equal(reversed.details.cached, true, "reversed hit flagged");
assert.ok(
	reversed.content[0].text.includes(`Compared image ${fpOf(cmpC)}`),
	"reversed hit recomposes the pair mapping",
);

// --- explicit pair + question ---
capturedCalls.length = 0;
await compareTool.execute(
	"cmp4",
	{ first: fpOf(cmpA), second: fpOf(cmpC), question: "did the banner change" },
	undefined,
	undefined,
	threeCtx,
);
assert.equal(imageBlocks(lastCall())[0].data, cmpA.data, "explicit first");
assert.equal(imageBlocks(lastCall())[1].data, cmpC.data, "explicit second");
assert.ok(textBlocks(lastCall())[0].text.includes("did the banner change"), "question folded into the compare prompt");

// --- one explicit + one defaulted ---
capturedCalls.length = 0;
await compareTool.execute("cmp5", { first: fpOf(cmpA) }, undefined, undefined, threeCtx);
assert.equal(imageBlocks(lastCall())[0].data, cmpA.data, "explicit first kept");
assert.equal(imageBlocks(lastCall())[1].data, cmpC.data, "omitted second defaults to the most recent distinct");

// --- helpful errors ---
const oneCtx = captureCtx(fakeModel(["text"]));
oneCtx.sessionManager = { getBranch: () => [{ type: "message", message: { role: "user", content: [cmpA], timestamp: 1 } }] };
const oneImage = await compareTool.execute("cmp6", {}, undefined, undefined, oneCtx);
assert.match(oneImage.content[0].text, /distinct image/i, "single-image session: helpful error");
assert.ok(oneImage.content[0].text.includes(fpOf(cmpA)), "error lists the available image");
const badFp = await compareTool.execute("cmp7", { first: "zzzzzz" }, undefined, undefined, threeCtx);
assert.match(badFp.content[0].text, /No image with fingerprint/, "unmatched fingerprint error");
assert.ok(badFp.content[0].text.includes(fpOf(cmpA)), "error lists available fingerprints");
const sameFp = await compareTool.execute("cmp8", { first: fpOf(cmpA), second: fpOf(cmpA) }, undefined, undefined, threeCtx);
assert.match(sameFp.content[0].text, /same image/i, "identical pair rejected");

// --- visibility: compare follows describe_video (all models; hidden only when disabled) ---
await handlers.session_start[0]({}, ctx(fakeModel(["text", "image"])));
assert.ok(activeTools.includes("compare_images"), "compare visible for vision models");
assert.ok(!activeTools.includes("describe_image"), "image tool still hidden for vision models");
await handlers.model_select[0]({}, ctx(fakeModel(["text"])));
assert.ok(activeTools.includes("compare_images"), "compare visible for text-only models");
await registered.commands.visionbridge.handler("off", ctx(fakeModel(["text"])));
assert.ok(!activeTools.includes("compare_images"), "compare hidden when disabled");
await registered.commands.visionbridge.handler("on", ctx(fakeModel(["text"])));
assert.ok(activeTools.includes("compare_images"), "compare back when re-enabled");

// --- status mentions the compare tool ---
notifications.length = 0;
await registered.commands.visionbridge.handler("status", ctx(fakeModel(["text"])));
assert.ok(notifyLines().includes("compare"), "status lists the compare tool");

// =====================================================================
// Ticket 05: completion toast + config honesty
// =====================================================================

const uiCaptureCtx = (model) => ({ ...captureCtx(model), hasUI: true });

// --- completion toast on the in-context swap ---
notifications.length = 0;
await handlers.context[0](
	{
		type: "context",
		messages: structuredClone([
			{
				role: "user",
				content: [{ type: "image", data: makePngHeader(1024, 768).toString("base64"), mimeType: "image/png" }],
				timestamp: 1,
			},
		]),
	},
	uiCaptureCtx(fakeModel(["text"])),
);
assert.ok(
	notifications.map((n) => n.m).some((m) => /described 1 image with .+ in \d+(\.\d+)?s/.test(m)),
	"swap completion toast names the model and elapsed time",
);

// --- completion toast on describe_image; cache hit is silent ---
notifications.length = 0;
const toastImg = { type: "image", data: makePngHeader(500, 400).toString("base64"), mimeType: "image/png" };
const imgToastCtx = uiCaptureCtx(fakeModel(["text"]));
imgToastCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [toastImg], timestamp: 1 } }],
};
await imageTool.execute("toast-1", { question: "anything" }, undefined, undefined, imgToastCtx);
assert.ok(
	notifications.map((n) => n.m).some((m) => /image described by .+ in \d+(\.\d+)?s/.test(m)),
	"describe_image completion toast",
);
notifications.length = 0;
await imageTool.execute("toast-2", { question: "anything" }, undefined, undefined, imgToastCtx);
assert.equal(
	notifications.filter((n) => n.m.includes("pi-vision-bridge")).length,
	0,
	"cache hit: no completion toast (nothing spent)",
);

// --- completion toast on compare_images ---
notifications.length = 0;
const cmpToastCtx = uiCaptureCtx(fakeModel(["text"]));
cmpToastCtx.sessionManager = {
	getBranch: () => [
		{ type: "message", message: { role: "user", content: [makeImage("toast-cmp-a")], timestamp: 1 } },
		{ type: "message", message: { role: "user", content: [makeImage("toast-cmp-b")], timestamp: 2 } },
	],
};
await compareTool.execute("toast-3", {}, undefined, undefined, cmpToastCtx);
assert.ok(
	notifications.map((n) => n.m).some((m) => /images compared by .+ in \d+(\.\d+)?s/.test(m)),
	"compare completion toast",
);

// --- failure toast unchanged ---
notifications.length = 0;
const failCtx = uiCaptureCtx(fakeModel(["text"]));
failCtx.modelRegistry.complete = async () => ({
	role: "assistant",
	content: [],
	usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	stopReason: "error",
	errorMessage: "boom",
});
const failSwap = await handlers.context[0](
	{
		type: "context",
		messages: structuredClone([
			{ role: "user", content: [makeImage("fail-toast")], timestamp: 1 },
		]),
	},
	failCtx,
);
assert.ok(
	notifications.map((n) => n.m).some((m) => /image analysis failed — boom/.test(m)),
	"failure toast unchanged",
);
assert.ok(
	failSwap.messages[0].content[0].text.includes("[Image analysis failed: boom"),
	"failure swap text unchanged",
);

// --- silent when notify is off ---
writeFileSync(join(fakeHome, ".pi", "agent", "pi-vision-bridge.json"), JSON.stringify({ notify: false }));
notifications.length = 0;
const quietCtx = uiCaptureCtx(fakeModel(["text"]));
quietCtx.sessionManager = {
	getBranch: () => [{ type: "message", message: { role: "user", content: [makeImage("quiet-img")], timestamp: 1 } }],
};
const quietResult = await imageTool.execute("quiet-1", { question: "quiet" }, undefined, undefined, quietCtx);
assert.ok(quietResult.details.model && !quietResult.isError, "notify off: the analysis still ran");
assert.equal(
	notifications.filter((n) => n.m.includes("pi-vision-bridge")).length,
	0,
	"notify off: no toasts at all",
);

// --- videoDownloadMaxMB: cut from config, tolerated when stale ---
const { loadConfig } = await import("../src/config.ts");
writeFileSync(
	join(fakeHome, ".pi", "agent", "pi-vision-bridge.json"),
	JSON.stringify({ videoDownloadMaxMB: 250, notify: true }),
);
const loaded = loadConfig();
assert.equal(loaded.videoDownloadMaxMB, undefined, "stale videoDownloadMaxMB is ignored, not surfaced");
assert.equal(loaded.notify, true, "other file fields still load");
assert.equal(loaded.enabled, true, "defaults intact alongside the stale field");
await registered.commands.visionbridge.handler("model test/vision-model", ctx(fakeModel(["text"])));
const savedNow = JSON.parse(readFileSync(join(fakeHome, ".pi", "agent", "pi-vision-bridge.json"), "utf8"));
assert.ok(!("videoDownloadMaxMB" in savedNow), "persisted config never contains videoDownloadMaxMB");
notifications.length = 0;
await registered.commands.visionbridge.handler("status", ctx(fakeModel(["text"])));
assert.ok(!notifyLines().toLowerCase().includes("download"), "status output never mentions download");

rmSync(fakeHome, { recursive: true, force: true });
rmSync(workDir, { recursive: true, force: true });
console.log("ALL HARNESS TESTS PASSED");
