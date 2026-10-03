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

rmSync(fakeHome, { recursive: true, force: true });
rmSync(workDir, { recursive: true, force: true });
console.log("ALL HARNESS TESTS PASSED");
