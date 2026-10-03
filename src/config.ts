/**
 * Configuration for pi-vision-bridge.
 *
 * Stored at ~/.pi/agent/pi-vision-bridge.json. Every field can be overridden
 * by environment variables so headless / print-mode usage needs no file:
 *
 *   PI_VISION_BRIDGE_MODEL       "provider/model-id"  explicit vision model
 *   PI_VISION_BRIDGE_OFF=1       disable the extension entirely
 *   PI_VISION_BRIDGE_MAX_TOKENS  description output cap
 *   PI_VISION_BRIDGE_CACHE_MAX   cache entry cap
 *   PI_VISION_BRIDGE_VIDEO_FRAMES  default frames sampled per video
 *   PI_VISION_BRIDGE_DEBUG=1     log debug info to stderr
 *
 * Atomic writes (tmp file + rename) keep the file from corrupting on crash.
 */

import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface BridgeConfig {
	/** Master switch. When false, media pass through untouched. */
	enabled: boolean;
	/** Explicit vision model as "provider/model-id". null = auto-select. */
	visionModel: string | null;
	/** Max output tokens for the description call. */
	maxTokens: number;
	/** Sampling temperature for the vision call. 0 = deterministic. */
	temperature: number;
	/** How many analyzed images to keep in the in-memory cache (LRU). */
	cacheMax: number;
	/** Show notify() toasts for analysis activity. */
	notify: boolean;
	/** Default frames sampled per video (bounded by the model's per-message image limit). */
	videoFrames: number;
	/** Max remote video download size in MB (local files are streamed, not downloaded). */
	videoDownloadMaxMB: number;
}

export const CONFIG_PATH = join(homedir(), ".pi", "agent", "pi-vision-bridge.json");

const DEFAULTS: BridgeConfig = {
	enabled: true,
	visionModel: null,
	maxTokens: 2048,
	temperature: 0,
	cacheMax: 64,
	notify: true,
	videoFrames: 10,
	videoDownloadMaxMB: 100,
};

export const debug = (...parts: unknown[]): void => {
	if (process.env.PI_VISION_BRIDGE_DEBUG === "1") {
		console.error("[pi-vision-bridge]", ...parts);
	}
};

function envFlag(name: string): boolean {
	return process.env[name] === "1" || process.env[name] === "true";
}

function envNumber(name: string): number | undefined {
	const raw = process.env[name];
	if (raw === undefined) return undefined;
	const n = Number(raw);
	return Number.isFinite(n) && n > 0 ? n : undefined;
}

export function isEnvDisabled(): boolean {
	return envFlag("PI_VISION_BRIDGE_OFF");
}

/** Effective config: defaults <- file <- environment. */
export function loadConfig(): BridgeConfig {
	const config: BridgeConfig = { ...DEFAULTS };

	if (existsSync(CONFIG_PATH)) {
		try {
			const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as Partial<BridgeConfig>;
			if (typeof raw.enabled === "boolean") config.enabled = raw.enabled;
			if (raw.visionModel === null || typeof raw.visionModel === "string") {
				config.visionModel = raw.visionModel;
			}
			if (typeof raw.maxTokens === "number" && raw.maxTokens > 0) config.maxTokens = raw.maxTokens;
			if (typeof raw.temperature === "number" && raw.temperature >= 0) config.temperature = raw.temperature;
			if (typeof raw.cacheMax === "number" && raw.cacheMax > 0) config.cacheMax = raw.cacheMax;
			if (typeof raw.notify === "boolean") config.notify = raw.notify;
			if (typeof raw.videoFrames === "number" && raw.videoFrames > 0) config.videoFrames = raw.videoFrames;
			if (typeof raw.videoDownloadMaxMB === "number" && raw.videoDownloadMaxMB > 0) {
				config.videoDownloadMaxMB = raw.videoDownloadMaxMB;
			}
		} catch (err) {
			debug("config read failed, using defaults:", err instanceof Error ? err.message : err);
		}
	}

	const envModel = process.env.PI_VISION_BRIDGE_MODEL?.trim();
	if (envModel) config.visionModel = envModel;
	const envMax = envNumber("PI_VISION_BRIDGE_MAX_TOKENS");
	if (envMax !== undefined) config.maxTokens = envMax;
	const envCache = envNumber("PI_VISION_BRIDGE_CACHE_MAX");
	if (envCache !== undefined) config.cacheMax = envCache;
	const envFrames = envNumber("PI_VISION_BRIDGE_VIDEO_FRAMES");
	if (envFrames !== undefined) config.videoFrames = envFrames;

	return config;
}

/** Persist the full config. Best-effort: failures are reported, not thrown. */
export function saveConfig(config: BridgeConfig): { ok: boolean; error?: string } {
	try {
		const dir = join(homedir(), ".pi", "agent");
		if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
		const tmpPath = `${CONFIG_PATH}.tmp`;
		writeFileSync(tmpPath, JSON.stringify(config, null, "\t") + "\n", "utf8");
		try {
			chmodSync(tmpPath, 0o600);
		} catch {
			// chmod is best-effort (Windows ignores it)
		}
		renameSync(tmpPath, CONFIG_PATH);
		return { ok: true };
	} catch (err) {
		return { ok: false, error: err instanceof Error ? err.message : String(err) };
	}
}

/** Stringify a config for status display. */
export function describeConfig(config: BridgeConfig): string {
	const lines = [
		`enabled: ${config.enabled}`,
		`vision model: ${config.visionModel ?? "auto-select"}`,
		`max tokens: ${config.maxTokens}`,
		`temperature: ${config.temperature}`,
		`cache cap: ${config.cacheMax}`,
		`video frames: ${config.videoFrames}`,
		`notify: ${config.notify}`,
	];
	const fileState = existsSync(CONFIG_PATH)
		? statSync(CONFIG_PATH).isFile()
			? `config file: ${CONFIG_PATH}`
			: "config file: invalid path"
		: "config file: not present (defaults)";
	return `${lines.join("\n")}\n${fileState}`;
}
