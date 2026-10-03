/**
 * Configuration for pi-vision-bridge.
 *
 * Stored at ~/.pi/agent/pi-vision-bridge.json. Every field can be overridden
 * by environment variables so headless / print-mode usage needs no file:
 *
 *   PI_VISION_BRIDGE_MODEL       "provider/model-id" or comma-separated
 *                                ordered list of vision models
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
	/**
	 * Ordered vision-model candidate list ("provider/model-id" entries).
	 * null = auto-select from the catalog. One entry = a pinned model with
	 * no fallback; several entries = health rotation in exactly this order.
	 */
	visionModels: string[] | null;
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
}

/** pi's agent dir, mirroring pi's own resolution (incl. its env override). */
function piAgentDir(): string {
	const envDir = process.env.PI_CODING_AGENT_DIR;
	if (envDir) return envDir.replace(/^~(?=$|[\/])/, homedir());
	return join(homedir(), ".pi", "agent");
}

export const CONFIG_PATH = join(piAgentDir(), "pi-vision-bridge.json");

const DEFAULTS: BridgeConfig = {
	enabled: true,
	visionModels: null,
	maxTokens: 2048,
	temperature: 0,
	cacheMax: 64,
	notify: true,
	videoFrames: 10,
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

/**
 * The optional "visionBridge" section in pi's own settings: global
 * (<agentDir>/settings.json) and project (<cwd>/.pi/settings.json), project
 * winning over global like pi's own merge. pi has no official
 * per-extension settings section, but unknown keys survive its loader, so a
 * section there works as a read layer. Returns validated known fields only;
 * unknown keys inside the section are ignored, never rejected.
 */
export function readPiSettingsSection(cwd: string | undefined): Partial<BridgeConfig> | undefined {
	const extract = (path: string): Record<string, unknown> | undefined => {
		try {
			if (!existsSync(path)) return undefined;
			const parsed = JSON.parse(readFileSync(path, "utf8")) as { visionBridge?: Record<string, unknown> };
			const raw = parsed?.visionBridge;
			return raw && typeof raw === "object" ? raw : undefined;
		} catch (err) {
			debug("pi settings read failed:", path, err instanceof Error ? err.message : err);
			return undefined;
		}
	};
	const globalRaw = extract(join(piAgentDir(), "settings.json"));
	const projectRaw = cwd ? extract(join(cwd, ".pi", "settings.json")) : undefined;
	if (!globalRaw && !projectRaw) return undefined;
	return validateSection({ ...globalRaw, ...projectRaw });
}

/** Keep only known, well-typed fields from a visionBridge section. */
function validateSection(raw: Record<string, unknown>): Partial<BridgeConfig> {
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
}

export function isEnvDisabled(): boolean {
	return envFlag("PI_VISION_BRIDGE_OFF");
}

/** Effective config: defaults <- file <- environment. */
export function loadConfig(): BridgeConfig {
	const config: BridgeConfig = { ...DEFAULTS };

	if (existsSync(CONFIG_PATH)) {
		try {
			const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as Partial<BridgeConfig> & {
				/** Legacy single-model field (pre-1.1). */
				visionModel?: unknown;
			};
			if (typeof raw.enabled === "boolean") config.enabled = raw.enabled;
			if (Array.isArray(raw.visionModels)) {
				const list = raw.visionModels
					.filter((s): s is string => typeof s === "string")
					.map((s) => s.trim())
					.filter((s) => s.length > 0);
				if (list.length > 0) config.visionModels = list;
			}
			// Legacy single-model field (pre-1.1): tolerated, treated as a one-entry pin.
			if (config.visionModels === null && typeof raw.visionModel === "string" && raw.visionModel.trim()) {
				config.visionModels = [raw.visionModel.trim()];
			}
			if (typeof raw.maxTokens === "number" && raw.maxTokens > 0) config.maxTokens = raw.maxTokens;
			if (typeof raw.temperature === "number" && raw.temperature >= 0) config.temperature = raw.temperature;
			if (typeof raw.cacheMax === "number" && raw.cacheMax > 0) config.cacheMax = raw.cacheMax;
			if (typeof raw.notify === "boolean") config.notify = raw.notify;
			if (typeof raw.videoFrames === "number" && raw.videoFrames > 0) config.videoFrames = raw.videoFrames;
			// Unknown fields (e.g. a stale videoDownloadMaxMB from older
			// versions) are ignored on load, never rejected.
		} catch (err) {
			debug("config read failed, using defaults:", err instanceof Error ? err.message : err);
		}
	}

	// One spec pins a single model; comma-separated specs give an ordered list.
	const envModels = process.env.PI_VISION_BRIDGE_MODEL?.split(",")
		.map((s) => s.trim())
		.filter((s) => s.length > 0);
	if (envModels && envModels.length > 0) config.visionModels = envModels;
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
		`vision models: ${config.visionModels ? config.visionModels.join(", ") : "auto-select"}`,
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
