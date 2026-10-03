/**
 * Prompt templates and model-facing text rendering for pi-vision-bridge.
 *
 * Everything the *models* read — the nested vision-analysis prompts and the
 * swap text that replaces an image for text-only models — lives here, so
 * analysis execution (src/vision.ts) stays prompt-agnostic: callers inject
 * a prompt spec and the seam routes it verbatim.
 */

/** The tool a text-only model calls to re-examine a swapped image. */
export const IMAGE_TOOL = "describe_image";

export const GENERIC_SYSTEM_PROMPT = `You are the vision stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

Produce one complete, standalone text description of the image. Priorities, in order:

1. TEXT FIRST: transcribe ALL visible text verbatim — error messages, code, terminal output, labels, buttons, headings, table cells, axis labels. Preserve line breaks, indentation, and layout order. Text accuracy matters more than anything else. Mark any character you are unsure about with "‹?›" instead of guessing.
2. STRUCTURE: if it is a screenshot or UI, describe the window/application, regions (header, sidebar, main content, dialogs), the key components, their states (selected, disabled, loading, error), and the overall layout top-to-bottom.
3. DATA: if it is a diagram or chart, describe nodes, labeled arrows and their direction, axes, series, units, and the important values.
4. VISUAL FACTS: salient colors, counts, sizes, and positions of notable elements.

Rules:
- Describe only what is visible. Explicitly mark inferences as uncertain; never invent text or details.
- Plain text/markdown only. No preamble like "This image shows", no advice, no questions, no concluding remarks.
- Be thorough but do not pad: every sentence must carry information about the image.`;

export const GENERIC_USER_PROMPT = "Describe this image.";

export function buildQuestionPrompt(question: string): string {
	return `Answer this question about the image, with the transcription precision of an OCR pass for anything textual it touches:\n\n${question}`;
}

/** The task-typed analysis modes available on describe_image. */
export const ANALYSIS_MODES = ["ocr", "error", "ui", "diagram", "chart"] as const;
export type AnalysisMode = (typeof ANALYSIS_MODES)[number];

export interface ModePrompt {
	systemPrompt: string;
	userText: string;
}

interface ModeTemplate extends ModePrompt {
	/** One-line "use when" guidance surfaced in the tool description. */
	guidance: string;
}

/** Curated prompt per analysis mode: a focused way of reading the image. */
const MODE_TEMPLATES: Record<AnalysisMode, ModeTemplate> = {
	ocr: {
		systemPrompt: `You are an OCR stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

Transcribe ALL text visible in the image, verbatim. Priorities, in order:

1. COMPLETENESS: every piece of text — headings, body, labels, buttons, table cells, terminal output, code, watermarks, small print. Nothing is skipped because it looks unimportant.
2. VERBATIM: preserve exact spelling, casing, punctuation, numbers, and symbols. Mark any character you are unsure about with "‹?›" instead of guessing.
3. LAYOUT: reproduce reading order and line structure; keep line breaks, indentation, and column alignment. For tables, render rows in order, columns separated by " | ".
4. NON-TEXT: at the end, one short line naming prominent non-text regions (photos, logos, icons) — no detail.

Rules:
- Plain text only. No preamble, no commentary, no questions, no concluding remarks.
- Never invent or autocomplete text that is not legible.`,
		userText: "Transcribe all text in this image verbatim.",
		guidance: "verbatim transcription of every visible word",
	},
	error: {
		systemPrompt: `You are the vision stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

The image likely shows a failure: an error dialog, a crash, a failed test, a stack trace, terminal or log output. Find the failure and transcribe it exactly. Priorities, in order:

1. THE ERROR MESSAGE: the primary error or diagnostic text, verbatim — the full message, any error code, and the severity level.
2. STACK TRACE / DETAIL: any stack trace, assertion, diff, expected-vs-actual, or detail lines, verbatim, in order, with line numbers if shown.
3. CONTEXT: the lines that locate the error — file paths, line numbers, timestamps, the running command, environment or version strings.
4. UI STATE: if it is a dialog or notification, its title, body, available buttons, and which one is emphasized or default.
5. NO ERROR VISIBLE: say so in one line, then give a brief factual summary of what the image does show.

Rules:
- Mark uncertain characters with "‹?›"; never invent text.
- Plain text/markdown only. No preamble, no advice, no fix suggestions, no questions.`,
		userText: "Extract the error message and all failure-related text from this image, verbatim.",
		guidance: "error messages and stack traces from failure screenshots",
	},
	ui: {
		systemPrompt: `You are the vision stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

The image is a user interface (application window, web page, dialog, terminal). Produce a precise UI inventory. Priorities, in order:

1. WINDOW: what application or page it is, title bar text, and the overall layout impression.
2. REGIONS: the major areas top-to-bottom and left-to-right — header, toolbars, sidebars, content, status bar, dialogs — with their placement.
3. COMPONENTS: every meaningful component per region — buttons, inputs, tabs, lists, tables, trees, menus, toggles — with its visible label and state (selected, focused, disabled, checked, loading, error).
4. TEXT: transcribe the important labels and values verbatim; mark uncertain characters with "‹?›".
5. CUES: focus rings, cursors, selection highlights, badges, counts.

Rules:
- Describe only what is visible; mark inferences as uncertain.
- Plain text/markdown only, organized by region. No preamble, no advice, no questions.`,
		userText: "Inventory this interface: regions, components, labels, and states.",
		guidance: "component and layout inventory of interface screenshots",
	},
	diagram: {
		systemPrompt: `You are the vision stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

The image is a diagram (flowchart, architecture, graph, UML, ER, mind map, org chart). Reconstruct its structure as text. Priorities, in order:

1. NODES: every node, box, or shape with its label verbatim (mark uncertain characters "‹?›"), and its type when distinguishable (process, decision, database, group, start, end).
2. EDGES: every arrow or connection — from which node, to which node, its direction, and its label if any. Include containment: nodes nested inside grouped regions.
3. FLOW: the overall path(s) — where it starts, the main sequence, branches at decisions, where it ends.
4. LEGEND: keys, annotations, numbering, and colors used as meaning.

Rules:
- Describe only what is visible; mark inferences as uncertain.
- Plain text/markdown only. No preamble, no advice, no questions.`,
		userText: "Reconstruct this diagram: nodes, arrows, and their labels.",
		guidance: "nodes, arrows, and relationships from diagrams",
	},
	chart: {
		systemPrompt: `You are the vision stage of a coding assistant that cannot view images directly. Another language model — not a human — will read your output as its only view of the image.

The image is a chart or plot. Extract its data faithfully. Priorities, in order:

1. TYPE: the kind of chart (bar, line, pie, scatter, area, heatmap, box, …) and what it plots.
2. AXES: each axis with its label, units, scale, and range; for time axes, the period covered.
3. SERIES: every series or category with its color or marker, and its values — exact where labeled, estimated from the scale otherwise (mark estimates "~").
4. NOTABLES: peaks, troughs, trends, outliers, crossings, annotations, and the legend verbatim.

Rules:
- Mark uncertain characters with "‹?›"; never invent data points.
- Plain text/markdown only. No preamble, no interpretation advice, no questions.`,
		userText: "Extract this chart's type, axes, series, and values.",
		guidance: "axes, series, and values from charts and plots",
	},
};

/** Human/mode-name line for docs and errors. */
export function modeGuidance(mode: AnalysisMode): string {
	return MODE_TEMPLATES[mode].guidance;
}

/** All modes with one-line guidance, for tool descriptions. */
export function modesGuidanceList(): string {
	return ANALYSIS_MODES.map((m) => `${m} (${MODE_TEMPLATES[m].guidance})`).join(", ");
}

/** Resolve a mode to its curated prompt, folding in an optional question. */
export function resolveModePrompt(mode: AnalysisMode, question?: string): ModePrompt {
	const template = MODE_TEMPLATES[mode];
	return {
		systemPrompt: template.systemPrompt,
		userText: question ? `${template.userText}\n\nFocused question: ${question}` : template.userText,
	};
}

/** Spec for the wrapper text that replaces an image for text-only models. */
export interface SwapTextSpec {
	/** Short fingerprint id of the image, as shown to the model. */
	fingerprint: string;
	/** The description produced by the vision model. */
	description: string;
	/** Attribution suffix for the message that carried the image, e.g. " (from bash output)". */
	origin?: string;
	/** Vision model key that produced the description, e.g. "provider/model-id". Omit when analysis failed. */
	describedBy?: string;
	/** Pixel dimensions of the image, when known. Absent/unknown renders nothing. */
	dimensions?: { width: number; height: number };
}

/**
 * Render the wrapper that replaces an image block in the outgoing request.
 * One helper, one format — the in-context swap and (later) region-aware
 * re-examination agree on it. Dimensions render only when known.
 */
export function buildSwapText(spec: SwapTextSpec): string {
	const dims = spec.dimensions ? `${spec.dimensions.width}x${spec.dimensions.height} px; ` : "";
	const describedBy = spec.describedBy ? `described by ${spec.describedBy}; ` : "";
	// The region zoom is only offered when the model can construct pixel
	// coordinates — i.e. when the dimensions were published above.
	const regionHint = spec.dimensions
		? `, or with region [x, y, w, h] in image pixels (against the dimensions above) to zoom into part of it`
		: "";
	return (
		`[Image ${spec.fingerprint}${spec.origin ?? ""} — ${dims}${describedBy}` +
		`this model cannot view images directly]\n${spec.description}\n` +
		`[end of image ${spec.fingerprint}; call ${IMAGE_TOOL} with fingerprint "${spec.fingerprint}" to re-examine it with a focused question${regionHint}]`
	);
}
