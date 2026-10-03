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
	return (
		`[Image ${spec.fingerprint}${spec.origin ?? ""} — ${dims}${describedBy}` +
		`this model cannot view images directly]\n${spec.description}\n` +
		`[end of image ${spec.fingerprint}; call ${IMAGE_TOOL} with fingerprint "${spec.fingerprint}" to re-examine it with a focused question]`
	);
}
