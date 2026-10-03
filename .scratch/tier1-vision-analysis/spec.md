# Tier 1 vision analysis upgrade

Source: conversation synthesis (research round + design discussion). No upstream issue.

## Scope

Three capability additions to pi-vision-bridge, plus the prefactor that makes
them clean vertical slices, plus a polish pass:

1. Task-typed analysis modes on `describe_image`
2. Image dimensions in the swap text + region zoom
3. A `compare_images` tool
4. Polish: completion toast, config honesty

## Decisions (settled in conversation — implement against these)

- **Mode taxonomy**: five modes — `ocr`, `error`, `ui`, `diagram`, `chart`.
  Not z.ai's full seven: codegen-from-UI (`ui_to_artifact`) is out of scope,
  and generic description remains the no-mode default.
- **Cache keys include every analysis-shaping input**: fingerprint + mode +
  question + region (and pair + question for compare). A mode or region is
  never free to reuse another analysis's cached text.
- **Region coordinates**: pixels `[x, y, w, h]` against the dimensions
  published in the swap text. Invalid boxes are clamped (with a notice in the
  result) rather than rejected, except zero-area, which is an error.
- **Compare shape**: a third tool, not a param on `describe_image` — clearer
  model-facing affordance, mirrors z.ai's `ui_diff_check`. Both images go in
  ONE vision call (two separate descriptions lose the spatial comparison).
- **Visibility**: `compare_images` follows `describe_video` — visible to all
  models, hidden only when the extension is disabled.
- **Deferred (do not build here)**: cross-session disk cache, intent-aware
  first-pass emphasis, remote video URLs, native `video_url` APIs.

## Out of scope

Audio analysis (no STT path in pi). Anything touching the video frame-sampling
policy.
