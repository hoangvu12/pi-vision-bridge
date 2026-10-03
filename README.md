# pi-vision-bridge

**Give non-vision pi models the ability to understand images AND video.**

- Paste an image into a session running a text-only model (DeepSeek, GLM, Qwen-Coder, gpt-oss, …) and a vision model describes it *in-context before the request leaves* — the text-only model answers as if it saw the image.
- Reference a video file (screen recording, demo, clip) and the `describe_video` tool watches it by sampling frames with ffmpeg and describing them through a vision model — with timestamps, scene by scene.
- Ask "what changed between these two images?" and the `compare_images` tool diffs them in one vision call — differences, what stayed the same, every change attributed to the right version.
- When the active model already supports images, the image bridge stays completely out of the way; the video tool remains available (nothing in pi can watch video natively).

```
you paste an image + question                     you reference a video + question
        |                                                |
pi-vision-bridge: model has no image input       pi-vision-bridge: injects a prompt hint
        |                                          (or the model calls the tool on its own)
vision model describes the image                           |
        |                                          ffmpeg samples ~10 frames (768px, JPEG)
description replaces the image                            |
in the outgoing request                          vision model describes the frames
        |                                          with timestamps, scene by scene
text-only model answers                          text-only model answers
(zero extra round-trips)                         (one tool round-trip)
```

Without the bridge, pi replaces images for non-vision models with a useless `(image omitted: model does not support images)` placeholder — and has no video story at all.

---

## Why this design (research summary)

This extension synthesizes the best mechanics from the OpenCode vision plugins ([opencode-image-vision](https://github.com/showlotus/opencode-image-vision), [opencode-image-proxy](https://github.com/samiulsami/opencode-image-proxy), [opencode-see-image](https://github.com/alfaoz/opencode-see-image)), [z.ai's vision-mcp-server](https://docs.z.ai/devpack/mcp/vision-mcp-server), pi's ecosystem (pi-vlm-proxy, pi-vision-watcher, pi-vision-tool), and video-sampling research:

| Decision | Rationale |
|---|---|
| **Eager in-context swap** (images, before the request) | Tool-based approaches cost 1+ extra model round-trips per image; eager analysis puts the description in the *first* request of the main model. Fastest possible path. |
| **Non-destructive** (per-request transform) | opencode-image-proxy patches images out of stored history — switch to a vision model later and the images are gone. Here the session keeps original images; switching models restores native vision instantly. |
| **Content-hash LRU cache** | The `context` event fires before *every* request (turns, retries, cache warming). Without a cache every turn would re-analyze every image; with it, each image is analyzed exactly once per process. |
| **Generic cached description + targeted `describe_image` tool** | Intent-aware descriptions (passing the current question to the vision model) break cache coherence across turns. The generic description is complete and cacheable; the tool provides the focused re-look ("quote the exact error text"). |
| **Video via frame sampling, not native video APIs** | pi's model layer has no video content type, so native video (z.ai's `video_url` for GLM-4.6V) would mean provider-specific raw HTTP + auth for exactly one provider. Frames work with **every** vision model through pi's own provider stack. Uniform sampling with a frame cap + 768px scaling + per-frame timestamps follows the video-understanding research (multigrid.ai, ffmpeg-cookbook) for token-efficient chronological coverage. |
| **Prompt hint on video references** | Videos can't enter pi messages, so they arrive as text path references. A `before_agent_start` prompt section tells the model a video file exists and to call the tool — discoverability without eager analysis. |
| **Ranked candidate list + health rotation** | A relay listing a model (the `/models` catalog) says nothing about its upstream actually serving it. Failed models are benched for 5 minutes and the analysis rotates to the next candidate; the winner sticks. (opencode-see-image's route-fallback pattern.) |
| **Route-diversified ranking** | At most 2 candidates per id prefix (upstream route), so one dead route can't fill the whole pool. |
| **No-tools helper call** | The vision model processes attacker-influenceable content (the media). It only ever receives the media + a fixed prompt — never tools, files, or bash (the pi-native fix for opencode-see-image issue #6). |
| **Prompt: factual, verbatim-OCR, structured** | Vision models paraphrase text unless told to transcribe exactly; factual framing measurably reduces hallucination; temperature 0 keeps descriptions deterministic. |
| **Auto-hide image tool for vision models** | Zero prompt tokens, zero latency: `describe_image` is removed from the active set and the context handler exits immediately. `describe_video` stays — no model in pi can watch video. |

## Install

Local development:

```bash
pi --extension ./extensions/index.ts          # try it once
```

Or register the folder in `~/.pi/agent/settings.json`:

```jsonc
{
  "extensions": ["C:/path/to/pi-vision-bridge"]
}
```

Or as a package (once published):

```bash
pi install npm:pi-vision-bridge
```

Requirements: at least one connected vision-capable model (any provider you've authenticated) — auto-picked, no extra API keys. Video support additionally needs **ffmpeg + ffprobe on PATH** (frame extraction and probing).

## How it works

### Images

1. **`context` event** — before every model request, if the active model lacks image input, every `ImageContent` in the conversation (pasted images *and* tool-result images) is replaced with:
   ```
   [Image a1b2c3d4e5 — 1920x1080 px; described by iroha/dashscope/qwen3.6-flash; this model cannot view images directly]
   <thorough description: verbatim text, layout, structure, data, colors>
   [end of image a1b2c3d4e5; call describe_image with fingerprint "a1b2c3d4e5" to re-examine it with a focused question, or with region [x, y, w, h] in image pixels (against the dimensions above) to zoom into part of it]
   ```
   Multiple images are analyzed in parallel; identical images (same content hash) are analyzed once and reused. Pixel dimensions are parsed from the image header (no model call, no re-encode) and published so the model can reason about layout and construct region queries; formats that can't be parsed cheaply degrade silently.

2. **`describe_image` tool** — visible only to non-vision models. Focused re-examination with a question; images are pulled from persisted session history, so it works turns later.
   - `mode` (optional) tunes the reading with a curated prompt: `ocr` (verbatim transcription of every visible word), `error` (error messages and stack traces), `ui` (component/layout inventory), `diagram` (nodes, arrows, relationships), `chart` (axes, series, values). No mode = the thorough generic description.
   - `region` (optional) `[x, y, w, h]` in image pixels zooms into part of the image: the region is cropped out and the crop alone goes to the vision model. Out-of-bounds boxes are clamped (noted in the result); zero-area boxes error. PNG is cropped natively; other formats use ffmpeg when available.
   - Answers are cached per (image, mode, region, question) — two modes or two regions over the same image are two analyses.

3. **`compare_images` tool** — visible to *all* models (a vision model still can't diff two images it was shown in different turns without re-attaching them). "What changed between these two images?" answered in **one** vision call with a diff-oriented prompt: every difference (text, layout, state, color), what stayed the same, and which version each difference belongs to. Takes the two fingerprints (defaults to the two most recent distinct images in the session when omitted) and an optional focus question. Cached per pair + question, order-insensitively.

### Video

4. **`describe_video` tool** — visible to *all* models:
   ```json
   {
     "path": "demo.mp4",
     "question": "at what timestamp does the error dialog appear",
     "max_frames": 10
   }
   ```
   - Probes the file (ffprobe: duration, dimensions), validates it's a real video.
   - Samples frames evenly across the duration (4–16 frames, default 10; one per ~2s for short clips; scaled to 768px JPEG), each labeled with its timestamp.
   - Sends all frames in one multi-image vision call with a video-specific prompt (scene-by-scene timeline, on-screen text transcription, changes between frames; explicitly notes audio is not analyzed).
   - Frame count is bounded by the model's per-message image limit (`inputLimits.images.maxPerMessage`) when declared.
   - Supported containers: mp4, mov, mkv, m4v, webm, avi. Local files are streamed (any size); per-(video, question) results are cached.

5. **Prompt hint** — when your message references a video file that exists (quoted or bare path), a system-prompt section tells the model to call the tool, so it knows to look without you repeating yourself.

### Model selection

6. **Ranked candidates** from your connected, authenticated, image-capable models. Scoring prefers cheap/fast tiers (`flash`, `mini`, `nano`, `haiku`, `lite`, `turbo`…), strong-OCR families, non-reasoning models, and low price — then diversifies across upstream routes. The best pick wins; on failure the next candidate takes over.

7. **Tool visibility sync** — on `session_start` / `model_select`: vision model → `describe_image` hidden; text-only → shown. `describe_video` and `compare_images` always shown (when enabled).

## Configuration

```bash
/visionbridge                                # status: config, ranked vision models, cache, tools
/visionbridge model google/gemini-2.5-flash  # pin a specific vision model (no fallback)
/visionbridge auto                           # back to auto-selection with fallback
/visionbridge on | off                       # master switch
/visionbridge cache clear                    # drop cached descriptions
```

Config file `~/.pi/agent/pi-vision-bridge.json` (written by the commands, atomic writes, `0600`):

```jsonc
{
  "enabled": true,
  "visionModel": null,        // "provider/model-id" or null = auto
  "maxTokens": 2048,          // description output cap
  "temperature": 0,           // deterministic descriptions
  "cacheMax": 64,             // LRU entries
  "notify": true,             // analysis toasts
  "videoFrames": 10           // default frames sampled per video
}
```

Environment overrides (useful in print/JSON mode):

| Variable | Effect |
|---|---|
| `PI_VISION_BRIDGE_MODEL` | Pin vision model `provider/model-id` |
| `PI_VISION_BRIDGE_OFF=1` | Disable the extension |
| `PI_VISION_BRIDGE_MAX_TOKENS` | Output cap |
| `PI_VISION_BRIDGE_CACHE_MAX` | Cache cap |
| `PI_VISION_BRIDGE_VIDEO_FRAMES` | Default video frames |
| `PI_VISION_BRIDGE_DEBUG=1` | Debug logging to stderr |

## Behavior reference

| Situation | Behavior |
|---|---|
| Active model has image input | Images pass through natively, `describe_image` hidden, `describe_video` still available |
| Active model is text-only | Images swapped for descriptions before the request |
| Vision model upstream down | Rotate to next ranked candidate; failures benched 5 min |
| All candidates down | Model sees a clear failure note + retry hint; it won't hallucinate |
| Same image again later | Cache hit — free |
| Need the exact words / an error's details / a chart's values | `describe_image` with `mode: ocr / error / ui / diagram / chart` |
| Need a closer look at part of an image | `describe_image` with `region [x, y, w, h]` — the crop alone is analyzed |
| "What changed between these screenshots?" | `compare_images` diffs the pair in one vision call |
| Video referenced in your message | Prompt hint tells the model to call `describe_video` |
| Video analyzed again (same question) | Cache hit — free |
| Switch to a vision model mid-session | Original images return natively (history was never modified) |
| ffmpeg missing | Video tool returns a clear error; images still work |

## Notes

- **Cost**: one flash-tier call per distinct image (~$0.001–0.01 on typical providers) and one multi-image call per video/question. The image description replaces tokens the main model couldn't use anyway.
- **Privacy**: media go only to the vision model you picked, through pi's own provider auth. No third-party services.
- **Aborts**: nested vision calls and ffmpeg respect the session abort signal.
- **JSON/print modes**: fully functional (no toasts without UI; the transform is mode-independent).
- **Audio**: not analyzed — pi has no speech-to-text path. The video prompt says so explicitly so the model doesn't speculate.
- **Future**: native `video_url` input for GLM-4.6V-style APIs would bypass sampling for providers that support it, at the cost of provider-specific raw HTTP calls; frame sampling was chosen for universality.

## Development

```bash
npm install                        # dev deps for typechecking
npx tsc --noEmit                   # typecheck
node test/harness.mjs              # mock-driven wiring tests (no network)
bash make-test-video.sh            # regenerate test-video.mp4 (3 labeled scenes)
PI_VISION_BRIDGE_DEBUG=1 pi --extension ./extensions/index.ts "@test-image.png" "what is this?"
PI_VISION_BRIDGE_DEBUG=1 pi --extension ./extensions/index.ts "watch test-video.mp4 and describe it"
```

## License

MIT
