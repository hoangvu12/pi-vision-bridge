# 03 — Dimensions in swap text + region zoom

**What to build:** Two halves that complete each other.

First, the swap text that replaces each image for text-only models publishes
that image's pixel dimensions (width x height, parsed from the image header —
no model call, no re-encode), so the model can reason about layout and
construct region queries. Formats that can't be parsed cheaply degrade
silently: no dimensions line, no failure.

Second, `describe_image` gains an optional `region` argument — `[x, y, w, h]`
in pixels against those published dimensions. The region is cropped out of
the image, the crop alone goes to the vision model, and the result is cached
per fingerprint + region + question. Boxes are clamped to the image bounds
(with a notice in the result when clamping changed the box); zero-area boxes
are a clear error. The result text states which region was analyzed so the
model can cite it.

**Blocked by:** 01 — Generalize the analysis seam.

**Status:** ready-for-agent

- [ ] Dimensions parsed from PNG and JPEG headers appear in the swap text
      (harness: tiny fixtures with known dimensions; assert the rendered
      wrapper contains them)
- [ ] Unknown or unparseable formats produce swap text without a dimensions
      line and no error (graceful degradation)
- [ ] `region` accepts `[x, y, w, h]`; out-of-bounds boxes are clamped and
      the result notes the clamping; zero/negative-area boxes return a clear
      error result
- [ ] A region analysis sends only the cropped image to the vision model
      (harness: mocked registry asserts a single image block; the crop step
      itself is stubbed at the seam)
- [ ] Region cropping works against real files in the end-to-end path (uses
      the repo's test image; crop artifacts are cleaned up afterwards)
- [ ] Cache key includes region: same image, two regions → two analyses
- [ ] End-to-end: zooming into a region of the test screenshot yields a
      description focused on that region's content
