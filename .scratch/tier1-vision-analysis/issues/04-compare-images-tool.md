# 04 — compare_images tool

**What to build:** A third tool, `compare_images`, that answers "what changed
between these two images?" in ONE vision call. Takes two image fingerprints
(defaults to the two most recent distinct images in the session when
omitted) and an optional focused question. Both images go to the vision model
together with a diff-oriented prompt — describe what differs (text, layout,
state, color), what stayed the same, and which version each difference
belongs to. Cached per image pair + question. Result reports usage from the
nested call.

Visibility follows `describe_video`, not `describe_image`: visible to all
models (text-only and vision alike), hidden only when the extension is
disabled — a vision model still can't diff two images it was shown in
different turns without re-attaching them.

**Blocked by:** 01 — Generalize the analysis seam.

**Status:** ready-for-agent

- [ ] Tool registered with the two-fingerprint + question schema;
      fingerprints optional, question optional
- [ ] Omitted fingerprints resolve to the two most recent distinct images in
      the session (harness: branch with three images, assert the two latest
      are chosen)
- [ ] Fewer than two images in the session, or an unmatched fingerprint,
      returns a helpful error listing what's available (mirrors
      `describe_image`'s retry affordance)
- [ ] One model call receives both images (harness: mocked registry asserts
      two image blocks in a single request, plus the diff-oriented prompt)
- [ ] Cache key is the (order-insensitive) pair + question: same pair asked
      again → cache hit; reversed order → also a hit
- [ ] Nested usage is reported in the tool result
- [ ] Visibility sync: shown for text-only AND vision models; hidden when
      the extension is disabled (harness: visibility matrix assertions)
- [ ] End-to-end: comparing two versions of a screenshot yields a
      change-focused answer (manual/demo check — two crops of the repo's
      test image work as the pair)
