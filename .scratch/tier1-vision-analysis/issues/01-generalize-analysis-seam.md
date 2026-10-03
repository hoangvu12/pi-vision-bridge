# 01 — Generalize the analysis seam

**What to build:** A behavior-identical prefactor of the vision-analysis
pathway so later tickets land as clean slices. The nested vision call (the one
routed through the ranked-candidate fallback machinery) becomes able to accept
multiple images in a single model request and an injectable prompt spec
(system prompt + user text), with the current fixed prompt as the default.
The swap-text construction (the wrapper the text-only model sees in place of
each image) moves into a single helper that takes a dimensions argument —
left unfilled for now — alongside the fingerprint, origin, and model
attribution it already renders.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] All existing harness tests pass unmodified
- [ ] The analysis call accepts two images in one model request (harness:
      mocked registry captures the outgoing message and asserts two image
      blocks plus the prompt text)
- [ ] The prompt is injectable: a custom system prompt and user text reach
      the mocked model call verbatim; with no override, the current generic
      image prompt is used (harness assertion)
- [ ] Swap text is produced by one helper; a dimensions argument exists in
      its signature and renders nothing when absent
- [ ] End-to-end behavior unchanged: pasted image + text-only model still
      answers from a swapped-in description; video path unaffected
- [ ] Health rotation, sticky winner, and caching behavior unchanged
      (existing tests cover this — they must stay green without edits)
