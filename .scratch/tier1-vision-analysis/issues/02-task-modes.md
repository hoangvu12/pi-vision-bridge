# 02 — Task-typed analysis modes

**What to build:** `describe_image` gains an optional `mode` argument with
five values — `ocr`, `error`, `ui`, `diagram`, `chart` — each backed by a
curated prompt template tuned for that reading (verbatim transcription for
ocr; error-message + stack-trace focus for error; component/layout inventory
for ui; nodes/arrows/relationships for diagram; axes/series/values for
chart). The cache key extends to fingerprint + mode + question, so two modes
over the same image are two analyses. The tool's model-facing description
teaches when to pick which mode. No mode means today's behavior: the generic
thorough description.

**Blocked by:** 01 — Generalize the analysis seam.

**Status:** ready-for-agent

- [ ] Tool schema accepts an optional `mode` restricted to the five values;
      anything else fails schema validation
- [ ] Each mode resolves to a distinct curated prompt that reaches the
      mocked model call (harness: capture the prompt per mode; assert they
      differ and carry mode-specific instructions)
- [ ] Cache key includes mode: same image + two different modes → two model
      calls; same image + same mode twice → one (harness with mocked registry
      call counting)
- [ ] Omitting `mode` produces exactly the current generic behavior
      (prompt and cache key unchanged from today's default path)
- [ ] Tool description and prompt snippet enumerate the modes with one-line
      "use when" guidance, so the model can pick without guessing
- [ ] The mode applies only to `describe_image`; the in-context swap and the
      video path are untouched
- [ ] End-to-end: an error-bearing screenshot re-examined with
      `mode=error` yields an error-focused reading (manual/demo check
      against the repo's test image)
