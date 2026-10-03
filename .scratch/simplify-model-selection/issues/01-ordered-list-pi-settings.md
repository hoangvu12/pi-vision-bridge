# 01 — Ordered model list + pi-settings config layer

**What to build:** Remove the name/price scoring from vision-model
selection; replace the single-model pin with a user-configurable ordered
candidate list; honor a `visionBridge` section in pi's own settings files
(global and project scope) as a config layer above the extension's own
file and below environment variables.

**Blocked by:** None.

**Status:** ready-for-agent

- [x] Scoring heuristics removed; auto mode uses catalog order with
      prefix diversification (harness: a "pro" model listed first stays
      first)
- [x] `visionModels` ordered list: one entry pins, several rotate in
      exactly that order (harness: first entry 503s, second answers,
      sticky winner skips the benched entry)
- [x] `visionBridge` section read from pi's settings files, project
      scope winning over global (harness: both scopes against real
      files); env vars still win; unknown keys in the section are
      ignored
- [x] Config refreshed per request, not just at construction (harness:
      the in-context swap path honors the section)
- [x] `/visionbridge model` accepts multiple specs (space or comma);
      status names the active config source; legacy single-model config
      values load as a one-entry pin
- [x] README: model selection, configuration, and behavior sections
      rewritten for the new layers
- [x] Typecheck and harness green; live end-to-end verified (settings
      section pins `iroha/dashscope/qwen-omni-turbo`, single vision call,
      no rotation)

## Comments

- Implemented across 1.1.0 (`9e456da`), 1.1.1 (`88082bc`), 1.1.2
  (`ee4760f`). The two patch releases fix bugs the live tests caught:
  1.1.0 read config only at construction, and its `getSettings`-based
  settings layer never worked because pi 0.87 does not actually expose
  `getSettings` to extensions at runtime (types declare it; a probe
  extension showed `undefined` on both the API and the event context).
  1.1.2 reads the settings files directly instead.
