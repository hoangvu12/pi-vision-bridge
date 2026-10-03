# Simplify vision-model selection

Source: conversation after live testing. The scoring heuristics ranked
models by name ("flash", cheap tiers, vision families) and price. That
is a guess about quality, and no verified list exists to ground it.
Decided: drop the guess, keep the proven availability machinery, and
give the user an ordered list instead.

## Decisions

- **Remove** the auto-scoring heuristics (`CHEAP_TIERS`, `VISION_FAMILIES`,
  `autoPickScore`): no name-based quality guessing.
- **Keep** health rotation, benching, sticky winner, and the in-process
  candidate memo — proven against real upstream outages.
- **Keep** prefix diversification in auto mode (availability insurance
  against one dead route filling the pool, not a quality guess).
- **Auto mode** (no config): candidates in registry catalog order,
  image-capable, authenticated, prefix-diversified.
- **User-configured ordered list**: replaces the single-model pin. One
  entry means a pinned model with no fallback; several entries mean
  health rotation in exactly that order.
- Config field becomes `visionModels: string[] | null`. The legacy
  single `visionModel` string in existing config files is tolerated on
  load and treated as a one-entry list. `PI_VISION_BRIDGE_MODEL` now
  accepts a comma-separated ordered list.

## Out of scope

Quality calibration probes (parked), cross-session disk cache of
health state, per-call model choice by the agent.
