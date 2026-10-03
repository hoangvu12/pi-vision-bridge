# pi-vision-bridge

Agent instructions for this repo. Human-facing docs live in `README.md`.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/` —
one directory per feature, one file per ticket. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles with their default label strings
(`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root,
created lazily by `/domain-modeling` as terms and decisions get resolved. See `docs/agents/domain.md`.
