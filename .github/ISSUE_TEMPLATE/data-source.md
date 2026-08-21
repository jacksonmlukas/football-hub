---
name: New data source
about: Adding an ingest path
labels: infra
---

## Source
URL, auth, quota, terms (redistribution allowed?)

## Quota budget
Calls per week. Confirm no per-team or per-game loop.

## Contract checklist
- [ ] Golden fixture saved to `tests/golden/fixtures/`
- [ ] `Contract` defined with all five fields, including `ranges`
- [ ] Contract test written red first
- [ ] Nightly golden test added
