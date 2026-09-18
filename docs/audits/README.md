# Audit records

Each file here is an audit **as delivered, on the date in its name, and is never edited.**
`2026-09-16-audit-iv.json` is audit IV; its `artifact_source` indexes I–III. Audit V is #325.

The `measurements_*` sections are the ADR-0007 case — the thing that produced the number, with its
reproduce snippet — and that is why the file is committed whole rather than as an extract. The `plan`
array is **not** a measurement; it is a proposal, and the tracker has already re-ordered it. Twenty
tickets cite `plan[N]` by index, so an edit here would silently re-point every one of them — the
label-versus-body defect ADR-0024 was written about, in a file. **Re-ordering and supersession live in
the milestones and in the tickets' own bodies, never here.** Known divergences from `plan` as written:

- Phase 1 step 4 (mark the `-qb` rows on the site) is superseded by #299: the `-qb` rows left the
  published path. No ticket exists for it and none should.
- Phase 2 step 5 (#311) is re-ordered to the front of Phase 2 (2026-09-16): the plan ran the gates
  whose t's the cluster fix restates, then fixed the cluster. Nine gate tickets are `blocked_by` #311.
- Phase 3 step 6 is split under ADR-0024 into #327 (measurement) and #322 (the re-decision).
- Findings R11 and R12 (#313, #314) and evaluation ranks 8 and 10 (#315, #316) appear in no plan
  step. #313–#315 are placed in Phase 2 on 2026-09-16 with reasons in their bodies; #316 sits in the
  **Audit IV — unplaced** milestone until Audit V places it.
- R13's evidence line *"the data is on disk"* (the pool's ledgers) is the reviewer's inference,
  not a verified fact: what was verified is that `fetch.pool.ledgers` exists with zero callers.
  Checked 2026-09-18: no `pool_state.json` has ever been written in this checkout and the pool's
  credentials exist in no agent environment. The finding's mechanism (rivals carry empty
  ledgers) is unaffected; its magnitude (M-R6, $17.53 → $13.06) is `verified: reported` on a
  synthetic board and is not a measurement of this pool. #317's estimate grows accordingly:
  capture the state first, through `--payload`, the same page-save #334 needs.
- `mutation_testing.starter_change` counts 14 mutants and 7 survivors but names only 12 by
  text (the 7 survivors and 5 of the killed); the other two came from a run in a `/tmp` copy
  that is not reproducible, so there is nothing to recover. **The live record from 2026-09-17
  is `scripts/mutate_starter_change.py`**: it carries the 12 the file names, applies each to
  the module and re-takes the count (12/12 killed on `333e59a`, #304). The count of 14 is not
  verifiable from this file and is not carried forward. `mutation_testing.pool` is complete at
  10/10 and needs no successor.
