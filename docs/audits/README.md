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
