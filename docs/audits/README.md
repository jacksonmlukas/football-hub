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

## 2026-09-20 method audit

`2026-09-20-method-audit.json` is indexed as a **separate method audit, not Audit V** (#373)
-- the "Audit V is #325" line above stays true, and #325 still owes the artifact read and the
Phase 2/3 re-plan. Its own `schema.relation_to_audit_v` says the same; filed as the milestone
"Method audit 2026-09-20". Known divergences from `plan` as written:

- S1 (#357) and S4 (#361) were confirmed against the repo's real code before filing, not left
  at the JSON's own `verified: read` / faithful-reimplementation caveat: S1 re-run against the
  real `summarise`/`per_season` on `bb334e0`, 0 disagreements in 2,000 draws, null ADOPT rate
  0.061 vs 2^-4 = 0.0625; S4 confirmed by direct read of
  `injury.observations`/`walk_forward`.
- `plan` phase 0 is in-season and dated; its steps are re-ordered as weeks pass, and
  re-orderings live in the tracker, never here.
- `overlap_with_audit_iv` lists 15 items this round found independently that are already on
  the tracker from audits I-IV. No S ticket was filed for any of them; if one of those
  R/B/C/G tickets is later closed as a duplicate of an S ticket, that is a divergence worth
  recording here.
- 2026-09-20 grilling session: the freeze (#326) was restated on its own axis ("blocks a
  modelling change landing") and #360, #361, #365, #366, #367, #369, #371 were given
  `blocked_by #326` edges; #357 is exempt as the freeze's own release condition -- a finding
  cannot be blocked by the freeze it is the release condition for. #375 (the phase-2
  posterior, `plan[2].steps[0]`) and #376 (S6's NOT-RUNNABLE remedy -- measure the
  draft/weekly gate ceilings) were filed. Adopted: #363 option 1, #335 (A)+(i) with
  `TIE_MIN_CLUSTERS = 12`, #364 option 3, #358 option 1, #349 (a), #372 as drafted.
- **S10, restated 2026-09-21 (rule 13):** the finding named `WEEKLY_TRIALS = 400` as the
  leverage study's trial count; the study spent 1600 per arm. Its actionable claim -- resolvable
  for want of CPU -- was refuted by the 4,000-trial re-run under #368. The mechanism was real,
  the constant wrong, the conclusion wrong. `docs/pool-leverage.md` carries the restatement and
  the disposition of the first run's 24-of-25 figure (superseded and unestablished, #167's
  form). The JSON is not edited.
- **S12, restated 2026-09-21 (rule 13):** the finding's third piece named the odds poll cadence as
  why the lines archive is thin. The cause is persistence, not cadence: `poll_odds` and the
  pre-existing `refresh` job both write captures into a gitignored `data/processed` store on an
  ephemeral runner, so no CI poll has ever persisted one and the durable archive has only ever
  been local runs. Raising the cadence to six polls a week added four more that also evaporate.
  #383 holds the decision on where a CI capture should live; #370 carries the correction.
- **The verdict's power figures, restated 2026-09-21 (rule 13):** the headline *"a sign test with
  14% power"* was derived from the repo's published standard errors against a faithful
  reimplementation. The rule it described no longer exists: S1 (#357) made the interval half
  independent of the sign half, and #335 added the tie-aware every-season half. Measured on the
  combined rule at 10,000 trials (`scripts/rule16_combined_power.py`, ADR-0019): null size 0.011
  and power 0.032 at delta = 2.0 pts/team-game for the draft gate; 0.019 and 0.19 at
  delta = 0.3 pts/team-week for the weekly gate. Power fell rather than rose, and the mechanism is
  that one tie abstains from both halves -- a single unresolvable season forces SHOW whatever the
  effect elsewhere. The draft gate's ADOPT branch is a named rule-16 exemption. `docs/gate-power.md`
  and ADR-0019 carry both the numbers and the mechanism.
- **Championship equity, restated 2026-09-21 (rule 13):** the audit reasons throughout from
  ADR-0009's REMOVE as this repo's headline removal. Under the rule S1 and #335 produced, the
  2026-09-21 draft-gate run reads **SHOW** -- won 0, tied 1, lost 3 of 4, with 2024's -5.99 inside
  its own noise at 20 rooms. The decision stands on judgment and three of four losses; the verdict
  word does not. `README.md` row 1, `docs/method.md` row 6, ADR-0009 and `docs/gate-power.md` carry
  the dated restatement beside their originals.
- **`method_rule_to_add`, superseded 2026-09-21:** the drafted clause landed as `docs/method.md`
  rule 17. A fourth instance in three days -- a decision rule, a test fixture, an operational `gh`
  query and a power harness, each a check whose outcome could not vary with the thing it checked --
  promoted the pattern to **rule 18**: every check gets a positive control, planted and seen to fire
  before the check is trusted. Rule 18 carries its own contract
  (`tests/contracts/test_every_check_has_a_positive_control.py`), walking every verdict, gate and
  guard in `src/hub`; rules 15 and 17 are named as its test-shaped and rule-shaped special cases.
- **Findings that landed in a different shape, 2026-09-21:** S5's remedy is wider than the finding --
  the width ledger is append-only *and* compares only at an identical `config_digest` and
  `data_digest`, and records per-season `gain`, `se`, `m` and disposition (#362, #382, `4c03934`).
  S9, S11 and S13 carry `blocked_by #326` edges rather than being phase-0 work. S8 and S8a are
  pre-registered and adoptable as unwired exhibits, and their stage 2 is blocked on
  snapshot-archive depth rather than on the freeze -- which makes #383 upstream of either rating
  gate ever being decidable.
