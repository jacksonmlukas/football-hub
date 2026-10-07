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
  effect elsewhere. **Restated 2026-10-06 (#418, rule 13):** those four figures are reproduced to
  the digit by the script today, and are the combined rule's power under a stand-in within-season
  spread (cluster SD set equal to the between-season `s`; a season SE of 0.060 on the weekly path
  against the recorded 0.41-0.55). At the gates' own recorded precision the same harness gives
  null 0.0001 / power 0.0008 (weekly, delta = 0.3) and null 0.0014 / power 0.0075 (draft,
  delta = 2.0). The mechanism and the exemption stand, more strongly; see `docs/gate-power.md`
  (*What contradicts a published figure*) for the attribution. The draft gate's ADOPT branch is
  a named rule-16 exemption. `docs/gate-power.md`
  and ADR-0019 carry both the numbers and the mechanism.
- **Championship equity, restated 2026-09-21 (rule 13):** the audit reasons throughout from
  ADR-0009's REMOVE as this repo's headline removal. Under the rule S1 and #335 produced, the
  2026-09-21 draft-gate run reads **SHOW** -- won 0, tied 1, lost 3 of 4, with 2024's -5.99 inside
  its own noise at 20 rooms. The decision stands on judgment and three of four losses; the verdict
  word does not. `README.md` row 1, `docs/method.md` row 6, ADR-0009 and `docs/gate-power.md` carry
  the dated restatement beside their originals. *(Restated 2026-10-06, #381, rule 13: under (C),
  ADR-0019's 2026-10-06 amendment, a re-run of the same recipe reads **REMOVE on 3 resolved of 4,
  1 abstained**; prior value SHOW. The same four places carry that restatement; the re-run is not
  bit-for-bit #376's, see `docs/gate-power.md`, where #429 attributes it to an unrecorded input,
  not code or seeds; today's run stands and the verdict word is unchanged.)*
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

## 2026-10-06 Audit V

`2026-10-06-audit-v.json` is **Audit V (#325)**: the post-freeze read of the week-3/4 artifacts.
It ran after `uv run python -m hub.audit_ready --season 2026 --week 4` printed *"READY for 2026
week 4: every predicted game is scored"* and exited 0, on head `3992074`. Findings V1–V6 are
#419–#424, all in the milestone "Audit V — in-season repairs". None is tagged `blocking`.
The re-plan lives in the tracker, not here: Phase 2 gains the four unplaced Phase-2-shaped
tickets and an edge order through #418 → #381. Phase 3 hands its survivor steps to "Dormant —
the 2027 pool (#379)". "Audit IV — unplaced" is emptied and closed. Known limits of the read,
stated in the file rather than discovered later:

- V1's mechanism is inferred from code. A summary read of the ESPN-derived roster parquet was
  refused by the session's permission system. The symptom is read directly off the published
  `roster.json` at six slate commits.
- Whether any 2026 in-season starter change has happened is not established. V2 is that the
  study's inputs cannot show one either way.

## 2026-10-06 Week-4 instrument check (#389) -- no verdict

`uv run python -m hub.instrument_check --week 4 --from 2026-09-30T00:00 --until 2026-10-06T00:00`
(`src/hub/instrument_check.py`). It checks the instrument and **produces no ADOPT, REMOVE or SHOW**;
no published number moves, so rule 13 is not engaged and the #326 freeze is not crossed. The reason
for the check is the k argument in #389: four scored weeks add no cluster, so a verdict then would
read SHOW whatever is true.

What ran on 2026-10-06, from a fixtures-and-HEAD checkout with no `data/`:

- **Criterion 2, persistence: FAIL for week 4, and it cannot be repaired.** The six polls
  scheduled in week 4's window (Wed 09-30 11:00Z through Mon 10-05 11:00Z) have no committed
  capture in a fresh clone of `HEAD`. #383 landed 2026-10-06 19:24Z; nothing scheduled before it was
  ever committed, and a market price cannot be re-captured. `state/odds/2026/` holds only #428's
  back-fill (8 polls, 2026-08-25 to 09-06, all 18 weeks, 1,834 rows), which the same run confirms
  reads from a fresh clone with the clone's own `hub.store`. So the reading **is unavailable for
  weeks 1 to 4 beyond that back-fill**, and that is the finding #389 said a failure would be. The slots are
  *scheduled* instants, not proof a run fired; the check cannot tell a lost run from an undelivered
  one, and for this window the difference does not change what is readable.
- **Positive control (rule 18)**, `tests/unit/test_instrument_check.py`: one capture removed from a
  committed archive turns criterion 2 red; so does one left untracked in the working tree. A
  window with no poll past its grace is `NOT-YET`, never `PASS`.
  `tests/contracts/test_odds_archive_reads_from_a_fresh_checkout.py` holds the back-fill's readability
  as a standing check.

**Not yet established, and what remains.** Criteria 1, 3 and 4 read the stamped paired parquet of a
gate run over real 2026 rows (`weekly_gate --run --out`, then `--paired`), which needs the pinned
nflverse data and was not run here: no `data/` exists in this worktree. They are `NOT-YET`, not
`PASS`, and the check exits 3 until they are supplied. The first in-season capture that persists is
the 2026-10-07 11:00Z slate run; after it, `--from 2026-10-07T00:00 --until <later>` is the first
window where criterion 2 can pass. Criterion 4's k is not yet reported for a real run.
