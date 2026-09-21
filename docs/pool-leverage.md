# Survivor pool: the rival-attrition leverage term

`hub.season.pool.leverage` prices what `hub.season.pool.weekly` omits from every published
figure: the current week's rival attrition, correlated with our own pick, which is the whole
economic case for a contrarian departure from the free pick (#161). This page is the record of
whether that term has been *resolved* -- distinguishable from zero at the repo's evidential
bar -- and it is restated here, dated, rather than edited over the top of itself, per
[method.md rule 13](method.md).

Not a slate artifact: nothing on this page is a pick, a dollar figure a reader acts on this
week, or a model constant. It is a study result about the instrument itself.

## 2026-09-11: the first run, 1600 trials per arm

Five candidates against the free pick T31, five default concentrations (1, 2, 4, 8, 16),
twenty-five comparisons, on the synthetic 32-team board `tests/unit/test_pool.py::_board`
builds -- weeks 1-14, week 1 decided, 21 entries, pot $420, seed 0, both pool rules at
`split` -- 415 seconds:

* **Not resolvable at two standard errors on the axis.** 2 of 25 comparisons cleared the bar
  (T30 at concentration 8.0, z 2.3; T28 at 16.0, z 2.5) against about 1.1 by chance. Unresolved,
  not zero.
* **The direction was consistent and grew with concentration.** 24 of 25 terms positive -- the
  attrition favoured the departure, as #161's argument predicts -- though the rows share their
  draws and are not twenty-four independent readings.
* **The resolution is the advanced arm's**, and its standard error is what set the bar:
  resolving a +$3 term at concentration 16 needed roughly 4,000 trials per arm against the
  1600 spent. `WEEKLY_TRIALS = 400` -- what the weekly path spends, not what this run spent --
  was never the constraint on the weekly path; it was simply the number nobody had raised for
  this study alone. This is `LEVERAGE` in `hub.season.pool`, the caveat string every weekly
  figure prints beside `pool_digest` when `--leverage` is not given.

## 2026-09-21: re-run at 4,000 trials per arm, under #368 (S10)

The module's own arithmetic said 4,000 trials per arm would resolve a +$3 term at
concentration 16 if the effect it measured in 1600 trials were real. `LEVERAGE_STUDY_TRIALS =
4000` (`hub.season.pool`, separate from `WEEKLY_TRIALS` so the weekly path's speed is
untouched) is that number, and `scripts/leverage_study.py` is the harness ADR-0007 asks for --
committed, tested by inspection against the weekly path's own tests, and re-runnable:

```
uv run python scripts/leverage_study.py --trials 4000
```

Same board formula, same seed (0), same week (2, "week 1 decided"), same entries (21), same
pot ($420), same rules (`split`/`split`), same five-candidate top and five-concentration axis
as the first run. On today's codebase the free pick at week 2 is **T30**, not T31 -- the
ranking the intervening commits since 2026-09-11 produce for this board differs from what
2026-09-11's code produced, which is itself informative (see caveats below).

| | 2026-09-11 (1600/arm) | 2026-09-21 (4000/arm) |
|---|---|---|
| comparisons clearing 2 s.e. | 2 of 25 | 2 of 25 |
| expected by chance | ~1.1 | ~1.1 |
| terms positive | 24 of 25 | **5 of 25** |
| `resolvable_on_the_axis` | False | False |
| verdict | unresolved | **unresolved** |

**Still not resolvable at 4,000 trials per arm.** 2 of 25 comparisons clear two standard
errors (T26 at concentration 2.0, term −$9.69 against a ±$8.99 bar, z ~2.16; T28 at
concentration 16.0, term −$4.29 against a ±$4.14 bar, z ~2.07), against the same ~1.1 chance
gives on 25 null comparisons -- so raising the trial count fourfold changed nothing about
whether the term clears the bar. `resolvable_on_the_axis` (which holds the *count* of hits to
its own null rather than reading any one row) says `False` at both trial counts. Both of this
run's resolvable hits are **negative** -- unlike the first run's two (T30 at 8.0, T28 at 16.0),
which were both positive.

**The direction-consistency finding does not reproduce.** The first run reported 24 of 25
terms positive, read as directional support for #161's argument even though unresolved. This
run finds 5 of 25 positive -- the opposite pattern. Both runs are honest about the rows sharing
their draws and not being that many independent readings (one seed; the arms meet the same
game results at every concentration; the five candidates at one concentration share the free
pick's arm) -- but a reader comparing the two tables should not carry forward "24 of 25
positive, growing with concentration" as a standing fact about this term. It was a property of
one seed's draw at 1600 trials on 2026-09-11's codebase, not a stable direction.

## What changed between the two runs, honestly

This is **not** a controlled replication that varied only the trial count. Ten months of
commits landed on `hub.season.pool` between the two runs (#152 stated the concentration axis,
#156 closed the both-sides-of-a-fixture gap, #159 built the pairing, #161 itself, and more).
The free pick moving from T31 to T30 on the identical board and week is direct evidence the
candidate ranking is not byte-identical to what it was on 2026-09-11 -- most plausibly a
correctness fix reordering which team `auto_pick` and the candidate list prefer, since the
board's win probabilities are computed by the same formula, copied verbatim into
`scripts/leverage_study.py`.

So this run answers "is the term resolvable at 4,000 trials on the codebase as it stands
today" -- yes to "still not resolvable", no additional signal that the 2026-09-11 sign
pattern was ever a property of the term rather than of that run's seed and code. A true
apples-to-apples replication would need to pin the exact commit the first run was made
against, which nobody recorded (ADR-0007's gap this ticket does not close: the *first* run
predates the harness that would have made it reproducible byte-for-byte).

## Where this leaves #161's caveat

`LEVERAGE` in `hub.season.pool` still states the 2026-09-11 finding and is still what every
`hub.season.pool` weekly figure prints beside `pool_digest` when `--leverage` is not given
(prior text kept, per this page's own opening rule). It is not rewritten here: the module
prints a *sentence a reader can check against this page*, not a live statistic, and updating
it is a separate, deliberate edit -- not a side effect of running a study. What this page adds
is that raising the trials fourfold did not resolve the term, so the "somebody should run
4,000" clause in that sentence is discharged without the sentence's headline changing: the
term is still unresolved, not zero, on both counts.

**Not adopted, not acted on.** No published survivor figure moves off this page. If a future
session wants `LEVERAGE`'s text itself to carry the 2026-09-21 finding, or wants a true pinned
replication, that is a follow-up ticket, not a rerun of this one.

## 2026-09-21, restated the same morning: what the second run does and does not say

Read back by the maintainer after the run above landed. Two corrections, both under
[rule 13](method.md), dated beside the text they correct rather than over it.

**S10 (`docs/audits/2026-09-20-method-audit.json`) was wrong in two ways, and the run proved
both.** The audit named `WEEKLY_TRIALS = 400` as the study's trial count; the study spent
**1600** per arm -- `WEEKLY_TRIALS` is the weekly path's number, and the first section above
already says so. And the audit's actionable claim -- *unresolved for want of CPU, resolvable by
an afternoon* -- is refuted: 4,000 trials per arm did not resolve it. The mechanism the audit
described (nobody had raised the study's own trial count) was real; the constant was wrong; the
conclusion drawn from it was wrong. The audit file is never edited; `docs/audits/README.md`
carries the dated pointer here.

**The 24/25 → 5/25 reversal is not a finding about leverage.** The section above says the free
pick moved **T31 → T30** on the same board, seed and week, because intervening commits changed
the ranking. So the two runs do not share an arm, a baseline, or necessarily a candidate set --
the comparison is confounded, and *"the direction-consistency finding does not reproduce"*
claims more than it can. The right disposition is [#167](https://github.com/jacksonmlukas/football-hub/issues/167)'s:
**the 24-of-25 figure is superseded and unestablished**, because no re-run of it is possible --
not because a re-run contradicted it. The 5/25 figure is a property of this run alone and is
not evidence about the first.

**Before any third run, pin the board.** #167's root cause applies verbatim: *a measurement whose
inputs are not pinned can be contradicted and never corrected.* The free pick and the candidate
set are to be committed as a fixture, the way `rank_tiers`' docstring was rebuilt onto one, so
that a third run disagreeing with the second is information and not the same lesson a third
time. That is a ticket, not this page.

## 2026-09-21: the board is pinned (#377) -- still no third run

`tests/golden/fixtures/pool_leverage_candidates.json` is that fixture: the free pick (**T30**)
and the five candidates (T28, T27, T26, T25, T24) `pool.candidate_ranking` returns for this
board, week and rules today, dated. `scripts/leverage_study.py` reads it and studies those six
teams on every run rather than re-ranking the board live, so a future run compares against this
page's two runs on the same arms instead of drifting the way the free pick drifted between them.
A ranking change is now a fixture diff, caught by
`test_the_committed_leverage_fixture_matches_todays_ranking` in `tests/unit/test_pool.py` before
it can confound a study -- the test this page's two runs above did not have. `--refit`
regenerates the fixture from today's ranking and prints the diff against the committed one; it
spends no trials.

**No third run is made here.** This section pins the board a third run would need; it does not
run one. `LEVERAGE_STUDY_TRIALS = 4000` and the verdict above -- unresolved, both runs -- stand
untouched. A future session that wants a true pinned replication runs
`uv run python scripts/leverage_study.py` against this fixture and restates this page under
[rule 13](method.md), the way the two runs above did.
