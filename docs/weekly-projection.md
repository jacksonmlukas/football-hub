# The Weekly projection

**Built 2026-08-27.** `hub.models.weekly` is the week-specific layer: everything else in this
repo projects one season-long per-game mean and applies it flat to all seventeen weeks.

Its shape was fixed by measurement rather than chosen. See
[weekly-screen.md](weekly-screen.md) for the screen and
[weekly-projection-plan.md](weekly-projection-plan.md) for the pre-registration.

    weekly Usage  = season-to-date Usage x exp(coef . snap_trend)
    weekly TDs    = weekly yards x the POSITION's touchdown rate
    weekly points = league scoring applied to the counts

**`f = 1` is the incumbent.** With `coef = 0` the multiplier is one and the projection is
exactly the flat one, so the null is the identity and this cannot be much worse than what it
adjusts. **The multiplier acts on counts, never on points.**

> **Restated 2026-09-11 — the Usage multiplier is the identity, under
> [method.md rule 13](method.md).** Issue #233 decided it and #248 implements it. On the
> `(yds_prior, ecr)` basis #229 settled, the snap-share trend is **+0.0142 at permutation
> *p* 0.24** at the published anchor 8, +0.0081 at *p* 0.66 at 12, and clears at anchor 10
> alone; a claim that survives one row filter and not another is a claim about the filter,
> and the pooled +0.038 was measured on a control basis found to concentrate the yardage
> confound. So the first line of the formula above now reads `weekly Usage = season-to-date
> Usage` — the incumbent this page names in the paragraph before this box — and the other
> two lines are unchanged. **The two coefficients' fates, side by side:**
>
> | | where it acted | on the settled basis | licence | in `hub.models.weekly` since #248 |
> |---|---|---|---|---|
> | `coef` on the snap-share trend | the Usage multiplier, `exp(coef · snap_trend)`, fitted each fold | +0.0142, *p* 0.24 at anchor 8; killed at 4, 6, 8 and 12; clears at 10 alone | **revoked** — #233 | **the identity.** `project` takes no coefficient; `fit_multiplier`, `multiplier` and the clip `[0.6, 1.6]` are gone with it, and the multiplier is exactly 1.0 for every player-week |
> | the touchdown term — the prior TD rate's finding, applied as the position's rate on projected yards | `weekly TDs = weekly yards × the POSITION's touchdown rate` | −0.012 at −2.49 se, 4/5 seasons — not a finding, rule 4 (#229) | **qualified, not withdrawn** — #229; not decided by #233 or here | **untouched.** `components.td_rate` did not move and `tds_hat` is byte-identical to what the `f = 1` arm always projected |
>
> **What moves on this page.** *The fitted multiplier*, below, is a coefficient nothing
> applies: its table stands as the record of what the fit returned and is no longer
> re-runnable from `--fit`, which fits nothing now; the commit before #248 landed is where
> it runs. Under *Gate A*, the **"the week"** row and the **"both together"** row are
> contrasts against an arm that no longer exists; **"the rebuild — `f = 1` vs flat"** is the
> weekly projection's own figure now, on the same held-out rows, and `--fit` prints it and
> nothing else. The *What moves* table under *Three specification errors* is unaffected —
> those were rebuild defects and were found on the `f = 1` arm.
>
> **What does not move.** The Panel, `panel.TREND_MIN_WEEK` (it is still the threshold the
> `snap_trend` column is dark before, and `hub.models.weekly` simply no longer reads it),
> every touchdown figure, the efficiency rule, the turnover term, and the shrinkage
> experiment's grids and objectives — `fit_shrink` no longer takes a coefficient and returns
> what it returned. The weekly blend gate's inputs still build and its **REMOVE** verdict
> stands; it is re-run once after this lands, on the machine, to record what the change cost
> — that figure is not on this page yet. No fitted constant moved: neither weekly module is
> in `FITTED_MODULES`, and `config_digest` / `fitted_digest` read the same before and after.
>
> **Where the record lives.** The screen keeps `snap_trend` in its family and prints its
> cell at every anchor with the note that it is not licensed
> ([weekly-screen.md](weekly-screen.md), *What this does and does not license*), and
> `--permute snap_trend` reproduces the measurement #233 decided on. That is the harness
> ADR-0007 asks for; the multiplier's own fit was a diagnostic and never a reason.

## Why these two terms and nothing else

Nine features were screened, two survived a joint screen, and screening those two against the
counts rather than the total showed they do not overlap:

> **Restated 2026-09-10 — the family is eight, under [method.md rule 13](method.md).** Issue
> #170 removed `wind` from the screen: it is game-time observed weather, read off the schedule
> as recorded conditions and screened as a week-*w* feature with a pre-stated sign, which is
> [method.md rule 2](method.md). It was killed at 4/5 seasons and never reached the joint
> screen, so **neither surviving term moves and no number in the table below is affected**.
> What moves is the size of the family the two came out of. See
> [weekly-screen.md](weekly-screen.md) for what the wind row itself published and why it
> cannot be re-run.

| | targets | receptions | carries | attempts | **touchdowns** |
|---|---|---|---|---|---|
| snap-share trend | +0.094 | +0.077 | +0.062 | +0.032 | **+0.018 killed** |
| prior TD rate | +0.008 | −0.000 | −0.005 | +0.005 | **−0.120** |

So one goes in the Usage multiplier and the other in the touchdown term, and nothing crosses.

**Efficiency is not projected.** Yards per carry persists at r = 0.108 year over year and
touchdowns per yard at ~0, so a model predicting this week's efficiency is predicting noise.
Each player's efficiency is held at his own accumulated rate and only his *opportunity* moves.
**Spread is not projected either** — `sd = k·√mu` stands per
[ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md).

## The fitted multiplier

> **Superseded 2026-09-11 (#248).** Nothing applies this coefficient: the multiplier is the
> identity, and the fit that produced the table below is not in the tree. The figures stand
> as the record of what it returned. See the restatement at the top of this page.

`coef` in `count = expected × exp(coef · snap_trend)`, least squares on the log ratio, fitted
on strictly earlier seasons. Stable across the walk-forward:

| held-out season | coef targets | coef carries |
|---|---|---|
| 2022 | 0.373 | 0.465 |
| 2023 | 0.414 | 0.503 |
| 2024 | 0.512 | 0.546 |
| 2025 | 0.542 | 0.523 |

Snap share is a fraction, so a **ten-point** rise in snap share is `exp(0.5 × 0.10) ≈ 1.05` —
about **five per cent** more targets. Small, which is the honest size of it, and the clip at
[0.6, 1.6] essentially never binds.

## Gate A, which is a diagnostic and not a gate

Three arms, because the first version had two and could not see its own subject.

| contrast | gain | se | seasons |
|---|---|---|---|
| **the week** — weekly vs `f = 1` | **+0.0103 MAE** | +2.8 | **4/4** |
| the rebuild — `f = 1` vs flat | +0.0634 MAE | +5.3 | 4/4 |
| both together — weekly vs flat | +0.0737 MAE | +5.9 | 4/4 |

All three clear both halves of the repo's usual bar. **None of them is the gate**: the flat
projection has no week-level term at all, so beating it is nearly free, and what decides
whether this ships is the lineup ([ADR-0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md)).

> **Restated 2026-09-11 (#248).** The weekly projection is the `f = 1` arm, so the second
> row is its figure and the first and third are contrasts against an arm that is gone.
> `--fit` prints the rebuild alone now. The +0.074 that
> [ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) and
> `CONTEXT.md` cite as the projection's accuracy gain is the third row; the figure that
> describes the shipped projection is the second, +0.0634. ADR-0016's decision does not move
> here: its gate scored the fitted arm at −0.304 and is re-run once after this lands, by the
> maintainer, to record what the arm now scores — the number is not on this page.

## Three specification errors, and what each was worth

Recorded because each one inverted or hid the result, and because the first two were found by
checking calibration rather than by anything failing.

**1. The diagnostic compared the wrong things.** It carried two arms — the fitted projection
against `ppg_before` — so it compared a *component rebuild* with a *points mean* and buried the
multiplier under every difference between two whole estimators. It reported **−0.0025 MAE at
1/4 seasons** and that number was about the rebuild, not about the week. The plan had already
said `f = 1` is the incumbent; the code did not implement it.

**2. Interceptions and fumbles were not projected at all.** `components.SCORING` prices both at
−2. Leaving them out over-projected quarterbacks by **+1.44 points a week**, which is almost
exactly what an interception a game costs. A projection missing two of the scoring components
is not projecting fantasy points.

**3. The efficiency floor compared a per-game mean against a total.** `MIN_UNITS = 8` is eight
*accumulated* units, and it was being tested against a per-game figure — so essentially every
receiver failed it, since nobody catches eight passes a game, and got the pooled yards-per-catch
rate instead of his own. That under-projected receivers by **−0.66 points a week**.

Fixing 2 and 3 moved the rebuild from −0.009 MAE (1/4 seasons) to +0.062 (4/4), and the week
from +0.0067 at 1.5 se to +0.0103 at 2.7 se. **The bugs were the entire result.** Per-position
bias, which is what surfaced them:

| | before | after |
|---|---|---|
| QB | +1.44 | −0.30 |
| RB | +0.58 | −0.23 |
| WR | −0.61 | −0.08 |
| TE | −0.25 | −0.47 |

The overall bias was −0.002 the whole time. Pooling hid all four.

## What is not done

**Gate B has been run and this does not set a lineup.** A lineup set off the Weekly projection
lost to one set off weekly consensus rank by **−0.684 points per team-week**, in all three
held-out seasons, CI [−1.519, +0.159] — *SHOW, NEVER RANK ON*. See
[weekly-gate.md](weekly-gate.md). *(As first measured; restated 2026-08-28 to −0.304, CI
[−1.043, +0.415], two of three seasons — the first run was one draw from a non-reproducible
board, improvements.md #18. The verdict did not move.)* Beating the flat projection on
accuracy and losing the lineup decision to a free public ranking is the screen/gate
distinction at its sharpest.

**The injury designation is not in the model, and it cannot go in here.** The plan admits it
to Phase 2 unscreened, having been measured at +0.170 MAE and 3.8 se. But the Gate A panel is
built from `player_stats`, which has no row for a player who did not play — so of **5,473 "Out"
designations across 2021-25, six reach the panel**. Doubtful is 764 against 2.

`hub.models.injury` scores an injury row with no stat row as *zero*: the player who did not
play is its entire subject, and here he is structurally absent. Fitting retention on these rows
would measure something much weaker — "what a Questionable player who played anyway retains" —
and would report it under the stronger result's name.

This is a direct consequence of the pre-registered treatment of inactive weeks (excluded from
Gate A, zero in Gate B), and it means **the injury term belongs in Gate B's construction**,
which builds a complete player-week grid where a missing row is a zero. `status` and `practice`
are carried on the panel for that purpose. There is a test pinning the constraint so it is not
quietly fitted here later.

## Reproduce

```bash
uv run python -m hub.models.weekly --fit
```
