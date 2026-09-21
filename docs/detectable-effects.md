# The detectable effect, for each of the fifteen measurements

**S2a (#359).** For each row of `docs/method.md`'s fifteen-measurement table, what that
design could have detected at 80% power: the between-season standard deviation `s`, the
number of held-out seasons `k`, and the resulting minimum detectable effect `delta = (t(0.975,
k-1) + z(0.80)) * s / sqrt(k)` -- `hub.models.experiment.minimum_detectable_effect`, unchanged
by #357 (S1); what #357 fixed was which interval `experiment.gate`'s ADOPT/REMOVE decision
reads, not the MDE formula itself, so a `delta` already published anywhere in this repo since
#45 (2026-09-07) is already the fixed rule's number and is cited rather than recomputed.

**One table, one axis, no decision.** This page computes what a design could resolve; it does
not say whether any measurement's own effect cleared that bar, and it changes no verdict.

**Where a row's inputs no longer exist, the row says so rather than estimating** -- several do.

## How `s` was obtained, and why two methods appear on this page

Three gates (`draft`, `interval_shape`, `weekly` blend) run through `experiment.summarise`,
which clusters on the season and reports a **bootstrapped** standard error directly; where that
figure is published in a doc, it is quoted here and `s = se_published * sqrt(k)` is shown for
comparability with the rows that have no such figure. Everything else on this page -- the
screens (partial correlations) and the two gates that predate or sit outside that machinery
(`spread`, `injury`) -- has no published season-clustered `se`, only per-season point figures,
so `s` is the **closed-form sample standard deviation** of those `k` numbers (`ddof=1`) and
`se = s / sqrt(k)`. The two methods agree asymptotically and disagree at small `k` the way a
bootstrap and a closed-form estimate of the same quantity always can; which method produced
each row's `s` is stated in its own row.

**Screens are not gates, and this page does not pretend otherwise.** A screen's effect is a
partial correlation `r`, not a paired points/MAE gain, so its `delta` is a detectable `r` on
this same formula's arithmetic -- read as "the smallest per-season swing in `r` this design
could resolve at 80% power," not as anything comparable across rows of different units. No
Fisher-z transform is applied; the formula is run on `r` directly, which is the same
approximation `docs/method.md` rule 14's family-wide count already accepts implicitly by
reading `t` off `r` without one. Units are stated in every row for this reason.

## The table

| # | measurement | kind | s | k | se | delta (80% power) | note |
|---|---|---|---|---|---|---|---|
| 1 | Expected-vs-actual points, next season (uniform weighting) | screen | 0.0456 | 4 | 0.0228 | **0.092** (r) | closed-form; per-season r 2021-22 +0.249, 2022-23 +0.152, 2023-24 +0.192, 2024-25 +0.242 ([lambda-sweep.md](lambda-sweep.md)) |
| 2 | Recency-weighted variant | screen | — | — | — | **not computed** | only the pooled r across k=4 seasons is published for each half-life (8wk +0.193, 4wk +0.147, 2wk +0.090, [lambda-sweep.md](lambda-sweep.md)); no per-season breakdown was found to compute `s` from |
| 3 | Depth-chart climb, next season | screen | 0.0607 | 4 | 0.0303 | **0.122** (r) | closed-form; per-season partial r 2021-22 +0.007, 2022-23 +0.074, 2023-24 +0.025, 2024-25 −0.072 ([depth-chart-signal.md](depth-chart-signal.md)) |
| 4 | Depth-chart climb, rest-of-season | screen | 0.104 | 4 | **0.052 (published)** | **0.209** (r) | `se` read directly off the published per-season table (week 10, "all": 2021 −0.040, 2022 +0.100, 2023 −0.049, 2024 +0.035, se=0.052, [depth-chart-signal.md](depth-chart-signal.md)) |
| 5 | Age | screen | 0.0638 | 5 | 0.0285 | **0.103** (r) | closed-form; per-season partial r −0.012, +0.037, −0.085, +0.010, +0.088 ([signal-screens.md](signal-screens.md)) |
| 6 | Championship equity as the objective (draft gate) | gate | 5.552 | 4 | **2.776 (published, back-derived)** | **11.17** (pts/team-game, published directly) | the **current, restated** figure (2026-09-16 hold-out re-run, #290): per-season −19.28 / −6.72 / −8.61 / −18.24, MDE 11.17 printed directly ([ADR-0009](adr/0009-championship-equity-does-not-pick.md)); the earlier shipped-constants figure (s implied 7.34, se 3.67, as #357's own table quotes) is superseded per rule 13 and not used here |
| 7 | VOR ordering | gate-shaped | — | — | — | **not computed** | only a pooled figure is published (−5.06, CI [−7.90, −2.18], 3 seasons × 40 drafts = 120 rows pooled, [market-value.md](market-value.md)); no per-season breakdown exists to compute a between-season `s` from |
| 8 | `edge`, the repo's original signal | unvalidatable | — | — | — | **no s/k exists** | closed as unvalidatable, not merely unmeasured: no historical ADP source exists to build one, and backtest boards carry no `adp`/`edge` column at all ([ADR-0010](adr/0010-edge-is-displayed-but-never-ranked-on.md)) |
| 9 | Volume model beating the market's mean | model | — | — | — | **not applicable** | fit on 2022-23, evaluated pooled on 2024-25 only -- one evaluation window, not multiple held-out seasons to cluster on ([volume-model.md](volume-model.md)) |
| 10 | Lineup optimiser | gate | — | — | — | **not computed** | `docs/gate-power.md` states this gate builds its paired frame from the network at run time and persists nothing, so a season-clustered `s` "cannot be measured offline"; stage 2's own comparator question is still open (#138) |
| 11 | Per-player weekly spread | gate (`hub.models.spread`) | 0.0033 | 5 | 0.0015 | **0.0054** (MAE) | closed-form; per-season gain (`own_k` vs `positional`) +0.0033, +0.0081, +0.0078, +0.0106, +0.0029 ([player-spread.md](player-spread.md)). Season-clustered, and deliberately not the pooled "1.8 se" this module's own `verdict()` prints, which is unclustered across every player-season row |
| 12 | Weekly injury retention | gate | — | — | — | **not computed** | only the pooled figure (+0.170 MAE at 3.8 se, n=3,687 player-weeks) and a qualitative "better in all three held-out seasons" are published, no per-season gain values ([weekly-injury.md](weekly-injury.md)) |
| 13 | Injury type on top of it | gate | 0.0410 | 3 | 0.0236 | **0.122** (MAE) | closed-form; per-season gain (type-adjusted vs retention) +0.0507, +0.0602, −0.0150 ([weekly-injury.md](weekly-injury.md)) |
| 14 | Snap-share trend | screen | 0.0592 | 4 | 0.0296 | **0.119** (r) | closed-form; per-season partial r beyond ECR at the published anchor (week ≥ 8/10) +0.2540, +0.1496, +0.2815, +0.2616 ([snap-trend-signal.md](snap-trend-signal.md)). The paired-points decision test on the same signal (top-1/top-3 by snap delta vs ECR) publishes only a win count, not per-season values, so no second `delta` is computed for it |
| 15 | Weekly projection vs weekly consensus rank (the weekly blend gate) | gate | 0.382 | 4 | **0.191 (published)** | **0.768** (pts/team-week, published directly) | current restated figure: per-season −0.452, −1.490, −0.868, −1.204, se and MDE printed directly ([weekly-blend-gate.md](weekly-blend-gate.md)); `s = se * sqrt(k)` reproduces #357's own table exactly |

## Reading this table

**Six rows have no `delta`: #2, #7, #8, #9, #10, #12.** Two shapes among them: #8 is closed as
unvalidatable and never will have one; the other five are measurements whose surviving
published record is a pooled figure, a qualitative season count, or (for #9) a single
evaluation window rather than multiple held-out seasons -- none of these facts changed because
this ticket ran, and none is estimated here rather than reported absent.

**Nine rows have a `delta`, on two different bases.** Three (#4, #6, #15) use a season-
clustered `se` this repo's own gates already publish; six (#1, #3, #5, #11, #13, #14) use a
closed-form `s` computed from per-season point figures because no season-clustered `se` was
ever published for them. The two are not interchangeable and each row says which it is.

**Nothing here is re-run, and no verdict moves.** Every number this page cites is already
published somewhere in `docs/`; this page's own contribution is the one arithmetic step
(`minimum_detectable_effect`) applied uniformly, and the six rows it could not apply that step
to because the inputs do not exist.
