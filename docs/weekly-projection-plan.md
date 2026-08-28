# Weekly component projection — the plan

**Written 2026-08-27, before any measurement.** Pre-registration, per
[method.md](method.md) rule #1: the decision rules, the incumbents and both gates are fixed
here, in writing, before a number exists to be tempted by.

---

## What is actually new here

Not "components instead of points". That is already the repo's position and it is built:
[`hub.models.components`](../src/hub/models/components.py) reconstructs points from
receptions, carries, yards and touchdowns, the distribution families behind each count were
*measured* rather than assumed, and the whole thing is written up in
[component-projection.md](component-projection.md).

What is new is **the week**. Every projection this repo makes today is one season-long
per-game mean applied flat to all seventeen weeks. Ja'Marr Chase is 19.6 points in Week 1 and
19.6 points in Week 12, against any opponent, in any game script, whatever his snap count did
last Sunday. There is no week-level term anywhere in the system.

So the honest name for this work is **a week-specific component model**, and the thing it has
to earn is the *week*, not the decomposition. Saying that precisely matters, because two
neighbouring things have already been measured and lost, and this must not quietly re-run
either of them.

## What the record already settles

Do not re-measure these. They are in the tree with their harnesses.

| already measured | result | so |
|---|---|---|
| Component-derived weekly **spread** vs `sd = k·√mu` | lost, 1.365 vs 1.140, P(better) 0.0% | the square-root law stays. This model supplies **means**, not a new spread law ([ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) keeps the losing code) |
| Volume/efficiency shrinkage toward a **positional** mean | null, −0.2% RMSE, P(better) 40.2% | shrinking a WR1 and a WR5 to the same place over-shrinks the studs |
| The same, toward a **pick-implied** prior | 99.9% against that baseline, but still loses to simply reading the pick | ships as a decomposition, not a projection ([volume-model.md](volume-model.md)) |
| Touchdown rate per yard, year over year | r = **−0.004** receiving, **−0.030** rushing | a player's own TD rate carries no information beyond his yardage. **Season-over-season only** — the week-over-week version is unmeasured and is a screen below, not an assumption |
| Efficiency persistence | ypc 0.108, ypt 0.369, catch rate 0.402 | do not try to predict weekly efficiency. Predict opportunity |
| Volume persistence | targets 0.805, carries 0.791 > points 0.775 | volume is the part that carries, which is the premise of all of this and it holds |

And two positives, which are the only two this repo has ever found in fourteen measurements:

* **Snap-share trend**, next three weeks in season: partial r **+0.236** beyond both PPG and
  ECR, 12/12 season-anchor cells, placebo-clean ([snap-trend-signal.md](snap-trend-signal.md)).
  Its *decision* then failed its gate and it is shown, never ranked on
  ([ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md)).
* **Injury retention** by (status, practice): **+0.170** MAE at 3.8 se
  ([weekly-injury.md](weekly-injury.md)).

Both are *timeliness* signals — something published on a Monday that consensus has not yet
absorbed. Every null asked consensus about information it had had all summer. That pattern is
the sharpest prior available for what follows, and it is stated here before the screen rather
than after it.

---

## The shape

One row per **(player, season, week)**. The model predicts a vector of counts, and league
scoring turns the vector into points — never the other way round.

```
                    week-level information
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   team volume          player share           efficiency
   (plays, pass/run)    (targets, carries)     (yds/target, TD rate)
        │                     │                     │
   implied team total    snap-share trend       ~nothing: ypc r=0.108,
   spread/game script    injury designation      TD/yd r≈0
   pace, rest, roof      depth-chart change
        └─────────────────────┴─────────────────────┘
                              │
                    component means per week
                              │
              measured families (component-projection.md):
              volume → negative binomial (var/mean 1.20–2.45)
              TDs    → binomial (var/mean 0.83–0.86, underdispersed)
              yards  → compound, units × yards-per-unit sampled apart
                              │
                     league scoring weights
              (espn.scoring_settings(), read from the league)
                              │
                        weekly points
```

The structural claim, which is testable and which the persistence table above already
supports: **week-level information moves team volume and player share, and barely touches
efficiency.** A model that tries to predict this week's yards per carry is predicting noise.

### The data is already here

No new fetch is needed to screen or to gate. This was the surprise while scoping.

| need | source | status |
|---|---|---|
| Weekly component **outcomes and expectations** | `ff_opportunity` — `rec_attempt`, `receptions`, `rush_attempt`, `*_yards_gained`, `*_touchdown`, and an `_exp` counterpart for each | **in the store**, 2021-25 |
| Closing **spread and total** per game | `schedules` — `spread_line`, `total_line` | in the fetch layer, free, historical |
| Rest, roof, surface, temp, wind, divisional | `schedules` | same call |
| Snaps / participation | `participation`, `ftn_charting` | fetched, unscreened by design ([improvements.md #4](improvements.md)) |
| Injury designation | `nfl.load_injuries` | already used by `hub.models.injury` |
| Defence vs position | `playoff_sos._dvp_from_stats` | built; currently season-aggregate over playoff weeks |
| Interceptions, fumbles, 2PC | `player_stats` weekly | needs the `cols` list widened past `fantasy_points_ppr` |

Implied team total is `total_line / 2 ± spread_line / 2`, computable for every historical game
at zero API cost. The odds API (`hub.fetch.odds`, `MARKET = "spreads"`, `CREDIT_FLOOR = 50`)
is only needed for *live* freshness in-season, and adding a `totals` market would double the
credits per pull — decide that at Phase 4, not now.

---

## Phase 0 — pre-registration

**Deliverable: this file, extended with the exact rules, before Phase 1 runs.**

### The two incumbents, and why there are two

**Floor incumbent — the flat projection.** Today's `proj_blend` applied unchanged to every
week. Beating it proves only that *week-level information exists*, which is close to
guaranteed: the incumbent has literally no week-to-week variation, so any signal that a
Thursday-night game exists will beat it. **A win here is necessary and means almost nothing.**

**Real incumbent — weekly consensus.** FantasyPros weekly ECR or ESPN's weekly projection: the
free thing a person would actually read on a Saturday. This is the incumbent that matters, it
is the one every previous null lost to, and per the pattern above it is beatable only where we
are *timelier*, not where we are cleverer. It needs a fetch (the board already scrapes
FantasyPros consensus; the weekly page is a different URL) and that fetch is the first
concrete task, because **a gate against the wrong incumbent is worse than no gate.**

### The metric

Paired absolute error on **player-week PPR points**, the same player-week scored by both arms.
Per-observation, not per-season means — which is what
[`experiment.paired_gain`](../src/hub/models/experiment.py) exists to take.

Secondary and reported, never gated on: per-component MAE, so a win can be attributed to
volume or to touchdowns rather than asserted.

### The gate, both halves

The repo's usual bar, unchanged: the arm must beat the incumbent **in every held-out season**
*and* clear **`experiment.MIN_SE` = 2.0 standard errors** on the paired difference. Written as
a `verdict()` with its losing branch unit-tested before the measurement runs, per method.md
rule #1 — the rule that a documented-but-unimplemented gate printed ADOPT on a 0.6% gain.

### The leakage rule needs a new tool

[method.md](method.md) rule #2 lives in `experiment.expanding_seasons`, which splits by
*season*. A weekly model needs the within-season version: predictors from weeks strictly
before `w`, outcome at `w`, never the same week. **Build `expanding_weeks` beside it**, under
the same AST guard, before any feature is computed. This is the exact failure mode that made
depth-chart climb read as 7.4 sigma, and a weekly model has seventeen times the surface for it.

### Repeated measures

A player's seventeen weeks are not seventeen observations — protocol item 3, which turned
noise into an apparent 4-sigma result once already. **Cluster by player** in every bootstrap,
and report the season-anchor cell structure the snap-trend screen used (sign must hold in
every cell, not on the pool).

---

## Phase 1 — the screen

Per the protocol: **minutes, not sessions, and no `src/` until something clears.** Ad hoc is
explicitly allowed here — ADR-0007's trigger is *citation*, not exploration.

Partial correlation of each candidate against player-week points, controlling for the flat
projection **and** for weekly consensus once it is fetched. One screen per feature, each with
its sign pre-stated:

| # | feature | pre-stated sign | asks consensus about |
|---|---|---|---|
| 1 | Implied team total | + | information it has had since the line opened |
| 2 | Spread / game script (pass-run split) | + for pass volume when trailing | same |
| 3 | Opponent DvP by position | + | same |
| 4 | Snap-share trend, last 3 weeks | + | **something published on a Monday** |
| 5 | Injury designation (status × practice) | − | **something published on a Friday** |
| 6 | Target-share trend | + | Monday |
| 7 | Rest days, roof, wind | wind − for passing | pre-priced |
| 8 | Week-over-week TD-rate persistence | pre-stated **null** | the season-level result says zero; the weekly version is untested and this is the honest way to test it |

Rows 4, 5 and 6 are the ones with a prior. Rows 1-3 and 7 are shared information and the
expectation is that they beat the *floor* incumbent handsomely and the *real* incumbent not at
all. **Recording that expectation now is the point of writing it down now.**

Kill criteria, pre-registered: a feature whose sign flips between seasons is a bug, not a
signal (protocol item 4). A feature that survives only in the pooled sample and not in the
per-season cells is a repeated-measures artifact.

---

## Phase 2 — the model, only for features that cleared

Committed code from this point, per ADR-0007: pure core, testable offline, results written
through `hub.store` and stamped with `config_digest`.

`hub/models/weekly.py`:

* `component_means(players, week_ctx)` — per-player, per-week expected counts, built as
  team volume × player share × efficiency, with only the surviving features moving any factor.
* Reuse the **measured** families from `components.py` for dispersion. They were measured
  against 12,673 player-weeks and there is no reason to re-derive them.
* Efficiency held at the player's shrunk season rate, per the persistence table — with the
  single exception of any efficiency feature that cleared Phase 1, which the table says will
  not happen.
* Aggregate with `espn.scoring_settings()` rather than a hardcoded PPR map, since the league's
  own weights are readable and `components.SCORING` is otherwise an assumption.

**Graceful degradation is a hard rule.** No odds on a Sunday morning must produce the flat
projection with a printed note, never an error and never a blank. Same pattern as
`board.build_or_last_good`.

---

## Phase 3 — the two gates, which are not the same question

**Gate A, the projection.** Walk-forward by season over 2021-25 via `expanding_seasons`,
per-player-week paired errors, both halves of the bar, against **both** incumbents reported
separately. Beating the flat projection and losing to weekly consensus is a null and must be
written up as one.

**Gate B, the decision.** A better projection is not a better lineup — that is the whole
screen/gate distinction, and [ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md)
is the local proof: a signal at partial r +0.24 lost its decision gate at −3.81 points.
`hub/season/lineup_gate.py` already exists for exactly this: set the weekly lineup off the new
projection, set it off the incumbent, score both against realised points, paired by team-week.
Metric is **points per team-week**, and the arms share a seed and a roster.

**Gate B can fail while Gate A passes, and if it does, the projection is shown and never
started on** — the same disposition the snap trend got.

---

## Phase 4 — operations

Only if Gate B passes.

* A weekly CLI in the shape of the draft-night tools: one command, an answer under a cap, and
  a last-good fallback.
* Live freshness needs a decision on the odds API `totals` market and its credit cost.
* `hub.models.conformal` is built and connected to nothing ([improvements.md #2](improvements.md)).
  Weekly player predictions are the first thing in this repo with enough volume to calibrate
  intervals against — 300+ player-weeks a week. That is a consumer, and it is the reason to
  sequence conformal *after* this rather than before.

---

## Timing

Nothing here is on the draft path, and none of it should start before **2026-09-03**. Phases 0
and 1 are historical and could run today; they will not, because the week before a draft is
not when you start a new model. Phase 1 in Week 1, Phase 2-3 across Weeks 2-4, Phase 4 only if
both gates pass.

The one thing worth doing early is the **weekly-consensus fetch**, since without it Phase 3
gates against the wrong incumbent, and that is the mistake that would make everything after it
worthless.

## What I expect to happen

Stated before measuring, so it cannot be adjusted afterwards:

1. The model beats the **flat** incumbent by a wide margin. This is nearly free and proves
   little.
2. It does **not** beat weekly consensus on market-derived features alone. Fourteen
   measurements say consensus prices shared information.
3. If anything survives against consensus it is the snap-share trend and the injury
   designation — the two things already measured positive, both timeliness.
4. Week-over-week TD rate comes back null, consistent with the season-level −0.004.
5. Gate B is the one at real risk of failing after Gate A passes, because a lineup is a
   max over a roster and a projection has to be wrong in the right *direction* to change it.

If (1) and (2) both land, the honest write-up is *"week-level information exists and consensus
already has it"* — which is a null, and the fifteenth measurement, and belongs in
[signal-screens.md](signal-screens.md) beside the others.
