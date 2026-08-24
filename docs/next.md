# What to do next, and why

**Written 2026-08-24, 10 days before the draft.** Ordered by what would change a decision,
not by what is interesting.

## The architecture: one prediction module, many readers

**Jackson's framing, and it is right.** Predict every stat for every player-game once, and
let everything else read from it. `championship-leverage.md` called this "one object, three
consumers"; the object is a distribution over *component stats*, not over fantasy points.

Most of it already exists, scattered:

| piece | where it lives today |
|---|---|
| stat → fantasy points | `models/components.py` |
| count and yardage distributions | `models/components.py` |
| pick → component decomposition | `models/volume.py` |
| dispersion laws, skew, correlated draws | `draft/season.py` |
| readers | `season/lineup.py`, `draft/optimize.py`, `season/survivor.py` |

What is missing is that they are not *one object*. `weekly_moments` hands back two moments,
the skewed correlated draw happens inside `simulate_weeks`, and `lineup.py` computes its own
group spread. Three implementations of one idea, which is how they drift.

### What unifying actually buys

Consistency is the boring reason. The two real ones are **new consumers that do not exist
yet**, and both are audits rather than edges:

**Sportsbook props become a scoreboard.** If the module predicts stats, every player-week
with a posted line is a money-backed test of it. Run once against the Week 1 opener:

| | |
|---|---|
| stat-lines matched | 12 |
| mean bias | **+3.5 yds, +16%** |
| MAE | 8.8 yds |

We over-project by about 16% against the market, and not uniformly — Darnold's passing line
is 30 yards *above* our number while most receivers are below theirs. That is a calibration
gap in us, and validating against history alone could never have shown it. It also does not
violate the standing decision in [decisions.md](decisions.md): the aim is not to beat the
book, it is to find out where we are wrong.

**Team scores become derivable.** Aggregate every player on a team and you get a team-score
distribution, and from two of them a game win probability — computed from stats rather than
read off a spread. `models/market.py` currently takes the spread as given. Having both means
they can be compared, which is a second audit surface.

### What it does not buy

It does not escape the circularity below. If one object both ranks candidates and scores the
season, that dependence is structural rather than accidental. The only fix is scoring against
*realised* outcomes, which is P0.

**Started 2026-08-24 at Jackson's direction**, against my recommendation to wait until after
the draft. Done as a pure refactor -- no number changes -- so the test suite is the proof that
the draft tool still works.

`hub/models/predict.py` now owns everything about what a player does in a week: `TALENT_CV`
and its per-position table, the square-root spread law, per-position skew, the Cornish-Fisher
draw, teammate correlation, and the component line behind a projection. `hub/draft/season.py`
keeps what a *league* does -- rosters, schedule, bracket, lineups -- and re-exports the moved
names, because breaking `calibrate`, `leverage`, `optimize` and the tests to be tidier is not
an improvement.

Teammate correlation moved too. It was defined in `components.py` and used two ways:
analytically by `lineup.py` for a closed-form spread, and by sampling in the simulator. Two
uses of one table is fine; the table living in the *scoring* module was not, since scoring is
how stats become points and who moves together is a prediction.

**The constraint this had to respect**, and did: component-derived spread is measurably
*worse* than the fitted square-root law (mean error 1.365 against 1.140, P(better) 0.0%), so
the object keeps the fitted laws for its moments and exposes components alongside them. A
unification that quietly swapped a validated number for a tidier derivation would be a
regression wearing a refactor's clothes.

Still to move, and deliberately not moved yet: the distribution constants in `components.py`
(`COUNT_DISPERSION`, `TD_DISPERSION`, `PER_UNIT_CV`) and `sample_weeks` belong here too, on
the same reasoning. That would leave `components.py` as pure scoring. Left for after the
draft -- it touches the props audit path and there is no reason to move it under a deadline.

## The finding that reorders everything

**The P(win) recommendation reproduces VOR, and VOR loses to the market.**

Rank correlation between the championship-equity lift and the quantity it is scored on:

| pick | vs `vor_proj` | vs `proj_blend` | vs market (−ADP) |
|---|---|---|---|
| 3 | **0.927** | 0.879 | 0.515 |
| 22 | **0.964** | 0.964 | 0.818 |

(Picks beyond 3 reuse the same candidate set, so treat pick 3 as the independent observation.)

That matters because [market-value.md](market-value.md) measured VOR ordering **losing** to
simply drafting the market by 5.06 points per team game on realised outcomes, P(better) 0.0%.
If the lift is 0.93 correlated with VOR, the board's headline recommendation may be actively
worse than "take the best available by ADP".

This is not proven — correlation with VOR is not the same as sharing VOR's fate, and the lift
does carry roster-construction information VOR does not. But it is the most important open
question in the repo, and everything below is ordered around it.

---

## P0 — Settle whether the objective helps or hurts

**Blocking. Nothing else matters as much.** Started 2026-08-24.

### Why it cannot be deferred

The two candidates disagree at **8 of 8** of my picks, and the market's choice falls inside
the P(win) TAKE tier only once. This is not a question about the third decimal place -- it
changes the pick every time.

### Design

**Arms**, both using the same room and the same seed so the comparison is paired:

- **A, market**: best available by ADP that fills an unfilled starting slot, lexicographically
  -- need gates, ADP breaks ties. The competent market-follower from
  [market-value.md](market-value.md), which lands on exactly 1/12 and is therefore a fair null
  rather than a strawman.
- **B, optimizer**: the top of `win_probability` over the `recommend()` shortlist.

**Room**: opponents draft ADP with noise and fill their own roster, same lexicographic rule.

**Seasons**: 2022, 2024, 2025. 2023 is excluded -- ESPN returns 92 projections for it against
394-527 for the others, an upstream gap.

**Outcome**: realised points. Best legal lineup from each roster using actual season
production, scored per team game. Never a projection -- the whole point is escaping the
circularity, which is structural now that one object both ranks and scores.

**Power**: 20 drafts per season, 3 seasons, 60 paired observations. The strategy backtest
detected a 5-point effect with 120 observations, so this can see an effect of that size but
not a small one -- which is itself worth knowing, since an effect too small to detect is too
small to headline.

**Cost control**: `n_draft_sims=6, n_season_sims=120` inside each optimizer call, well below
the shipped 24x300. Noisier per call, and that noise is part of what is being tested: if the
recommendation is unstable at cheap settings it is not a usable recommendation.

### Result (2026-08-24): no detectable difference, and a self-correction

| run | optimizer settings | n | optimizer − market | 95% CI | P(better) |
|---|---|---|---|---|---|
| first | 6 × 120 | 60 | **−5.79** | [−9.17, −2.45] | 0.0% |
| check | 12 × 250 | 15 | −1.91 | [−7.85, +3.14] | 27.1% |
| **powered** | **12 × 250** | **36** | **+0.04** | **[−3.64, +3.58]** | **51.0%** |

**The first result was an artifact of running the optimizer at a quarter of its shipped
budget** (6 draft rollouts × 120 seasons, against 24 × 300 live). The gap shrank as the
budget rose and vanished at adequate power. The risk was flagged in this document before the
run and nearly acted on anyway, which is the more useful lesson: a pre-registered caveat is
worthless unless it is actually checked before the conclusion ships.

At realistic settings both arms beat the room by about the same amount — market +3.11,
optimizer +3.15. **This is not "the market wins."** It is "no effect larger than about 3.6
points is detectable at n=36", which is absence of evidence rather than evidence of
equivalence.

**Action, per the rule fixed below:** lead with the market, keep championship equity as a
tiebreaker. Not because the optimizer is harmful — it is not, and the earlier claim that it
might be is withdrawn — but because it cannot demonstrate an edge over an alternative that is
instant and far simpler, and the burden is on the complicated thing.

**What this does not settle.** The candidate shortlist was top-8 by VOR rather than
`recommend()`'s scarcity/value logic, so all three runs may understate the optimizer. That
limitation does not shrink with more samples, and with the result sitting on zero it is
exactly the kind of thing that could tip it. It is the obvious next test if the tiebreaker
role is ever revisited.

### Decision rule, fixed before the numbers are in

| result | action |
|---|---|
| CI excludes zero, favouring **B** | keep championship equity as the headline, fix the circularity (P1) |
| CI excludes zero, favouring **A** | lead with the market plus the measured corrections; demote equity to a tiebreaker |
| **CI contains zero** | lead with the market. An objective that cannot demonstrate it beats ADP should not be the headline |

The third row is the one worth pre-registering. A null is the most likely outcome at this
sample size, and it has an action rather than being a disappointment to explain away.

Extend the realised-outcome backtest from *strategies* to the actual optimiser. At each of my
picks in 2022/2024/2025, compare what `win_probability` recommends against what the market
recommends, play the draft out, and score the final roster on realised points.

Three outcomes, each with a clear action:

- **P(win) beats the market** → keep it, and fix the circularity (P1).
- **P(win) ties the market** → keep it for the roster-construction information, drop the
  claim that it beats ADP, and lead with the market.
- **P(win) loses** → lead with the market plus the measured corrections, and demote
  championship equity to a tiebreaker.

Cost: expensive (nested simulation inside a backtest) but tractable with reduced sims. This
is the one thing worth being slow about.

## P1 — Break the circularity *(only if P0 says keep it)*

`optimize.py` scores the season on `proj_blend` and ranks candidates on `proj_blend`. The
size of that bias is now measured: 17.25 pp of apparent championship equity, which is larger
than any real effect in the repo.

Options, cheapest first: score seasons on a market-implied mu while ranking on `proj_blend`;
or bootstrap-perturb the scoring mu per simulation so no candidate is scored on exactly the
number it was ranked on. Then check whether the lift ordering survives.

## P2 — Calibrate the opponent model

`simulate_remaining_draft` models opponents as ADP plus assumed noise. Measured against three
real drafts, actual picks deviate from the market's implied order with a standard deviation of
**29–42 picks**, which is far more than the model assumes.

Caveat that makes this a project rather than a number: that spread was measured against
*projection* rank, not ADP, so part of it is projection≠ADP rather than room noise. Measuring
it properly needs historical ADP, which ESPN returns as a 169 sentinel for 69-78% of players
([decisions.md](decisions.md)). Worth doing because a wrong opponent model changes who is
available at your next pick, which is one of the few things the market cannot price for you.

## P3 — Draft-night rehearsal

The deadline is real and untested end to end under failure.

- Time every command against a 90-second clock, including the slowest path.
- Kill the network mid-build and confirm it serves last-good state rather than a traceback,
  which `CLAUDE.md` requires and the draft path has never been tested for.
- Feed deliberately bad input: a mistyped pick (now warns), a duplicate pick, an out-of-order
  sync, a pick of someone already gone.
- Write the runbook: exact commands, in order, with the fallback for each.

Cheap, and the only item here whose value is certain.

## P4 — Playoff schedule into the objective

`championship-leverage.md` calls weeks 15-17 strength of schedule "the largest underexploited
edge", and [six-of-twelve.md](six-of-twelve.md) showed seeding is worth 4.4x under 6-of-12.
The simulator models neither: `_round_robin` has no opponent quality and playoff weeks reuse
regular-season draws.

Today SoS is a tiebreaker column the drafter reads by eye. Putting it in the objective is what
would let it price a pick.

## P4b — Props as a standing calibration check *(cheap, do now)*

The one-game run above is n=12. Widen it as more of the Week 1 slate gets priced, and record
the bias per stat and per position. If we are 16% high on receiving yards everywhere, that is
a correction to make; if it is noise at n=12, that is worth knowing before trusting the
number. Costs ~4 credits per event and errors are free.

## P5 — Market prices for weekly lineups *(after the draft)*

The odds key now works, and the API carries the **full component set including volume** —
`player_receptions`, `player_rush_attempts`, and the yardage and touchdown markets. Volume is
the half that persists (targets 0.805, carries 0.791) and touchdowns are the noise, so a
market pricing receptions and carries is pricing exactly the part worth having.

`hub/season/lineup.py` currently takes mu and sd from ESPN's projection. Taking them from
money-backed prices is the strongest input upgrade available anywhere in the repo — and it is
for the seventeen weeks *after* the draft, so it does not compete with P0-P3.

Quota is the constraint: 64 credits a week for a full slate against 500 a month. Pulling only
your roster and your opponent's fits comfortably, and is all a lineup decision needs.

## P6 — Validate the `edge` column

`edge` (expert consensus rank against ADP) is the repo's original draft edge and has never
been tested against outcomes. The P0 harness tests it for free once it exists.

---

## Explicitly not doing

- **More projection modelling.** Four documented nulls and a standing decision
  ([decisions.md](decisions.md)): do not backtest for edges, backtest to audit.
- **The draft-room extension.** Researched, filed as a nice-to-have at Jackson's direction.
  More moving parts than anything in the repo, ten days out, with `--taken` already tested
  end to end against 192 real picks.
- **Season-long player props.** Settled: The Odds API does not carry them.
