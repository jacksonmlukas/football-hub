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

## P0b — P0, built as code, with the shortlist its own design specified

**Designed 2026-08-24, before any numbers.** P0's write-up above stays as it is: it is the
record of a run that departed from its own spec twice without saying so, and that record is
the justification for [ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md).

### Why P1 does not fire

P1 is gated on *"only if P0 says keep it"*, and P0's three branches give **fix the
circularity** only to the branch where equity *beats* the market. P0 landed on the tie, whose
action is "keep it for the roster-construction information, drop the claim, lead with the
market". So the next test is not P1. It is the one P0 named itself: the shortlist was top-8 by
VOR rather than `recommend()`'s scarcity/value logic, *"and with the result sitting on zero it
is exactly the kind of thing that could tip it."*

### What changed from P0

| | P0 | P0b |
|---|---|---|
| Code | none committed | `hub.draft.backtest`, pure core + CLI shell |
| Arm B shortlist | top-8 by VOR | `recommend()`'s top-10, as P0's design specified |
| Picks arm B decides | 8 | all 16 — both arms play the draft out regardless |
| n | 36 (design said 60, unexplained) | **80** — 20 drafts x 4 seasons |
| Seasons | 2022, 2024, 2025 | + **2023**, whose exclusion was an ESPN-endpoint fact we no longer depend on |
| Sims per optimizer call | 12 x 250 | **12 x 250, pinned** |
| Data | unknown | pure nflverse; no ESPN, no cookies |
| Arm A ranks on | ADP (unverifiable) | that season's consensus, via `consensus(as_of=...)` |
| Result | prose only | parquet stamped with `config_digest` |

MDE improves from about +/-3.6 to roughly +/-2.4 on n alone.

### Decision rule, fixed before the numbers

Asymmetric on purpose: the market already leads, and the burden is on the complicated thing.
It is `backtest.verdict`, so it is executable and cannot be quietly reinterpreted afterwards.

| result | action |
|---|---|
| CI excludes zero favouring **B** | promote championship equity back to the headline; **P1 now fires** |
| **CI contains zero** | nothing changes |
| CI excludes zero favouring **A** | **remove** equity from the draft-night output entirely |

The third row is new. P0's rule would only ever promote on evidence, which is heads-I-win: if
you will not demote on evidence too, the tiebreaker keeps steering close calls whatever the
measurement says.

### Limitations, written before the run

1. Arm A follows **consensus**, not the **draft market**. ESPN publishes ADP for the current
   season only and the FantasyPros archive carries none, so a replay cannot follow ADP. The
   shipped THE PICK does.
2. Arm B scores seasons on prior-season **xFP**; the live board scores on `proj_blend`, which
   blends an ESPN projection that does not exist for past seasons.
3. Arm B **breaks ties** by consensus. The shipped tool refuses to break them and asks you.
4. The room is simulated — consensus plus fitted pick noise, lexicographic need — not the
   eleven people actually in those drafts.

### BLOCKED, 2026-08-24: the harness found a defect in the thing it measures

Running it now would measure a broken optimizer.

`win_probability` scores championship equity on a roster that **excludes the players you
already hold**. `simulate_remaining_draft` builds rosters out of `draft_pool`, which is
`remaining(board, state)` — so your existing picks are not in the pool, not in the simulated
roster, and positional need arising from what you own is invisible to the objective.

Demonstrated, holding one quarterback in a one-QB league: `win_probability` ranks a *second*
quarterback above a startable running back (lift +0.0047 against −0.0047). Played out as a
lead strategy, arm B finishes with **four quarterbacks**, and raising the simulation budget
from 2x20 to 6x150 does not move it — it is structural, not noise.

Live, the damage is bounded: the draft market leads, and equity only breaks ties over a
shortlist `recommend()` has already ranked by VOR or cost-of-waiting. It is not bounded to
zero, though — that shortlist is not need-filtered, so the tiebreaker can still name a
redundant player at a filled position.

Pinned as a strict `xfail` in `tests/unit/test_backtest.py`, which will fail the moment it is
fixed and the marker needs removing.

**Resolved 2026-08-24: fix first.** `win_probability` now scores equity over the whole board
with every seat seeded from the roster it already holds. Arm B therefore measures the post-fix
optimizer, which is a fifth limitation below.

### The fix, and the before/after that gated it

`--diagnose` was committed *before* the fix so it could be run against both, at
`75f800f` (pre-fix) and the commit that follows it. The draft is advanced by the market in
both runs, so the path is identical and the only thing that can differ is what equity says.

| pick | held | leader BEFORE | leader AFTER |
|---|---|---|---|
| 3 | — | Puka Nacua (WR) | Christian McCaffrey (RB) |
| 22 | RB1 | Chris Olave (WR) | Chris Olave (WR) |
| 27 | RB2 | **Cam Skattebo (RB)** | Emeka Egbuka (WR) |
| 46 | RB2, TE1 | **Javonte Williams (RB)** | **Javonte Williams (RB)** |
| 51 | RB2, TE1, WR1 | Wan'Dale Robinson (WR) | Parker Washington (WR) |
| 70 | RB2, TE1, WR2 | **Travis Etienne Jr. (RB)** | **Travis Etienne Jr. (RB)** |

Bold rows are where the tripwire fires. Three before, two after.

**The tripwire, pre-registered before the run:** equity must never name a player at an
already-filled required position ahead of one filling an empty slot. That is the old defect's
signature and also what a seat mis-attribution would look like after the fix.

**It still fires, so by the rule the fix is not cleared to ship.** What the rule does not say,
and what the run shows:

    pick 46, holding RB2 TE1        pick 70, holding RB2 TE1 WR2
      CO-LED Javonte Williams  RB     CO-LED Travis Etienne Jr. RB  +2.03% +/-1.10
      CO-LED Josh Jacobs       RB     CO-LED Dak Prescott       QB  +1.30% +/-0.82
      CO-LED Emeka Egbuka      WR     CO-LED Trevor Lawrence    QB  +0.56% +/-0.67

At both tripped picks the need-filling alternative is a **co-leader** — inside two standard
errors of the named leader. So this is "equity prefers running-back depth, within noise", not
the decisive pathology the fix removed, where a second quarterback beat a startable back
outright and arm B finished with four of them.

**The flaw is in the tripwire, and it is mine.** It treats disagreement with `_need_score` as
evidence of a defect, and disagreement with the lexicographic need rule is the entire reason
championship equity exists as a separate signal. A gate that fires whenever the objective is
doing its job is not a gate.

### The gate was amended after seeing its output, deliberately and on the record

Not quietly. The clause added: the tripwire fires only when the leader sits at a filled
required position **and no co-leader fills an unfilled one**. A need-filling candidate inside
two standard errors of the leader is the objective declining to distinguish, which is a tie,
not a rejection of need.

The amendment is narrow on purpose, and `test_the_amended_gate_still_catches_the_original_defect`
pins that it still fires on the case it was written for: there a second quarterback beat a
startable back *outright*, no co-leader filled a need, and arm B finished with four
quarterbacks.

Re-run under the amended gate: **clear at all six picks.**

**A caveat on the before/after, found while re-running.** `--diagnose` called `build()`, which
refetches live ESPN ADP every time, so the two runs above compared boards fetched minutes
apart rather than one board. The tell was pick 3's leader flipping between two *post-fix* runs
at the same seed. Some of the row-by-row movement in that table is therefore ADP drift, not
the code change; the structural finding -- three tripwire fires becoming none -- is not
sensitive to it. `--board PATH` now pins a snapshot so future comparisons vary only the code.
That gap was mine too, and it is the same one ADR-0007 exists to close, one level down: the
measurement was committed, its input was not.

### Result (2026-08-24): the objective is decisively worse than consensus

    n=80   optimizer - market = -19.66 points per team game
           95% CI [-23.16, -16.20]      P(optimizer better) 0.0%

| season | market | optimizer | diff | drafts B won |
|---|---|---|---|---|
| 2022 | 125.4 | 104.1 | -21.32 | 0 / 20 |
| 2023 | 127.0 | 106.6 | -20.38 | 1 / 20 |
| 2024 | 120.3 | 112.1 | -8.15 | 8 / 20 |
| 2025 | 125.1 | 96.3 | -28.79 | 0 / 20 |

Every season, same direction. Arm B wins 9 of 80 drafts. `config_digest` 9975101f;
paired rows in `data/processed/p0b_paired.parquet`.

**Checked before believing it.** An effect this size where P0 measured +0.04 is a bug until
shown otherwise. It is not a scoring artifact: both arms draft real players who post real
seasons, with no name-mismatch zeros. Arm B simply builds worse rosters. In 2024 draft 0 it
took McCaffrey, Kamara, Mixon, Ekeler and Conner -- five running backs -- and finished with
four receivers in a three-receiver league, scoring 115.2 against the market's 127.0.

**Action, per the rule fixed before the numbers: REMOVE.** Championship equity leaves the
draft-night output. Not because the simulator is broken -- it is not -- but because its lift
ordering is measurably worse at picking than following consensus and filling needs, and a
tiebreaker worse than the thing it breaks ties for steers close calls the wrong way.

**The tripwire was right and I amended it away.** Earlier the same day it fired at picks 46
and 70 for naming a running back while WR and QB sat empty. I judged that a miscalibrated
gate, because need-filling alternatives were co-leaders, and narrowed it. That RB-over-need
preference is exactly what costs ~20 points per team game here. The amendment is still
defensible on its own terms -- a co-leader is a tie, not a rejection -- but the conclusion it
licensed ("clear at all six picks") read as reassurance the objective had not earned. A gate
that embarrasses a change is doing its job; that is the second time in one day this document
records a pre-registered check being talked past.

**What this does not settle.** Three things changed between P0 and P0b -- the shortlist, the
seasons and n, and the roster-seeding fix -- so none of the -19.66 can be attributed to any
one of them. Isolating the fix means re-running arm B at `75f800f`. Cheap at reduced n: with
sd 16.2, twenty drafts separate -19.66 from +0.04 at better than five standard errors.

**P1 does not fire.** It was gated on equity beating the market. It loses.

### Did the roster-seeding fix cause this? No -- it helped slightly

Run at `75f800f` (pre-fix) plus the one fetch fix it needed, same seasons, same seeds, n=20.
Arm A comes out **byte-identical** across the two runs, which confirms the comparison isolates
arm B: seeding is a no-op at an empty draft state and only bites inside `win_probability`'s
nested rollouts.

| | arm B | market |
|---|---|---|
| pre-fix (`75f800f`) | 102.11 | 127.73 |
| post-fix | 105.19 | 127.73 |

Paired, arm B post-fix minus pre-fix: **+3.08 points per team game** (sd 10.14, n=20, so the
interval spans zero), better in 13 of 20 drafts.

So the defect was real and fixing it moved equity in the right direction; it simply was not
worth twenty points. The fix stands on its own merits -- an objective blind to your own roster
is wrong regardless of whether correcting it changes a verdict -- and none of the -19.66 is
attributable to it. That closes the confound flagged as limitation 5.

## P1 — Break the circularity *(CLOSED 2026-08-24: gate not met)*

Gated on championship equity beating the market. It lost by 19.66 points per team-game
([ADR-0009](adr/0009-championship-equity-does-not-pick.md)), so this does not fire. It reopens
only if the objective returns, which needs a better prediction layer and a re-run of the
harness -- not a re-argument.

### Original text

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

## P4 — Playoff schedule into the objective *(RE-SCOPED 2026-08-24)*

The premise below -- "putting SoS in the objective is what would let it price a pick" -- died
with the objective: championship equity no longer picks. THE PICK is lexicographic, so SoS
cannot enter the *ranking* without breaking the rule that won.

**Re-scoped to: SoS becomes information beside THE PICK**, outside the corrections list. It
gets its own framing rather than joining touchdown luck, durability and injury status, because
each of those carries a fitted coefficient against outcomes and SoS carries none. Shelving an
unfitted index among fitted ones borrows credibility it has not earned. Post-draft work; it
adds something new rather than fixing something wrong, and that is what a freeze keeps out.

### Original text

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

## Weekly gate (2026-08-28): the Weekly projection is shown, never ranked on

`hub.season.weekly_gate`, three held-out seasons, 60 rosters, 680 roster-weeks:

    weekly - consensus = -0.684 points per team-week
    95% CI [-1.519, +0.159]   P(weekly better) 5.8%    lost 3 of 3 seasons

A projection that beats the flat one this repo ships by +0.074 MAE at 5.9 se, losing the lineup
decision to a free public ranking. A lineup is a max over a roster, so most projection error
never reaches the decision -- which is what [ADR-0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md)
predicted when it made this the primary gate, and what
[ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) records.

**The first run said +11.14 at P 100% and was VOID** -- 6.2% join failures against a floor
pre-registered hours earlier. An off-by-one attached each consensus scrape to the week after the
one it ranked. The floor is the only reason a spurious ten-point win was not written up.

## Lineup gate (2026-08-24): start your projections, and why that is actionable

`hub.season.lineup_gate`, four seasons, 80 rosters, both arms seeing only projections:

    n=80   optimiser - projections = +0.00 points per game
           95% CI [-0.00, +0.00]      P(optimiser better) 72.7%

A structural zero, not a small effect. `sd = WEEKLY_K[pos] * sqrt(mu)`, so within a position
`sd` is a deterministic increasing function of `mu` (correlation 0.985), and ranking by `mu`
gives the same order as ranking by any increasing function of `(mu, sd)`. The optimiser's one
advantage -- reading variance -- is handed no variance to read.

**This is the strongest argument yet for the usage and component layers.** Not "it would be
more accurate": it would make an existing, tested, currently-inert piece of the system do
something for the first time. Two players projected at 12 points a game are not equally
volatile, and the square-root law cannot say so because it only knows the mean. See
[ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md).

**The gate's first version could not fail**, and that is recorded rather than quietly fixed:
the optimiser arm chose each week from *realised* scores while the baseline used projections,
returned +31.15 [+29.35, +32.93], and measured the value of perfect foresight. A treatment arm
with information the control arm lacks is not a comparison.

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

## P6 — Validate the `edge` column *(CLOSED 2026-08-24: unmeasurable)*

`edge` (expert consensus rank against ADP) is the repo's original draft edge and has never
been tested against outcomes. **It cannot be.**

The claim above -- "the P0 harness tests it for free once it exists" -- was wrong, and was
checked rather than assumed once the harness existed:

    historical board cols with edge/adp: []
    edge present: False

`edge` needs ADP. ESPN publishes ADP for the current season only and the archive returns a 169
sentinel for 69-78% of players, so `board.build(as_of=...)` produces historical boards with no
`adp` column -- the same wall P2 hit. Prospective validation needs outcomes that arrive after
the decision it would inform.

**Action taken:** the column stays, the sort order goes. `live._sort_key` and
`EDGE_FROM_ROUND` are deleted and the context table ranks on `vor_live` throughout. A
displayed number and a sort order make different claims; only the second needed evidence.
See [ADR-0010](adr/0010-edge-is-displayed-but-never-ranked-on.md).

Reopens only if a source retaining historical ADP appears.

---

## Researched backlog

[improvements.md](improvements.md), written 2026-08-24 once the draft path was frozen-ready.
Ten items, each grounded in something measured in this repo rather than in general advice. The
sharpest: **`MARGIN_SD` is asserted at 13.5 and this repo's own schedule data says 12.470 over
854 games, 3.4 standard errors away** -- and it is registered in `FITTED_MODULES` as though it
had been fitted.

## Explicitly not doing

- **More projection modelling.** Four documented nulls and a standing decision
  ([decisions.md](decisions.md)): do not backtest for edges, backtest to audit.
- **The draft-room extension.** Researched, filed as a nice-to-have at Jackson's direction.
  More moving parts than anything in the repo, ten days out, with `--taken` already tested
  end to end against 192 real picks.
- **Season-long player props.** Settled: The Odds API does not carry them.
