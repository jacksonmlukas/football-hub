# The snap-share trend: the first screen here that did not come back null

**Measured 2026-08-25.** Five signals have been screened in this repo and all five were null
([signal-screens.md](signal-screens.md)). This one is not, and it survives the control that
matters.

## The claim

The trade press says a snap-share jump precedes production by one to two weeks. That is a
testable sentence, so it was tested rather than believed or dismissed.

The prior was low. [depth-chart-signal.md](depth-chart-signal.md) screened depth-chart climb at
both horizons and found nothing — partial r of +0.008 beyond ECR. But a depth chart is a coarse
ordinal that nflverse derives; snap share is a continuous record of what a coach actually did
last Sunday, and being null on the first is weak evidence about the second.

## The screen

Following [signal-screens.md](signal-screens.md)'s protocol, which was assembled from the two
errors that produced it:

* **Predictor strictly before the outcome.** Snap share is averaged over weeks *w−1..w* against
  weeks *1..w−2*; the outcome is points in weeks *w+1..w+3*. No overlap.
* **Repeated measures.** One row per player per anchor week, so no player appears twice inside
  any single screen. Anchors are reported separately and never pooled with each other.
* **Sign consistency across seasons**, reported for every cell.
* **A bucketed cut as well as a linear one**, since a correlation cannot see a cliff.
* **Absent weeks score zero, not dropped.** Requiring a future stat row conditions on
  surviving, and surviving correlates with the predictor. Scoring the missing weeks as zero is
  both the fantasy truth and the conservative choice — it *raised* the measured effect
  (anchor 10: +0.221 → +0.234), which is what says the filter was hiding some of it.

2022-25 regular season, ~1,470 player-anchors per anchor week.

## It survives the control that matters

`CONTEXT.md` requires a signal be screened for predictive power **beyond ECR**. Controlling
only for a player's own scoring answers a weaker question. So both were run: partial r against
season-to-date PPG, and against PPG *plus* the rest-of-season consensus rank as it stood on the
last gameday of week *w*.

| anchor | season | n | r \| ppg | r \| ppg + ECR | se away |
|---|---|---|---|---|---|
| 8 | 2022 | 331 | +0.1367 | +0.1373 | 2.5 |
| 8 | 2023 | 345 | +0.1802 | +0.1801 | 3.3 |
| 8 | 2024 | 356 | +0.2400 | +0.2372 | 4.4 |
| 8 | 2025 | 344 | +0.1473 | +0.1426 | 2.6 |
| 10 | 2022 | 345 | +0.2542 | +0.2540 | 4.7 |
| 10 | 2023 | 359 | +0.1572 | +0.1496 | 2.8 |
| 10 | 2024 | 360 | +0.2761 | +0.2815 | 5.3 |
| 10 | 2025 | 348 | +0.2618 | +0.2616 | 4.9 |
| 12 | 2022 | 352 | +0.1767 | +0.1790 | 3.3 |
| 12 | 2023 | 379 | +0.1793 | +0.1774 | 3.4 |
| 12 | 2024 | 363 | +0.2543 | +0.2486 | 4.7 |
| 12 | 2025 | 350 | +0.2377 | +0.2272 | 4.2 |

Pooled: **+0.175** (anchor 8), **+0.236** (anchor 10), **+0.208** (anchor 12). **Twelve of
twelve cells positive**, sign consistent at every anchor.

**Adding the ECR control changes the answer by less than 0.01 anywhere.** Consensus is not
pricing this. That is the whole finding: the five nulls before it were all cases where the
information was already in the board.

## The placebo, because this repo has been fooled before

The depth-chart signal looked real at 7.4 sigma and was leakage. So the same pipeline was run
200 times per season with the predictor permuted across players:

| season | real | permuted mean | permuted sd | max abs permuted |
|---|---|---|---|---|
| 2022 | +0.2540 | +0.0010 | 0.0541 | 0.1302 |
| 2023 | +0.1496 | −0.0023 | 0.0498 | 0.1431 |
| 2024 | +0.2815 | +0.0020 | 0.0532 | 0.1686 |
| 2025 | +0.2616 | −0.0005 | 0.0572 | 0.1403 |

Centred on zero, so the pipeline is not manufacturing the effect. Note 2023 is the weak season:
its +0.1496 is barely above the largest of 200 permutations. Three of four seasons are far
clear; one is not.

## It is stronger where it would actually be used

The screen above is mostly rostered players. A waiver decision is about the tail. Restricting to
players ranked outside the consensus top 100:

| anchor | pooled r | n | seasons positive |
|---|---|---|---|
| 10 | **+0.2537** | 1,052 | 4/4 |
| 12 | **+0.2360** | 1,076 | 4/4 |

And bucketed, which is the form a decision actually takes — mean residual points per game over
the next three weeks, after removing PPG and ECR:

| | snap jump > 15pp | everyone else |
|---|---|---|
| 2022 | +1.01 | −0.16 |
| 2023 | +0.98 | −0.16 |
| 2024 | +1.36 | −0.24 |
| 2025 | +2.15 | −0.38 |

**A gap of roughly 1.2 to 2.5 points a game**, in a subgroup the consensus has already ranked
outside its top 100.

## Where it does not work

Anchors 4 and 6 are null and flip sign between seasons (pooled −0.057 and +0.097). Early in a
season there are too few weeks to form both windows — at week 4 the comparison is weeks 3-4
against weeks 1-2 — and season-to-date PPG is a weak control on three games. **The signal needs
about eight weeks of snaps before it says anything.** Reported because a signal that works
everywhere is usually a bug.

## What this does not establish

* **It does not establish beating the waiver market.** The control is the published consensus,
  not eleven managers bidding FAAB on Tuesday night. Those are different opponents, and this
  repo's record is a long argument that the distinction decides everything.
* **The horizon is three weeks**, not rest-of-season. A waiver claim is usually a longer bet.
* **No mechanism is asserted.** "Coaches trust him more" is a satisfying story and this screen
  tested none of it — [signal-screens.md](signal-screens.md) point 6, which was got wrong here
  once already.
* Snap counts key on `pfr_player_id` and everything else keys on `gsis_id`; the crosswalk
  matches 99.8%, and the missing 0.2% is not characterised.

## Gated as a decision, 2026-08-25: shown, never ranked on

The screen is not the decision. Acting on it was gated against the obvious alternative — best
available by consensus rank — over 40 paired weekly claims among players owned in under 50% of
leagues, scored on the next three weeks:

    top-1 by snap delta   -3.81 points vs by ECR, wins 1/4 seasons
    top-3 by snap delta   -3.35 points vs by ECR, wins 1/4 seasons

**Worse than the thing it was meant to improve on**, and the reason is structural: a *partial*
correlation says delta adds information **given** ECR. Ranking by delta alone throws ECR away.

The rule the screen does support — ECR chooses the candidate set, delta chooses within it —
comes out at +0.66 to +1.32 points at 0.2 to 0.5 se, positive at every pool size and
significant at none. A single waiver decision has a standard deviation near 20 points over
three weeks, so resolving a one-point edge would take about **1,800 decisions, or 44 seasons**.

That is an inconclusive result, not a negative one, and the asymmetry is worth stating plainly:
the *screen* is well powered because it sees ~1,400 players per anchor; the *decision* is not,
because it makes one claim a week.

See [ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md).

## What was built: nothing, yet

Per the protocol and the precedent in [depth-chart-signal.md](depth-chart-signal.md), a screen
result is not a module. What it earns is the *right* to design one — this is the first signal
in the repo to earn that.

The shape would be a weekly report of available players ranked by snap delta, and it needs its
own gate, which is not this screen: does acting on it beat taking the best available player by
consensus rank? That is a different question with a different control, and it is the one that
decides whether any of this is worth a waiver claim.

## Reproduce

The screen is two scripts in the scratchpad, not a module, deliberately. To re-run it, the
inputs are `nfl.load_snap_counts`, `nfl.load_player_stats`, `nfl.load_ff_rankings("all")`
filtered to `redraft-overall`, and `nfl.load_ff_playerids` for both crosswalks
(`pfr_id`→`gsis_id` and `fantasypros_id`→`gsis_id`).
