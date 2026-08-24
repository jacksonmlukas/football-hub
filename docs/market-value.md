# Where the value is, and where it is not

**Measured 2026-08-24.** Jackson's premise — the market prices outcomes well — holds, and
holds harder than expected. Nothing in this repo's decision layer beats simply drafting the
market when scored on what actually happened.

## The result

Three seasons (2022, 2024, 2025), 40 simulated drafts each, slot 3. The room drafts the
market's own ordering with noise and fills its roster. I draft by strategy S. **Every roster
is scored on realised points**, not on a projection.

| strategy | my starters | vs room average | best roster in the league |
|---|---|---|---|
| **adp_need** — follow the market | **99.3** | **−0.31** | **8.3%** |
| vor_need — value over replacement | 94.2 | −5.37 | 1.7% |
| proj_need — raw projected points | 80.6 | −18.61 | 0.0% |

`vor_need` against `adp_need`: **−5.06 points per team game, 95% CI [−7.90, −2.18],
P(better) = 0.0%.**

The baseline lands at 8.3% — exactly 1/12 — which is the sanity check: a competent drafter
following the market gets his fair share of titles and no more, and no less.

## The in-simulation result said the opposite, and it was an artifact

Run the same comparison inside `champion_probability`, scoring seasons on `proj_blend`:

| strategy | scored on `proj_blend` | scored on ADP-implied | scored on **realised** |
|---|---|---|---|
| vor_need | **+17.25 pp** | −4.66 pp | **−5.06 pts/gm** |

`optimize.py` has always carried the caveat that "the season is scored on the same projection
the greedy ranks on". This puts a number on it: **it is worth about 17 points of apparent
championship equity**, which is larger than any real effect this repo has measured.

Note the middle column is circular in the opposite direction — scoring on an ADP-implied mu
flatters the strategy that ranks on ADP for exactly the same reason. Neither in-simulation
column is evidence. Only the realised one is.

## Two bugs found on the way, both of which would have produced a wrong answer

**A strawman opponent.** The first version had the room draft pure ADP with no roster
awareness, so opposing teams finished with five quarterbacks and no tight end. Beating that
handed every strategy +24 points. Real rooms fill their roster; once they did, the baseline
moved from a flattering 8.03% to a fair 8.55%.

**An additive need weight.** Roster need was added to value as `VOR + 6*need`, but VOR is in
points (0–15) and the room's key is in pick numbers (1–250), so the weights were on
incomparable scales. My own VOR strategy drafted **4.9 quarterbacks** in a one-quarterback
league. That is not a finding about VOR, it is a finding about the harness. `hub.draft.optimize`
gets this right with a lexicographic rule — an unfilled starting slot outranks any amount of
value, and the strategy only breaks ties within a need tier — and adopting it fixed the
composition. The headline result survived the fix, which is the only reason it is reported.

## What this means

**Do not try to out-project the market.** Every attempt in this repo has failed the same way:
the market alone scores 3.578 RMSE against the best component model's 3.474 at 87%
([volume-model.md](volume-model.md)), and now the market's *draft ordering* beats
value-over-replacement on realised outcomes.

**The market's ordering already contains the scarcity argument.** VOR exists to say that a
running back is worth more than his raw points because replacement is thinner. ADP already
knows that — it is why running backs go early — and imposing VOR on top double-counts it.

**The remaining edges are the ones the market cannot price**, and they are narrow:

- **Corrections the projection demonstrably omits**, each measured against the market rather
  than replacing it: touchdown luck at QB and WR ([td-luck.md](td-luck.md)), durability at QB
  and WR ([durability.md](durability.md)), being ruled out today.
- **Things that are not about the player at all** — your slot, who survives to your next pick,
  weeks 15-17 schedule under a 6-of-12 bracket ([six-of-twelve.md](six-of-twelve.md)).

That is a much smaller claim than "our model beats ADP", and it is the one the evidence
supports.

## The scoring weights belong to the league

Prompted by the same principle — fantasy points are an aggregate of real stats, so the
*weights* are a league setting rather than a property of this repo. `components.SCORING` was
hardcoded as full PPR and merely assumed correct.

Checked against `mSettings`: this league matches on all nine items, with no per-position
overrides. That was luck until it was verified. `make draft` now reads the league's weights
and prints a loud mismatch if they ever diverge — a commissioner moving to half-PPR would
otherwise mis-score every projection, simulation and pick without a word.

## What would change the verdict

Three seasons and one league. The 2023 season is missing because ESPN returns only 92
projections for it against 394-527 for the others, which is an upstream gap and not something
this repo can fix.

The comparison is also of *strategies*, not of the full `win_probability` optimiser — that one
is too expensive to run inside a 120-draft backtest. It ranks on the same `proj_blend` that
`vor_need` does, so there is no reason to expect it to escape the same verdict, but that is an
inference rather than a measurement.

## Reproduce

The harness lives outside `src/` deliberately: it is a one-off screen, and the repo's pattern
is that screens produce documents rather than modules ([signal-screens.md](signal-screens.md)).
