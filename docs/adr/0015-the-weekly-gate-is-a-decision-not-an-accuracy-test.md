# The weekly gate is a decision, not an accuracy test

**Status:** accepted 2026-08-27.

**Decision.** The Weekly projection is gated on whether a lineup set off it beats a lineup set
off weekly consensus rank, paired in points per team-week. Its **accuracy** against consensus
is not gated and is not measured, because it cannot be. A per-player-week accuracy comparison
against the *flat* projection is still computed and reported, and it decides nothing.

See [weekly-projection-plan.md](../weekly-projection-plan.md) for the full pre-registration.

## Why

The obvious design is the accuracy gate: project player-week points, compare mean absolute
error against the consensus projection, adopt if it wins. It cannot be run.

`nflreadpy.load_ff_rankings("all")` ships six seasons of weekly FantasyPros consensus —
`weekly-qb/rb/wr/te` plus a cross-position `weekly-op` page, 15–20 scrape dates a season across
2020–2025, ~467 ranked players a date. It carries `ecr`, `sd`, `best`, `worst`, and
`player_opponent`. **`r2p_pts` is null on every historical weekly row.**

So the only incumbent worth beating exists as a *ranking*. There is no incumbent points
projection to take a paired difference against, and the repo's whole record says the incumbent
that matters is the free public one: fourteen measurements, and the five nulls all lost to
consensus rather than to a straw man.

## Considered, and rejected

**Fit a rank → points curve and keep the accuracy gate.** Rejected because it puts free
parameters on the *incumbent's* side of the comparison. The curve would be fitted by us, on our
data, and would absorb an unknown part of the effect being measured — the arm we are trying to
beat would contain a model we built. A gate whose control arm has fitted parameters is not
measuring what it claims to.

**Score the accuracy gate on a rank metric — Spearman, or top-N precision.** Rejected because
it measures ranking quality across all ~467 ranked players, and the decision only ever touches
the eight you start out of the sixteen you roster. A model can win Spearman on the tail and
never change a lineup. That is [ADR-0013](0013-the-snap-trend-is-shown-and-never-ranked-on.md)
restated: a correlation says a quantity *adds to* the board and never that it should *be* the
board.

**Use ESPN's weekly projections as the incumbent instead.** They are points, which would make
the accuracy gate runnable. Rejected on availability: no history in the fetch layer, and
reconstructing it would mean trusting a source we cannot re-pull for past seasons — which is
the reproducibility failure [ADR-0007](0007-measurements-that-steer-the-product-are-committed-code.md)
exists to prevent.

## The trade-off, which is real

A decision gate is **lower powered and narrower** than an accuracy gate, and the cost falls
entirely on false negatives.

A lineup is a max over a roster. Most projection error never reaches the decision: being wrong
about a player you were never going to start, or wrong in a direction that does not change the
order, costs nothing here. So a genuinely more accurate Weekly projection can return a null on
this gate, and that null will be real about *lineups* while being wrong about *projections*.

We take that. The alternative failure — adopting on an accuracy number that moves no decision —
is the one this repo has actually made, and it costs more: a well-built, well-tested model
shipping while being confidently worse than a one-line rule.

## Consequences

- **The lineup rule is fixed at *start your highest projections*.** This gate varies the
  projection under an identical search; it does not vary the search. It therefore does not
  reopen [ADR-0012](0012-the-lineup-optimiser-waits-for-real-variance.md), whose structural
  zero came from varying the search over identical projections.
- **The accuracy diagnostic is still computed**, against the flat projection only, so a gate
  result can be attributed rather than asserted. A lineup win with no accuracy win is a ranking
  artifact and is to be treated as one.
- **Join quality becomes load-bearing on one side.** A rostered player absent from the weekly
  consensus page is ranked last, which is the incumbent saying "do not start him". A join
  failure therefore does not add noise — it forces a bench on the incumbent's arm and biases
  the result toward us. Hence the pre-registered 2% void floor, and unmatched players reported
  by name.
- **A future architecture review proposing "just fit rank → points and run a proper accuracy
  gate" should read this first.** That is the specific suggestion this ADR exists to answer.

## What would change it

FantasyPros backfilling `r2p_pts` on the historical weekly pages, or any source of *published,
re-pullable, per-week* consensus points across several seasons. Then the accuracy gate becomes
runnable against a control arm with no parameters of ours in it, and it should be run **in
addition** — never instead. The decision gate stays primary either way, because the reason it
is primary is not the missing column.
