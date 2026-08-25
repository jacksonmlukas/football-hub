# `edge` is displayed but never ranked on

**Status:** accepted 2026-08-24.

**Decision.** `edge` — expert consensus rank minus ESPN ADP — stays as a column on the board
and in the poller's context table. Nothing sorts by it. `live._sort_key` and
`EDGE_FROM_ROUND` are deleted, and P6 ("validate the `edge` column") is closed as
**unmeasurable** rather than undone.

## Why it cannot be measured

`edge` requires ADP. ESPN publishes ADP for the current season only, and the historical
archive returns a 169 sentinel for 69–78% of players ([decisions.md](../decisions.md)) — the
same wall P2 hit. So the backtest harness in `hub.draft.backtest`, which reconstructs
historical boards through `board.build(as_of=...)`, builds them with **no `adp` column at
all**. The column `edge` would be scored on does not exist on any frame that can be scored:

    historical board cols with edge/adp: []
    edge present: False

`next.md` claimed "the P0 harness tests it for free once it exists". That was wrong. Nor can
it be validated prospectively — that needs outcomes which, by definition, arrive after the
decision it would inform.

## Why the column stays and the sort goes

A displayed number and a sort order make different claims. The column says *"the market and
the experts disagree about this player by this much"*, which is true, cheap and checkable. The
sort order says *"players the market undervalues are the ones to take"*, which is a hypothesis
about outcomes, and it is the hypothesis that cannot be tested.

This repo demoted VOR ordering to context on evidence (−5.06 pts/team-game,
[market-value.md](../market-value.md)) and removed championship equity on evidence (−19.66,
[ADR-0009](0009-championship-equity-does-not-pick.md)). `edge` is a weaker case than either:
there is no evidence and there cannot be. Ranking on it would mean the one signal that
survives is the one nobody could check.

## What was deleted with it

`EDGE_FROM_ROUND = 4` and `_sort_key`. The threshold had a real story behind it — sorting by
edge at pick 3 once surfaced Godwin, Downs and Jordan Love, all at negative VOR, as the top of
a first-round board — but it was a number chosen to contain a problem rather than measured.
With one sort key, `_sort_key` was a pass-through.

The failure that story describes is still guarded: `rank()` filters `vor_live > 0`, so a
below-replacement player cannot reach the table however large his edge.

## Reopening

Would need historical ADP, from a source that retains it. If one appears, the harness already
has the shape — add an arm that ranks descending on `edge` and score it against consensus the
way P0b scored championship equity. Until then this is not an open question, it is a closed
one.
