# Touchdown luck: last season's actuals against the current board

**Built 2026-08-24**, `hub/draft/regression.py`, surfaced on `make draft`.

This is the "fit on previous years' actual stats, find value on the current board" cut. It
is not a projection — it is one specific way the room misprices, tested directly.

## The chain

Three things, each measured separately, in this order:

1. **Touchdown rate per yard does not persist.** Year over year: **+0.004** receiving,
   **−0.030** rushing ([component-projection.md](component-projection.md)). Volume persists
   at 0.79–0.81; touchdown rate is indistinguishable from zero.
2. **Fully regressing it beats carrying points forward**, and the fitted optimal shrink is
   **1.0**, not a partial one ([volume-model.md](volume-model.md)).
3. **The market does not fully regress it.** That is the part that turns a modelling fact
   into an edge, and it is the part below.

## Does the room already discount last year's touchdowns?

Two regressions on the same standardised predictors — prior-season *yardage points* and
prior-season *touchdown points*. One predicts where the player went in next season's draft;
the other predicts what he actually scored. `td/vol` is what a touchdown point is worth
relative to a yardage point.

| position | n | market td/vol | truth td/vol | gap | 95% CI | P(market overpays) |
|---|---|---|---|---|---|---|
| **QB** | 73 | **1.02** | **−0.05** | **+1.07** | [+0.21, +2.11] | **98.9%** |
| RB | 165 | 0.35 | 0.10 | +0.25 | [−0.17, +0.55] | 91.2% |
| WR | 221 | 0.16 | 0.06 | +0.10 | [−0.11, +0.32] | 83.4% |
| TE | 67 | 0.27 | 0.42 | −0.14 | [−1.08, +0.25] | 26.5% |

**Quarterbacks are the clear case.** The room prices last year's passing touchdowns almost
as heavily as volume (1.02), and their true predictive weight is *negative* (−0.05). That is
a large gap and it clears significance on its own.

**Do not pool these.** The pooled regression gives a *market* ratio of −0.17 against a
*truth* ratio of +0.67 — apparently the opposite conclusion. That is Simpson's paradox:
quarterbacks earn far more touchdown points than skill players and are drafted much later,
so mixing positions inverts the sign. The first version of this test reported the pooled row
and it was meaningless.

## How much to believe

Honestly: **quarterbacks yes, running backs and receivers directionally, tight ends no.**

RB and WR are at 91% and 83%, which is below anything this repo calls a result, and QB is one
of four positions tested — the same multiple-comparisons trap that stopped the volume model
from shipping a QB-only win in [component-projection.md](component-projection.md).

What makes this case stronger than that one, and worth shipping where the other was not:

- The **mechanism was measured first and independently**. Zero touchdown persistence was
  established before this test was run, and it predicts this exact sign.
- **Three of four positions show the predicted direction**, which is a pattern rather than a
  lone outlier. TE is the exception and TE has the smallest sample by some way.
- The QB gap is **large**, not marginal — the room's weight is twenty times the true one.

It is still one league's four drafts. Treat the quarterback column as a real finding and the
rest as a tiebreaker.

## It is not the column the board already had

`fp_over_expected` measures realised points against expected points from *opportunity*.
This measures realised touchdowns against the *yardage that produced them*. On the live
board they correlate at **+0.16** — and since the two are signed in opposite directions, a
genuine overlap would show as a strong negative. They are different cuts and both are on the
board.

## On the live 2026 board

96 drafted players inside ADP 120 have a 2025 season to measure. Points per game above what
their yardage supports:

| | player | pos | ADP | td luck |
|---|---|---|---|---|
| FADE | Matthew Stafford | QB | 79.9 | +4.01 |
| FADE | Davante Adams | WR | 46.1 | +3.95 |
| FADE | Brock Purdy | QB | 102.9 | +3.94 |
| FADE | Josh Allen | QB | 22.1 | +3.59 |
| BUY | CeeDee Lamb | WR | 10.7 | −1.63 |
| BUY | Justin Jefferson | WR | 12.1 | −1.55 |
| BUY | Breece Hall | RB | 33.6 | −1.39 |
| BUY | Kenneth Walker III | RB | 22.1 | −1.32 |

Jefferson appearing here *and* topping the xFP−FP list is the two independent instruments
agreeing on him, which is worth more than either alone.

## What this does not say

It does not say Josh Allen is a bad pick. It says the part of his 2025 that came from an
unsustainable touchdown rate will not repeat, and the room is paying for it anyway. Whether
the rest of his profile justifies pick 22 is a different question and this column does not
answer it.

## Reproduce

```bash
make draft
```
