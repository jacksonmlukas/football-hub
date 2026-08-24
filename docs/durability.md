# Durability: availability as a per-player trait

**Built 2026-08-24**, `hub/draft/durability.py`.

`TALENT_CV` already carried availability, but only as a population average — it was fitted on
points per *team* game, so missed time sits inside it at the positional level. Every running
back therefore carried identical durability risk. A back with a long injury history and one
who has never missed a snap were the same player to the simulation.

## Availability persists more than the folklore says

Games missed, year over year, across 1,531 player-season pairs with a real prior role:

| | |
|---|---|
| Pearson r | **+0.407** |
| Spearman r | +0.344 |

The practical form:

| missed last season | P(miss 3+ this season) |
|---|---|
| 0 | 41.3% |
| 1–2 | 45.3% |
| 3–5 | 59.1% |
| **6+** | **76.2%** |

Base rate 55.3%. A 35-point spread between the extremes, which is a good deal stronger than
the common claim that injury proneness is hindsight.

**One season is enough.** A two-year history scores R² 0.4597 against 0.4614 for last season
alone, and the second year's coefficient is −0.040 against −0.159. Almost all of the signal is
in the most recent season, so the model uses that and nothing older.

## The projection does not fully price it

`ppg_next ~ proj_ppg + prior games missed`. A projection that already discounted durability
would leave nothing for missed games to explain:

| target | beta(proj) | beta(missed) | 95% CI | P(<0) |
|---|---|---|---|---|
| points per **game played** | 0.753 | −0.087 | [−0.169, −0.002] | 97.9% |
| points per **team game** | 0.664 | −0.186 | [−0.266, −0.104] | **100.0%** |

Per team game is the fantasy question — what he delivers across a season, not what he does on
the days he suits up — and there the effect is unambiguous. Roughly 3.2 points of season
output per prior game missed.

The per-game row is worth noting on its own: fragile players score slightly less *even when
they play*, which is consistent with playing hurt.

## The surprise: running backs are already priced

| position | n | beta(missed) | 95% CI | P(<0) | applied |
|---|---|---|---|---|---|
| **QB** | 121 | **−0.457** | [−0.645, −0.257] | **100.0%** | yes |
| **WR** | 274 | **−0.151** | [−0.266, −0.045] | **99.6%** | yes |
| TE | 101 | −0.097 | [−0.242, +0.062] | 89.0% | no |
| **RB** | 158 | −0.065 | [−0.293, +0.167] | 71.2% | **no** |

**Running backs — the position everyone worries about — show essentially nothing.** The
natural reading is that the market already discounts running back durability, because
everyone knows running backs break. There is no residual left to take, and marking them down
again would be double-counting the market's own discount.

The inefficiency is at quarterback and receiver, where durability is not the first thing a
drafter thinks about.

## Today's injury news is a different quantity, and is not priced

A player hurt *now* is not the same as a player who was fragile *last year*. The board
surfaces current designations from ESPN and deliberately leaves them out of the projection,
because **there is nothing to fit a coefficient on** — no history of preseason designations
against outcomes. Inventing a markdown would be worse than showing the drafter the flag.

It is also low-information at this date: **21 of the top 120 by ADP are QUESTIONABLE** in late
August, which is close to saying nothing.

## What it says about the pick-3 tie

Both candidates carry a designation today, and neither is flagged by the trait:

| player | missed 2025 | status today | priced? |
|---|---|---|---|
| Puka Nacua | 1 | QUESTIONABLE | trait says nothing |
| Christian McCaffrey | 0 | QUESTIONABLE | trait says nothing; RB unpriced regardless |

So the durability model, having been built precisely to answer this, says **the concern is
smaller than intuition suggests for both**. McCaffrey's reputation rests on 2023–24, and a
second prior season adds essentially nothing measurable over the most recent one. The pick-3
verdict is unchanged.

Current status remains genuinely unmodelled, and is exactly the kind of thing the board
already suggests using to break a tie the simulation cannot.

## Biggest markdowns on the live board

| player | pos | ADP | missed | markdown |
|---|---|---|---|---|
| Jayden Daniels | QB | 50.9 | 10 | −4.57 ppg |
| Joe Burrow | QB | 58.8 | 9 | −4.11 ppg |
| Malik Nabers | WR | 34.3 | 13 | −1.96 ppg |
| Garrett Wilson | WR | 36.5 | 10 | −1.51 ppg |

## Reproduce

```bash
make draft
```
