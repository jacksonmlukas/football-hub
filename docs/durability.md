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

## Today's injury news is a different quantity, and is priced separately

The first version of this said current status could not be priced because there was nothing
to fit a coefficient on. That was asserted rather than checked, and it was wrong. nflverse
publishes weekly injury reports back to 2019, and a **week-1 designation is the closest
historical analogue to an August one** — so the coefficient can be fitted.

`total_next/17 ~ proj_ppg + designation`, 1,263 player-seasons:

| designation | n | beta | 95% CI | P(<0) | applied |
|---|---|---|---|---|---|
| **Out / Doubtful** | 27 | **−1.631** | [−2.554, −0.736] | **100.0%** | yes |
| Questionable | 36 | −0.949 | [−2.495, +0.459] | 90.2% | **no** |

Mean games played backs it up: 7.7 for Out and 6.7 for Doubtful, against 11.7 for the
undesignated.

**Questionable stays unpriced, and the reason is not mainly the p-value.** A week-1
QUESTIONABLE and an August QUESTIONABLE are different populations:

| designation | August 2026 board | historical week 1 |
|---|---|---|
| QUESTIONABLE | **12.6%** | **2.9%** |
| OUT + DOUBTFUL + IR | 4.1% | 2.1% |

August QUESTIONABLE is **4.4× more common**. Applying a coefficient estimated on the much
sicker week-1 group to an eighth of the August board would be worse than leaving it for
judgment. Out, Doubtful and IR transfer far better and are applied.

**IR has no coefficient of its own** — nobody on injured reserve appears on a practice
report. Starting a season there means missing at least four games by rule, so it is at least
as severe as Out; borrowing that number understates it, which is the direction to be wrong
in.

Unlike the durability *trait*, the injury markdown applies at every position. Being ruled out
is news, not a trait the market has had years to discount.

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

Both carry QUESTIONABLE, which is the one designation that is measured and deliberately not
priced — so for these two it remains a judgment call, and exactly the kind of thing the board
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
