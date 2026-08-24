# projection_lambda: the sweep, and the answer

Phase 2.4 of `docs/foundation-plan.md`. Run 2026-08-23.

**Result: `projection_lambda = 0.0`. The regression signal does not improve on consensus,
and the incumbent 0.08 was making the board measurably worse.**

## The question

`hub.draft.projection` nudges the consensus board by last season's expected-vs-actual gap:

```
adj_ecr = ecr * exp(-lam * z)
```

The reasoning behind it is sound and is worth restating, because the reasoning is not what
failed. Human rankers anchor on realised fantasy points, which bake in touchdown variance
that does not repeat. ffopportunity gives expected points from opportunity alone, so the
gap is a clean signal consensus should underweight. `lam` was set to 0.08 by judgment.

## The holdout

- **Signal:** 2024 expected-vs-actual gap, standardised within position.
- **Board:** the last FantasyPros redraft-overall ECR snapshot before the 2025 season —
  what a drafter would actually have had in hand.
- **Truth:** 2025 realised PPR points.
- **Coverage:** 553 players with both a preseason rank and a 2025 finish; 437 of them
  (79%) carry a 2024 signal.
- **Metric:** points captured by the top 50 of the resulting board. The first 50 picks
  decide a season, and scoring the whole board lets deep-bench noise drown the part that
  matters. Spearman over the full board is reported alongside.

## The sweep

| lam | spearman | top50 pts | delta | bootstrap | sigma |
|---|---|---|---|---|---|
| **0.00** | **0.7450** | **11,656** | — | — | — |
| 0.02 | 0.7449 | 11,656 | +0.0 | +0.5 ± 4.8 | +0.11 |
| 0.04 | 0.7436 | 11,656 | +0.0 | −17.0 ± 7.8 | −2.18 |
| 0.06 | 0.7417 | 11,609 | −46.9 | −30.8 ± 10.5 | −2.93 |
| **0.08** | 0.7386 | 11,609 | **−46.9** | **−34.6 ± 11.6** | **−2.99** |
| 0.12 | 0.7312 | 11,700 | +44.5 | −37.3 ± 14.8 | −2.51 |
| 0.16 | 0.7205 | 11,656 | +0.1 | −41.5 ± 14.6 | −2.84 |
| 0.24 | 0.6956 | 11,352 | −304.1 | −161.3 ± 18.1 | −8.91 |
| 0.32 | 0.6721 | 11,224 | −432.2 | −228.5 ± 21.0 | −10.90 |

Two readings, and they agree.

**Spearman falls monotonically with lambda.** 0.7450 → 0.6721. Every nudge makes the
whole-board ordering worse, with no interior optimum. That is not a badly-chosen lambda;
that is a signal pointing the wrong way.

**Every lambda at or above 0.04 is significantly negative** under bootstrap, and the
incumbent 0.08 sits at **−3.0 sigma**. It was not neutral. It was costing about 35 points
of top-50 production.

## The trap this nearly walked into

On the single realisation, **lambda 0.12 looks like the winner at +44.5 points.** Under
300 bootstrap resamples it is −37.3 ± 14.8, or −2.5 sigma. The apparent gain was one
favourable draw of which players happened to land inside the top 50.

An earlier version of the selection rule took the best point estimate and would have
returned 0.12 — a confident, evidence-flavoured recommendation to make the board worse.
The unit test that caught it builds a holdout from pure noise and asserts the sweep returns
zero; the first implementation returned 0.05. The bootstrap gate exists because of that
test, not because of foresight.

## Why the signal fails, and what would change it

The reasoning is not obviously wrong, so the negative result deserves a candidate
explanation rather than a shrug. The most likely one: **by the time FantasyPros publishes a
preseason board, the regression is already priced.** Rankers see the same expected-points
data. The gap this exploits was real in an era when it was not public; ffopportunity being
free and widely used may have closed it.

That is a hypothesis, not a finding. What would test it: run the same holdout on 2022→2023
and 2023→2024. If lambda is near zero in every year, the signal is priced. If it is
positive in older pairs and zero recently, it decayed — which is a more interesting result
and a genuine portfolio artifact.

## What changed

- `projection_lambda` set to **0.0** in `conf/config.yaml` and `hub.config`.
- `DEFAULT_LAMBDA` in `hub.draft.projection` set to 0.0, so the module is a no-op at its
  default rather than silently applying a harmful adjustment if it is ever wired up.

`hub.draft.projection` is currently imported by nothing except its own test — the board has
never used it. That turns out to have been lucky rather than wise, and this sweep is the
argument for leaving it that way until a multi-season version says otherwise.

## Reproduce

```bash
uv run python -m hub.draft.tune --sweep
```
