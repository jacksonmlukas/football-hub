# projection_lambda: three holdouts, and the answer

Phase 2.4 of `docs/foundation-plan.md`. Single-season run 2026-08-23, extended to three
season pairs the same day.

**Result: `projection_lambda = 0.0`.** The decision is unchanged from the first run. The
*reason* is not, and the difference matters.

## The question

`hub.draft.projection` nudges the consensus board by last season's expected-vs-actual gap:

```
adj_ecr = ecr * exp(-lam * z)
```

Human rankers anchor on realised fantasy points, which bake in touchdown variance that does
not repeat. ffopportunity gives expected points from opportunity alone, so the gap is a
signal consensus should in principle underweight. `lam` was 0.08 by judgment.

## Method

- **Signal:** season N expected-vs-actual gap, standardised within position.
- **Board:** the last FantasyPros redraft-overall ECR snapshot before season N+1 opened —
  what a drafter would actually have had in hand. Window bounded on **both** sides, so a
  player ranked in an earlier preseason but not this one cannot carry a stale rank forward.
- **Truth:** season N+1 realised PPR points.
- **Metric:** points captured by the top 50 of the resulting board, bootstrapped over 300
  resamples. Spearman over the full board reported alongside.

## Three holdouts

Delta in top-50 points against the untouched consensus board, bootstrap mean per season
pair, then pooled across the three.

| lam | 22→23 | 23→24 | 24→25 | mean | sd | t |
|---|---|---|---|---|---|---|
| 0.00 | — | — | — | — | — | — |
| 0.02 | +10.8 | +71.1 | −14.6 | +22.5 | 44.0 | +0.88 |
| 0.04 | −22.0 | +126.6 | −35.6 | +23.0 | 90.0 | +0.44 |
| 0.06 | −47.5 | +161.6 | −71.7 | +14.1 | 128.3 | +0.19 |
| **0.08** | **−87.1** | **+183.7** | **−93.6** | **+1.0** | **158.3** | **+0.01** |
| 0.12 | −77.9 | +232.8 | −168.6 | −4.6 | 210.5 | −0.04 |
| 0.16 | −37.6 | +213.2 | −166.2 | +3.1 | 192.9 | +0.03 |
| 0.24 | −80.0 | +122.3 | −193.3 | −50.3 | 159.9 | −0.55 |
| 0.32 | −104.7 | −282.9 | −252.6 | −213.4 | 95.3 | **−3.88** |

Best lambda by year: **0.02, 0.12, 0.00**.

## What this actually says

**The sign is not stable.** 2023→2024 is strongly positive at every moderate lambda, by as
much as +233 points. The two seasons bracketing it are negative by comparable margins. One
season in three, this adjustment was worth a great deal; the other two it cost.

**Pooled, the effect is indistinguishable from zero.** At the incumbent 0.08 the mean across
three years is +1.0 points with a standard deviation of 158. Every lambda from 0.02 to 0.24
has |t| < 0.9. There is no evidence here for any nonzero nudge.

**The one thing all three years agree on is that a big nudge is bad.** At lambda 0.32 every
season is negative, mean −213, t = −3.88. Spearman also falls monotonically with lambda in
all three years (0.716→0.669, 0.741→0.666, 0.716→0.656) without exception.

That last pair of facts is worth sitting with, because they are in tension. The whole-board
ordering gets *worse* with lambda in every season, including 2023→2024 — yet that season's
top-50 points got substantially *better*. The adjustment is reshuffling inside the top of
the board, and in one year out of three that reshuffle happened to be right. That is a
description of variance, not of edge.

## Correcting the single-season conclusion

The first run used only 2024→2025 and I offered an explanation: that the regression is
already priced by the time a preseason board is published, since ffopportunity is free and
widely used.

**Three seasons do not support that.** A fully priced signal would sit near zero every
year. Instead one year shows a large positive effect. Whatever is happening is closer to
regime dependence — or to nothing at all plus noise large enough to look like both.

Two numbers also moved between runs, because the first version of `holdout` bounded the ECR
window on only one side and let players carry a stale rank forward. The 2024→2025 holdout
went from 553 players to 430, and the 0.08 delta from −34.6 to −93.6. Same sign, larger
magnitude. The earlier table in this file was computed on the leakier holdout and has been
replaced.

## The trap, still worth recording

On the 2024→2025 single realisation, lambda 0.12 looked like the winner at +44.5 points.
Bootstrapped it was −37.3 ± 14.8. The first selection rule took the best point estimate and
would have returned it — a confident, evidence-flavoured recommendation to make the board
worse. The unit test that caught it builds a holdout from pure noise and asserts the sweep
returns zero; that first implementation returned 0.05.

The three-season view is the same lesson one level up. A single holdout said "clearly
negative, this signal is priced." Three say "unstable, and the single-season story was
overconfident."

## What would change the answer

Not more lambda values. More seasons, and a reason. Specifically:

- Extend to 2019→2020 and 2020→2021. `load_ff_rankings("all")` reaches back to 2019-12-27,
  so two more pairs are available without new data.
- Ask what was different about 2023→2024 rather than averaging it away. If the effect
  concentrates in a particular position or a particular part of the board, that is a real
  finding and the current whole-board metric is hiding it.

Both are post-draft work. Neither changes what to do on Sep 3.

## What changed in the code

- `projection_lambda` set to **0.0** in `conf/config.yaml` and `hub.config`.
- `DEFAULT_LAMBDA` in `hub.draft.projection` set to 0.0, so the module is a no-op at its
  default rather than silently applying an adjustment with an unstable sign.
- `holdout()` now bounds the ECR window on both sides.

`hub.draft.projection` is imported by nothing except its own test — the board has never
used it. Given a signal whose sign flips year to year, that stays the right state.

## Reproduce

```bash
uv run python -m hub.draft.tune --sweep
uv run python -m hub.draft.tune --sweep --signal-season 2023 --board-season 2024
```
