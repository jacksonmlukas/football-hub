# Market-implied shrinkage: the tripwire fired, and the number is promising anyway

**Run 2026-08-28**, at the user's request, having been declined in
[weekly-shrinkage.md](weekly-shrinkage.md) as the third try at the same rescue. It is the
fourth. That is written at the top because it is the most important thing about it.

## The result

Points per team-week, weekly arm minus consensus rank:

| shrink target | frozen rosters | with waiver churn |
|---|---|---|
| none | −0.304 [−1.068, +0.394] | −3.790 [−5.199, −2.361] |
| positional mean (`mae`) | −0.235 [−0.894, +0.434] | −3.773 [−5.211, −2.277] |
| positional mean (`tail`) | −0.474 [−1.200, +0.253] | −12.442 [−15.006, −10.063] |
| **market-implied (`mae`)** | **+0.159** [−0.433, +0.736] | **−2.450** [−3.637, −1.290] |
| market-implied (`tail`) | — | −1.329 [−2.477, −0.091] |
| *market prior alone* | *−1.146* [−2.062, −0.255] | *−1.985* [−3.542, −0.492] |

**+0.159 is the first positive point estimate in the whole programme.** The verdict is still
**SHOW, NEVER RANK ON** — the interval contains zero and it won 2 of 3 seasons.

## The tripwire fired

Pre-registered before the run:

> **the churn gate at least halves its loss, from −3.79 to better than −1.9, and stays
> negative. The frozen gate moves less than 0.3.** If the frozen gate moves more than that,
> something other than thin-sample shrinkage changed and the run is suspect.

| | predicted | measured | |
|---|---|---|---|
| churn | better than −1.9 | **−2.450** | missed — a 35% improvement, not a halving |
| frozen | moves < 0.3 | **moves 0.46** | **tripwire fired** |

The tripwire is right, and it is right about something real: **this is no longer shrinkage.**
`w = n/(n+k)` never reaches 1, so a market signal is blended into *every* player's projection,
thick-sample and thin alike. The estimator is not "the same model with thin samples
regularised" — it is a new model, a market/Usage blend, and a frozen gate full of thick-sample
players had no business moving half a point under a thin-sample fix.

**The rule is not "have a tripwire". It is do not touch it after you have seen what it caught**
([ADR-0009](adr/0009-championship-equity-does-not-pick.md)). So it is not explained away, and
the frozen figure is not reported as a gate result.

## It is not the market being imported, which is worth knowing

The obvious reading of a gain that appears the moment market information enters is that the
market is carrying it. It is not:

| arm | frozen |
|---|---|
| in-season Usage alone | −0.304 |
| **market prior alone** | **−1.146** |
| the two blended | **+0.159** |

**The blend beats both of its components**, and the market prior on its own is the worst arm
tried. So the gain is genuine complementarity — a preseason rank knows things about a
thin-sample player that his four games do not, and his four games know things the August
market could not. That is the diagnosis `component-projection.md` made in August — *"a WR1 and
a WR5 do not regress toward the same place, and the market already prices that"* — landing
where it was pointed.

Note also that the churn gate improves by the same mechanism in the same direction
(−3.79 → −2.45), and the market prior alone does better on churn (−1.99) than the model did,
which is consistent: waiver decisions are exactly where in-season history is thinnest.

## Why the curves are refitted rather than imported

`hub.models.volume.VOLUME_CURVE` is the right functional form and the wrong numbers: it is
frozen at a fit on **2022-25**, which is exactly the span held out here. Using it would
evaluate a prior on the seasons it was fitted on. The same shape — `log(1 + count per game) =
a + b·log(rank)`, per position — is refitted inside every walk-forward fold on strictly earlier
seasons, against the board's preseason `ecr`, with the rank clamped to the fitted range.

**And the rank is the August one, never the Friday one.** Shrinking toward the weekly ranking
would make the arm partly *be* the incumbent it is measured against, and a win could be
entirely the borrowed half. There is a test that hands the function both and checks which one
moves the answer.

## What this licenses

**Nothing, yet.** A positive number arrived at on the fourth variant, after a tripwire fired,
is not a result — it is a hypothesis with a good prior. What it licenses is a *new*
pre-registration:

> The **market/Usage blend** is a model in its own right, not a regularised version of the
> Weekly projection. Pre-register it as such — its own null (which is *not* `f ≡ 1`, since the
> blend at `k = 0` is a different object), its own fitted `k`, its own gate — and run it once,
> cleanly, on both gates.

Until then [ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) stands
unchanged: shown, never ranked on.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run --shrink mae-market
uv run python -m hub.season.weekly_gate --run --churn --shrink mae-market
uv run python -m hub.season.weekly_gate --run --shrink market-only   # the probe
```
