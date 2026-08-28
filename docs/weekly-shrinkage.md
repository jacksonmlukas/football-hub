# Shrinking thin samples: the pre-registered fix does nothing, and the aggressive one is worse

**Run 2026-08-28.** The experiment [ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md)
and [weekly-projection-plan.md](weekly-projection-plan.md) pre-registered after the waiver gate
failed on a winner's curse: shrink each player's projection toward his positional mean by his
sample size, fit on training seasons, and re-run both gates.

## The result

Points per team-week, weekly arm minus consensus rank:

| shrinkage | frozen rosters | with waiver churn |
|---|---|---|
| **none** | **−0.304** [−1.068, +0.394], 1/3 | **−3.790** [−5.199, −2.361], 0/3 |
| **`mae`** — as pre-registered | −0.235 [−0.894, +0.434], 2/3 | −3.773 [−5.211, −2.277], 0/3 |
| `tail` — exploratory | −0.474 [−1.200, +0.253], 2/3 | **−12.442** [−15.006, −10.063], 0/3 |

**The pre-registered shrinkage changes nothing** — −3.790 to −3.773 on the gate it was
designed to rescue. **The aggressive version makes it three times worse.**

Both verdicts are unchanged: SHOW on the frozen gate, REMOVE on the churn variant.

## Why the pre-registered version does nothing

It fits `volume_k = 0` — **no volume shrinkage at all** — in every held-out season:

| held-out season | volume_k | eff_k |
|---|---|---|
| 2022 | 0.0 | 128.0 |
| 2023 | 0.0 | 16.0 |
| 2024 | 0.0 | 8.0 |
| 2025 | 0.0 | 8.0 |

Zero is in the grid on purpose, so declining to shrink is a candidate the fit can pick. It
picked it, four times out of four, and on accuracy the shrunk arm is **−0.0042 MAE at −0.8 se,
2/4 seasons** — a null, very slightly the wrong way.

**The objective was blind to the defect.** The projection is *already unbiased at every sample
size* — −0.36 points at one game played, −0.09 at nine or more. Mean absolute error has nothing
to reward, because the winner's curse does not live in the mean. It lives at the **maximum over
hundreds of candidates**, and the mean is dominated by the bulk.

Pre-registering the fitting objective along with the estimator was right, and it is what makes
this a clean null rather than a search. But the objective and the defect were never matched.

## Why the aggressive version is worse

`tail` fits the same estimator against the bias among the **top decile by projection** in
training — the slice a waiver decision reads. It works, in the narrow sense: held-out tail bias
among thin-sample players goes from +1.08 to −0.81 in 2022, +0.85 to −0.69 in 2023.

And it costs **16% of MAE** — 4.72 to 5.50 — because fixing the top by pulling everything
toward positional means flattens the discrimination the decision needs. The churn gate goes
from −3.79 to −12.44. **Removing the bias by removing the signal is worse than the bias.**

## What this actually is

Uniform shrinkage toward a **single positional mean** is the wrong instrument, and this repo
already knew that at a different grain. [component-projection.md](component-projection.md),
written 2026-08-24 on *season-long* volume:

> The likely reason shrinking hurts skill positions: shrinking toward a single positional mean
> over-shrinks the studs. A WR1 and a WR5 do not regress toward the same place, and the market
> already prices that. A volume model that shrank toward an **ADP-implied** prior rather than a
> positional average is a different and more promising thing.

That is the same failure, reproduced at weekly grain by an independent route four days later.
[volume-model.md](volume-model.md) records that swapping the positional mean for a pick-implied
prior turned the season-level null into a 99.9% result against the same baseline — and still
did not beat simply reading the pick.

## Against the pre-registration

> *Shrinkage should help the churn gate materially and the frozen gate barely, since frozen
> rosters are all thick-sample. If it helps the frozen gate too, something else changed and the
> run is suspect.*

**The frozen half held**: −0.304 → −0.235, well inside the interval, as predicted. **The
material half failed**: the churn gate moved by 0.017 points. The suspect-run tripwire did not
fire, which is the one piece of good news — nothing moved that should not have.

## What would be next, and it is not being run tonight

Shrink toward a **market-implied** prior — the preseason board's own projection for that player,
or his consensus rank — rather than a positional average. It is the fix both this and the
season-level work point at.

Two reasons to stop here rather than run it. It is now the *third* variant of the same rescue
attempt, and a rescue attempted three times is a result being negotiated with. And shrinking
toward consensus imports the incumbent into the arm being measured against it, which needs its
own pre-registration to be worth anything.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run --churn --shrink mae
uv run python -m hub.season.weekly_gate --run --churn --shrink tail
```
