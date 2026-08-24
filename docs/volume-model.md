# The volume model, anchored on the market's pick

**Built 2026-08-24**, `hub/models/volume.py`. The result is mixed and the mixed part is the
point, so it is stated first.

| | held-out RMSE |
|---|---|
| carry points forward | 3.787 |
| components, touchdowns regressed | 3.720 |
| **market's pick alone** | **3.578** |
| blend of the two point predictions | 3.482 |
| **components + pick-implied prior** | **3.474** |

- **It fixed the volume model.** Against carrying volume forward: **−0.247 RMSE,
  95% CI [−0.385, −0.104], 99.9%.** The earlier null was the anchor, not the idea.
- **It did not beat reading the pick.** Against the market alone: −0.105,
  95% CI [−0.294, +0.079], **87%** — below anything this repo calls a result.
- **It ties a trivial blend.** Averaging the component prediction and the market prediction
  scores 3.482 against the structural model's 3.474. That is the tell: the structure is not
  buying anything the arithmetic did not.

So the model ships as a **decomposition**, not a replacement projection.

## Why the positional-mean version failed and this one works

[component-projection.md](component-projection.md) screened shrinking volume toward a
positional average and got a null (3.311 against 3.303). The diagnosis there was that a WR1
and a WR5 do not regress toward the same place, and the market already prices where each one
regresses to.

That diagnosis holds. Swapping the positional mean for a pick-implied prior turns a null into
a 99.9% result against the same baseline. The mechanism was right; the anchor was wrong.

Fitted on 526 player-season pairs from this league's own drafts, 2022-25, joined to nflverse
through the `ff_playerids` crosswalk (a proper ID join, not name matching — 90% of each draft
matches). Held out by season: curves and shrink weights fitted on target seasons 2022-23,
evaluated on 2024-25.

**Volume is trusted less than efficiency** — keep 0.5 of a player's own volume against 0.7 of
his own efficiency. Volume is what a changed situation moves most, and the pick is the only
input that knows the situation changed at all.

Implied per-game volume, as a sanity check on the curves:

| | pick 5 | pick 40 | pick 120 |
|---|---|---|---|
| RB | 19.6 car / 4.4 tgt | 11.3 car / 2.8 tgt | 8.3 car / 2.2 tgt |
| WR | 10.6 tgt | 6.8 tgt | 5.3 tgt |

## What ships, and why that shape

`decompose(pick, position, target_ppg)` returns the component line that produces a given
points projection. **The market's mean is reproduced exactly rather than second-guessed** —
the screen never showed this model beating it — and only the *shape* comes from here.

That shape is the thing a points projection cannot give and
[component-projection.md](component-projection.md) needs: `sample_weeks` requires components,
and the draft board has only points. This is the bridge between them.

`pick_prior` and `project` are also exposed, since the evaluation above is about them and a
future volume model will want to beat them rather than start over.

## Guards worth knowing about

**Extrapolation is clamped to the observed pick range per position.** This league has never
drafted a tight end before pick 11 or a quarterback before 24. Unclamped, the tight end curve
claims 10.8 targets a game at pick 3 — more than any real tight end sees — purely by
extrapolating past every data point it has.

**Efficiency is fitted linear in log(pick), not log-log.** Yards per carry goes negative (a
receiver with one carry for −5 yards), and `log(y + 1)` on that is NaN. The first version of
the screen did exactly that to 188 of 272 training rows, which surfaced as an
`SVD did not converge` rather than as a wrong answer.

## What would change the verdict

The 87% against the market is the number to watch. It is a point estimate favouring the model
by about 3% of RMSE on 254 held-out pairs, and this league has only four drafts to learn from.
More seasons would settle it either way, and the honest reading today is *suggestive, not
established*.

Two things that might turn it into a result rather than more data:

- **Real ADP rather than realised draft pick.** A pick is where he actually went in one
  twelve-person room, which is ADP plus that room's noise. ESPN publishes ADP, which is the
  cleaner signal and is already on the board.
- **Situation features the pick cannot fully price** — a depth chart, a coaching change, a
  vacated target share. That was screened once as a standalone signal and came back null
  ([depth-chart-signal.md](depth-chart-signal.md)); as a *correction to a market anchor* it
  is a different question and has not been tested.

## Reproduce

```bash
uv run python -m hub.draft.calibrate     # the draft-pick side of the join
```
