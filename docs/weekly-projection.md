# The Weekly projection

**Built 2026-08-27.** `hub.models.weekly` is the week-specific layer: everything else in this
repo projects one season-long per-game mean and applies it flat to all seventeen weeks.

Its shape was fixed by measurement rather than chosen. See
[weekly-screen.md](weekly-screen.md) for the screen and
[weekly-projection-plan.md](weekly-projection-plan.md) for the pre-registration.

    weekly Usage  = season-to-date Usage x exp(coef . snap_trend)
    weekly TDs    = weekly yards x the POSITION's touchdown rate
    weekly points = league scoring applied to the counts

**`f = 1` is the incumbent.** With `coef = 0` the multiplier is one and the projection is
exactly the flat one, so the null is the identity and this cannot be much worse than what it
adjusts. **The multiplier acts on counts, never on points.**

## Why these two terms and nothing else

Nine features were screened, two survived a joint screen, and screening those two against the
counts rather than the total showed they do not overlap:

| | targets | receptions | carries | attempts | **touchdowns** |
|---|---|---|---|---|---|
| snap-share trend | +0.094 | +0.077 | +0.062 | +0.032 | **+0.018 killed** |
| prior TD rate | +0.008 | −0.000 | −0.005 | +0.005 | **−0.120** |

So one goes in the Usage multiplier and the other in the touchdown term, and nothing crosses.

**Efficiency is not projected.** Yards per carry persists at r = 0.108 year over year and
touchdowns per yard at ~0, so a model predicting this week's efficiency is predicting noise.
Each player's efficiency is held at his own accumulated rate and only his *opportunity* moves.
**Spread is not projected either** — `sd = k·√mu` stands per
[ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md).

## The fitted multiplier

`coef` in `count = expected × exp(coef · snap_trend)`, least squares on the log ratio, fitted
on strictly earlier seasons. Stable across the walk-forward:

| held-out season | coef targets | coef carries |
|---|---|---|
| 2022 | 0.313 | 0.486 |
| 2023 | 0.382 | 0.534 |
| 2024 | 0.508 | 0.550 |
| 2025 | 0.550 | 0.528 |

Snap share is a fraction, so a **ten-point** rise in snap share is `exp(0.5 × 0.10) ≈ 1.05` —
about **five per cent** more targets. Small, which is the honest size of it, and the clip at
[0.6, 1.6] essentially never binds.

## Gate A, which is a diagnostic and not a gate

Three arms, because the first version had two and could not see its own subject.

| contrast | gain | se | seasons |
|---|---|---|---|
| **the week** — weekly vs `f = 1` | **+0.0103 MAE** | +2.7 | **4/4** |
| the rebuild — `f = 1` vs flat | +0.0624 MAE | +5.0 | 4/4 |
| both together — weekly vs flat | +0.0727 MAE | +5.6 | 4/4 |

All three clear both halves of the repo's usual bar. **None of them is the gate**: the flat
projection has no week-level term at all, so beating it is nearly free, and what decides
whether this ships is the lineup ([ADR-0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md)).

## Three specification errors, and what each was worth

Recorded because each one inverted or hid the result, and because the first two were found by
checking calibration rather than by anything failing.

**1. The diagnostic compared the wrong things.** It carried two arms — the fitted projection
against `ppg_before` — so it compared a *component rebuild* with a *points mean* and buried the
multiplier under every difference between two whole estimators. It reported **−0.0025 MAE at
1/4 seasons** and that number was about the rebuild, not about the week. The plan had already
said `f = 1` is the incumbent; the code did not implement it.

**2. Interceptions and fumbles were not projected at all.** `components.SCORING` prices both at
−2. Leaving them out over-projected quarterbacks by **+1.44 points a week**, which is almost
exactly what an interception a game costs. A projection missing two of the scoring components
is not projecting fantasy points.

**3. The efficiency floor compared a per-game mean against a total.** `MIN_UNITS = 8` is eight
*accumulated* units, and it was being tested against a per-game figure — so essentially every
receiver failed it, since nobody catches eight passes a game, and got the pooled yards-per-catch
rate instead of his own. That under-projected receivers by **−0.66 points a week**.

Fixing 2 and 3 moved the rebuild from −0.009 MAE (1/4 seasons) to +0.062 (4/4), and the week
from +0.0067 at 1.5 se to +0.0103 at 2.7 se. **The bugs were the entire result.** Per-position
bias, which is what surfaced them:

| | before | after |
|---|---|---|
| QB | +1.44 | −0.30 |
| RB | +0.58 | −0.23 |
| WR | −0.61 | −0.08 |
| TE | −0.25 | −0.47 |

The overall bias was −0.002 the whole time. Pooling hid all four.

## What is not done

**Gate B has not been run**, so nothing here sets a lineup. That needs the roster harness in
`hub/season/lineup_gate.py` pointed at weekly consensus as the incumbent, and it is the only
thing that decides whether this ships.

**The injury designation is not in the model, and it cannot go in here.** The plan admits it
to Phase 2 unscreened, having been measured at +0.170 MAE and 3.8 se. But the Gate A panel is
built from `player_stats`, which has no row for a player who did not play — so of **5,473 "Out"
designations across 2021-25, six reach the panel**. Doubtful is 764 against 2.

`hub.models.injury` scores an injury row with no stat row as *zero*: the player who did not
play is its entire subject, and here he is structurally absent. Fitting retention on these rows
would measure something much weaker — "what a Questionable player who played anyway retains" —
and would report it under the stronger result's name.

This is a direct consequence of the pre-registered treatment of inactive weeks (excluded from
Gate A, zero in Gate B), and it means **the injury term belongs in Gate B's construction**,
which builds a complete player-week grid where a missing row is a zero. `status` and `practice`
are carried on the panel for that purpose. There is a test pinning the constraint so it is not
quietly fitted here later.

## Reproduce

```bash
uv run python -m hub.models.weekly --fit
```
