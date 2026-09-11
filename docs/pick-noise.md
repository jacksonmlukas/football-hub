# Pick noise, refitted — and the board either side of it

**Re-run 2026-09-07**, under [method.md rule 13](method.md) — *a measurement that contradicts a
published number is not finished until the published number moves*. This is the third of that
rule's three incidents, and the one it names as open.

The estimator was repaired on 2026-09-04 (`02488c0`). The estimate was not re-run, and a test
asserting the old pair held the two apart. Nothing here is a new method: it is the same fitter,
on the same four drafts, finally asked what it says.

## The result

`sigma(pick) = a + b * pick`, fitted by `fit_pick_noise(league, [2022, 2023, 2024, 2025])`.

| | intercept `a` | slope `b` | slope 95% CI | population fitted |
|---|---|---|---|---|
| **shipped now** (2026-09-11, #155) | **2.51** | **0.150** | **[0.143, 0.156]** | 612 picks with `ecr <= 168`, 4 drafts |
| superseded 2026-09-11 | 1.31 | 0.169 | [0.159, 0.179] | 672 picks inside a 204-pick pool, 4 drafts |
| superseded 2026-09-07 | 1.00 | 0.253 | *none published* | stated as "734 picks", over the whole consensus list |

Unrounded, the current fit returns `a = 2.5135`, `b = 0.14950`. The constants ship rounded to
the precision `noise_from_picks` prints.

## Restated 2026-09-11: the drafted sample is one-sided past rank 168 (#155)

The 2026-09-07 fit ran on every matched pick inside the draftable pool. That sample is
conditioned on being *drafted*, and near the back of the pool the conditioning is one-sided: a
player ranked 190 who went undrafted has no pick number and is absent, so the ones who remain
at that rank are exactly the ones who beat it. Measured on the 672 picks as the mean **signed**
deviation `pick − ecr` by rank bin — zero under no censoring, negative where only early-goers
survive the pool boundary:

| ecr bin | n | mean signed `pick − ecr` | mean absolute |
|---|---|---|---|
| 1 – 145 | 545 | within ±3 at every bin | — |
| 145 – 169 | 71 | −5.1 | 20.7 |
| 170 – 192 | 37 | **−22.8** | **26.6** |
| 193 – 204 | 19 | **−36.2** | **36.2** |

In the last two bins the signed mean *equals* the absolute mean: every observed player went
earlier than his rank, which is what a sample containing one tail looks like. Those rows were
fitting the slope to a one-sided residual and inflating it. The fit is now restricted to
`ecr <= 168`, the last bin before the signature appears, and the ceiling swept so the choice is
visible:

| fit ceiling | n | `a` | `b` | 95% CI | sigma at pick 100 |
|---|---|---|---|---|---|
| 204 (superseded) | 672 | 1.31 | 0.169 | [0.159, 0.179] | 18.2 |
| **168** | **612** | **2.51** | **0.150** | **[0.143, 0.156]** | **17.5** |
| 145 | 545 | 2.83 | 0.143 | [0.116, 0.170] | 17.1 |
| 120 | 460 | 2.33 | 0.156 | [0.138, 0.174] | 17.9 |

The superseded interval and the restated one do not overlap, so the censored tail's effect on
the slope is resolvable at two standard errors clustered on the draft. Below 168 the slope is
stable across cuts and the intervals overlap; the cut is where the data stops being two-sided,
not where the slope is prettiest.

**The axis, stated correctly.** The fit reads `ecr`; `_sigma` applies it to `mu_pick`. For the
historical drafts these are the *same number* — no draft market exists for a past preseason
(ADR-0010), so `blended_adp`'s `w · adp + (1 − w) · ecr` is the identity — and the fitted and
applied axes coincide on the population the fit uses. On the live board, where ADP exists, they
differ, and that is a limitation this fit carries rather than one it can remove. The docstring
that said "fitted on a pick number" was describing the intent and not the code. Modelling the
censoring instead (a Tobit) was declined: it needs a counterfactual pick number for players
never picked, which the data does not contain.

**The board either side of it, and it barely moves.** Same served board of 457 rows, 20,000
sims, seed 0, `w = 0.5`, the four scarcity turns from slot 3. Not one player moves by more than
3.6 points of survival probability at any turn — against the 2026-09-07 refit, which moved
forty-nine by more than five at turn 75 → 94. The direction is split, and the split is the
crossover: the restated law has a higher intercept and a shallower slope, so it crosses the
old one at pick 17 — half a pick *wider* at the top, tighter from the second round on.

| turn | max mover | before | after |
|---|---|---|---|
| 3 → 22 | Drake London | 0.093 | 0.129 |
| 27 → 46 | Javonte Williams | 0.150 | 0.161 |
| 51 → 70 | Christian Watson | 0.491 | 0.481 |
| 75 → 94 | Trevor Lawrence | 0.309 | 0.294 |

So the 2026-09-07 refit was the correction that mattered, and this is the refinement that
says what population it was measured on. The scarcity ordering does not change at any turn.

The interval is bootstrapped over **drafts**, n = 4 — not over the 672 picks. One manager
reaching in round two moves every later pick in that room, so a pick-level interval would be
several times too tight. That is [gate-power.md](gate-power.md)'s error, one layer down.

> **What caused the move.** Two defects in `fit_pick_noise`, both inflating the slope, both
> fixed in `02488c0` and neither re-run afterwards:
>
> 1. It fitted an `ecr` rank over a 300-plus-player consensus list, while `_sigma` applies the
>    result to `mu_pick`, an expected **pick number**. Fitting on one axis and predicting on
>    another stretched the x-range by half again and flattened the slope to cover ranks that
>    are not picks at all. The fit now runs on the draftable pool: 672 of the 732 matched
>    picks, the ones whose consensus rank was ever a pick number.
> 2. It refitted through `max(sigma_hat − a, 0)`, which zeroes every residual under the pinned
>    intercept instead of letting it pull the slope down — so the slope was fitted to the upper
>    envelope of the data.
>
> **And a third thing, which is why the intercept moved too.** 1.00 was exactly `MIN_SIGMA`.
> `_constrained` pins the intercept there whenever the unconstrained line wants to go negative,
> so the published intercept was the floor speaking rather than the data. Neither shipped
> number had been identified by the corrected fit. At 1.31 the constraint no longer binds.

> **The direction reverses, and that is the finding.** The superseded comment argued the fit
> *widened* a `2.0 + 0.18 * mu` prior that was over-confident about who survives deep on the
> board. Repaired, the fit is **narrower than that prior at every pick in the pool**:
>
> | at pick | prior | superseded fit | shipped now |
> |---|---|---|---|
> | 3 | 2.5 | 1.8 | 1.8 |
> | 24 | 6.3 | 7.1 | 5.4 |
> | 100 | 20.0 | 26.3 | 18.2 |
> | 204 | 38.7 | 52.6 | 35.7 |
>
> A narrower sigma means the simulated room drafts closer to consensus, so a player ranked
> ahead of your next turn is *more* certainly gone. Availability falls, `cost_of_waiting`
> rises, and the correction pushes the board **toward** scarcity — the opposite of what the
> superseded comment concluded from the same repair.

## A board built before and after

`readable()` board of 2026-09-04, 457 rows, availability at 20,000 sims, seed 0, `w = 0.5`.
Slot 3 of 12, so the scarcity turns are the ones with a 19-pick wait: 3 → 22, 27 → 46, 51 → 70,
75 → 94. Every figure below is `P(still on the board at my next turn)`.

**Turn 3 → 22.** Three players move by more than 5 points and none by 10. The top ten by
`cost_of_waiting` do not reorder.

| player | before | after |
|---|---|---|
| Drake London | 0.175 | **0.093** |
| Nico Collins | 0.449 | **0.371** |
| Trey McBride | 0.438 | **0.369** |
| Saquon Barkley | 0.314 | **0.267** |
| Omarion Hampton | 0.500 | **0.456** |
| Garrett Wilson | 0.898 | **0.943** |

**Turn 27 → 46.** Nineteen move by more than 5 points, nine by more than 10, and the top ten
reorders for the first time.

| player | before | after |
|---|---|---|
| Kyren Williams | 0.297 | **0.159** |
| Javonte Williams | 0.286 | **0.150** |
| Breece Hall | 0.272 | **0.136** |
| Tetairoa McMillan | 0.313 | **0.180** |
| Ladd McConkey | 0.345 | **0.213** |
| Zay Flowers | 0.242 | **0.117** |

**Turn 51 → 70.** Thirty-one move by more than 5 points, twelve by more than 10.

| player | before | after | rank by `cost_of_waiting` |
|---|---|---|---|
| Rome Odunze | 0.344 | **0.191** | 8 → 3 |
| Bucky Irving | 0.215 | **0.075** | 9 → 6 |
| Jameson Williams | 0.261 | **0.108** | 23 → 17 |
| David Montgomery | 0.315 | **0.158** | — |
| Luther Burden III | 0.317 | **0.161** | — |
| Emeka Egbuka | — | — | 2 → 1 |

**Turn 75 → 94.** Forty-nine move by more than 5 points, twenty-two by more than 10.

| player | before | after | rank by `cost_of_waiting` |
|---|---|---|---|
| Courtland Sutton | 0.341 | **0.196** | 9 → 4 |
| DK Metcalf | 0.316 | **0.168** | 19 → 12 |
| Jaylen Warren | 0.278 | **0.130** | 20 → 14 |
| Sam LaPorta | 0.280 | **0.130** | — |
| Tucker Kraft | 0.365 | **0.221** | — |
| Tony Pollard | 0.386 | **0.242** | — |

Every direction is the same one, and the size grows with the pick number, which is what a slope
correction of −0.084 has to look like. Round one is nearly untouched — sigma at pick 22 differs
by less than two picks — and by round seven a fifth of the players you might wait on have moved
by more than ten points of survival probability.

**The rank column is the part that changes a decision.** `cost_of_waiting` is `VOR × P(gone by
your next turn)`, so a falling availability lifts a player up the scarcity board. Rome Odunze,
Courtland Sutton, DK Metcalf, Jaylen Warren, Bucky Irving and Emeka Egbuka all rise; nobody
falls far, because the correction moves in one direction for everyone and only the size differs.

## What does not move

The `w = 0.5` mixed-room prior: `fit_espn_weight` still cannot be estimated, for the reason its
docstring gives. `MIN_SIGMA` is unchanged at 1.0 — what changed is that the shipped intercept is
no longer sitting on it. `evaluate.OPP_NOISE` stays 1.0, a scale over this base rather than an
absolute sigma, so the three readers still resolve to one dispersion.

## The room's scale, as a sensitivity (#49) — built, not yet run

The simulated room draws each opponent's perceived pick as `mu_pick + N(0, scale × sigma)`,
with `sigma` the fitted law above. `scale` is a **multiplier on that fitted sigma**, never an
absolute number of picks: 0.5 is a room following consensus twice as closely as the law says
a room does, 1.5 one following it half again as loosely, and 1.0 is the law itself — the only
value any published draft-gate figure was played at. `optimize.simulate_remaining_draft`'s own
comment says the ECR-fitted spread credits an ESPN room with more error than the premise
allows, that every point of it becomes edge for the greedy, and that the result should be read
as a sensitivity rather than a measurement. It had never been varied.

`hub.draft.backtest --noise-scales 0.5,1.0,1.5` now runs one gate per scale through
`experiment.run_gate` and prints one row each: the scale, the paired mean, the season-clustered
interval, the MDE and the verdict, with the four stamps a paired frame carries. The scale
reaches both the room the two arms are played in and the rollouts arm B evaluates inside it,
from one argument, so a row varies the knob and not the gap between what arm B believes about
the room and what the room is. `--ceiling` adds the foresight arm per scale.

**The real sweep has not been run.** It needs the four boards and roughly three times the
hours one gate run takes; what is committed is the runner, the option, and a synthetic test
that two scales give different paired means on one seed and the same scale reproduces
exactly. No table is published here until a run produces one, and the published draft-gate
figures in [gate-power.md](gate-power.md) are the `scale = 1.0` row of a table whose other
two rows do not yet exist.

`config_digest` moves from `281b7b7a` to `ab32cf62` and `fitted_digest` from `d5598b96` to
`3d6fc111`. That is [ADR-0006](adr/0006-fitted-constants-live-with-their-provenance.md) working:
a refit is supposed to move the model version. The prediction artifacts already committed under
`site/data/` keep their `281b7b7a` stamp, because each records the version that produced it.

## Reproduce

```bash
uv run python -m hub.draft.board --fit-noise      # two most recent drafts
```

The figures above come from all four drafts, which is the span the superseded constants claimed:

```python
from hub.draft.availability import historical_picks, noise_from_picks
from hub.fetch.espn import resolve_league_id

df = historical_picks(resolve_league_id(), [2022, 2023, 2024, 2025])
(a, b), said = noise_from_picks(df, draws=2000, seed=0)
```

It needs an ESPN session — `historical_picks` is network-bound, which is why
`noise_from_picks` takes a frame and returns its sentence rather than printing one. The 2026-09-07
run matched 732 of 792 picks against a rank scraped inside their own preseason
(189/204, 187/204, 178/192, 178/192), and 672 of those fell inside the 204-pick pool.
