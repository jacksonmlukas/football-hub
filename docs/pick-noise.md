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

~~The superseded interval and the restated one do not overlap, so the censored tail's effect on
the slope is resolvable at two standard errors clustered on the draft.~~ **Withdrawn 2026-09-13
(#287)**, see the section below. Below 168 the slope is stable across cuts and the intervals
overlap; the cut is where the data stops being two-sided, not where the slope is prettiest —
and since #287 it is stated as an assumption with its sensitivity beside it.

## Restated 2026-09-13: what four clusters can say, and the ceiling as an assumption (#287)

Two corrections to the section above, from audit III's finding Q16.

**The "resolvably different" claim is withdrawn.** The 204 fit and the 168 fit are nested
subsets of the same four drafts — 612 of the 672 rows, 91%, are in both — so their bootstrap
resamples move together, and non-overlap of the two intervals is not evidence of a difference
at any level. It was read above as the censored tail's effect being resolvable at two standard
errors. It is not; what the signed-deviation table shows is a *signature* in the last two bins,
and the ceiling is the cut before it. That is a stated assumption, not a finding the interval
established.

**The interval states its resample count.** Four drafts resampled with replacement give
4⁴ = 256 ordered resamples and 35 distinct multisets, so the 2,000 draws in
`noise_from_picks` revisit at most 256 of them and the 2.5th and 97.5th percentiles are those
of a small discrete set. The fitter's sentence now says so on every run, beside the interval
and beside the statement that it cannot resolve a difference between nested subsets.
The published `PICK_NOISE_SLOPE_CI` [0.143, 0.156] is unchanged; what changed is what it is
read as — the band four observations of this league's drafting support, no more.

**The ceiling is swept, not asserted.** `scripts/fit_pick_noise.py --ceilings 120,144,168,192,216`
runs `sweep_ceilings` and prints one row per cut. It needs an ESPN session; the 2026-09-13
lane had none, so the cells below that were not run are marked rather than filled. The 120 and
168 rows are the 2026-09-11 sweep's; 216 collapses to the 204-pick pool (`min(pool, ceiling)`)
and is that row.

| fit ceiling | n | `a` | `b` | 95% CI | sigma at pick 100 |
|---|---|---|---|---|---|
| 120 | 460 | 2.33 | 0.156 | [0.138, 0.174] | 17.9 |
| 144 | *not established* — the 2026-09-11 sweep ran 145: 545, 2.83, 0.143, [0.116, 0.170], 17.1 | | | | |
| **168 (shipped)** | **612** | **2.51** | **0.150** | **[0.143, 0.156]** | **17.5** |
| 192 | *not established* | | | | |
| 216 (= the 204-pick pool) | 672 | 1.31 | 0.169 | [0.159, 0.179] | 18.2 |

Across the cells that exist the slope spans 0.143–0.169 and sigma at pick 100 spans 17.1–18.2,
about one pick. Whether 192 sits on a cliff between 168 and the pool is the cell the sweep is
for; if it does, that is a separate decision and not a number this restatement moves.
`PICK_NOISE_INTERCEPT`, `PICK_NOISE_SLOPE` and `PICK_NOISE_SLOPE_CI` ship unchanged.

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

## The room's scale, as a sensitivity (#49) — built; run 2026-10-08, see below

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

### The table, run 2026-10-08 (#323; the paragraph above is kept as the record of before)

`uv run python -m hub.draft.backtest --seasons 2022,2023,2024,2025 --drafts 20 --seed 0 --ceiling
--noise-scales S --workers 4`, once per scale (S = 0.5, 1.0, 1.5; each run is the same boards and
the same seed, so a row is what the in-one-command sweep prints), on `main` at the merge of #405.
Points per team game, optimizer minus market, paired over 80 drafts in 4 held-out seasons; the
interval is the percentile bootstrap clustered on the season, the MDE is season-clustered at 80%
power, and the ceiling is the perfect-foresight arm played in the same room.

| room scale | optimizer − market | 95% CI | MDE | ceiling | verdict |
|---|---|---|---|---|---|
| 0.5 (room follows consensus twice as closely) | −16.19 | [−21.72, −9.58] | +12.98 | +25.12 | REMOVE |
| **1.0 (the fitted law)** | **−11.02** | **[−14.84, −7.20]** | **+7.84** | **+21.90** | **REMOVE** |
| 1.5 (room follows it half again as loosely) | −15.97 | [−22.43, −9.76] | +13.83 | +16.56 | REMOVE |

> **Amended 2026-10-09 (#467; the table and paragraph below are kept as written).** The 2025
> season in every row above was run with absence priced at zero (a truncated 2024 stats cache entry).
> The 1.0 row re-run on the fixed cache is -13.49 [-18.36, -8.62], MDE 9.90, ceiling +21.90, REMOVE
> (4 of 4 resolved); see `docs/gate-power.md`, *Amended 2026-10-09 (#467)*. The 0.5 and 1.5 rows
> were not re-run.

**What it says.** The 1.0 row reproduces the figure `docs/gate-power.md` restated on 2026-10-06
(−11.02 [−14.84, −7.20], MDE 7.84, ceiling +21.90), to the digit. The optimizer loses to the market
at every scale and every interval excludes zero on the losing side, so the verdict does not depend
on the room's noise: the sensitivity the code comment on `simulate_remaining_draft` asked for
changes no verdict. The mean is *not* monotone in the scale (the fitted law is the best of the
three, not the worst or a bound), so the table does not say "less noise, less loss"; each
off-1.0 row is a worse room for the greedy than the one it was fitted in, in both directions. The
MDEs at 0.5 and 1.5 (13.0, 13.8) are larger than at 1.0 (the paired differences are wider there); the ceiling shrinks as the room loosens (25.1, 21.9, 16.6), which is a reading and not a tested
mechanism. Three scales on one axis, one seed:
the table bounds the knob's effect at the grid points, it does not interpolate between them.

**The grid was fixed before the run (rule 1).** The three scales, 0.5 / 1.0 / 1.5, are
`backtest.NOISE_SCALES` (a `not_an_input` declaration, "the axis the pick-noise sensitivity
sweeps") and are the example in the #49 paragraph above (`--noise-scales 0.5,1.0,1.5`), both
written when the runner was built and before any row existed. They were not chosen after
seeing a result, and no other scale was run.

Stamps, the same on all three rows: `cfg_digest` `96fee74f`, `board_digest` `fa974b0d`,
`data_digest` printed as `unpinned` by the runner. The commit stamp `7aee8823` is this branch's
HEAD when the runs started; the code that ran is `main` at the #405 merge (`15cad00`) plus
#323's changes, none of which touch the draft gate. No constant moved, so `config_digest` is
unchanged by this run.

**From the #49 refit (earlier history, not this run's stamps; text kept).**
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
