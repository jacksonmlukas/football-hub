# IMPUTE_CV, measured on rookies

**Measured 2026-09-13 (#277).** `hub.models.predict.IMPUTE_CV` / `IMPUTE_CV_BY_POS` say how
wrong `board._impute_xfp`'s rank curve is, as a share of the projection it invents, and the
flag adds that in quadrature to the talent spread of every imputed player. The shipped values
(pooled 0.260) were measured on 2026-09-11 by blanking observed **veterans** inside the top 200
and re-imputing them from the rest — a scratchpad script that is not in the tree, on a
population the board never imputes. Audit III's Q11 named both defects. This is the committed
successor: `scripts/fit_impute_cv.py` (`hub.draft.impute_cv`), on the players the board
actually imputes.

## The population

Rookies are players with **no prior NFL season** who appear **inside the top 200 by consensus**
on that season's board, for the boards 2021–2025. Two readings of the disposition's wording,
each stated in the module docstring:

* *by consensus, not by ADP* — ESPN publishes ADP for the current season only, so a past board
  has none; consensus is the rank the curve reads;
* *"no prior NFL season" is "first weekly line in nflverse player stats is the board season"* —
  nflverse's `rookie_year` lives on a rosters table no loader here carries. The weekly stats are
  cached 2019 onward, so a 2021 board looks back two seasons. A player rostered but lineless for
  a whole season before his first line would be mis-read as a rookie; inside the top 200 that is
  rare.

2019 has no preseason consensus (the archive starts 2019-12-27) and 2020's as-of is not in the
cache, so **boards 2019 and 2020 are not established** and the run is 2021–2025.

Each board's counts, in the order the rules apply:

| board | top 200 | rookies | imputed | with a season line | ≥ 8 games |
|---|---|---|---|---|---|
| 2021 | 187 | 17 | 17 | 17 | 15 |
| 2022 | 185 | 21 | 21 | 21 | 17 |
| 2023 | 188 | 21 | 21 | 21 | 19 |
| 2024 | 180 | 21 | 21 | 21 | 19 |
| 2025 | 178 | 24 | 24 | 24 | 22 |

No rookie carried a prior xFP (zero join collisions); two imputed players inside the top 200
(2022, 2024) have no weekly line in any season and cannot be classified by the proxy.

## The result

`sd(realised / imputed − 1)`, the shipped constant's statistic. n = **92 rookies over 5
seasons** (the clusters). Two realised quantities, both from the season's own `ff_opportunity`
rows: PPR points per game (the disposition's) and the season's own xFP per game (what the
curve imputes, and what the veteran measurement compared against).

| | QB (8) | RB (36) | WR (42) | TE (6) | pooled (92) |
|---|---|---|---|---|---|
| **shipped** (veterans, blanked) | 0.217 | 0.334 | 0.223 | 0.220 | **0.260** |
| rookies, vs realised PPG | 0.186 | 0.423 | 0.379 | 0.378 | **0.395** |
| rookies, vs the season's own xFP/game | 0.164 | 0.359 | 0.288 | 0.319 | **0.324** |
| median residual, vs xFP | −0.11 | −0.08 | −0.08 | +0.21 | −0.07 |

**The interval is clustered on the season.** Rookies inside one season share a board, a curve
fitted on that year's veterans and one realisation of the year, so the 92 rows are not 92
independent observations — the error [gate-power.md](gate-power.md) is about. The pooled CV is
measured once per season and the estimate is the mean of the five under a t on 4 degrees of
freedom (t(0.975, 4) = 2.78), the form the gates use:

| | per season (2021 · 2022 · 2023 · 2024 · 2025) | mean | se | 95% | vs shipped 0.260 |
|---|---|---|---|---|---|
| vs the season's own xFP/game | 0.187 · 0.310 · 0.365 · 0.405 · 0.306 | **0.315** | 0.037 | [0.213, 0.416] | +1.5 t — **does not clear** |
| vs realised PPG | 0.279 · 0.311 · 0.505 · 0.511 · 0.309 | 0.383 | 0.051 | [0.241, 0.525] | +2.4 t — does not clear |

So the rookie number sits above the shipped one on every reading and in four of five seasons
on the like-for-like, and **at five clusters neither reading clears the two-sided 95% bar**.
The secondary, on the unit the shipped number quoted its own se on — over players — is
se 0.026 (vs xFP) and 0.031 (vs PPG), which would read as 2.5 and 4.4 se; that is the unit
that treats rows within a season as independent, and it is reported so the two can be read
side by side, not as the interval. RB and WR carry the difference; QB and TE are eight and
six players and say nothing on their own. The curve also sits low for rookies — a median
residual of −0.07 to −0.11 against the veterans' −0.03 — which is the direction the
disposition predicted: a rookie's rank carries no production information, so the curve fitted
on veterans is optimistic about him.

**`≥ 8 games` conditions on the outcome.** A rookie who lost the job or was hurt by
October has fewer than eight games and is dropped, and those are disproportionately the
seasons whose realised rate would sit far below the imputed one — the busts. Selecting on
survival removes part of the downside tail, so the CV measured here is biased **low**: a
lower bound on the rookie dispersion, the same direction the veteran number was. Three to
four rookies a board fall to the floor (the counts table); relaxing it re-admits per-game
rates on a handful of games, which is noise of a different kind, and the choice is stated
rather than tuned.

## What moves

Nothing, by this run. The disposition keeps `IMPUTE_CV` shipping meanwhile — the veteran
number is a lower bound on rookie dispersion, so the flag already under-states risk, and
withdrawing to `TALENT_CV_BY_POS` would under-state it more. Moving the constant to the
rookie number is a decision with its own digest move and before-and-after; the constant's
docstring names this measurement as its successor and carries the table above.

## Reproduce

```bash
uv run python scripts/fit_impute_cv.py
uv run python scripts/fit_impute_cv.py --exclude-season 2024    # hold-out, #294
```

Reads the archive through `board_as_of`, `expected_points` and `nflverse.load`; nothing is
written. `--min-games` is 8 (half a season, the rule `docs/weekly-spread.md` fitted under).
