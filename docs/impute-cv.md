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
| rookies, vs realised PPG | 0.186 | 0.423 | 0.379 | 0.378 | **0.395** (se 0.031 over players) |
| rookies, vs the season's own xFP/game | 0.164 | 0.359 | 0.288 | 0.319 | **0.324** (se 0.026) |
| median residual, vs xFP | −0.11 | −0.08 | −0.08 | +0.21 | −0.07 |

The like-for-like row (vs xFP) sits **0.064 above the shipped pooled value, 2.5 standard errors
over players**; against realised points, 0.135 above at 4.4 se. RB and WR carry the difference;
QB and TE are eight and six players and say nothing on their own. The curve also sits low for
rookies — a median residual of −0.07 to −0.11 against the veterans' −0.03 — which is the
direction the disposition predicted: a rookie's rank carries no production information, so
the curve fitted on veterans is optimistic about him.

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
