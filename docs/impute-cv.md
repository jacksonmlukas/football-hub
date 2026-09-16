# IMPUTE_CV, measured on rookies

**Measured 2026-09-13 (#277); adopted 2026-09-16 (#298).** `hub.models.predict.IMPUTE_CV` /
`IMPUTE_CV_BY_POS` say how wrong `board._impute_xfp`'s rank curve is, as a share of the
projection it invents, and the flag adds that in quadrature to the talent spread of every
imputed player. What ships is this measurement:

| | QB | RB | WR | TE | pooled |
|---|---|---|---|---|---|
| **`IMPUTE_CV_BY_POS` / `IMPUTE_CV`** | 0.315 | **0.359** | **0.288** | 0.315 | **0.315** |
| was, 2026-09-11 to 2026-09-16 | 0.217 | 0.334 | 0.223 | 0.220 | 0.260 |

RB and WR carry their own values; QB and TE are eight and six rookies, too thin to split, and
take the pooled value. The pooled value is the season-clustered estimate (the mean of the five
per-season pooled CVs, below), not the 0.324 over players.

The values that shipped before (pooled 0.260) were measured on 2026-09-11 by blanking observed
**veterans** inside the top 200 and re-imputing them from the rest — a scratchpad script that
is not in the tree, on a population the board never imputes. Audit III's Q11 named both
defects. `scripts/fit_impute_cv.py` (`hub.draft.impute_cv`) is the committed successor, on the
players the board actually imputes, and #298 moved the constants to it: a mis-measured constant
replaced with the right-population estimate, not a challenger against an incumbent, so the
house rule for challengers did not apply and the interval below was not asked to clear a bar.

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
| veterans, blanked (shipped before #298) | 0.217 | 0.334 | 0.223 | 0.220 | **0.260** |
| rookies, vs realised PPG | 0.186 | 0.423 | 0.379 | 0.378 | **0.395** |
| rookies, vs the season's own xFP/game | 0.164 | 0.359 | 0.288 | 0.319 | **0.324** |
| median residual, vs xFP | −0.11 | −0.08 | −0.08 | +0.21 | −0.07 |

**The interval is clustered on the season.** Rookies inside one season share a board, a curve
fitted on that year's veterans and one realisation of the year, so the 92 rows are not 92
independent observations — the error [gate-power.md](gate-power.md) is about. The pooled CV is
measured once per season and the estimate is the mean of the five under a t on 4 degrees of
freedom (t(0.975, 4) = 2.78), the form the gates use:

| | per season (2021 · 2022 · 2023 · 2024 · 2025) | mean | se | 95% | vs the veteran 0.260 |
|---|---|---|---|---|---|
| vs the season's own xFP/game | 0.187 · 0.310 · 0.365 · 0.405 · 0.306 | **0.315** | 0.037 | [0.213, 0.416] | +1.5 t — **does not clear** |
| vs realised PPG | 0.279 · 0.311 · 0.505 · 0.511 · 0.309 | 0.383 | 0.051 | [0.241, 0.525] | +2.4 t — does not clear |

So the rookie number sits above the veteran one on every reading and in four of five seasons
on the like-for-like, and **at five clusters neither reading clears the two-sided 95% bar**
— which is why the adoption (#298) is on population and not on this test. The secondary, on
the unit the veteran number quoted its own se on — over players — is se 0.026 (vs xFP) and
0.031 (vs PPG), which would read as 2.5 and 4.4 se; that is the unit that treats rows within a
season as independent, and it is reported so the two can be read side by side, not as the
interval. RB and WR carry the difference; QB and TE are eight and six players and say nothing
on their own, which is why they ship at the pooled value. The curve also sits low for rookies
— a median residual of −0.07 to −0.11 against the veterans' −0.03 — which is the direction
the disposition predicted: a rookie's rank carries no production information, so the curve
fitted on veterans is optimistic about him.

**`≥ 8 games` conditions on the outcome.** A rookie who lost the job or was hurt by
October has fewer than eight games and is dropped, and those are disproportionately the
seasons whose realised rate would sit far below the imputed one — the busts. Selecting on
survival removes part of the downside tail, so the CV measured here is biased **low**: a
lower bound on the rookie dispersion, the same direction the veteran number was. Three to
four rookies a board fall to the floor (the counts table); relaxing it re-admits per-game
rates on a handful of games, which is noise of a different kind, and the choice is stated
rather than tuned.

## What moved

**2026-09-16 (#298): the constants.** `IMPUTE_CV` 0.260 → 0.315 and `IMPUTE_CV_BY_POS`
to the table at the top; `config_digest` `b1f69382` → `8dbae43a`, `fitted_digest` `04c2d997`
→ `983eb30f`, recorded in `tests/unit/test_config.py`'s pin as a model change. What it
changes on a board is the talent spread of every imputed player,
`hypot(TALENT_CV_BY_POS, IMPUTE_CV_BY_POS)`: QB 0.295 → 0.373, RB 0.506 → 0.523, WR 0.382 →
0.423, TE 0.284 → 0.363; an observed player is unchanged. The one caller that passes the flag
is the championship-equity exhibit (`hub.exhibits.championship_equity`), so that is where a
rookie's title odds widen; the backtest's season simulator does not read the flag, and the
#197 frozen-board pin holds because that fixture carries no `xfp_imputed` column.

The 2026-09-13 run itself moved nothing — that was the measurement, and the move was a
decision with its own before-and-after, which is this section.

**The hold-out sets still record the constant as not refitted** (`conf/holdout/*.json`,
#294): the shipped value is this script's measurement on all five seasons, and a four-season
refit has one cluster fewer than the decision was taken on, with QB and TE pooled by the
decision rather than by the run. The reason carries the hold-out's own pooled number so the
replay's run line shows what it did not use.

## Reproduce

```bash
uv run python scripts/fit_impute_cv.py
uv run python scripts/fit_impute_cv.py --exclude-season 2024    # hold-out, #294
```

Reads the archive through `board_as_of`, `expected_points` and `nflverse.load`; nothing is
written unless `--record` is given with a hold-out. `--min-games` is 8 (half a season, the
rule `docs/weekly-spread.md` fitted under).
