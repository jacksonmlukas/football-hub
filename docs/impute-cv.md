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
on their own, which is why they ship at the pooled value. The curve also sits *high* for rookies
— a median residual of −0.07 to −0.11 against the veterans' −0.03 — which is the direction
the disposition predicted: a rookie's rank carries no production information, so the curve
fitted on veterans is optimistic about him. (Restated 2026-10-07, #315 and #462. Prior text:
"The curve also sits low for rookies". The residual is `realised / imputed − 1`, so a negative
median is a curve above the outcome. See "What moved".)

**`≥ 8 games` conditions on the outcome.** A rookie who lost the job or was hurt by
October has fewer than eight games and is dropped, and those are disproportionately the
seasons whose realised rate would sit far below the imputed one — the busts. Selecting on
survival removes part of the downside tail, so the CV measured here is biased **low**: a
lower bound on the rookie dispersion, the same direction the veteran number was. Three to
four rookies a board fall to the floor (the counts table); relaxing it re-admits per-game
rates on a handful of games, which is noise of a different kind, and the choice is stated
rather than tuned.

## What moved

> **Withdrawn 2026-10-07: item 3 of #315 did not land and `_impute_xfp` is unchanged.** The
> maintainer moved it to #462, so nothing below shipped: the shipped curve is still the rolling
> median followed by a running minimum, and `site/data/draft_board.json` is unaffected. The
> text and measurements are kept as the record of the trial; read "now" and "is now" as "in the
> trial". The one thing that stands is the sign correction in "The result" ("sits high").

**2026-10-07 (#315, item 3, trial, not shipped — see #462): the curve.** `_impute_xfp` smoothed with a rolling median and
then forced the curve non-increasing with a running minimum. That clamp lowers every point
that follows a rise and never raises one, so it sat at or below the smoothed data wherever
the data rose. It is now pool-adjacent-violators (`board._pava_decreasing`): the violating
neighbours are replaced by their mean, which is the least-squares non-increasing fit and
keeps the sum of the smoothed values (the residuals of the monotone fit average zero). The
rolling median is kept, since it is what stops a single outlier setting a rookie's projection;
only the clamp changed. Measured with `scripts/fit_impute_cv.py` on the same five boards,
before → after (median residual, `sd(realised / imputed − 1)` against the season's own
xFP/game):

| | QB | RB | WR | TE | pooled |
|---|---|---|---|---|---|
| median residual, rookies | −0.11 → −0.11 | −0.08 → −0.10 | −0.08 → −0.10 | +0.21 → +0.21 | **−0.07 → −0.08** |
| CV, rookies | 0.164 → 0.164 | 0.359 → 0.358 | 0.288 → 0.283 | 0.319 → 0.318 | 0.324 → 0.322 |
| season-clustered mean CV | | | | | 0.315 → 0.314 (se 0.037 → 0.035) |

**The median residual on rookies moved away from zero, not toward it, and the sentence above
that read it as the curve sitting low had the sign backwards.** The residual is
`realised / imputed − 1`, so a negative median is a curve that sits *high* for rookies —
which is the explanation this page already gave (a rookie's rank carries no production
information, so a curve fitted on veterans is optimistic about him). PAVA lifts the curve
where the clamp lowered it, so a curve that was already high got slightly higher for RB and
WR, and the rookie residual widened by about 0.01. The fix is to the veteran-side bias, and
there it works: blanking the observed veterans inside the top 200 in ten rank-interleaved
folds and re-imputing them (918 players over the same five boards), the mean residual is
+0.0057 under the clamp and +0.0017 under PAVA, the median is 0.000 under both, and the CV
0.217 → 0.216. The size of the whole effect is small next to the rookie gap, which is the
finding: the clamp's one-sided bias was real and tiny, and the rookie residual is a
different thing (rank carries no information about a rookie) that no monotone fit removes.
Pooling the raw values without the median (PAVA alone) was also measured: rookie pooled
median −0.066, CV 0.322, veteran CV 0.231 — a noisier curve for no better residual, so not
adopted.

`IMPUTE_CV` / `IMPUTE_CV_BY_POS` are **not refitted**: the pooled value moved 0.315 → 0.314 and
the per-position ones by at most 0.005 (WR 0.288 → 0.283), inside the se. A refit would move
`config_digest` for a change smaller than the noise in the estimate; it stays a separate
decision. On the published board (2026-09-04, 300 rows, 43 imputed) the imputed
`xfp_per_game` rises by a mean of +0.20 per game (median +0.15, max +0.60; 27 of 43 rows
change, none fall) — RB +0.24, WR +0.19, QB +0.23, TE +0.05, against imputed levels of 7 to 13.
That figure is the artifact's rows re-imputed from the 300 published rows, not a re-run of
`make draft`, so the published `site/data/draft_board.json` carries the old curve until the
next build.

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

> **Amended 2026-10-08 (#320, option B adopted by the maintainer; prior text above kept).**
> The paragraph above is the state from #294 to this date and is no longer true of
> `conf/holdout/`: **the hold-out now binds both constants.** Its reason, "a four-season refit
> has one cluster fewer than the decision was taken on", disqualified every leave-one-season-out
> fit by construction (a held-out fit always has one cluster fewer) and so could never rebind
> anything; it left the rookie measurement in-sample on the hold-out draft path, which was half
> of the width discrepancy `docs/gate-power.md`'s 2026-09-21 note chased (the 11.10 vs 10.40).
>
> `scripts/fit_impute_cv.py --exclude-season N --point-in-time --record` refits on the seasons
> **strictly before N** (method.md rule 2), not on every season but N: a 2023 replay is not
> handed the 2024-25 rookies. That is a different fit from the other four scripts' (they leave
> one season out and keep the later ones), and it is said on the command each set records. It
> costs clusters on the early seasons, which is why the table is read with its `k`:
>
> | replayed season | fit on | clusters | n | `IMPUTE_CV` | QB | RB | WR | TE |
> |---|---|---|---|---|---|---|---|---|
> | 2022 | 2021 | 1 | 15 | 0.187 | 0.187 | 0.187 | 0.187 | 0.187 |
> | 2023 | 2021-22 | 2 | 32 | 0.248 | 0.248 | 0.248 | 0.248 | 0.248 |
> | 2024 | 2021-23 | 3 | 51 | 0.287 | 0.287 | 0.287 | **0.273** | 0.287 |
> | 2025 | 2021-24 | 4 | 70 | 0.317 | 0.317 | **0.379** | **0.279** | 0.317 |
> | *shipped (2021-25)* | | *5* | *92* | *0.315* | *0.315* | *0.359* | *0.288* | *0.315* |
>
> `IMPUTE_CV` is the shipped statistic: the mean of the per-season pooled CVs against the
> season's own xFP per game (on one fit season, that season's pooled CV). A position keeps its
> own CV only with at least `MIN_POS_N` = 20 rookies and takes the refit pooled value
> otherwise -- the shipped split (RB 36 and WR 42 own, QB 8 and TE 6 pooled) as a threshold,
> fixed before the run. So the early seasons' by-position tables are the pooled value four
> times: that is what 15 and 32 rookies can say, not a missing measurement. **What this
> says about the constant:** 2021 was the quietest rookie class (0.187) and the number climbs
> as seasons are added (0.187, 0.310, 0.365, 0.405 per season, 2021-24), so the replayed
> 2022-23 seasons carry a rookie error well below the shipped 0.315 -- the in-sample value
> flattered no one there and *over*-stated the error the replayer could have known. The
> previously recorded leave-one-out pooled numbers (0.329, 0.314, 0.302, 0.327) are not
> comparable: they were over players, from a fit that included later seasons.
>
> **A correction to "What moved" above.** It says the backtest's season simulator does not read
> the `xfp_imputed` flag and that the exhibit is the only caller. The exhibit's `win_probability`
> *is* arm B's scorer in `hub.draft.backtest`, and the board it is handed carries the flag, so
> the draft gate reads `IMPUTE_CV_BY_POS` on every imputed row. The 2026-10-08 hold-out replay
> confirms it moved (docs/gate-power.md): the #197 frozen-board pin is the fixture that carries
> no flag, which is why it held. The weekly forward arm does not read it
> (docs/weekly-forward.md, 2026-10-08).

## Reproduce

```bash
uv run python scripts/fit_impute_cv.py
uv run python scripts/fit_impute_cv.py --exclude-season 2024    # print only, leave-one-out
uv run python scripts/fit_impute_cv.py --exclude-season 2024 --point-in-time --record  # #320
```

Reads the archive through `board_as_of`, `expected_points` and `nflverse.load`; nothing is
written unless `--record` is given with a hold-out. `--min-games` is 8 (half a season, the
rule `docs/weekly-spread.md` fitted under).
