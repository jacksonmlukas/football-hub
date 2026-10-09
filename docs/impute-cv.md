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
median is a curve above the outcome. See "What moved".) **Restated 2026-10-09 (#327; the sentences
above kept):** the explanation that follows the dash is not supported like with like -- the
median residual of a rookie imputed from his own season's curve is +0.02, and the -0.07 was
the curve of the previous season sitting 11-12% high; see "The like-for-like measurement".

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

## Like with like: the pre-registration (#327, 2026-10-09; written before any number is taken)

The shipped 0.315 compares a **last-year curve** with **this year's outcome**: the as-of board
for season Y imputes a rookie from the rank-to-xFP curve of Y-1's veterans
(`board._impute_xfp`), and the residual is taken against the rookie's own Y xFP per game.
`IMPUTE_CV` is the *imputation error alone* (#322, the kind question, answered), so this
measurement imputes and realises on the same season. Fixed here, before the run:

1. **The like-for-like imputation.** Take the same as-of board. Keep each *observed* player's
   rank and replace his Y-1 xFP per game by his **season-Y** xFP per game (`expected_points(Y)`,
   the quantity the realised side reads); a veteran with no Y line, and every rookie and every
   other imputed row, goes null. Run the shipped `board._impute_xfp` unchanged on that frame and
   read the rookie's value off it. The curve is the Y-veterans' curve at the rookie's rank, the
   rookie sits outside the fit, and the only thing that differs from the shipped basis is the
   season the curve was drawn from.
2. **Same rookies on both bases.** The drift-carrying and like-for-like bases are computed on the
   identical rows (same rookies, same `MIN_GAMES`), so the four cells differ by basis and
   statistic only.
3. **The 2x2.** Per position and pooled, on each basis: `sd(r)` (ddof 1, the shipped statistic)
   and `RMS(r) = sqrt(mean(r^2))`, `r = realised / imputed - 1`, with the median and the mean
   (the bias) beside them. Each is reported over rookies and as the **mean of the per-season
   pooled values** (the season-clustered form #298 adopted: k seasons, t on k-1 df); the
   clustered pooled cell is the one that reads beside 0.315.
4. **The drift component.** `d = imputed_like_for_like / imputed_shipped - 1` on the same rookies
   is the year passing, as the curve sees it. Reported as sd, RMS, median and mean, and as
   `sqrt(max(0, RMS_drift_basis^2 - RMS_like_for_like^2))`, the quadrature remainder, labelled an
   approximation (the two terms are not independent).
5. **#322's cells.** Tripwire: the like-for-like **sd**, season-clustered, against 0.315 -- above
   it is a bug until shown otherwise (rule 9), and is said as such. Adoption reads the
   like-for-like **RMS**, season-clustered, with the median residual printed beside it. This
   ticket writes no constant.
6. **Population (R18's four defects).** First seasons are keyed on `player_id` from the
   **regular-season** weekly stats (`season_type == "REG"`), grouped by id so row order cannot
   matter. A board name resolves to an id through the stats' own normalised names; a name that
   resolves to more than one id is **ambiguous**, excluded and tallied (never silently the
   earliest of both). An imputed top-200 player whose first line comes in a *later* season than
   the board's is a **delayed debut**: tallied, not measured on this board. The realised row is
   picked by id in a deterministic order, not `unique(keep="first")`.
7. **The BuildReport.** The measurement takes `board_as_of`'s pair and refuses a board whose
   `BuildReport` shows an advisory stage that can run on a historical build (SoS, touchdown
   luck, durability, byes) not having run. The live-only checks and the ADP market stage do not
   apply to an as-of board by construction and are not required.
8. **Controls (rule 18).** The drift-carrying cell is the shipped measurement and must reproduce
   0.315 (clustered sd) / 0.324 (over rookies) on the shipped 92-rookie population, or the
   population change is named by the tallies that moved. Tests plant: a shifted curve (moves the
   drift component, not the like-for-like residual), a two-id name collision (excluded, not
   reclassified), a shuffled stats order (same answer), a delayed debut (tallied) and an absorbed
   stage (refused). There is no verdict here to be underpowered (rule 16): the number is reported
   with its season-clustered interval at k = 5.

> **Amended 2026-10-09, before any number was read (prior text above kept).** Two things the
> run found in the pre-registration itself. (a) *Item 7*: the 2021 board cannot build the
> durability stage -- it reads the 2020 preseason consensus and the archive begins 2020-10-16
> (`ContractViolation: ... a season before 2021 cannot be replayed`) -- so "refuse a board that
> did not build every stage" taken literally refuses 2021 and the five-season basis with it.
> `ARCHIVE_UNBUILDABLE = {2021: ("durability",)}` excuses that one stage on that one board, by
> name; the same absence on 2022-25 and any other stage on 2021 is refused (tested). The imputation
> reads rank and xFP and never the durability columns. (b) *Item 6*: a board name resolves to an
> id over the fantasy positions only, so a linebacker who shares a quarterback's name is not a
> collision.

## The like-for-like measurement (#327, run 2026-10-09; the constant is not changed)

`scripts/fit_impute_cv.py`, the five boards 2021-2025, `--min-games 8`, nothing held out; every
board's `BuildReport` read (2021 excused durability by name, 2022-25 built all four stages). Rows
in each cell are the **same 92 rookies**. Reproduction control: the drift-carrying cell is the
shipped measurement and reproduces it to the digit -- clustered sd **0.315**, over rookies
**0.324**, n = 92 (RB 36 / WR 42 / QB 8 / TE 6) -- so the population fixes moved no row in or
out of the 92. They did change two tallies: 2025 has 23 imputed rookies, not 24, because Travis
Hunter's nflverse position is CB and he has no fantasy-position line (he is in `no_line`; the
count of rookies who played eight games is unchanged at 22); and 2021 now shows one imputed player whose first
line comes a season late (a delayed debut). Ambiguous names: none on any board.

**The 2x2** -- pooled, relative error `realised / imputed - 1` against the season's own xFP per
game; the season-clustered mean of the five per-season values first, over all 92 rookies in
brackets:

| | drift-carrying (last season's curve) | like-for-like (own season's curve) |
|---|---|---|
| sd (the shipped statistic) | **0.315** [0.324] | **0.359** [0.370] |
| RMS relative error | 0.320 [0.333] | **0.356** [0.370] |
| median residual | -0.069 | **+0.020** |
| mean residual (bias) | -0.084 | +0.038 |

Intervals (t on 4 df, k = 5): like-for-like sd 95% [0.245, 0.473], +1.1 t from 0.315; like-for-like
RMS [0.243, 0.470], +1.0 t; drift-carrying RMS [0.222, 0.418]. **None of the four separates from
0.315.** Per season (2021 to 2025), like-for-like sd 0.197 · 0.397 · 0.401 · 0.425 · 0.376 and RMS
0.195 · 0.396 · 0.404 · 0.414 · 0.370.

By position, like-for-like (drift-carrying in brackets), over rookies:

| | n | sd | RMS | median | bias | season-clustered RMS |
|---|---|---|---|---|---|---|
| QB | 8 | 0.178 [0.164] | 0.167 [0.172] | -0.02 | -0.01 | 0.153 [-0.112, 0.417], k = 3 |
| RB | 36 | 0.446 [0.359] | 0.440 [0.376] | +0.02 | +0.03 | 0.397 [0.190, 0.605] |
| WR | 42 | 0.316 [0.288] | 0.312 [0.303] | -0.03 | +0.01 | 0.306 [0.251, 0.362] |
| TE | 6 | 0.306 [0.319] | 0.471 [0.409] | +0.33 | +0.38 | 0.480 [-1.341, 2.302], k = 2 |
| pooled | 92 | 0.370 [0.324] | 0.370 [0.333] | +0.02 | +0.04 | 0.356 [0.243, 0.470] |

RB's interval [0.190, 0.605] contains WR's [0.251, 0.362] and the pool's [0.243, 0.470]; no
position interval excludes the pooled mean, which is what #322's "positions at the pool unless
their season-clustered intervals separate" reads. QB and TE
are too thin to say anything (k = 3 and k = 2 seasons with two rookies).

**The drift component.** At the same rank, this season's curve over last season's
(`imputed_lfl / imputed - 1`), pooled: **mean -0.115, median -0.120, sd 0.100, RMS 0.152** (QB
-0.07, RB -0.14, WR -0.11, TE -0.07). The year passing lowers the curve by about 11-12% at every
rank, and varies across rookies by 10% around that. As an RMS added to the like-for-like error
(`sqrt(RMS_drift-carrying^2 - RMS_like-for-like^2)`) it is **0.000**: the drift-carrying RMS
(0.320) is *below* the like-for-like (0.356), so there is nothing left to subtract.

**The tripwire (#322) trips: like-for-like sd 0.359 is above 0.315.** "A bug until shown otherwise"
(rule 9), and it is shown otherwise, by arithmetic and not by argument. The relative error divides
by the imputed value, and `1 + r_drift = (1 + r_like) * q` row by row, `q = imputed_lfl /
imputed` (an identity, asserted in the run's own check). The curve of last season sits 11.5%
*above* this season's, `mean q = 0.885`, so every drift-carrying relative error is the like-for-like
one shrunk by about 0.885 -- sd included. A pure level shift would leave the drift-carrying sd at
`0.885 * 0.370 = 0.328`; it is 0.324 over rookies. **The sign argument behind the tripwire, that
removing drift can only lower the number, holds for an additive drift and fails for this
multiplicative one**: the drift is a curve sitting high, and a curve sitting high makes every
relative error smaller, not larger. The pre-registration did not state that argument's premise; it
is stated here, after the run, and a reading of #322 that treats 0.359 as a bug would be wrong
for this reason and for no other.

**Restatement (rule 13), 2026-10-09; the text above kept.** "The curve also sits *high* for
rookies ... the direction the disposition predicted: a rookie's rank carries no production
information, so the curve fitted on veterans is optimistic about him" (The result, above, and
#315/#462) attributes the median residual of -0.07 to the rookies. Like with like it is **+0.020**:
imputed from his own season's veterans a rookie lands on the curve, slightly above it (bias
+0.038), not below it. The -0.07 was the year passing (the drift's median is -0.120), not a
rookie effect. **What this does not touch:** the rookie error itself is not small -- 0.37 over
rookies, 0.36 clustered, larger than the veteran 0.260 on any basis -- and no published number,
fitted constant or digest moved (the shipped 0.315 is unchanged; `config_digest` is unchanged).

**For #322, not decided here.** Its adoption cell, like-for-like RMS clustered, is **0.356**
[0.243, 0.470], median residual **+0.020** beside it, against the shipped 0.315: higher, not lower,
which is the opposite of the direction the ticket expected and follows from the identity above.
The over-rookies RMS is 0.370. One thing the maintainer may want to weigh before that write
(the kind question is answered and this does not reopen it): the board a draft sees imputes a rookie
from last season's curve, so the projection the constant is applied to carries the 11.5% level
shift, and a relative dispersion measured like with like is applied to a base that is not the one
it was measured on. Whether that matters is a question about the consumer and not about this
measurement; it is noted on #327 for #322.

**Controls (rule 18).** Unit tests plant each condition and confirm the check fires: a curve
shifted 0.8x moves the drift component and leaves a rookie who earns exactly his own-season
curve at residual zero; the like-for-like value is unchanged when last season's curve is moved or
poisoned; a two-id name collision is excluded and tallied and a linebacker's name is not a
collision; a delayed debut is tallied; stats and realised rows in reverse order give the same
rows; each absorbed stage is refused and the 2021 durability excuse is exactly that; and a
level shift reproduces `q * sd`. Mutating the like-for-like fill to read last season's value, and
removing the ambiguity exclusion, each fail a test.

**The hold-out sets did not move.** `--exclude-season N --point-in-time` (print only) for
2022-2025 reproduces the table in the 2026-10-08 amendment to the digit, because the 92 rows are
the same rows and the refit reads the drift-carrying basis, as it did.

## Reproduce

```bash
uv run python scripts/fit_impute_cv.py
uv run python scripts/fit_impute_cv.py --exclude-season 2024    # print only, leave-one-out
uv run python scripts/fit_impute_cv.py --exclude-season 2024 --point-in-time --record  # #320
```

Reads the archive through `board_as_of`, `expected_points` and `nflverse.load`; nothing is
written unless `--record` is given with a hold-out. `--min-games` is 8 (half a season, the
rule `docs/weekly-spread.md` fitted under).
