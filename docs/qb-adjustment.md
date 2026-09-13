# The quarterback adjustment, validated before it is built

**Measured 2026-09-12** (#218). The game layer has no quarterback awareness: ratings pass the
live price through, and where the staleness field says there is none, nothing adjusts. #218
proposes a quarterback-adjusted team rating for exactly those games, consumed from a
published source rather than refitted, and made adoption conditional on one free
validation: 538 shipped both a base Elo win probability and a quarterback-adjusted one for
every game, so the Brier difference between the two columns *is* the value of the
adjustment, out of sample, with nothing this repo built.

## The rule, fixed before the number

Adopt only if the mean per-season Brier delta (base − quarterback-adjusted) over the last ten
regular seasons the file holds, 2013–2022, is positive with a 95% t interval (9 df) that
excludes zero. Ties score 0.5; playoff games excluded.

## The data

`nfl_elo.csv`, 17,379 games 1920–2022, `elo_prob1` and `qbelo_prob1` beside the scores. 538
is gone and `projects.fivethirtyeight.com` now serves an ABC News page, so the file is the
Wayback Machine's capture of `https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv` at
timestamp `20230301000000` (the last full season, 2022, included). Not committed — 5.9 MB
of someone else's data — and the fetch is three lines of `urllib` against that URL. Quarterback-adjusted probabilities exist from 1950; the ten-season
window is the one the rule names.

## The result

| season | games | base Brier | QB-adjusted | delta |
|---|---|---|---|---|
| 2013 | 256 | 0.2162 | 0.2091 | +0.0071 |
| 2014 | 256 | 0.2061 | 0.2091 | −0.0030 |
| 2015 | 256 | 0.2261 | 0.2216 | +0.0045 |
| 2016 | 256 | 0.2189 | 0.2092 | +0.0097 |
| 2017 | 256 | 0.2166 | 0.2100 | +0.0065 |
| 2018 | 256 | 0.2200 | 0.2122 | +0.0078 |
| 2019 | 256 | 0.2223 | 0.2169 | +0.0054 |
| 2020 | 256 | 0.2179 | 0.2076 | +0.0103 |
| 2021 | 272 | 0.2343 | 0.2306 | +0.0037 |
| 2022 | 271 | 0.2243 | 0.2189 | +0.0054 |

**Mean delta +0.0057, se 0.0012, 95% t CI [+0.0031, +0.0084], positive in 9 of 10 seasons.**
The rule says adopt. For scale: the base model's Brier is 0.2203 against a coin flip's 0.2500,
so the adjustment recovers about 19% of the distance the base model covers — 2.6% of its
error. Over all 73 seasons with the column it is +0.0019 and positive in 44, which is the
long-run figure and not the one the rule reads; the last decade is where the quarterback
matters most and where the data this repo would consume comes from.

## What it licenses, and what it does not

It licenses building #218 as specified: a quarterback-adjusted team rating **only where the
staleness field says there is no live price**, consumed from `greerreNFL/nfeloqb` (538's
method on nflfastR data, published twice weekly in season) through a fetch CLI with a
contract, a cache and last-good, relative to what the team rating already embeds and
decaying at 10% a game so a long-tenured backup is not double-counted *(the decay is the
source's own, not this repo's — restated under #268 below)*. A game with a live
price is unchanged: the best published quarterback-adjusted Elo is +0.01 MAE against the
closing line after fifteen years, and its own method blends 65% market.

It does not license ranking on it anywhere a price exists, and it does not measure *this*
repo's implementation — that is #218's last criterion, the change to the published survivor
and pickem numbers reported once.

## As built (2026-09-12, #218)

**Where it acts.** `hub.models.ratings.rated_games` is the one seam, read by the weekly fit
and by `hub.season.survivor.grid_from_schedule`. A game is adjusted only where the staleness
field marks no live price: a snapshot quote whose run of unchanged polls began more than
`STALE_AFTER_DAYS` (7) before the moment asked, or the moving field, which nothing polls. The
threshold is a stated choice from #210's measurement — the current week's median run was 1.8
days, every week from 2 out the full 12.2 — and lives in `hub.models.quarterback` beside the
other two numbers, all three in the model digest (`config_digest` moved `e3d549ab` →
`08ceee28`). A game with a live price is unchanged, and `tests/unit/test_quarterback.py`
holds the row byte for byte.

**The construction.** Per team, from `hub.fetch.nfeloqb`'s state:

    points = 0.132 × (qb_value − embedded) × 0.9 ** tenure
    embedded = arrival_value − arrival_adj / 3.3

`qb_value` is the current starter's rolling value; `arrival_*` are read off the first row of
his current run of starts, so `embedded` is the team's rolling quarterback value when he
arrived — what the rating already priced in — recovered from the source's own adjustment on
that row. `tenure` counts the games he has played in the run; a starter named for the coming
game and yet to start it has none and gets the full gap. 0.132 points per value unit is
538's 3.3 Elo per unit over 25 Elo per point, declared rather than fitted; the recipe notes the
other published conversion (21.5) disagrees by 16%. The home spread moves by the home side's
points minus the away side's, only when both teams are in the state.

**What the row says.** `adjusted_by = "nfeloqb"` and `qb_adjustment` (points added to the
home spread), null on every row the rule did not reach; the version string gains `-qb` on
exactly those rows, so they file to their own partition and the track record can tell them
from the passthrough. *(Since #284 the model string moves the same way: `market_baseline-qb`
on an adjusted row, `market_baseline` where the betting market set the number, and the run
line says how many games were adjusted rather than `passthrough` on every run.)*

**What the run says.** Once per run, `hub.models.ratings --fit` and `hub.season.survivor`
print `quarterback adjustment: N of M priced games touched (no live price); mean |change| X
points, Y home win probability`. On the fixture, one game of two: 4.98 points, 0.141 win probability
*(superseded under #268 below: 4.80 points, 0.137)*. **The live
figure is not recorded here** — no pull of nfeloqb has been made from this repo — and it is
the maintainer's to run: `uv run python -m hub.fetch.nfeloqb --refresh`, then the two CLIs
above against the store that holds the archive.

## Restated 2026-09-12 (#268) — the construction is the source's, and the estimator above is withdrawn

The construction under **As built** is superseded. It is kept as published, per
[method.md](method.md) rule 13; what follows is what replaced it and why.

**The defect.** The shipped estimator subtracted an *arrival-time* baseline from a *current*
value and decayed the difference by tenure. The difference carries the starter's own value
drift since he arrived — a quantity with nothing to do with the gap between this quarterback
and what the team rating already embeds, which is what the module exists to price. The
source's own `qbN_adj` is already that gap: already relative, already in Elo, already decayed
by nfelo's own update rule. Rebuilding it from the arrival row was re-deriving a published
column, wrongly.

**Measured on the live cache, 2026-09-12**, reproduced independently on the 32 live teams:
mean absolute error **0.4** spread points, worst **3.6**, four sign flips. The worst case is
not a backup — it is Baltimore, an established starter with no quarterback change, at
**+4.018** where the source says **+0.404**: roughly eleven points of home win probability,
on a row the module selected precisely because the betting market had not moved. The error
is exactly zero when tenure is zero, by algebra: when the arrival row *is* the latest row the
formula collapses to the right one. Every fixture in the unit tests set that condition, so the
suite was green and the estimator had never been exercised on the input it exists to handle —
[method.md](method.md) rule 15, written on this incident.

**The construction now.** Per team, from the latest row of `hub.fetch.nfeloqb`'s state:

    points = qb_adj / 25

Nothing else enters it. `qb_adj / 25` over the cached file reproduces the published 538 and
nfelo magnitudes to the digit (n = 5,554: min −9.70, p1 −5.37, median +0.08, p95 +1.19,
max +2.88). `POINTS_PER_VALUE`, `DECAY_PER_GAME`, `ELO_PER_VALUE`, `arrival_value`,
`arrival_adj` are gone from the estimator and the state; `tenure` stays in the state as a
reported field the estimator does not read. `config_digest` moved `08ceee28` → `fd28e5e5`.

**What moves.** Every published prediction and survivor pick that carried the adjustment —
every row with `adjusted_by = "nfeloqb"` — was rated from the withdrawn formula and is
restated here, once: on the live teams the correction is under half a point for most and up
to 3.6 for the worst, in either direction. The fixture figure the run line reports moves from
4.98 points / 0.141 win probability to **4.80 / 0.137** — small on the fixture because its
starters' values barely drifted, which is exactly the property that hid the defect. The
week-six double-counting argument the module docstring made is withdrawn with the formula: it
is true against a rating that updates and false against the frozen rows this module reaches.

Every state fixture in `tests/unit/test_quarterback.py` now carries a tenure above zero and a
starter whose value drifted, and `test_points_per_team_off_the_captured_file_are_the_sources_adjustment_over_25`
holds all 32 teams of the frozen live capture to the source's column. **Not re-run from this
repo:** the per-team check on the live cache itself, which is the maintainer's file; the
capture is the last 300 rows of it.

**What the first live pull must confirm** is listed in `hub.fetch.nfeloqb`'s docstring: the
URL, the column names, that the coming week is listed with expected starters and null scores,
and the team abbreviations. `NFELOQB` says `verified_against_live=False` until then.

**One limit that is #210's rather than this ticket's.** The staleness field accrues only where
the snapshot archive persists. The Actions runner starts each run with an empty store and
takes one fresh snapshot, so there every quote is seconds old and reads as live, and the
adjustment reaches the published numbers through the maintainer's own runs against the local
archive, or once the runner keeps one. #210's second criterion — `price_source` distinguishing
live from stale in the coalesce — is still open, and this change does not decide it.

## Restated 2026-09-12 (#281) — the row arrives labelled, and the cut is declared once

#210's second criterion is now met, and it moves where the rule above is applied. The
coalesce in `hub.schedule.priced_games` used to rank any snapshot above the moving field, so
the quote frozen for twelve days that #210 measured outranked a field upstream still
refreshes. `price_source` is now three-way: **`live`** (a snapshot whose run of unchanged
polls began within `STALE_AFTER_DAYS` of the moment asked), **`schedule`** (the moving
field — no snapshot, or a snapshot gone stale), and **`stale`** (a stale snapshot with no
moving field behind it: used, because dropping the game would shrink the season, and said
so). A stale snapshot that the moving field beats keeps its `snapshot_spread` and staleness
columns on the row, and `priced_at` is null, since no snapshot priced it.

**The cut is declared once.** `quarterback.live_price(at)` is the expression over
`unmoved_since`, applied in `priced_games` and nowhere else; `quarterback.apply` no longer
takes the moment and reads the label — a row is adjusted exactly when `price_source` is
`stale` or `schedule`. The two consumers of the cut cannot disagree about a row, which is what
#281 asked for, and what #251 settles about the cut itself (the poll age rather than the
quote's last move; what a change in book set does to the run) changes in that one
expression. `STALE_AFTER_DAYS` stays in this module and in the digest: `config_digest`
and `fitted_digest` are unmoved (`a1e669b9`, `9be7844c`).

What this changes on a run: a game both sources price and whose snapshot is stale is now
priced from the moving field — the same number where the two agree, which
`tests/golden/test_line_agreement.py` holds — and its version string reads `-schedule-qb`
where it read `-snapshot-qb`. The ratings run line counts the three: `N from a live
snapshot, M from a stale one, K from the moving field`. Weeks published before #281 carry
`snapshot`, which `schedule.PROVENANCE` still classifies so the publisher can read them
back. The survivor grid rates from the same seam and so reads the three-way label; carrying
it onto the grid's own rows is the survivor lane's.
