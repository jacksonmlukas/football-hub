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
decaying at 10% a game so a long-tenured backup is not double-counted. A game with a live
price is unchanged: the best published quarterback-adjusted Elo is +0.01 MAE against the
closing line after fifteen years, and its own method blends 65% market.

It does not license ranking on it anywhere a price exists, and it does not measure *this*
repo's implementation — that is #218's last criterion, the change to the published survivor
and pickem numbers reported once.
