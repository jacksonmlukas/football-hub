# MARGIN_SD: the number that was asserted for decades

**Fitted 2026-08-24**, `hub/models/margin.py`. `MARGIN_SD` moves from **13.5 to 12.741**.

The accuracy gain is small and the provenance gain is the point.

## What it was

`hub/models/market.py` carried:

```python
# Standard deviation of the actual margin around the closing spread. Stable across decades
# of NFL results and ...
MARGIN_SD = 13.5
```

No fit, no interval, no write-up. It converts every closing spread into a win probability, so
it is the most load-bearing constant in the NFL path — and `hub.models.market` sits in
`config.FITTED_MODULES`, meaning the number was hashed into every model version *as though it
had been measured*. [ADR-0006](adr/0006-fitted-constants-live-with-their-provenance.md) draws
the line between a measurement and a choice. This one was registered on the right side of it
and living on the wrong side.

The data had been in the fetch layer the whole time. `schedules` carries `spread_line` and
`result` back to 1999.

## Sign convention, checked first

`result == home_score - away_score` for every completed game, verified rather than assumed, and
`spread_line` is home-relative too. So the residual is `result - spread_line` with no sign
juggling — and getting that backwards would have produced a plausible-looking systematic bias
rather than an error.

The mean residual is **+0.096** over the full sample. On the 2022-24 slice it looked like
+0.70, which is 1.6 se and would have been easy to write up as home-field advantage. It is not
anything.

## What it is

| window | n | sd | se | 13.5 is |
|---|---|---|---|---|
| all, 1999-2025 | 7261 | 13.214 | 0.110 | +2.6 se |
| trailing 15 | 4096 | 12.982 | 0.143 | +3.6 se |
| trailing 10 | 3018 | 12.741 | 0.164 | +4.6 se |
| trailing 5 | 1424 | 12.648 | 0.237 | +3.6 se |

**It is not stable, which is what the assertion got wrong.** Per-season sd runs from 11.5 to
14.4, and the trend is **-0.037 a year at -2.4 se** — the market has got sharper. "Stable
across decades" was the one claim the data most directly contradicts.

## The gate, fixed before any log-loss was computed

Three candidates — the incumbent 13.5, an expanding all-history fit, and a trailing ten-season
fit — scored by **held-out log-loss**, walking forward one season at a time and fitting only on
strictly earlier seasons. A candidate had to *beat* the incumbent. Ties went to the incumbent,
because replacing a constant hashed into every model version for no measured gain is churn.

26 held-out seasons, 1999-2025:

| candidate | mean held-out log-loss | gain over 13.5 | significance | seasons better |
|---|---|---|---|---|
| incumbent 13.5 | 0.61363 | — | — | — |
| all-history | 0.61349 | +0.000143 | 2.81 se | 19/26 |
| **trailing 10** | **0.61343** | **+0.000199** | 2.06 se | 15/26 |

The rule selects on mean log-loss, so **trailing 10 wins and 12.741 is adopted**.

## Being honest about the size of this

**0.0002 log-loss is tiny.** Across a 285-game season it is 0.057 nats — about five games'
worth of the difference between calling a 3-point favourite at 58.8% and at 59.5%. The
dispersion was roughly 6% too wide and log-loss barely noticed.

So this is not an accuracy win worth celebrating. It is a **provenance** win: a number that was
hashed as fitted is now fitted, carries an interval, has a window, and has a test guarding it
against a silent revert. `docs/improvements.md` originally framed this as a correction that
mattered predictively; that framing was too strong and is corrected here.

## Two things the rule did not handle well

Recorded rather than fixed after the fact, because retuning a gate once you have seen its
output is the failure this repo has already caught twice.

1. **It did not require significance.** "Beats the incumbent" fired on a 2.06 se gain. A future
   gate of this shape should require a margin, not just a sign.
2. **It selected on mean, not consistency.** All-history was better in 19 of 26 seasons at
   2.81 se; trailing-10 was better in 15 of 26 at 2.06 se. The rule picked the larger mean
   gain, which is the noisier of the two.

> **Brought to the house rule 2026-09-12 (#285).** Both items above were the pre-registration
> of this change. `verdict` now reads `experiment.gate` over a season-clustered bootstrap
> (ADR-0019): a candidate is adopted only if it beats the incumbent in **every** held-out
> season and the interval on its mean gain excludes zero. Under that rule the 2026-08-24
> adoption would not have fired -- 15 of 26 seasons fails the every-season half, and so
> would all-history's 19 of 26. **12.741 stands anyway**: it is the live incumbent, the rule
> gates the next change to it rather than re-litigating this one, and a re-run today scores
> the challengers against 12.741, not 13.5. Whether the constant should return to 13.5 on the
> strength of a rule it never faced is a human decision, not one this change makes. No
> published figure moves: `MARGIN_SD`, `hub.models.market` and `config_digest` are untouched.

**And a trailing window goes stale.** All-history would not have. Re-run after each season:

```bash
uv run python -m hub.models.margin --fit
```

## Reproduce

```bash
uv run python -m hub.models.margin --fit
```

Everything in `hub/models/margin.py` except `main()` is pure and offline; 23 tests in
`tests/unit/test_margin.py` were written before the walk-forward was run.

---

# The shape: mass on the key numbers, measured 2026-09-11

Issue #185. The width above was settled on 2026-08-24; this is the other half of the same
constant — not how wide the margin distribution is, but whether it is smooth. Football margins
are not: games end on 3 and 7 far more often than any bell curve puts them there, and the
single Gaussian is the only measured input to the survivor and pool pipeline, which multiplies
two dozen of these probabilities together. **The Gaussian stays**, and this section says why in
the order `docs/method.md` rule 8 requires — the ceiling first.

## The ceiling, before anything was built

Every consumer of this distribution reads one number from it: `P(margin > 0 | spread)`.
`season/survivor.py` and `MarketBaseline` both price as `normal_cdf(spread / MARGIN_SD)` and
neither reads a cover probability, a push, or any quantile but the median. So a *perfect*
margin distribution is worth exactly what a perfect `P(win | spread)` is worth, and that can be
bounded without building anything: replace the Gaussian's price with the favourite's realised
win rate in one-point buckets of |spread|, in sample, and score both.

| trailing 10 seasons (2017–2026), 2,488 games | log-loss | gain per game |
|---|---|---|
| Gaussian, `Phi(spread / 12.741)` | 0.60774 | — |
| perfect `P(win \| spread)` by one-point bucket, in sample | 0.60214 | **0.00560** |

**0.0056 a game, 1.6 nats a season, in sample.** That is 28 times the gain the width refit
above delivered, and it is an upper bound flattered by construction. It is also where survivor
lives: 7-point-or-better favourites are priced at 0.774 and win 0.806, and a plan of eighteen
such picks survives at 0.0099 under the price and 0.0206 at the realised rate — a factor of
two, in the safe direction. There was headroom worth measuring against.

## The histogram, from our own sample

Reproduced rather than cited, over the spine's own window, with the spine centred on each
game's closing spread:

| \|margin\| | empirical share | spine's share | excess |
|---|---|---|---|
| 3 | **0.1503** | 0.0546 | **+1.751** |
| 6 | 0.0663 | 0.0512 | +0.296 |
| 7 | 0.0828 | 0.0496 | +0.671 |
| 10 | 0.0490 | 0.0437 | +0.121 |
| 14 | 0.0518 | 0.0346 | +0.499 |

**15.0% of games end on exactly 3**, 2.75 times what the Gaussian says, which reproduces the
~14.9% the ticket asked for. The lumpiness is real and it is large. The values are
`FITTED_KEY_EXCESS` in `hub/models/margin.py`.

## The model, and the pre-registered rule

The shape #185's amendment named: keep `Phi(spread / 12.741)` as the spine, multiply the
spine's mass in `(k − ½, k + ½]` by `1 + excess[k]` at each key number `k` and at `−k`,
renormalise. `lumpy_home_win_prob` is that in closed form — every term is a Gaussian cell —
and with every excess at zero it is the plain price exactly, which a test holds.

The rule, fixed before the walk-forward ran: the lumpy price must beat the Gaussian on **mean
held-out log-loss**, bumps refitted each season on the trailing ten seasons of strictly
earlier data, spine unchanged in both arms. A tie or a loss keeps the Gaussian.

## What it found

27 held-out seasons, 2000–2026:

| arm | mean held-out log-loss gain | se across seasons | seasons better |
|---|---|---|---|
| lumpy over Gaussian | **−0.00094** | 0.00032 | 7 of 27 |

**KEEP the Gaussian.** The lumpy price is worse, at −2.9 standard errors, in 20 of 27 seasons.
One of the seven it wins is 2026 at two games; without it the mean is −0.0011 and 6 of 26. It
closes none of the ceiling — it moves away from it.

> **Re-derived under the house rule 2026-09-12 (#285).** `shape_verdict` adopted on
> `mean > 0`, the weakness the width section above had already named and this section then
> repeated. It now reads `experiment.gate` the way every other gate in the tree does: every
> held-out season and an interval excluding zero. The recorded verdict is unchanged and needs
> no re-run to say so -- 7 of 27 fails the every-season half by itself, and a mean of −0.00094
> at 0.00032 se fails the other. `test_the_recorded_shape_verdict_is_unchanged_under_the_house_rule`
> holds that from the recorded constants. Note that with the season as the cluster, a sign
> that holds in every season implies an interval above zero (every resample of positive
> per-season gains is positive), so the every-season half is the binding one here as it is in
> every gate that clusters on the season.

Calibration by spread bucket, **held out** (the lumpy price for each season fitted on earlier
seasons only), both sides of every game, seasons 2017–2026:

| home spread | sides | Gaussian | lumpy | actual |
|---|---|---|---|---|
| 0 to 3 | 572 | 0.557 | 0.553 | 0.526 |
| 3 to 6 | 955 | 0.617 | 0.609 | 0.617 |
| 6 to 9 | 583 | 0.708 | 0.695 | **0.750** |
| 9 to 14 | 289 | 0.803 | 0.788 | **0.858** |
| 14+ | 92 | 0.884 | 0.870 | 0.891 |
| **favourites of 7+** | **732** | **0.774** | **0.760** | **0.806** |

Season-long survival for a plan of eighteen such favourites, beside the current model as the
ticket asked: **Gaussian 0.0099, lumpy 0.0071, realised 0.0206.** The corrected model is lower,
as #185 pre-registered — and it is lower because it is *further from the truth*, not closer.

> **Labelled 2026-09-12 (#286).** Those three are the favourites row raised to the
> eighteenth power: an independence bound — one rate every week, weeks independent, off the
> 732 games in the row — and not a measurement of any plan. They carry no interval because
> the sampling error on the rate is the smaller of the figure's two errors and the other is
> the assumption. `hub.models.margin.survival_line` prints them under that label now; the
> comparison between the three, which is what the paragraph above reads off them, does not
> depend on the assumption, since all three make it.

## Why, and this is the finding

A game whose line sits on 3 or 7 does price differently under the lumpy distribution: a
3-point favourite moves from 0.593 to 0.586, a 7-point favourite from 0.709 to 0.695. That is
the direction the histogram forces and the **opposite** of the direction the data show.

The mechanism is arithmetic. The excess at 3 is nearly **symmetric about the spread**: a
favourite wins by exactly 3 at 2.9 times the spine's rate and loses by exactly 3 at 2.5 times
it. Every key number sits within 14 points of the centre, so the mass a symmetric bump adds
lands on *both* sides of zero, and for any favourite the losing side gains proportionally more
than the winning side already holds. Every bump pulls every favourite toward one half. Signed
bumps — fitted separately for a favourite winning by `k` and losing by `k` — do no better
(−0.00084, better in 4 of 27), because the asymmetry is small.

Where the miss in the calibration table actually lives is the **location**, not the shape: on
the same window the favourite-relative residual averages −1.56 points on pick'ems (n=569,
se 0.53) and +0.78 at 6 to 9 (n=583, se 0.54) and +0.91 at 9 to 14 (n=289, se 0.72), with the
dispersion flat across buckets at 12.3–13.1. Small favourites trail the number and mid-range
favourites beat it — the pick'em figure is the only one past two standard errors on its own,
and the pattern is the same one #176 graded in win probability. No distribution symmetric
about the spread can price that, and a key-number shape is symmetric about the spread by
construction. That is a different ticket — a spread-dependent centre is a claim about the
market, not about football scoring — and it is not decided here.

Key numbers matter for `P(margin > line)` — covers and pushes — which nothing in this repo
prices. If something ever does, the histogram above is where it starts.

## What moved

Nothing that prices a game. `MARGIN_SD` is unchanged, `hub.models.market` is unchanged, and
`config_digest` is unchanged at `2c7620f9`: `hub.models.margin` is not in `FITTED_MODULES`,
and the numbers added to it are outputs of a fit that tests guard, not inputs to a prediction.
`test_the_live_shape_is_the_one_the_record_supports` holds that `survivor` and `market` read
no lumpy price while the recorded gain is negative beyond two standard errors.

## Reproduce

```bash
uv run python -m hub.models.margin --shape
```

Ceiling, histogram, walk-forward, calibration table, survival and verdict, in that order. The
eleven mutations that prove the tests are listed in the message of the commit that landed
this section.
