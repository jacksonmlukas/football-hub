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
