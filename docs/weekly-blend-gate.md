# The market/Usage blend, gated once

**Run 2026-08-29**, under [ADR-0017](adr/0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md),
which fixed every rule and was committed *before* the run — the commit order is the record.

**2,000 roster-weeks over 160 rosters, four held-out seasons, 79 covered weeks.
Join failure 0.1% against a 2% floor.**

## The result

| gate | weekly − consensus | 95% CI | seasons won | verdict |
|---|---|---|---|---|
| **frozen** *(primary)* | **+0.711** | **[+0.313, +1.129]** | 3/4 | **SHOW, NEVER RANK ON** |
| churn | −2.006 | [−2.761, −1.229] | 1/4 | SHOW, NEVER RANK ON |

**The interval excludes zero and favours the blend, for the first time in the programme.** It
still does not adopt, because the bar has two halves and the every-season half fails: it lost
2025 by −0.497.

## The thing that matters most is the trend

| season | frozen gain | |
|---|---|---|
| **2022** | **+1.894** | **never scored before — the one out-of-sample season** |
| 2023 | +0.961 | |
| 2024 | +0.487 | |
| **2025** | **−0.497** | the season closest to the one being drafted |

**The edge decays monotonically and is negative in the most recent season.** Whatever this is,
it was worth two points a team-week in 2022 and is worth nothing now. For a 2026 draft that is
the decision-relevant fact, and it points the same way as the verdict.

No mechanism is asserted for the decay — [signal-screens.md](signal-screens.md) point 6, which
this repo got wrong once already. The obvious candidate is that weekly consensus improved; it
is untested.

## Against the pre-registration

| | pre-stated | measured | |
|---|---|---|---|
| frozen | positive but small, **interval contains zero** | +0.711, **interval excludes zero** | direction right, precision wrong |
| churn | between −1.5 and −3 | **−2.006** | right |
| 2022 vs the rest | in line | +1.894, agrees with 2023–24 | right; 2025 is the dissenter |

The interval tightened partly because rosters were doubled to 40 — pre-registered for exactly
that reason, before the numbers.

**No tripwire fired.** Join failure 0.1%; 2022 did not disagree with the other three as a
block; the churn gate did not come out positive.

## What this settles

ADR-0017 fixed the consequence of each outcome before the run:

> **SHOW**: ADR-0016 stands, the blend is printed beside consensus, and **the rescue attempts
> end.** Five variants is where a negotiation with a result becomes a search for one.

So: [ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) stands. Nothing
sets a lineup. **The weekly programme is closed.**

What it produced, in order: four screened signals, a component model that beats the flat
projection by +0.074 MAE at 5.9 se, a lineup gate it loses, a winner's curse diagnosed at the
waiver pool, and a market/Usage blend that beats a free public ranking by +0.711 points a
team-week on average and is going backwards. Fifteen measurements, and consensus has won
fourteen and a half.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run --seasons 2021,2022,2023,2024,2025 \
  --drafts 40 --shrink mae-market
uv run python -m hub.season.weekly_gate --run --seasons 2021,2022,2023,2024,2025 \
  --drafts 40 --churn --shrink mae-market
```
