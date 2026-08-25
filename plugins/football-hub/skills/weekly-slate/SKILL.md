---
name: weekly-slate
description: Use when running the weekly in-season refresh — fetching new data, refitting ratings, publishing the dashboard, or setting lineups, pickem, and survivor. Triggers on "run the slate", "week N refresh", "set my lineup", "survivor pick", "publish the dashboard".
---

# Weekly Slate Ritual

One command, run twice a week. Tuesday for the refit, Sunday morning for the lock.

```
make slate WEEK=<n>
```

## What it does, in order

1. `hub.fetch.nflverse --refresh` pulls new play-by-play and ff_opportunity. No quota cost.
2. `hub.fetch.cfbd --week N` uses bulk week endpoints only, roughly 5-8 calls. Never loops.
3. `hub.fetch.odds --snapshot` takes one pull, one market, one region. Watch
   `x-requests-remaining` in the response header.
4. `hub.models.ratings --fit` refits state-space ratings on data through week N-1.
5. `hub.models.conformal --recalibrate` updates the rolling calibration window.
6. `hub.publish` writes `site/data/*.json`, commits, and Pages redeploys.

## Cadence

| Day | Action |
|---|---|
| Tue | Full refit. Research and refactors belong here, not later in the week. |
| Wed | Waivers, using `xfp` trend rather than last week's points. |
| Sun AM | Odds snapshot, lineup lock, survivor and pickem submit. Nothing else. |
| Sun PM | Watch only. (`make live` was removed 2026-08-24: it invoked a CLI that does not exist. `hub.fetch.espn.poll()` is real; its entry point is roadmap -- see docs/gaps.md.) |

**Reserve Sunday's token budget.** The weekly cap bites before the 5-hour window does. If the
meter is past 70% on Wednesday, stop building and coast to lineup day.

## The locked-number rule

Model outputs freeze at the Sunday snapshot. Live scores and win probability move all afternoon,
but they are information rather than signal, because the market half of the hybrid stopped
updating at lock. Never revise a decision on in-game movement the model never saw.

## Degraded mode

If a fetch fails, the pipeline serves last-good state from `data/processed/` and marks the
dashboard panel stale. That is correct behavior. Do not fix it mid-slate. Open an issue and move
on. A stale number you know is stale beats a fresh number you cannot trust.

## When not to use

Draft week goes through `draft-day-ops`. Model changes go through `model-eval` first.
