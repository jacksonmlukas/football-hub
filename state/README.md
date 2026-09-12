# Quota state

What this repo has spent against two metered third-party accounts:

- `odds.json` — the last balance The Odds API reported in `x-requests-remaining`, and when.
  `hub.fetch.odds` refuses the next pull below `CREDIT_FLOOR` on this number.
- `cfbd-quota.json` — CFBD calls made, by billing month, against the 1,000-a-month free tier.
- `gate-width.json` — the last season-clustered interval width each backtest gate produced,
  keyed by gate. `hub.models.experiment.review_width` flags the next run's interval as
  needing review when it narrows against this by more than the interval could honestly
  narrow. The `draft` row is the shipped-constants run (−11.07 [−16.92, −5.22]) and the
  `weekly` row the post-#248 run (−0.825 [−1.152, −0.541]), both from 2026-09-12 and both
  written up in `docs/gate-power.md`.
- `interval_coverage.json` — what `hub.models.coverage --measure --survivor --write` last
  measured: whether the weekly player interval covers at nominal, and the survivor
  favourite's price. `hub.publish` carries it into `track_record.json` and the slate commits
  it before gating on it (#273).

**Committed on purpose.** Both lived under `data/raw/`, which `.gitignore` excludes as
redistributed third-party data — correct for a cached payload and fatal for a counter. Every
scheduled run therefore started with no record of what had been spent, so the odds floor could
never refuse and the CFBD counter reset on every run. A guard that cannot fire reads as a guard.

Not an Actions cache, for the same reason: a cache can be evicted, and a cache miss resets a
counter silently — the same failure arriving by a different route.

Nothing here is third-party data. These are counts and a balance about this repo's own
accounts; `tests/contracts/test_quota_state_survives_a_runner.py` holds that line.
