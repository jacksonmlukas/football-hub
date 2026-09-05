# Quota state

What this repo has spent against two metered third-party accounts:

- `odds.json` — the last balance The Odds API reported in `x-requests-remaining`, and when.
  `hub.fetch.odds` refuses the next pull below `CREDIT_FLOOR` on this number.
- `cfbd-quota.json` — CFBD calls made, by billing month, against the 1,000-a-month free tier.

**Committed on purpose.** Both lived under `data/raw/`, which `.gitignore` excludes as
redistributed third-party data — correct for a cached payload and fatal for a counter. Every
scheduled run therefore started with no record of what had been spent, so the odds floor could
never refuse and the CFBD counter reset on every run. A guard that cannot fire reads as a guard.

Not an Actions cache, for the same reason: a cache can be evicted, and a cache miss resets a
counter silently — the same failure arriving by a different route.

Nothing here is third-party data. These are counts and a balance about this repo's own
accounts; `tests/contracts/test_quota_state_survives_a_runner.py` holds that line.
