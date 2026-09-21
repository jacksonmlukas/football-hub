# Quota state

What this repo has spent against two metered third-party accounts:

- `odds.json` — the last balance The Odds API reported in `x-requests-remaining`, and when.
  `hub.fetch.odds` refuses the next pull below `CREDIT_FLOOR` on this number, unless the
  reading is from an earlier month than the pull -- the quota resets monthly, so that one
  is unknown and one pull may re-read it (#264).
- `cfbd-quota.json` — CFBD calls made, by billing month, against the 1,000-a-month free tier.
- `gate-width.json` — **an append-only ledger, since #362 (S5)**, of every season-clustered
  interval width a backtest gate has produced. `hub.models.experiment.review_width` appends
  one entry per gate run, never overwrites one, keyed by `(gate, config_digest, data_digest,
  timestamp)`, with the run's own verdict recorded alongside the width and `requires_review`
  set when the interval narrowed against that gate's *most recent* previous entry by more
  than the interval could honestly narrow (#45's clustering argument predicts widening).

  **Why it moved off one record per gate.** The dict shape it replaced held exactly one row
  per gate name, overwritten on every run — so the file could never say how many times a gate
  had been run, and two runs whose numbers disagreed left only the second one behind. #362
  found this as the mechanism behind #B2's forking paths: not only across anchors and bases,
  but across re-runs nothing could count.

  **The two `requires_review: true` flags this ledger inherited, discharged.** Both predate
  #362 and are carried forward as the ledger's first entries, read off `git log --follow` on
  the old file (the pre-#362 shape recorded no `config_digest`/`data_digest`/timestamp, so
  those three fields on the pre-#362 entries below say so rather than guess):

  - **`draft`, commit `7e6625f` (2026-09-16).** The hold-out re-run of ADR-0009's figure:
    −13.21 [−18.76, −7.67], narrowed from the prior entry's 11.69 to 11.10 (ratio 0.95).
    **Published and reviewed**: ADR-0009 quotes both the before and after intervals side by
    side and its own text acknowledges the narrowing explicitly ("inside the noise of four
    ...") — the review this flag asks for was already done in prose on 2026-09-16, just never
    reflected by clearing the flag, because nothing before #362 could clear one without
    overwriting the record the review was about.
  - **`weekly`, commit `fb68c4e` (2026-09-12), never re-run since.** −0.825 [−1.152, −0.541].
    `fb68c4e`'s own message says `WIDTH_STATE` "was sitting untracked" before that commit --
    the file already existed locally, so this entry's `requires_review: true` most likely
    reflects a narrowing against an earlier, untracked local run this ledger never recorded
    (git holds no earlier commit for the file, so there is no way to recover what that run
    was). Carried forward as recorded on disk rather than reconstructed, per `docs/method.md`
    rule 13: what cannot be re-run is restated as unestablished, not guessed at. **Published**:
    `docs/weekly-blend-gate.md` quotes the identical interval and its REMOVE, 0-of-4 verdict --
    the number stands regardless of what its own `requires_review` flag was reacting to.

  Both historical entries' numbers are unchanged from what shipped before #362; this ledger
  only changes how many rows hold them and what each row can now say about itself.
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
