# A measurement that steers the product must be committed code

**Status:** accepted 2026-08-24.

**Decision.** If a number changes what this repo recommends, ships, or claims, the code that
produced it is committed, tested, and re-runnable from the CLI. An exploratory number that
changes nothing may stay ad hoc — but the moment it is cited as a reason, it needs a harness.

## Why

P0 asked whether championship equity beats the draft market, got +0.04 points per team game
with a 95% CI of [−3.64, +3.58] at n=36, and on that basis demoted championship equity from
the headline to a tiebreaker. That is the most consequential measurement in the repo.

Both P0 commits — `cf02d0f` (design) and `063194c` (result) — touch `docs/next.md` and nothing
else. There is no harness in `src/`, none in `scripts/`, and no artifact in `data/processed`.
The number cannot be reproduced, and two departures from its own pre-registered design went
unrecorded because nothing existed to record them:

- **The shortlist.** The design specified arm B as the top of `win_probability` over
  `recommend()`'s shortlist. The run used top-8 by VOR.
- **The sample size.** The design pre-registered 60 paired observations. The run delivered 36,
  unexplained.

Neither was noticed until an architecture review went looking for the harness five days later.
A pre-registered design is only a control if something checks the run against it, and prose
cannot.

## The trade-off, which is real

P0 got its answer in an afternoon *because* it skipped this. A committed harness is slower to
first number, and with a draft on 2026-09-03 that cost is not hypothetical. The judgement is
that a fast number you cannot re-run is worth less than a slower one you can — not in general,
but specifically for numbers that steer a decision you will have to defend to yourself later.

This is the same principle as [ADR-0004](0004-hydra-config-digest.md) and
[ADR-0006](0006-fitted-constants-live-with-their-provenance.md), one level up: those make a
*prediction* traceable to the model that made it; this makes a *decision* traceable to the
measurement that drove it.

## Consequences

- The harness is a module with a pure core, so its statistics are testable without the
  network. A backtest that only runs against live ESPN is one you will not re-run.
- Results are written through `hub.store` and stamped with `config_digest`, so a later
  disagreement can be attributed to changed code or changed constants.
- It does not apply to exploration. Screening a signal, sizing an effect, sanity-checking an
  API — all fine ad hoc. The trigger is *citation*: when the number becomes a reason.
