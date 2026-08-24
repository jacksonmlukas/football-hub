# Foundation plan

## The problem this solves

The repo currently promises more than it delivers. `CLAUDE.md`, the skills, `SETUP.md`, and the
Makefile all reference modules that do not exist. Every time an agent follows one of those paths
it dead-ends and improvises, which is why work keeps collapsing into narrow one-off fixes.

**Principle: nothing referenced anywhere in the repo may be missing.** Close that gap before
building anything new.

## Definition of done

The foundation is complete when all four hold:

1. Every module named in `CLAUDE.md`, `Makefile`, `SETUP.md`, `docs/`, and every SKILL.md exists
   and runs, even if the implementation is deliberately naive.
2. `make slate WEEK=1` runs end to end without a missing-module error.
3. Every fetch path validates through a `Contract` and writes through `hub.store`.
4. `uv run pytest -q` and `uv run pyrefly check` are both green, with coverage at or above 80%.

**Naive-but-real beats sophisticated-but-absent.** A `ratings` module that returns the market
prior unchanged is a valid Phase 1 deliverable. A missing one is not.

## Anti-goals for this phase

Do not start these until the foundation is done:

- Track A Bayesian state-space ratings
- Track B sequence model
- The championship-leverage simulator (`docs/championship-leverage.md`)
- Any modeling sophistication beyond a working stub

Those are what the foundation is *for*. Building them on dangling references is how you get a
system nobody can run in October.

---

## Phase 0 — Inventory

Do this first and in one session. It is cheap and it sizes everything else.

- [ ] **0.1 Dangling-reference audit.** Grep the whole repo for `hub.` module paths, `make`
      targets, and CLI flags mentioned in docs/skills. Produce `docs/gaps.md` listing every
      referenced-but-missing symbol with the file and line that references it.
      *Done when:* the list is exhaustive and each row says exists / missing / partial.
- [ ] **0.2 Reconcile.** For each gap, decide: build it, or delete the reference. Deleting is a
      legitimate outcome and often the right one.
      *Done when:* every row has a decision and Phase 1-5 todos below are updated to match.

---

## Phase 1 — Data spine (before Sep 3)

Everything downstream reads from here. Build it once, correctly.

- [x] **1.1 `hub.inspect`** _(done 2026-08-23)_ — the summarizing CLI that `.claude/hooks/guard_data_reads.py`
      already tells agents to use. It is referenced in an error message and does not exist,
      which is the worst kind of gap.
      Flags: `--schema`, `--head N --cols a,b,c`, `--describe`, `--nulls`.
      *Done when:* the hook's suggested command actually works, and it never prints more than
      ~40 lines regardless of input size.
- [ ] **1.2 `hub.fetch.nflverse`** — thin wrapper over `nflreadpy` with column selection
      enforced, caching to `data/raw/`, and a `Contract` on every return.
      *Done when:* `python -m hub.fetch.nflverse --refresh` writes parquet through `hub.store`
      and prints a summary only. Loading full-width pbp without `--cols` must raise.
- [ ] **1.3 `hub.fetch.cfbd`** — bulk week endpoints only, plus `--quota` reporting calls used
      this month against the 1,000 budget. Never loops over teams (see `docs/cfbd-quota.md`).
      *Done when:* `make check` prints quota status; a per-team loop is impossible by
      construction, not by convention.
- [ ] **1.4 `hub.fetch.odds`** — one market, one region, one pull. Logs
      `x-requests-remaining` and refuses to run if remaining credits fall below a floor.
      *Done when:* it stores a closing-line snapshot per game and cannot silently burn quota.
- [ ] **1.5 Contracts for every source.** Each fetch function validates before returning.
      Golden fixtures saved under `tests/golden/fixtures/`.
      *Done when:* `tests/contracts/` covers all four sources against frozen fixtures.
- [x] **1.6 `hub.store` proven against real data.** _(done 2026-08-23)_ It has only ever seen a synthetic smoke
      test. Run the ASOF join (`AS_OF_LINES`) against real predictions and real lines.
      *Done when:* the as-of join returns correct results on a case built to break a naive
      week-level join.

---

## Phase 2 — Draft path (hard deadline Sep 3)

This phase has a date attached. If Phase 1 slips, this still ships.

- [ ] **2.1 `--show-slots` on `hub.draft.board`** — referenced in `SETUP.md` step 5 and missing.
      Reads roster composition from the league API, falls back to config, and prints which
      source it used.
      *Done when:* it reports 12 teams / 3 WR / slot 3 and flags disagreement with
      `conf/config.yaml`.
- [ ] **2.2 Weeks 15-17 strength of schedule.** `hub.draft.playoff_sos` — map each NFL team's
      weeks 15-17 opponents from `nflreadpy.load_schedules()`, score opposing defense strength,
      join onto the board as `wk15_17_sos`.
      *Highest-value pre-draft build.* See `docs/championship-leverage.md`.
      *Done when:* the column appears on the board and the top and bottom five pass a sanity
      read.
- [ ] **2.3 `hub.draft.live`** — the draft-day poller. Referenced by `draft-day-ops`, `SETUP.md`,
      and the Makefile. Polls the ESPN draft endpoint, removes drafted players, recomputes VOR
      and replacement level, flags positional runs, prints the mode from `draft_mode()`.
      *Done when:* a dry run against last season's completed draft reproduces the pick sequence
      and never takes more than 2 seconds to refresh.
- [ ] **2.4 Tune `projection_lambda`.** Hydra multirun over a 2024-to-2025 holdout.
      *Done when:* lambda is set from evidence rather than the 0.08 judgment call, and the
      sweep output is committed.
- [ ] **2.5 Full dry run.** Replay last year's draft end to end with the live poller running.
      *Done when:* you find at least one bug. If you find none, the dry run was not realistic.

---

## Phase 3 — Model spine (after the draft)

- [ ] **3.1 `MarketBaseline`** implementing `Forecaster`. Converts closing line to win
      probability and margin. This is the benchmark everything else is scored against, so it
      must exist before any model does.
      *Done when:* it satisfies `isinstance(m, Forecaster)` and passes `validate_predictions`.
- [ ] **3.2 `hub.models.ratings`** — starts as a passthrough returning the market prior. Real
      Bayesian work is Track A and comes later.
      *Done when:* `make slate` calls it without error and it writes versioned predictions.
- [ ] **3.3 `hub.models.conformal`** — wire the existing `Conformalized` wrapper to a rolling
      calibration window driven by `conf/`.
      *Done when:* `--recalibrate` runs and reports empirical vs nominal coverage.
- [ ] **3.4 `hub.models.eval`** — the comparison harness from the `model-eval` skill.
      `--compare a,b --split temporal`, log-loss delta with a bootstrap interval, reliability
      diagram.
      *Done when:* it can score `MarketBaseline` against itself and report a delta of zero with
      an interval containing zero. That is the correctness test.
- [ ] **3.5 Prediction provenance.** Every model writes to
      `data/processed/preds/{model}/{ts}.parquet` with model hash and config digest.
      *Done when:* two runs differing only in a hyperparameter produce distinguishable rows.

---

## Phase 4 — Publish and dashboard

- [ ] **4.1 `hub.publish`** — writes `site/data/*.json` from the processed store.
      *Done when:* `make slate` produces every JSON the site reads.
- [ ] **4.2 The actual dashboard.** `site/index.html`, vanilla JS + uPlot, no build step.
      Panels: draft board, weekly slate, survivor, track record. Serves last-good state and
      marks stale panels rather than erroring.
      *Done when:* `make serve` shows real data and a killed poller degrades visibly but does
      not break the page.
- [ ] **4.3 Live overlay.** Browser polls `live.json`; model numbers stay frozen at Sunday lock.
      *Done when:* the UI makes the frozen-vs-live distinction obvious. See the locked-number
      rule in `weekly-slate`.

---

## Phase 5 — Season operations

- [ ] **5.1 Survivor IP solver** — `pulp`, 18-week assignment, use-each-team-once.
      *Done when:* it returns a feasible full-season assignment. Pool-aware objective is gated
      on getting pool config.
- [ ] **5.2 Weekly lineup optimizer** with the ceiling preference from
      `docs/championship-leverage.md`.
- [ ] **5.4 League transaction-history spike** — count real trades in prior seasons via
      `espn-api`. Twenty minutes, and it decides whether the trade evaluator is worth building.
- [ ] **5.5 Enable the watchdog cron** after the repo goes public Sep 4.

> **5.3 was dropped** (2026-08-23). It proposed `hub.draft.review` on the grounds that `SETUP.md`
> referenced it. `SETUP.md` does not, and the symbol appears nowhere in the repo — it was a new
> feature wearing a dangling reference's clothes, so it did not belong in a phase whose whole
> premise is closing gaps. Numbering left as-is so the surrounding items keep their labels.
> See `docs/gaps.md`.

---

## How to work this with Claude Code

**One phase per session, one todo at a time.** The failure mode you have been hitting is that a
broad prompt sends it chasing a specific until context fills.

Session opener that works:

```
Read docs/foundation-plan.md. We are working Phase 1 only. Start with 1.1 and stop when its
"done when" is satisfied. Use Plan mode first: show me the approach before writing anything.
Do not touch anything outside the files 1.1 needs.
```

Rules worth holding:

- **Plan mode before any multi-file change.** Approve the plan, then switch to Accept edits.
- **One worktree per phase** if you run parallel sessions. Cmd+N gives you an isolated one.
- **Demand evidence.** "Show me the test output," never "it works." The TDD hook makes this
  automatic, so read what it prints.
- **Stop at the "done when."** If a todo says done, it is done. Scope creep here is how the
  foundation never finishes.
- Reserve Sunday's token budget once the season starts. Foundation work is Monday-Wednesday.

## Order of operations

Phase 0, then Phase 1 and Phase 2 interleaved before Sep 3, with Phase 2 winning any conflict
because it has a date. Phases 3 through 5 after the draft.

If time gets tight, the minimum viable Sep 3 state is: 2.1, 2.2, 2.3, and 2.5. Everything else
can wait.
