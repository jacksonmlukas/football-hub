# Dangling-reference audit

Phase 0.1 of `docs/foundation-plan.md`. Every `hub.*` module path, `make` target, CLI flag, and
file path referenced anywhere outside `src/`, checked against what exists.

Audited 2026-08-23. **11 days to the draft.**

## Method

- `hub.*` paths: regex over `docs/`, `.claude/`, `plugins/`, `conf/`, `.github/`, `CLAUDE.md`,
  `SETUP.md`, `README.md`, `Makefile`, then `python -m <mod> --help` on each to separate
  "missing" from "exists but not invocable".
- `make` targets: defined-in-Makefile vs mentioned-in-prose.
- Paths: literal existence check.
- `docs/foundation-plan.md` is excluded as a reference source. It describes intent, so counting
  it would make every planned item look like an existing dangling reference.

## Summary

| Status | Count |
|---|---|
| Missing module, referenced by something that runs | **9** (all 9 resolved as of 2026-08-25) |
| Exists but not invocable as documented | **2** |
| Missing path | **2** |
| Already built, plan is stale | **2** |
| False positives (do not re-audit) | **8** |

---

## Missing modules

Each of these is named by a Makefile target, a SKILL.md, or a hook error message. Following any
of them dead-ends.

| Symbol | Referenced from | Status | Decision (0.2) |
|---|---|---|---|
| `hub.inspect` | `.claude/hooks/guard_data_reads.py:31,32` | ~~MISSING~~ **DONE 2026-08-23** | Built. `--schema`, `--head N --cols`, `--describe`, `--nulls`, default overview. 95% covered. |
| `hub.fetch.nflverse` | `Makefile:10`, `weekly-slate/SKILL.md:16` | ~~MISSING~~ **DONE 2026-08-23** | Built. Refuses wide sources without `cols`, contracts on return, writes week partitions through `hub.store`. |
| `hub.fetch.cfbd` | `Makefile:11,18`, `weekly-slate/SKILL.md:17` | ~~MISSING~~ **DONE 2026-08-23** | Built. Three independent loop guards, per-month quota accounting, `make check` works with no key. |
| `hub.fetch.odds` | `weekly-slate/SKILL.md:18` | ~~MISSING~~ **DONE 2026-08-23** | Built. One market, one region, credit floor, maps to nflverse game ids. |
| `hub.models.ratings` | `Makefile:12`, `weekly-slate/SKILL.md:20` | ~~MISSING~~ **DONE 2026-08-23** | Passthrough over `MarketBaseline`. `make slate WEEK=1` now runs end to end. |
| `hub.models.conformal` | `weekly-slate/SKILL.md:21` | ~~MISSING~~ **DONE 2026-08-24** | Built. Rolling calibration, `--recalibrate --model --alpha --window`. Exists and runs; what it still lacks is a *consumer* — [improvements.md #2](improvements.md). |
| `hub.models.eval` | `model-eval/SKILL.md:26` | ~~MISSING~~ **DONE** | Built. `--compare a,b --split temporal`, exactly as the SKILL.md invokes it. |
| `hub.publish` | `track-record.md:12,24`, `weekly-slate/SKILL.md:22` | ~~MISSING~~ **DONE 2026-08-24** | Writes preds/track_record/live plus a manifest; never blanks an artifact. |
| `hub.draft.live` | `draft-day-ops/SKILL.md:25`, `SETUP.md:227` | ~~MISSING~~ **DONE 2026-08-23** | Built. Live replacement level, run detection, `--replay` rehearsal. 192 picks at 7ms/refresh. |

### Why `hub.inspect` is the worst one

`.claude/hooks/guard_data_reads.py` blocks any command that looks like a raw data read and tells
the agent to run `python -m hub.inspect <dataset> --schema` instead. That module has never
existed. So the single rule the repo most wants enforced fails at the exact moment it fires, and
the agent is left to improvise — which is the behaviour this whole plan is trying to stop.

It fired during this audit, on a `cat > docs/foundation-plan.md` heredoc, because the *document
text* contains the string `data/raw/`. So it also has a false-positive problem: it matches on
command text rather than on an actual read of a file under `data/`.

### Resolved 2026-08-23

`src/hub/inspect.py` exists, and the hook was fixed alongside it. Three separate defects, all
of which had to go for "the suggested command actually works" to be true:

1. **`[^|]*` → `[^|\n]*`.** A reader command and its argument share a line; heredoc bodies and
   commit messages do not. This was matching a `| tail -1` on line one against a `data/` mention
   twenty lines later, and refused any command whose text merely described the cache layout.
2. **The hint named bare `python`,** which the modern-python PATH shim intercepts. Following the
   advice verbatim hit a second wall. Now `uv run python -m hub.inspect`.
3. **The guard blocked the escape hatch it recommends.** `--head` contains `head`, so
   `hub.inspect --head 5 <path>.parquet` matched the reader pattern. An `ALLOWED` pattern now
   exempts any command invoking `hub.inspect`. This is guidance rather than a sandbox — someone
   determined to `cat` a parquet can append a mention of `hub.inspect` — and keeping the
   recommended path unobstructed is worth more than closing that.

Found only by writing the verification commands from the plan and watching the hook refuse them.
A guard that fires while you are writing prose teaches you to ignore it, which is how a
guardrail stops working.

---

## Exists but not invocable as documented

Worse than missing in one respect: the module is present, so a reader assumes the path works.

| Symbol | Problem | Referenced from | Decision (0.2) |
|---|---|---|---|
| `hub.fetch.espn` | No `__main__`, no argparse. | ~~`Makefile:15` (`make live`)~~ | **Resolved 2026-08-24 by deletion.** The `make live` target was removed rather than the CLI built: a documented command that silently does nothing is worse than an absent one, and the repo goes public on Sep 4. `poll()` still exists and is real; the entry point is post-draft work, when the in-season path is the live one. |
| `make setup` | Runs `uv pip install -e .` — the legacy interface, blocked by the `trailofbits/modern-python` PATH shim. Also omits `[dev]`, so it installs no test tooling. | ~~`Makefile:4`~~ | **Resolved 2026-08-24.** Both `Makefile` and `SETUP.md` now use `uv sync --all-extras`, which also creates the venv. The stale `46 passed` check in SETUP.md was refreshed at the same time. |
| Three SKILL.md commands | Invoked bare `python -m hub....`, which the `trailofbits/modern-python` PATH shim intercepts: it prints `ERROR: Use `uv run python -m ...`` and does nothing. Same defect the hook message had. | ~~`model-eval/SKILL.md:26`, `draft-day-ops/SKILL.md:17,25`~~ | **Resolved 2026-08-25.** All three now say `uv run python`. `draft-day-ops` is the one used on draft night, so it was the one that mattered. The remaining bare-`python` mentions in `gaps.md`, `foundation-plan.md` and ADR-0004 are historical narrative and are left alone. |

---

## Missing paths

| Path | Referenced from | Status | Decision (0.2) |
|---|---|---|---|
| `data/processed/roster.parquet` | `hub/season/lineup.py:183` (the `--roster` default) | **MISSING, recorded 2026-08-25** | Nothing writes it. The weekly lineup CLI has therefore never been runnable end to end, and used to exit with a bare polars `FileNotFoundError`. It now prints what the file should contain and that the optimiser is [+0.00 against starting your highest projections](adr/0012-the-lineup-optimiser-waits-for-real-variance.md), so building the producer is not urgent. Post-draft: the roster is knowable only after Sep 3. |
| `tests/golden/fixtures/` | `data-contracts/SKILL.md:29`, `tests/contracts/test_contracts.py:4` | ~~empty~~ **DONE 2026-08-23** | 7 fixtures: 4 real captures, 3 hand-built for keyless sources. `pytest -m golden` runs the live diff. |
| `site/index.html` | `make serve`, `going-public.md:17`, `SETUP.md:141`, `draft-day-ops/SKILL.md:49` | ~~MISSING~~ **DONE 2026-08-24** | Four panels, no build step, stale panels named individually. |

Not gaps, despite looking like ones:

- `CONTEXT.md`, `docs/adr/` — `docs/agents/domain.md` explicitly says to proceed silently when
  absent, and that `/domain-modeling` creates them lazily. Working as designed.
- `.scratch/` — only relevant to the local-markdown issue tracker, and GitHub was chosen.

---

## Already built — the plan is stale

Both were completed on 2026-08-21, before the plan was written.

| Plan item | Reality |
|---|---|
| **2.1** `--show-slots` "referenced in SETUP.md step 5 and missing" | **Implemented.** Reports `teams 12 | slot 3 | QB1 RB2 WR3 TE1 FLEX1`, picks 3/22/27/46/51/70/75, and the 19-5 wait alternation. |
| **2.2** playoff SoS as `hub.draft.playoff_sos`, column `wk15_17_sos` | **Implemented.** 435 of 452 players placed, range 0.75–1.31, `--sos` view with same-tier swaps. |

**Naming conflict — resolved 2026-08-23 in the plan's favour.** The code shipped as
`hub.draft.schedule` / `sos_1517` and was renamed to `hub.draft.playoff_sos` / `wk15_17_sos` to
match. `schedule` was a poor name anyway: it suggested a schedule loader rather than a strength
metric, and it collided conceptually with `nflreadpy.load_schedules()`, which the module calls.

One thing worth knowing for future renames in this repo: the column rename silently no-op'd on
the first attempt because BSD `sed` on macOS does not support `\b` word boundaries. The pattern
matched nothing and `sed` exited 0. Grep for the old name afterwards rather than trusting the
exit code.

---

## Plan claims not supported by the repo

| Claim | Finding |
|---|---|
| **5.3** `hub.draft.review` — "Referenced in `SETUP.md`" | **It is not.** Grep for `review` in `SETUP.md` returns nothing. `hub.draft.review` appears nowhere in the repo except the plan. It was a new feature wearing a dangling reference's clothes. **Dropped from the plan 2026-08-23.** |

---

## False positives — do not re-audit

My first pass flagged these because `github.` contains the substring `hub.`.

| Match | Actually |
|---|---|
| `hub.repository`, `hub.repository_owner`, `hub.event_name`, `hub.event.repository.name` | GitHub Actions expressions in `.github/workflows/watchdog.yml` |
| `hub.io` | `username.github.io` — Pages URL |
| `hub.zip` | `football-hub.zip` — a download filename in `SETUP.md` |
| `hub.duckdb` | `data/processed/hub.duckdb` — the database file, not a module |
| `make playoffs` | The English phrase "8 make playoffs" in `championship-leverage.md` |

---

## Coverage against definition-of-done item 4

Target is 80%. Actual is **69%** (171 tests, 888 statements, 274 uncovered).

| Module | Cover | Note |
|---|---|---|
| `hub/store.py` | ~~0%~~ **97%** | Resolved 2026-08-23 by plan item 1.6 |
| `hub/fetch/espn.py` | 26% | Network paths; the parsers are covered, the fetchers are not |
| `hub/draft/board.py` | 45% | Almost all of the miss is `main()` — the CLI has no test |
| `hub/draft/availability.py` | 72% | `historical_picks` is network-bound |
| everything else | 81–100% | |

Two corrections to the plan fall out of this:

- **1.6 said `hub.store` "has only ever seen a synthetic smoke test."** There was no smoke
  test: coverage was zero and nothing imported it. **Resolved 2026-08-23.** Three defects
  surfaced once it was exercised: `sql()` took no parameters, so `AS_OF_LINES` -- the module's
  own canonical query, carrying a `?` -- could not be run through the module's own helper;
  there was no injection point for the data root, so nothing could be tested without
  reassigning module globals; and the `:02d` zero-padding in `LAYOUT` turned out to be
  load-bearing for week ordering (Hive keys arrive as strings, so an unpadded week 9 sorts
  after week 10) with nothing documenting or testing it. `--verify` now round-trips 285 real
  2025 games through the catalog and the as-of join.
- **The 80% gate is a Phase 1 blocker, not a finishing touch.** Three of the four worst-covered
  modules are exactly what Phase 1 builds on. Writing the fetch modules first and back-filling
  tests later would bake the gap in.

## What this implies for sequencing

`make slate WEEK=1` — the plan's definition-of-done item 2 — requires **three** missing modules
(`fetch.nflverse`, `fetch.cfbd`, `models.ratings`) plus `hub.store` wiring. That is the single
largest cluster and it is all Phase 1.

The draft path is in better shape than the plan assumes. Of the four items in the
minimum-viable-Sep-3 set (2.1, 2.2, 2.3, 2.5), **two are already done**. What remains is
`hub.draft.live` and the dry run — and the dry run cannot start until the poller exists.

Suggested order given 11 days:

1. `hub.inspect` (1.1) — small, unblocks the hook, and every later session benefits.
2. `hub.draft.live` (2.3) — the only remaining dated item.
3. `make setup` / `hub.fetch.espn` CLI — both are one-liners that make documented commands true.
4. Full dry run (2.5).
5. Everything else after Sep 3.
