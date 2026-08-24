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
| Missing module, referenced by something that runs | **9** |
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
| `hub.inspect` | `.claude/hooks/guard_data_reads.py:31,32` | MISSING | **Build.** Highest priority of the nine — see below. |
| `hub.fetch.nflverse` | `Makefile:10`, `weekly-slate/SKILL.md:16` | MISSING | **Build.** Blocks `make slate`. |
| `hub.fetch.cfbd` | `Makefile:11,18`, `weekly-slate/SKILL.md:17` | MISSING | **Build.** Blocks `make slate` and `make check`. |
| `hub.fetch.odds` | `weekly-slate/SKILL.md:18` | MISSING | **Build**, but after the draft. Nothing pre-Sep 3 needs odds. |
| `hub.models.ratings` | `Makefile:12`, `weekly-slate/SKILL.md:20` | MISSING | **Build as passthrough.** Blocks `make slate`. |
| `hub.models.conformal` | `weekly-slate/SKILL.md:21` | MISSING | **Build.** `Conformalized` already exists in `models/base.py`; this is the CLI around it. |
| `hub.models.eval` | `model-eval/SKILL.md:26` | MISSING | **Build**, post-draft. |
| `hub.publish` | `track-record.md:12,24`, `weekly-slate/SKILL.md:22` | MISSING | **Build**, post-draft. |
| `hub.draft.live` | `draft-day-ops/SKILL.md:25`, `SETUP.md:227` | MISSING | **Build.** Has a date: this is the draft-day poller. |

### Why `hub.inspect` is the worst one

`.claude/hooks/guard_data_reads.py` blocks any command that looks like a raw data read and tells
the agent to run `python -m hub.inspect <dataset> --schema` instead. That module has never
existed. So the single rule the repo most wants enforced fails at the exact moment it fires, and
the agent is left to improvise — which is the behaviour this whole plan is trying to stop.

It fired during this audit, on a `cat > docs/foundation-plan.md` heredoc, because the *document
text* contains the string `data/raw/`. So it also has a false-positive problem: it matches on
command text rather than on an actual read of a file under `data/`.

---

## Exists but not invocable as documented

Worse than missing in one respect: the module is present, so a reader assumes the path works.

| Symbol | Problem | Referenced from | Decision (0.2) |
|---|---|---|---|
| `hub.fetch.espn` | No `__main__`, no argparse. `python -m hub.fetch.espn --poll --interval 45` does nothing. | `Makefile:15` (`make live`) | **Build the CLI.** The `poll()` function already exists; it just has no entry point. |
| `make setup` | Runs `uv pip install -e .` — the legacy interface, blocked by the `trailofbits/modern-python` PATH shim. Also omits `[dev]`, so it installs no test tooling. | `Makefile:4` | **Fix to `uv sync`.** Same issue in `SETUP.md:55`. |

---

## Missing paths

| Path | Referenced from | Status | Decision (0.2) |
|---|---|---|---|
| `tests/golden/fixtures/` | `data-contracts/SKILL.md:29`, `tests/contracts/test_contracts.py:4` | `tests/golden/` exists but is **empty** | **Build** alongside the fetch modules — a golden test with no fixture is decorative. |
| `site/index.html` | `make serve`, `going-public.md:17`, `SETUP.md:141`, `draft-day-ops/SKILL.md:49` | MISSING — `site/` holds only `data/draft_board.json` | **Build**, post-draft. `make serve` currently returns a directory listing, which is not an error but is not a dashboard either. |

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
| **2.2** playoff SoS as `hub.draft.playoff_sos`, column `wk15_17_sos` | **Implemented** as `hub.draft.schedule`, column `sos_1517`. 435 of 452 players placed, range 0.75–1.31, `--sos` view with same-tier swaps. |

**Naming conflict to resolve:** the plan says `hub.draft.playoff_sos` / `wk15_17_sos`; the code
says `hub.draft.schedule` / `sos_1517`. One of them should move. Renaming the code is a
~10-minute change across three files and a test module; amending the plan is free. No opinion
worth defending either way, but leaving both names in circulation is how the next session builds
it a second time.

---

## Plan claims not supported by the repo

| Claim | Finding |
|---|---|
| **5.3** `hub.draft.review` — "Referenced in `SETUP.md`" | **It is not.** Grep for `review` in `SETUP.md` returns nothing. `hub.draft.review` appears nowhere in the repo except the plan. It is a new feature, not a dangling reference, and should be re-labelled as such — otherwise it competes for "foundation" priority it has not earned. |

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
