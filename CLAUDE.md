# Football Hub — Agent Rules

## Hard rules (violating these is the main failure mode)

1. **Never read a data file into context.** No `cat`, `head`, `view`, or `pl.read_parquet(...)` printed
   raw for anything in `data/`. One `load_pbp()` is 40k+ tokens. Always go through a CLI that
   prints a *summary*.
2. **No MCP servers for data.** Every data source is a CLI under `src/hub/fetch/`. A Bash call costs
   ~40 tokens; an MCP server costs 8-15k tokens of tool definitions in every session, all season.
3. **Never loop over teams or games when hitting CFBD.** Always pull the week-level or season-level
   bulk endpoint and filter locally. Looping is what burns the 1,000/month free quota. See
   `docs/cfbd-quota.md`.
4. **Scan work goes to a Haiku subagent.** Injury sweeps, line-movement diffs, parsing large ESPN
   payloads. Their context is discarded; yours is not.

## Model routing (Max 5x)

| Task | Model |
|---|---|
| Architecture, Bayesian model spec, subtle debugging | Opus (~2x/week; separate weekly cap) |
| Build loops, scripts, refactors | Sonnet (default, ~90% of work) |
| Scan subagents, payload parsing, news sweeps | Haiku |

**Reserve Sunday.** Research and refactors Mon-Wed. Thu onward stays clear so the weekly cap never
eats lineup day. Check `claude.ai/settings/usage` before starting anything heavy.

## Compact instructions

When summarizing this conversation, preserve: model specs and their fitted parameters, go/no-go gate
outcomes, quota consumption to date, and any decision that overrode a documented default. Drop:
file contents, stack traces already resolved, exploratory dead ends.

## Graceful degradation

Every module must produce a usable answer with zero attention. If a fetch fails, serve last-good
state from `data/processed/` rather than erroring. Systems that need an operator die in October.

## Worktrees

A worktree starts with no `.venv`; the first `uv run` builds one. That environment is the one the
rules assume, so **there is no setup step** — run the gates directly:

```bash
uv run ruff check src tests        # lint
uv run pyrefly check src tests     # types (name the paths; see below)
uv run pytest tests/unit -q        # tests
```

Three things make that true, and each of them was a session someone lost:

- The toolchain (`pytest`, `ruff`, `pyrefly`, `pytest-cov`, `hypothesis`) is a `[dependency-groups]`
  group in `pyproject.toml`, **not an extra**. uv installs default groups on every `uv sync` and
  every `uv run`; it installs an extra only when asked. As an extra it was in no venv uv ever built
  on its own. If you are tempted to move it back, read the comment above it first.
- Never trust `uv run pytest` to fail loudly. With no pytest in the venv, `uv run` falls through to
  `PATH` and runs whatever global pytest exists — against a venv missing the project's deps, so it
  reports `ModuleNotFoundError` on a first-party import. That is an environment fault dressed as a
  code error, and only having pytest in the venv prevents it.
- `pyrefly check` with no paths resolves its own file set in project mode, where it honours
  `.git/info/exclude` — which ignores `.claude/worktrees/`. Inside a worktree that matches zero
  files. **Always name `src tests`.** `.claude/hooks/tdd_gate.sh` does, and it must stay that way;
  do not "fix" it by editing the exclude file, which is what keeps agent worktrees out of the
  primary checkout's `git status`.

`.env` is gitignored and is carried in by `.worktreeinclude`. Nothing else the gates need is
gitignored. The suite is already written for a tree with no `data/` — that is the fresh-clone case
`test_board_build.py` and `test_board_main.py` exist to hold — so a worktree runs it unmodified.

## Agent skills

### Issue tracker

Issues live as GitHub issues in the private `football-hub` repo, driven by the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, unchanged: `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
