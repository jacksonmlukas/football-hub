# Claude Code setup

Everything below is one-time. The project plugin loads automatically from `.claude/settings.json`
once you trust the folder.

## 1. Install the language server binary first

`pyright-lsp` is the highest-value install here and it does not fetch its own binary.

```bash
npm install -g pyright
```

## 2. Install plugins

The official marketplace (`claude-plugins-official`) is added automatically. Inside Claude Code:

```
/plugin install pyright-lsp@claude-plugins-official
/plugin install security-guidance@claude-plugins-official
/plugin install commit-commands@claude-plugins-official
/plugin install pr-review-toolkit@claude-plugins-official
/plugin install github@claude-plugins-official
/plugin install plugin-dev@claude-plugins-official
```

Then `/reload-plugins` (add `--force` if it warns about the prompt cache).

Why each one:

| Plugin | What it buys |
|---|---|
| `pyright-lsp` | Claude sees type errors and missing imports after every edit and fixes them in the same turn. With Pyrefly in the TDD hook, this closes the write-verify loop. |
| `security-guidance` | Reviews each change for vulnerabilities. This repo is public and handles ESPN session cookies. |
| `commit-commands` | `/commit-commands:commit` — stage, message, commit. |
| `pr-review-toolkit` | Self-review agents, which is the only review this repo gets. |
| `github` | Issues and PRs without leaving the session. Pairs with GitHub Projects tracking. |
| `plugin-dev` | For extending `plugins/football-hub/`. |

The community marketplace needs adding by hand if you want it:

```
/plugin marketplace add anthropics/claude-plugins-community
```

## 3. Verify the project plugin loaded

```
/plugin list
```

You should see `football-hub@football-hub-local` with four skills: `draft-day-ops`,
`weekly-slate`, `model-eval`, `data-contracts`. Skills are namespaced, so invoke as
`/football-hub:weekly-slate`.

If skills do not appear: `rm -rf ~/.claude/plugins/cache`, restart, reinstall.

## 4. Monthly audit

Run `/plugin`, open **Installed**, and look under **Not used recently** — plugins you have not
invoked in two weeks across ten sessions still cost startup and context. Uninstall them.

The **Discover** tab shows a per-plugin **Context cost** estimate before you install. Check it.
Good setups run roughly eight audited skills. Bad ones run thirty.

## MCP policy for this repo

MCP tool *definitions* are cheap now, because Tool Search defers them and loads only what a task
needs. MCP tool *outputs* are still expensive, since a call returns the whole payload rather than
a summary.

So:

- **MCP is fine** for interactive, occasional work returning a handful of records: GitHub,
  project management, error monitoring.
- **MCP is wrong** for data fetching. nflverse, CFBD, ESPN, and odds stay as CLIs forever,
  because a CLI prints a 20-line summary where an MCP tool returns 60 KB of JSON.

## Worktrees

Three concurrent sessions, no more. Above that you are bottlenecked on your own review capacity.

```bash
claude --worktree bayes     # Track A: state-space ratings
claude --worktree eval      # Track C: conformal + backtest harness
# main checkout             # draft board + weekly slate
```

Track B (sequence model) stays out until October.

## Hooks

Two are wired in `.claude/settings.json`. CLAUDE.md is a request; hooks are a guarantee.

- `guard_data_reads.py` blocks `Read`/`cat` against `data/` or any parquet, and redirects to a
  summarizing CLI. This is the token rule made unbreakable.
- `tdd_gate.sh` runs the fast suite after every edit and exits 2 on failure, so Claude sees red
  immediately instead of at the end of a long session.
