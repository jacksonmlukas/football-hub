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
