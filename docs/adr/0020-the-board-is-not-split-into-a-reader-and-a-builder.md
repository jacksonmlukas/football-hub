# The Board is not split into a reader and a builder

**Status:** accepted 2026-09-04. Investigated under issue #19.

**Decision.** `hub.draft.board` keeps both building a Board and reading one. No reader-only
module is extracted, and the three function-local imports of it stay as they are.

## Why this is written down

The proposal has surfaced in two consecutive architecture reviews on the same day, each time
from the same true observations: the module is the largest in the repo, it is the only one at
the top of both fan-in and fan-out, and three call sites in `hub.season` reach it through
function-local imports. A third review will find the same three facts. This records what
measuring them showed, so the reasoning is not re-derived from scratch.

## What the measurements said

**The reader seam would be cheap, and that is true.** A prototype module holding only the
last-good read imports in **67 ms** against the Board's **273 ms**.

**It would also help almost nobody.** `board_as_of` is not a reader. It calls `build`, and
therefore needs the consensus and ADP fetchers, and it is what four of the five cross-package
call sites use — both gates and the backtest. A reader seam does nothing for any of them.

That leaves exactly **one** cross-package call site that genuinely reads: the Sunday roster.
`hub.draft.live` reads too, but it already lives beside the Board. One adapter is a
hypothetical seam.

**And the cost is small.** The first review reported 631 ms for the Board import. That was a
cold-cache artifact of a single measurement; best-of-three in fresh processes gives 273 ms, and
the *marginal* cost on the only live path — `hub.publish`, which the Pages deploy runs every ten
minutes — is **55 ms**, because publish already loads most of what the Board needs.

## The three function-local imports, classified

None is cycle avoidance: `hub.draft.board` imports nothing from `hub.season`.

| Site | Verdict | Evidence |
|---|---|---|
| `season.roster.fetch` | load-bearing lazy-loading | `publish` imports `roster` at module level, so eager would put the Board on the ten-minute deploy path |
| `weekly_gate_data.preseason_ranks` | load-bearing lazy-loading | the module imports in 155 ms; with the Board eagerly, 279 ms |
| `weekly_gate_data.assemble_universe` | same | same import, same function-local block that loads that function's whole network world |

`hub.season.lineup_gate` imports the Board at module level and costs 275 ms, which is fine for
a harness nobody runs on a clock. The pattern is consistent rather than accidental: modules on
a live path defer, harnesses do not.

## The argument that decided it

Cost was not the deciding factor; 55 ms would be worth removing if the change were free.

**`last_good` is Board knowledge and belongs with the Board.** Splitting it out puts "the
Board" in two modules, which costs navigability — the thing an architecture review is supposed
to buy. Compare `hub.paths`, extracted for a nearly identical-looking reason: what moved there
was a `Path`, which never belonged to the Board in the first place. Reading a Board does.

The existing arrangement already solves the real problem. One documented function-local import
on the one live path is a smaller thing to understand than a second module holding one function.

## What would reopen this

A second cross-package reader. Two would make the seam real by the rule the first one fails,
and the measurement above would then be an argument for the split rather than against it.

Not: another review noticing the line count, the fan-out, or the local imports. Those are the
observations that produced this ADR.
