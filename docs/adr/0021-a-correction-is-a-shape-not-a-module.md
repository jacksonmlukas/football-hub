# A Correction is a shape, not a module

**Status:** accepted 2026-09-04. Investigated under issue #20.

**Decision.** `hub.draft.durability` and `hub.draft.regression` keep their own
`prior_season`, `attach` and `correct_projection`. No `Correction` module or protocol is
introduced. What is genuinely shared is already in `hub.draft.prior_signal`.

## Why it was proposed

`CONTEXT.md` defines a **Correction**, two modules implement one, and they expose the same
four public names in the same order. This repo has fixed that exact pattern four times — the
**Panel**, the **Artifact**, the **Gate**, the **Cohort** — each a glossary term with no module
behind it. A fifth looked likely.

## Why it does not hold

**It was already decided, and the reason was already written down.** `hub.draft.prior_signal`
exists because of `docs/improvements.md` #15, and its module docstring answers this question
directly: the two `correct_projection` bodies *are not the same function*, and folding them
into one shape with flags would be the abstraction that costs more than the duplication. The
review that proposed this matched on four repeated function names without reading the module
that exists to explain them.

**The bodies differ in kind, not in detail.** Durability prices two things — a season-long
trait applied at two positions, and today's designation applied at all of them, because being
ruled out is news rather than a trait the market has had years to discount. Touchdown luck
prices one term at two positions. A shared body would need a flag for the second term, and a
flag whose two branches are "one term" and "two terms over different position sets" is a
`Correction` that has to be read alongside both callers to be understood.

**And there is a hard constraint the proposal did not weigh.** `config.fitted_constants` keys
every fitted number by its **module**: `durability.BETA`, `durability.INJURY_BETA`,
`regression.TD_LUCK_BETA`, and four more. Moving them into a shared module renames every key
and moves `fitted_digest`, which is hashed into every model version string — so every recorded
prediction's provenance would be invalidated to tidy four functions. That is
[ADR-0006](0006-fitted-constants-live-with-their-provenance.md)'s line, and a `Correction`
module could only avoid it by leaving the constants behind, at which point what is shared is a
four-line sequence and nothing else.

## What a third Correction would actually cost

Answered against the code rather than in principle, because that is the only thing a shared
shape would buy:

| Step | Cost today |
|---|---|
| `prior_season` | bespoke — different columns per signal, and no abstraction removes a fetch |
| the signal itself | bespoke — the entire content of a Correction |
| `attach` | **one line**, already `prior_signal.join_by_player` |
| `correct_projection` | **about five lines**, already using `prior_signal.priced` |

So a third Correction costs roughly six lines of shape. That is the whole prize, and it is
smaller than the flag it would take to win it.

## Why the four successful instances do not generalise

They were all *one idea implemented N times identically*, where the copies had drifted or
demonstrably could: the Gate's three verdicts had already diverged, the Cohort's seed formula
was hand-copied into two places, the Panel was reached into with function-local imports, the
Artifact's freshness contract was written three ways in one function.

This is *one sequence followed twice with genuinely different bodies*, and its shared parts
have already been extracted. The pattern that succeeded four times is "the same code in two
places". This is "the same shape, different code", which is what a domain looks like when two
things are members of one category rather than copies of one implementation.

## What would reopen this

A third Correction whose `correct_projection` turns out to be the same body as one of the two
existing ones. That would be evidence of a real duplicate rather than a shared shape — and it
would still have to answer the `fitted_digest` question before anything moved.
