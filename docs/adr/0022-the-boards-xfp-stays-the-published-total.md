# The Board's xFP stays the published total, with the rebuild beside it

**Status:** accepted 2026-09-04. Closes the spec in issue #1.

**Decision.** `xfp_per_game` remains the source's own pre-summed expected-points column
divided by games. The component rebuild is carried alongside as `xfp_components_per_game`,
with all nine mapped components per game under an `exp_` prefix. Nothing ranks on the rebuild.

## What the spec asked for, and why it changed

Issue #1 proposed building the Board's xFP the way the **Weekly projection** is built —
project the component stats, aggregate them through the league's scoring — and **gating** the
result before adoption, because it would move `proj_blend`, VOR, the corrected ADP, the draft
optimiser and the basis of [ADR-0009](0009-championship-equity-does-not-pick.md).

The gate was pre-registered and then not run, because measuring first showed it could not say
anything. Rebuilding the total from its parts reproduces the published total to a **mean
absolute difference of 0.017 points a player-week, max 0.213**, over 6,054 player-weeks. On the
current season's board the gap is **0.006 points a game, max 0.033**, across 365 players.

A gate on a difference that size returns "no detectable difference" by construction. It would
cost a pre-registration and four seasons of harness to conclude what one measurement already
shows, and — worse — it would produce a *null* that a later reader could mistake for evidence
that components do not help, when what it really measured was that two arithmetic routes to
the same sum agree.

## What replaced it

**Evidence that the plumbing is faithful**, asserted nightly rather than measured once:
`tests/golden/test_component_equivalence.py` rebuilds the total from its parts across five
seasons and fails if the two drift. It is a claim that decays silently — a renamed upstream
column, or a newly scored component nobody maps, would widen it without breaking anything
else — so it is a test rather than a paragraph.

## What the components actually bought

Not accuracy, and the spec was wrong to expect it there. What they bought is **attribution**:
a disagreement now decomposes into components that sum to the gap exactly, groupable by
position. Pooled over 1,187 player-seasons, the projection over-shoots by **+0.709 points a
game**, of which receiving yards (+0.194) and receptions (+0.186) are more than half.

"We are a point high on this player" became "we have him at two more receptions", which was
user story 10 and is the part of the spec that mattered.

**One limit, recorded because a reader will hit it.** The decomposition needs parts on both
sides. ESPN publishes a total and no parts, so against ESPN the gap decomposes on our side
only — it says what our number is made of, not which stat we disagree about. The sum-to-the-gap
property holds against the realised outcome, which is where it is asserted.

## Why this is an ADR and not only a comment

The reasoning already existed in a module docstring, in
[component-projection.md](../component-projection.md), and in a comment on the issue. None of
those is where a future architecture review looks. Two of the three candidates in this repo's
second architecture review were re-proposals of settled questions, and both needed an ADR to
stop a third. A spec with twenty-six user stories that a reader could mistake for unfinished
needs it more, not less.

## What would reopen this

A component projection that is **not** an arithmetic rearrangement of the published total —
one that projects usage forward the way `hub.models.weekly` does, rather than aggregating the
source's own expected stats. That is a different quantity, it could genuinely differ from the
published total, and a gate on it would not be vacuous.

Explicitly not a reason to reopen: that the Board projects points rather than stats. It carries
both, and the one it ranks on is the one that agrees with the source to six thousandths of a
point a game.
