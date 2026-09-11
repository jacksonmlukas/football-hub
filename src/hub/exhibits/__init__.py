"""Measurements the product no longer reads, kept so they can be re-run.

Two ADRs pull opposite ways over the code in this package and neither is wrong. ADR-0009
removed championship equity from the draft-night output -- it lost to following the draft
market by 17 to 20 points a team-game, four seasons of four -- and ADR-0007 requires that a
measurement which steered a decision stay committed, tested and re-runnable, so the code
could not simply be deleted. What neither said was *where* such code should live, and the
cost of leaving it inside `hub.draft` was measured on 2026-09-06: roughly a thousand lines
sat beside the code the product reads, indistinguishable from it at a glance, downstream of
no product decision, and two defects accumulated in them for 270 commits without any
product-level signal because nothing that ships could complain (#198).

**An exhibit is a dependency of a record, not of the product.** The only thing in `src/`
allowed to import from here is the harness that re-runs the measurement --
`hub.draft.backtest`, which ADR-0009 names as the way to reopen it. `hub.draft.board`,
`hub.draft.live` and `hub.publish` do not, and
`tests/contracts/test_the_exhibit_is_not_a_dependency.py` is what keeps that true rather than
a census. The direction is one-way: an exhibit imports the simulators it measures with
(`hub.draft.optimize`, `hub.draft.season`, `hub.models.predict`); nothing in those packages
imports an exhibit.

One module per measurement, named for the thing measured and carrying its own write-up:

  * `championship_equity` -- the objective ADR-0009 gated and removed; the record is
    `docs/adr/0009-championship-equity-does-not-pick.md` and the harness `hub.draft.backtest`.
  * `leverage` -- what this league's six-of-twelve bracket rewards, `docs/six-of-twelve.md`;
    a CLI with no importer, which is the state it was found in and the right one for it.

What does *not* belong here is anything a Gate still reads. `hub.draft.cohort` draws the
rosters two season-side Gates are scored on through `optimize.simulate_remaining_draft`,
and that stays where it is: the room simulator is machinery the live Gates run on, not a
removed measurement, however close it sits to one.
"""
