"""Every gate names its within-season repeated-measure unit, and this is what stops one of
them drifting back to guessing.

ADR-0019's #335 amendment: a season counts as a *win* only if its gain clears its own noise,
and "its own noise" is a bootstrap over the within-season repeated-measure unit --
`docs/method.md` rule 3's unit, different for every gate (the draft room, the roster, the
player, the event). `experiment.run_gate` has no default for `within`, the same way it has
none for `cluster` (`tests/contracts/test_gates_cluster_on_the_season.py`) -- a caller that
omitted it would get a `TypeError`, not a guess.

**#387.** Before, this file AST-walked five call sites for a hand-kept `HARNESSES` dict --
which excluded `hub.models.margin` entirely and had drifted (its own docstring said so:
"coverage's gate is not in `test_gates_cluster_on_the_season.py`'s `HARNESSES` -- an earlier
omission"). Every gate now declares one or more `Harness` (`hub.models.experiment.Harness`),
whose `within` field *is* the within-season unit a `run_gate`/`gate` call reads, so this reads
it off the harness `tests/gate_harnesses.py` discovers instead of re-parsing a call's AST.
"""
import pytest
from gate_harnesses import all_harnesses


@pytest.mark.parametrize("key", sorted(all_harnesses()))
def test_the_within_unit_is_a_declared_non_empty_tuple(key):
    within = all_harnesses()[key].within
    assert isinstance(within, tuple) and within, (
        f"{key}'s within-season unit ({within!r}) must be a declared, non-empty tuple of "
        f"column names")


# The per-gate units ADR-0019's #335 amendment states, so a reader has one place naming what
# `within` resolves to at each harness without opening six files. Keyed by `module.NAME`
# rather than by a short nickname, so `hub.models.margin`'s two harnesses -- both a real no-op,
# per #335 item 2 -- are each named rather than folded into one entry.
DECLARED_UNITS: dict[str, tuple[str, ...]] = {
    "hub.draft.backtest.HARNESS": ("draft",),
    "hub.models.coverage.SHAPE_HARNESS": ("player_id",),
    "hub.models.starter_change.HARNESS": ("season",),  # one cluster/season: a real no-op
    "hub.models.margin.SHAPE_HARNESS": ("season",),    # one row/season: a real no-op
    "hub.models.margin.WIDTH_HARNESS": ("season",),    # one row/season: a real no-op
    "hub.season.weekly_gate.HARNESS": ("roster",),
    "hub.season.lineup_gate.HARNESS": ("roster",),
    # #343: the three comparisons that reached the Gate by hand before this.
    "hub.models.spread.HARNESS": ("player_id",),
    "hub.models.component_error.HARNESS": ("player_id",),
    # #360 (2026-10-07): the player, rule 3's unit. The type comparison's declared no-op
    # (`season`) is discharged, and the retention-vs-`out_zero` gate that replaces the
    # `verdict` argmin reads the same unit.
    "hub.models.injury.HARNESS": ("gsis_id",),
    "hub.models.injury.RETENTION_HARNESS": ("gsis_id",),
}


def test_the_declared_units_table_covers_every_harness():
    """The table above is a reader's aid, not a second source of truth -- but a harness this
    file has never heard of is exactly the drift #387 replaces, so the two have to agree on
    which harnesses exist before either can say anything about what they resolve to."""
    assert set(DECLARED_UNITS) == set(all_harnesses()), (
        f"DECLARED_UNITS and the discovered harnesses disagree: "
        f"{set(DECLARED_UNITS) ^ set(all_harnesses())}")


@pytest.mark.parametrize("key", sorted(all_harnesses()))
def test_the_within_unit_matches_the_amendment_s_table(key):
    got = tuple(all_harnesses()[key].within)
    assert got == DECLARED_UNITS[key], (
        f"{key} resamples {got}, not {DECLARED_UNITS[key]} -- update this table and the "
        f"ADR-0019 amendment together, never one without the other")
