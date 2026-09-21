"""#378: a season dropping from the weekly gate's paired frame has to be loud.

Lane D's 2026-09-21 `hub.season.weekly_gate --run --ceiling` reported three clusters against
the four `docs/weekly-blend-gate.md` publishes and the ADR-0019 amendment's rule-16 table
reads this gate at. Season 2022 was simply absent from the per-season table the run printed --
`weekly_gate_data.assemble_universe` intersects the seasons its projection frame actually
covers with the seasons it was asked for (`set(proj["season"].unique()) & set(seasons)`), and
a season that produced no projection rows drops out of that intersection with nothing said.

The root cause was the CLI's own `--seasons` default: `expanding_seasons` always consumes the
*earliest* season in what it is handed as training data for the one after it, so a caller
naming exactly the four seasons it wants scored gets three -- the earliest is spent training
the second rather than being a fifth, sacrificial season nobody asked it to score. That one
drop is expected on every call this gate makes. What #378 asks for is that it is the *only*
kind of drop that can happen without the run saying so.

Two things are held here: the mechanism exists in the source (`SeasonDropped`, raised, and a
line printed naming what was requested against what was scored), and it actually fires,
offline, against the frozen panel archive -- a source-only check would pass on a `raise` that
guards the wrong condition or never runs.
"""
from __future__ import annotations

import inspect
import pathlib

import panelarchive as arc
import pytest

from hub.season import weekly_gate, weekly_gate_data

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "season"


def test_the_assembly_declares_a_dedicated_refusal():
    """Not a bare `ValueError` -- a name a caller can catch specifically, the same shape
    `experiment.CorrectionMissing` already uses for the sibling refusal this module makes
    (a Board built from a subset of the Corrections)."""
    assert issubclass(weekly_gate_data.SeasonDropped, RuntimeError)


def test_assemble_universe_raises_it_rather_than_returning_a_thinner_frame():
    src = (SRC / "weekly_gate_data.py").read_text()
    body = src[src.index("def assemble_universe"):]
    assert "raise SeasonDropped" in body, (
        "assemble_universe no longer raises SeasonDropped -- a season with no row is once "
        "again absorbed by the `set(...) & set(seasons)` intersection with nothing said")


def test_the_run_prints_what_it_asked_for_beside_what_it_got():
    """Loud on every run, not only on the failure path -- the walk-forward buffer drops a
    season on every call this gate makes, and a reader should not have to already know that
    to notice it happening."""
    src = inspect.getsource(weekly_gate.main)
    assert "seasons requested" in src and "scored" in src, (
        "weekly_gate.main no longer prints the seasons it requested against the seasons it "
        "scored -- the #378 acceptance criterion is that this is stated on every run, not "
        "just when something goes missing unexpectedly")


def test_the_cli_default_matches_the_seasons_the_published_gate_reads():
    """`docs/weekly-blend-gate.md`'s Reproduce section names five seasons so the walk-forward
    buffer is a fifth, sacrificial one and all four held-out seasons score. A default that
    named only the four -- the pre-#378 shape -- would silently reproduce the bug on every
    run nobody customised `--seasons` for, #376's included."""
    ap_default = None
    for line in inspect.getsource(weekly_gate.main).splitlines():
        if '"--seasons"' in line and "default=" in line:
            ap_default = line
            break
    assert ap_default is not None, "no --seasons argument found to check the default of"
    assert "2021" in ap_default, (
        f"--seasons' default does not carry a walk-forward buffer season ahead of the four "
        f"held-out ones: {ap_default.strip()!r}")


def test_a_season_with_no_row_is_refused_offline(monkeypatch, tmp_path):
    """The behavioural proof: requesting a season the archive covers nothing for -- rather
    than asserted by reading the source -- actually raises, against the same frozen panel
    archive `tests/unit/test_weekly_gate_data.py` runs the rest of this module's tests on."""
    arc.install(monkeypatch, tmp_path, board=True)
    with pytest.raises(weekly_gate_data.SeasonDropped, match=r"2025"):
        weekly_gate_data.assemble_universe([*arc.SEASONS, 2025])


def test_the_walk_forward_buffer_itself_still_runs_offline(monkeypatch, tmp_path):
    """And the one drop that is *not* a bug -- the earliest of the archive's two seasons,
    consumed as training -- must not trip the same refusal."""
    arc.install(monkeypatch, tmp_path, board=True)
    universe = weekly_gate_data.assemble_universe(arc.SEASONS)
    assert sorted(universe.realised) == [max(arc.SEASONS)]
