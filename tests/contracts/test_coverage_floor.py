"""Every module has a committed coverage floor, so none is silently exempt.

The floor itself is enforced by `scripts/coverage_ratchet.py` after the suite, because a test
cannot read the coverage of the run it is part of. What a test *can* hold is the list: a
module that exists and has no entry would otherwise be invisible to the ratchet, which
compares only what it finds in both places.

That is the same gap in miniature that `tests/coveragefloor.py` exists to close. A gate that
silently covers fewer modules than it did yesterday is the failure mode this repo keeps
finding, so the membership is asserted here rather than left to the script's own bookkeeping.
"""
from __future__ import annotations

from pathlib import Path

from coveragefloor import FLOOR

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"


def _modules() -> set[str]:
    """Every module coverage would report on, named the way it names them."""
    return {str(p.relative_to(ROOT)) for p in SRC.rglob("*.py")}


def test_every_module_has_a_committed_floor():
    missing = sorted(_modules() - set(FLOOR))
    assert not missing, (
        f"these modules have no committed coverage floor, so the ratchet cannot see them "
        f"getting worse: {missing}. Run "
        f"`uv run pytest tests/unit tests/contracts --cov=hub --cov-report=json:cov.json` "
        f"then `uv run python scripts/coverage_ratchet.py cov.json --update`.")


def test_the_floor_names_no_module_that_is_gone():
    """A deleted module leaving its entry behind is how a floor drifts into fiction -- the
    ratchet would keep comparing against a file nobody can make worse."""
    stale = sorted(set(FLOOR) - _modules())
    assert not stale, (
        f"the floor names modules that no longer exist: {stale}. Re-run with --update.")


def test_a_floor_is_a_count_of_statements_not_a_percentage():
    """Guards the unit, because the two are indistinguishable at a glance and the script
    compares with `>`. A percentage read as a count would invert the gate for every module
    above 50% -- it would demand coverage *fall*."""
    assert all(isinstance(v, int) and v >= 0 for v in FLOOR.values()), (
        "a floor is a count of statements a module may leave untested")
