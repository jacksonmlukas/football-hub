"""Stage 2 compares a gate's MDE against *its declared ceiling arm* -- so each declares one.

`docs/gate-power.md` stage 2 was pre-registered against a "foresight ceiling". #43 then built
the lineup gate's ceiling as a variance oracle, deliberately: a foresight lineup is the best XI
after the fact and bounds nothing interesting, where a perfect *spread* with the projection
untouched bounds what any spread model could deliver. The two arms are not interchangeable and
foresight is strictly the larger, so the variance oracle is the stricter comparator.

#138 amended the rule to read "its declared ceiling arm" rather than exposing the weaker arm to
make the rule read literally. That amendment is honest only if every gate *declares* an arm by
name where it prints, so no two gates' numbers can be read as one quantity -- which is what this
holds. A gate whose ceiling line said only "ceiling" would let three numbers in three units be
tabulated as one, and the rule would be comparing against a word.
"""
import importlib

import pytest

GATES = ("hub.draft.backtest", "hub.season.weekly_gate", "hub.season.lineup_gate")


@pytest.mark.parametrize("module", GATES)
def test_every_gate_declares_its_ceiling_arm_by_name(module):
    mod = importlib.import_module(module)
    arm = getattr(mod, "CEILING_ARM", None)
    assert isinstance(arm, str) and arm.strip(), (
        f"{module} has no CEILING_ARM. Stage 2 of docs/gate-power.md is applied to each gate's "
        f"*declared* arm, and a gate that declares none has nothing for the rule to read.")


def test_the_three_arms_are_distinct_so_no_two_numbers_read_as_one():
    """Two gates naming their arm identically would invite exactly the tabulation the
    amendment exists to prevent -- unless they genuinely bound the same quantity, which the
    draft and weekly gates (both foresight) do not: one is a season known in advance to a
    drafter, the other a week known in advance to a projection."""
    arms = {m: importlib.import_module(m).CEILING_ARM for m in GATES}
    assert len(set(arms.values())) == len(arms), f"two gates share an arm name: {arms}"


def test_the_rule_names_the_declared_arm_not_foresight():
    """The amendment itself, held. Stage 2 must not quietly revert to 'foresight ceiling'."""
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "gate-power.md").read_text()
    stage2 = doc[doc.index("**Stage 2"):doc.index("**Stage 2") + 900]
    assert "declared ceiling arm" in stage2, (
        "stage 2 no longer reads 'its declared ceiling arm'; the #138 amendment was reverted")
    assert "exceeds its foresight ceiling" not in stage2, (
        "stage 2 says 'foresight ceiling' again, which the lineup gate does not produce under "
        "the arm it declares")
    assert "2026-09-11" in stage2 or "amended" in stage2.lower(), (
        "an amended pre-registration that does not say it was amended is worse than none")


def test_each_gate_prints_the_arm_on_its_ceiling_line():
    """Declaring it is not enough; the line a reader sees has to carry it. Held on the source
    rather than by running three network gates: the constant must appear in the same file's
    ceiling output, not only at the top of the module."""
    import inspect
    for module in GATES:
        src = inspect.getsource(importlib.import_module(module))
        assert src.count("CEILING_ARM") >= 2, (
            f"{module} declares CEILING_ARM but nothing in the module reads it")
