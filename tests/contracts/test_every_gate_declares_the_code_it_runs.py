"""Every gate's `Harness` names the code its arms run, so the Ledger can see a code change (#435).

`config_digest` hashes config and fitted constants, not code: #361 changed the injury retention
baseline and the new interval was compared with the old, because the key had no term for it.
`Harness.arm_modules` is that term, declared beside the arms. A harness that declares none would
record `code_digest=None` -- "no modules declared" -- and compare with other such runs forever,
which is the defect again with the gap moved. So this holds every harness found by
`gate_harnesses.all_harnesses()` (the walk, not a list) to three things: it declares modules,
it declares the module it lives in, and every name resolves to a source file `code_digest` can
hash.

Rule 18: `_check` is run against a planted harness with no modules, one that omits its own
module, and one naming a module that does not exist -- each must fail with its own message.
"""
import pytest
from gate_harnesses import all_harnesses

from hub.ledger import code_digest
from hub.models.experiment import Actions, Harness

_ACTIONS = Actions(adopt="A", remove="R", show="S")


def _check(key: str, harness: Harness) -> None:
    home = key.rsplit(".", 1)[0]
    assert harness.arm_modules, f"{key} declares no arm_modules, so its ledger key cannot see code"
    assert home in harness.arm_modules, (
        f"{key} does not name its own module {home!r} in arm_modules {harness.arm_modules}")
    code_digest(harness.arm_modules)  # raises, naming the module, if one does not resolve


@pytest.mark.parametrize("key", sorted(all_harnesses()))
def test_each_harness_declares_the_modules_its_arms_run(key):
    _check(key, all_harnesses()[key])


def _planted(*modules: str) -> Harness:
    return Harness(name="planted", arm_a="a", arm_b="b", within=("x",), ceiling_arm="an arm",
                   actions=_ACTIONS, arm_modules=modules)


def test_a_planted_harness_with_no_modules_fails():
    with pytest.raises(AssertionError, match="declares no arm_modules"):
        _check("hub.models.planted.PLANTED", _planted())


def test_a_planted_harness_that_omits_its_own_module_fails():
    with pytest.raises(AssertionError, match="does not name its own module"):
        _check("hub.models.planted.PLANTED", _planted("hub.models.market"))


def test_a_planted_harness_naming_a_module_that_does_not_exist_fails():
    with pytest.raises(ValueError, match="not a module with a source file"):
        _check("hub.models.planted.PLANTED",
               _planted("hub.models.planted", "hub.models.nothing_here"))
