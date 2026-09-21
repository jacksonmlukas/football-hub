"""`scripts/rule16_combined_power.py` produced ADR-0019's rule-16 table, and then stopped
reproducing it: #363 landed behind it in the same lane and made every trial NOT-RUNNABLE for
want of a ceiling, so the harness returned an ADOPT rate of 0 whatever the inputs. Two lanes
found it independently on 2026-09-21. This is the check that catches that shape -- a harness
whose outcome cannot vary with the rule it is about -- and it is written the way rule 15 asks:
plant the case the harness must adopt on, and watch it adopt."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rule16_combined_power.py"


def _load():
    spec = importlib.util.spec_from_file_location("rule16_combined_power", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_harness_adopts_on_an_effect_it_cannot_miss():
    """An effect a hundred standard errors wide, with every season a confident win, must
    ADOPT on essentially every trial. A rate of zero here is not a strict rule; it is a
    harness whose ADOPT branch has been made unreachable by something ahead of it."""
    mod = _load()
    rate = mod._adopt_rate(k=4, s=1.0, m=20, delta=100.0, trials=20, seed=1)
    assert rate > 0.9, rate


def test_the_harness_does_not_adopt_under_the_null_more_than_the_rule_allows():
    """The other side: under the null the rate is small. Not a size test -- 60 trials is
    not one -- but the pair together shows the outcome varies with delta, which is the
    property the docstring is about."""
    mod = _load()
    rate = mod._adopt_rate(k=4, s=1.0, m=20, delta=0.0, trials=60, seed=2)
    assert rate < 0.2, rate
