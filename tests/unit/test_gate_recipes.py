"""The ledger key carries each gate run's arm (#384), through each real gate's `Harness.run`.

`tests/unit/test_ledger.py` holds the Ledger's own control (same digests, different recipes, not
compared). What it cannot hold is that a *gate* hands the Ledger a recipe at all: `config_digest`
and `data_digest` do not see `--churn`, `--drafts`, `--seed` and the rest, so a gate that passes
no recipe compares two arms at one digest pair -- the spurious REQUIRES REVIEW (or the masked
narrowing) the digest condition was meant to end. These tests run each gate's real `Harness`
twice over one in-memory `Ledger` and assert on the lines the run prints.

All offline; the frames are synthetic.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from gate_harnesses import GATE_MODULES, gate_harnesses

from hub.draft import backtest
from hub.ledger import Ledger, recipe
from hub.models import coverage, starter_change
from hub.season import lineup_gate, weekly_gate

_HARNESSES = gate_harnesses()


def _frame(within: tuple[str, ...], *, scale: float = 1.0) -> pl.DataFrame:
    """A season-effect frame carrying every column a harness's `within` names; `scale`
    widens the spread so a second run's interval is narrower or wider than the first."""
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2022, 2026):
        level = rng.normal(0.0, 2.0 * scale)
        for k in range(6):
            rows.append({"season": season, **dict.fromkeys(within, k),
                         "diff": float(level + rng.normal(0.0, scale))})
    return pl.DataFrame(rows)


def _width_lines(run: Any) -> list[str]:
    return [ln for ln in run.lines if "interval width" in ln or "REQUIRES REVIEW" in ln]


@pytest.fixture(params=sorted(_HARNESSES), ids=sorted(_HARNESSES))
def harness(request):
    h = _HARNESSES[request.param]
    return h._replace(bootstrap=100, ledger=Ledger(path=None))


def test_two_arms_of_one_gate_at_one_digest_pair_are_not_compared(harness):
    """The ticket's case through each real gate: the second run is far narrower than the first
    (a `REQUIRES REVIEW` if compared) and at the same digests -- the frames' stamps are of the
    same config and pins -- but a different arm. It must not compare, and must say why."""
    wide = harness.run(_frame(harness.within, scale=3.0), recipe="arm=a")
    narrow = harness.run(_frame(harness.within, scale=0.1), recipe="arm=b")
    assert not _width_lines(wide), "a first run has no history"
    text = "\n".join(narrow.lines)
    assert "REQUIRES REVIEW" not in text
    assert "1 earlier run(s) of this gate at another recipe" in text


def test_the_same_arm_still_compares_and_the_narrowing_sentence_is_unchanged(harness):
    """The other half: one arm run twice compares, and says what it said before recipes
    existed -- the lines are those of a recipe-less pair (`recipe=None` on both)."""
    def lines(arm: str | None) -> list[str]:
        h = harness._replace(ledger=Ledger(path=None))
        h.run(_frame(harness.within, scale=3.0), recipe=arm)
        return _width_lines(h.run(_frame(harness.within, scale=0.1), recipe=arm))

    with_arm, without = lines("arm=a"), lines(None)
    assert with_arm and "REQUIRES REVIEW" in "\n".join(with_arm)
    assert with_arm == without


def test_a_row_of_no_declared_arm_is_another_recipe_from_a_declared_one(harness):
    """A `recipe: null` row (written between #385 and this change) at the same digests is a
    different arm from a declared one: named as such, not compared."""
    harness.run(_frame(harness.within, scale=3.0), recipe=None)
    got = harness.run(_frame(harness.within, scale=0.1), recipe="arm=a")
    assert "another recipe" in "\n".join(got.lines)
    assert "REQUIRES REVIEW" not in "\n".join(got.lines)


# --- the recipe builder, and each gate's own flags ---------------------------------------


def test_the_recipe_is_one_sorted_spelling():
    assert recipe(b=1, a=True) == recipe(a=True, b=1) == "a=yes,b=1"
    assert recipe(shrink=None, lcb=1.0, seasons=[2022, 2023]) == "lcb=1,seasons=2022+2023,shrink=none"
    assert recipe(s={"b", "a"}) == "s=a+b"
    assert recipe() == ""


_BACKTEST: dict[str, Any] = {"seasons": [2022, 2023], "drafts": 20, "rounds": 3, "draft_sims": 12,
                                 "season_sims": 250, "seed": 0, "holdout": False, "ceiling": False}
_WEEKLY: dict[str, Any] = {"seasons": [2022, 2023], "drafts": 20, "seed": 0, "churn": False,
                               "open_pool": False, "unrestricted": False, "lcb": 0.0, "expected": False,
                               "shrink": None, "ceiling": False, "holdout": False}
_LINEUP: dict[str, Any] = {"seasons": [2022, 2023], "drafts": 20, "seed": 0, "ceiling_arm": None,
                               "parameter_uncertainty": False}

_CASES: list[tuple[str, Callable[..., str], dict[str, Any], str, Any]] = [
    (f"{mod}.{key}", fn, base, key, flip)
    for mod, fn, base, flips in (
        ("backtest", backtest.run_recipe, _BACKTEST,
         {"seasons": [2022], "drafts": 21, "rounds": 4, "draft_sims": 13, "season_sims": 251, "seed": 1,
              "holdout": True, "ceiling": True}),
        ("weekly", weekly_gate.run_recipe, _WEEKLY,
         {"seasons": [2022], "drafts": 21, "seed": 1, "churn": True, "open_pool": True,
              "unrestricted": True, "lcb": 1.0, "expected": True, "shrink": "mae", "ceiling": True,
              "holdout": True}),
        ("lineup", lineup_gate.run_recipe, _LINEUP,
         {"seasons": [2022], "drafts": 21, "seed": 1, "ceiling_arm": "oracle",
              "parameter_uncertainty": True}),
    )
    for key, flip in flips.items()
]


@pytest.mark.parametrize(("label", "fn", "base", "key", "flip"), _CASES,
                         ids=[c[0] for c in _CASES])
def test_every_flag_of_a_gate_that_changes_what_it_measured_is_in_its_recipe(
        label, fn, base, key, flip):
    """Each flag, flipped alone, moves the recipe -- so no flag the gate takes can change the
    arm and keep the key."""
    assert fn(**base) != fn(**(base | {key: flip})), f"{label} is not in the recipe"


def test_the_lineup_ceiling_arm_choice_is_in_the_recipe():
    """`--ceiling-arm`'s two choices are two recipes, and no `--ceiling` is a third."""
    arms = sorted(lineup_gate.CEILING_ARMS)
    got = {lineup_gate.run_recipe(seasons=[2022], drafts=20, seed=0, ceiling_arm=a,
                                  parameter_uncertainty=False) for a in [None, *arms]}
    assert len(got) == len(arms) + 1


def test_the_shape_gate_and_the_quarterback_gate_carry_their_arm():
    """`coverage.shape_law` and `starter_change.run` build their recipe in place; assert it on
    the ledger rows two real runs leave."""
    ledger = Ledger(path=None)
    h = coverage.SHAPE_HARNESS._replace(bootstrap=100, ledger=ledger)
    h.run(_frame(h.within, scale=3.0), recipe=recipe(min_weeks=1, min_prior=1, min_mu=0.0,
                                                     seed=0))
    assert [e.recipe for e in ledger._entries] == ["min_mu=0,min_prior=1,min_weeks=1,seed=0"]
    sc_ledger = Ledger(path=None)
    paired = pl.DataFrame({"season": [2026, 2027, 2028] * 2, "diff": [0.1, 0.2, 0.3] * 2,
                           "ceiling": [0.5] * 6})
    starter_change.run(paired, needed=3, ceiling=False, ledger=sc_ledger)
    assert [e.recipe for e in sc_ledger._entries] == ["ceiling=no"]


# --- no gate's run goes without one ------------------------------------------------------


def _harness_run_calls(module: str) -> list[ast.Call]:
    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run" and isinstance(n.func.value, ast.Name)
            and "harness" in n.func.value.id.lower()]


def test_every_gate_run_names_its_arm():
    """Every `Harness.run` call in a gate module passes `recipe=` -- a gate that dropped it
    would compare two arms at one digest pair again, with every test above still green because
    they hand the recipe in themselves. `hub.models.margin` only `decide`s (no ledger)."""
    calls = {m: _harness_run_calls(m) for m in GATE_MODULES}
    assert sum(len(c) for c in calls.values()) >= 6, "the walk found no gate runs to check"
    for module, found in calls.items():
        for call in found:
            assert any(kw.arg == "recipe" for kw in call.keywords), (
                f"{module}:{call.lineno} runs a Harness with no recipe; two arms of this "
                f"gate at one digest pair would compare (#384)")
