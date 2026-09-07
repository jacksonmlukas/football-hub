"""The gates' toolchain must be in the environment uv builds without being asked.

A fresh git worktree has no `.venv`; the first `uv run` creates one. uv installs the default
`[dependency-groups]` on every sync and every run, and installs an extra only when someone
names it — so declaring the toolchain as an extra meant no venv uv built on its own ever
contained it. Four agents in one session each rediscovered that before they could check their
own work.

The whole of #142's fix is which table five package names sit in. That is a one-line revert
that reads like a formatting preference, and `docs/decisions.md` carried a row recommending
exactly that revert until this landed. So the arrangement is asserted rather than trusted.

This reads the declaration, not the environment. Asserting that `ruff` is importable here
would pass for the wrong reason: by the time this file runs, pytest is already running, which
means the environment was built correctly whatever the declaration says.
"""
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Everything `.claude/hooks/tdd_gate.sh` and the `ci.yml` `test` job invoke, plus what the
# suite imports. `pytest-cov` is here because `--cov-fail-under=88` is a gate too.
GATE_TOOLING = {"pytest", "pytest-cov", "ruff", "pyrefly", "hypothesis"}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _names(requirements: list[str]) -> set[str]:
    """`pyrefly>=1.0` -> `pyrefly`. Enough for a table this small."""
    return {r.split(">")[0].split("=")[0].split("[")[0].strip() for r in requirements}


def test_the_toolchain_is_a_default_dependency_group(pyproject):
    group = pyproject.get("dependency-groups", {}).get("dev")
    assert group is not None, (
        "`dev` is not a [dependency-groups] group. As an extra it is absent from every venv "
        "uv builds on its own, which is every fresh worktree — see #142."
    )
    assert GATE_TOOLING <= _names(group), (
        f"the dev group is missing {GATE_TOOLING - _names(group)}; a gate whose tool is not "
        "installed by default is a gate a worktree cannot run"
    )


def test_the_toolchain_is_not_an_extra(pyproject):
    """An extra named `dev` would put uv back to installing it only when asked."""
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    assert "dev" not in extras, (
        "`dev` is an extra again. `uv sync --all-extras` would still install it, so CI stays "
        "green and only worktrees break — which is how this went unnoticed for four tickets."
    )
    # The two real extras stay extras: nothing under src/ or tests/ imports them, so a venv
    # without them still runs every gate, and torch is not worth downloading to lint.
    assert set(extras) == {"bayes", "seq"}, f"unexpected extras: {sorted(extras)}"


def test_the_gate_does_not_tell_anyone_to_sync_an_extra_that_is_gone(tmp_path):
    """`uv sync --extra dev` now exits non-zero. The hook must not hand that out as the fix."""
    hook = (ROOT / ".claude" / "hooks" / "tdd_gate.sh").read_text()
    assert "--extra dev" not in hook, (
        "the hook's recovery hint names an extra that no longer exists; an agent following it "
        "gets an error instead of an environment"
    )
    assert "uv sync" in hook, "the hook should still say how to rebuild a broken environment"
