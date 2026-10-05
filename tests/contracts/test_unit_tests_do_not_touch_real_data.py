"""Method rule 18 for the data-dir guard in `tests/conftest.py` (#410): a unit test that touches
a path under the repo's `data/` fails, and that is seen, not assumed.

The guard is an autouse fixture, and a fixture that cannot be seen to fire has not been seen to
work -- the failure it exists to catch is invisible by default, because a fresh clone has no
`data/` and so nothing there ever reads wrong. The guard is therefore planted: two throwaway
tests are run through a child pytest that loads this repo's conftest as a plugin. One stats a
real `data/` path and must come back red naming it; one stays in its `tmp_path` and must come
back green, so the red is the guard and not the harness.

The plant uses a path that does not exist on purpose. The leak is the *path*, not the file, so
the guard has to fire on a checkout with no `data/` as well as on one with.
"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import REDIRECTED_DEFAULTS, WRAPPED_CALLS

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, body: str, where: str = "") -> subprocess.CompletedProcess[str]:
    planted = tmp_path / where / "test_planted.py"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(textwrap.dedent(body))
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(planted), "-q",
         "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path),
         "-p", "tests.conftest", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)


def _callers(out: str) -> list[str]:
    """Every `file:line` a guard's message names after `via`, from a child pytest's output."""
    segments = re.findall(r"via ([\w.]+:\d+(?: <- [\w.]+:\d+)*)", out)
    assert segments, out
    return [frame for seg in segments for frame in seg.split(" <- ")]


def test_the_data_guard_names_its_repo_callers_not_itself(tmp_path):
    # #414: `_repo_frames()` counted its own frame and the guard closure, both in conftest.py,
    # so the three slots held `conftest.py <- conftest.py` and the `hub/` frame saying which
    # default to redirect was gone. Nothing failed: only the message got worse. The plant
    # reaches `inspect._available`, a `hub/` function that stats its argument.
    got = _run(tmp_path, """
        from hub import inspect
        from hub.paths import DATA

        def test_leaks():
            try:
                inspect._available(DATA / "planted_leak")
            except Exception:
                pass
        """)
    assert got.returncode != 0, got.stdout
    frames = _callers(got.stdout)
    assert not [f for f in frames if f.startswith("conftest.py:")], frames
    assert any(f.startswith("inspect.py:") for f in frames), frames


def test_the_network_guard_names_its_repo_callers_not_itself(tmp_path):
    # The same defect in the network guard. The plant lives under a `tests/` directory, which is
    # what the helper's filter takes for this repo's own code; 203.0.113.0/24 is TEST-NET-3 and
    # the guard raises before any packet is sent.
    got = _run(tmp_path, """
        import socket

        def test_reaches():
            try:
                socket.socket().connect(("203.0.113.1", 9))
            except Exception:
                pass
        """, where="tests")
    assert got.returncode != 0, got.stdout
    frames = _callers(got.stdout)
    assert not [f for f in frames if f.startswith("conftest.py:")], frames
    assert any(f.startswith("test_planted.py:") for f in frames), frames


def test_a_test_that_stays_in_its_tmp_path_passes(tmp_path):
    got = _run(tmp_path, """
        def test_clean(tmp_path):
            (tmp_path / "roster.parquet").exists()
        """)
    assert got.returncode == 0, got.stdout


# One planted leak per wrapped entry point (#411). Dropping a wrapper from the guard's tuple
# turned nothing red before: `Path.exists()` only ever reached `os.stat`. Each plant swallows
# the guard's raise, as this repo's broad `except Exception` handlers do, so the only thing that
# can turn it red is the recorded teardown failure. The path does not exist, on purpose. `os.stat`
# is an arm like the rest (#415): it used to be a separate `Path.exists()` test with no swallowing
# `except`, which proved the raise, not the record.
_ARMS = {
    "os.stat": "os.stat(leak)",
    "os.lstat": "os.lstat(leak)",
    "os.scandir": "os.scandir(leak)",
    "os.listdir": "os.listdir(leak)",
    "os.open": "os.open(leak, os.O_RDONLY)",
    "builtins.open": "builtins.open(leak)",
    "io.open": "io.open(leak)",
}


def test_every_wrapped_entry_point_has_a_planted_arm():
    # #415: this table restated by hand what the guard wraps, so a wrapped call added in conftest
    # left the contract green with the new arm unplanted -- the gap #411 existed to close, one
    # level up. The guard's own list is `WRAPPED_CALLS`; the plants must be exactly it.
    assert set(_ARMS) == set(WRAPPED_CALLS), (
        f"unplanted: {sorted(set(WRAPPED_CALLS) - set(_ARMS))}, "
        f"no longer wrapped: {sorted(set(_ARMS) - set(WRAPPED_CALLS))}")


@pytest.mark.parametrize("arm", _ARMS)
def test_every_wrapped_entry_point_fails_a_test_that_reaches_real_data(tmp_path, arm):
    got = _run(tmp_path, f"""
        import builtins, io, os
        from hub.paths import DATA

        def test_leaks():
            leak = DATA / "planted_leak"
            try:
                {_ARMS[arm]}
            except Exception:
                pass
        """)
    assert got.returncode != 0, got.stdout
    assert "touched the repo's real data/ directory" in got.stdout, got.stdout
    assert f"{arm}(" in got.stdout, got.stdout


# One control per redirect (#411). Each constant is a module-level default a test reaches when it
# names no path; the guard points it at an absent tmp path. The planted test uses the default as
# production code does, and must come back green. With the redirect dropped it resolves under the
# real `data/` and goes red -- seen by mutation, recorded in the commit that added this. Keyed by
# `module.attribute`, as the guard's `REDIRECTED_DEFAULTS` is; the value is the plant's import.
_DEFAULTS = {
    "hub.publish.ROSTER_PARQUET": ("from hub import publish as m", "m.ROSTER_PARQUET"),
    "hub.inspect.DATA": ("from hub import inspect as m", "m.DATA"),
    "hub.inspect.RAW": ("from hub import inspect as m", "m.RAW"),
    "hub.draft.state.STATE": ("from hub.draft import state as m", "m.STATE"),
}


def test_every_redirected_default_has_a_planted_control():
    assert set(_DEFAULTS) == set(REDIRECTED_DEFAULTS), (
        f"unplanted: {sorted(set(REDIRECTED_DEFAULTS) - set(_DEFAULTS))}, "
        f"no longer redirected: {sorted(set(_DEFAULTS) - set(REDIRECTED_DEFAULTS))}")


@pytest.mark.parametrize("default", _DEFAULTS)
def test_a_module_default_is_redirected_away_from_real_data(tmp_path, default):
    imp, attr = _DEFAULTS[default]
    got = _run(tmp_path, f"""
        def test_uses_the_default():
            {imp}
            {attr}.exists()
        """)
    assert got.returncode == 0, got.stdout
