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

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "test_planted.py").write_text(textwrap.dedent(body))
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "test_planted.py"), "-q",
         "-c", str(tmp_path / "pytest.ini"), "--rootdir", str(tmp_path),
         "-p", "tests.conftest", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_a_test_that_stats_a_real_data_path_fails(tmp_path):
    got = _run(tmp_path, """
        from hub.paths import ROSTER_PARQUET

        def test_leaks():
            # The shape of the original offender: a swallowed `.exists()` on a real default.
            ROSTER_PARQUET.exists()
        """)
    assert got.returncode != 0, got.stdout
    assert "touched the repo's real data/ directory" in got.stdout, got.stdout
    assert "roster.parquet" in got.stdout


def test_a_test_that_stays_in_its_tmp_path_passes(tmp_path):
    got = _run(tmp_path, """
        def test_clean(tmp_path):
            (tmp_path / "roster.parquet").exists()
        """)
    assert got.returncode == 0, got.stdout


# One planted leak per wrapped entry point (#411). Dropping a wrapper from the guard's tuple
# turned nothing red before: `Path.exists()` above only ever reached `os.stat`. Each plant swallows
# the guard's raise, as this repo's broad `except Exception` handlers do, so the only thing that
# can turn it red is the recorded teardown failure. The path does not exist, on purpose.
_ARMS = {
    "os.lstat": "os.lstat(leak)",
    "os.scandir": "os.scandir(leak)",
    "os.listdir": "os.listdir(leak)",
    "os.open": "os.open(leak, os.O_RDONLY)",
    "builtins.open": "builtins.open(leak)",
    "io.open": "io.open(leak)",
}


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
# real `data/` and goes red -- seen by mutation, recorded in the commit that added this.
_DEFAULTS = {
    "publish.ROSTER_PARQUET": ("from hub import publish as m", "m.ROSTER_PARQUET"),
    "hub.inspect.DATA": ("from hub import inspect as m", "m.DATA"),
    "hub.inspect.RAW": ("from hub import inspect as m", "m.RAW"),
    "hub.draft.state.STATE": ("from hub.draft import state as m", "m.STATE"),
}


@pytest.mark.parametrize("default", _DEFAULTS)
def test_a_module_default_is_redirected_away_from_real_data(tmp_path, default):
    imp, attr = _DEFAULTS[default]
    got = _run(tmp_path, f"""
        def test_uses_the_default():
            {imp}
            {attr}.exists()
        """)
    assert got.returncode == 0, got.stdout
