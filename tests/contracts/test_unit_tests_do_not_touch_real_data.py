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
