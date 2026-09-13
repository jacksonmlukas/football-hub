"""Every writer of last-good state goes through `hub.atomic` (#258).

`CLAUDE.md` promises last-good state when a fetch fails. A writer that truncates its final
path before it has the bytes turns a mid-write kill into a file the next read refuses or
parses as garbage -- the fallback destroyed by the outage it exists to survive. `hub.atomic`
writes to a scratch name and renames over the target in one call; this file holds the writers
to it.

**Named modules rather than all of `src/hub`**, on the same argument as
`test_sources_are_routed.py`: these are the files a reader serves as last-good -- the store's
partitions, the fetchers' caches and stamps, the site's artifacts, the board, the coverage
verdict. A CLI's `--out` scratch table (`backtest`, `margin`, `injury`, ...) is written for an
operator who asked for it and read by nobody unattended, and a list that claimed the whole
tree would be a list of those exceptions instead. This one only grows: a module joins it
when it is routed and never leaves.

The AST, not a text search: the call this forbids is named in this docstring and in every
module's own comments.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = "atomic"
DIRECT = frozenset({"write_text", "write_bytes", "write_parquet"})

ROUTED = (
    "store.py",
    "publish.py",
    "fetch/bigten.py",
    "fetch/cfbd.py",
    "fetch/espn.py",
    "fetch/nfeloqb.py",
    "fetch/nflverse.py",
    "fetch/odds.py",
    "fetch/pool.py",
    "draft/board.py",
    "models/coverage.py",
)


def _direct_writes(source: str) -> list[int]:
    """Lines calling `.write_text`, `.write_bytes` or `.write_parquet` on anything but the
    helper module itself (`atomic.write_text(...)` is the routed spelling)."""
    return [
        n.lineno for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in DIRECT
        and not (isinstance(n.func.value, ast.Name) and n.func.value.id == HELPER)
    ]


@pytest.mark.parametrize("module", ROUTED)
def test_a_routed_writer_does_not_write_its_final_path_directly(module):
    direct = _direct_writes((ROOT / "src" / "hub" / module).read_text())
    assert not direct, (
        f"src/hub/{module} writes a final path directly at line(s) {direct}. A process "
        f"killed mid-write leaves a truncated file there, and the next read serves garbage "
        f"or refuses the last-good state it exists to serve. Go through `hub.atomic`.")


def test_the_helper_is_the_one_place_that_writes():
    """The helper itself is where the three calls are allowed, and it has to make them --
    a helper that stopped writing would leave every routed module green and inert."""
    found = _direct_writes((ROOT / "src" / "hub" / f"{HELPER}.py").read_text())
    assert len(found) == 3, f"hub.{HELPER} should hold exactly the three writes, found {found}"


def test_the_routed_list_only_grows():
    missing = [m for m in ROUTED if not (ROOT / "src" / "hub" / m).exists()]
    assert not missing, f"the routed list names modules that are gone: {missing}"


def test_the_scan_sees_a_direct_write():
    """The scan against a planted call, so a green run is evidence of routing and not of a
    scanner that matches nothing (the failure `test_guards_are_load_bearing.py` catalogues)."""
    assert _direct_writes("p.write_text(s)\nq.write_parquet(t)\n") == [1, 2]
    assert _direct_writes("atomic.write_text(p, s)\natomic.write_parquet(df, p)\n") == []
