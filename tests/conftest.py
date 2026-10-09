"""The unit and contract suites do not reach the network, and now they cannot.

`pyproject.toml` sets `addopts = -m 'not golden'` and `tests/golden/` exists precisely so
live APIs are hit in one deliberate place. The implication a reader takes from that -- and
the reason the golden marker exists at all -- was not enforced anywhere.

It was not true, either. Nine command-line entry points driven against a deliberately absent
world exited zero under the full suite and non-zero when run alone. The absent world was
identical in both cases: sockets blocked, the HTTP adapter blocked, credentials deleted,
caches redirected. What differed is that something earlier in the suite had already fetched
for real, leaving the answer in the fetch library's in-process cache and a live connection in
the pool -- so a later test asking for absent data was quietly served present data (#117).

That is worse than a slow suite. A test can pass because of live data rather than because of
its fixture, with nothing distinguishing the two, and everything this repo does to freeze
captures -- the contracts, the panel archive, the scoreboard fixtures -- assumes the suite
reads them rather than the wire. It is also invisible to per-file runs: each of those nine
passes alone, and only the whole-suite ordering exposes it.

**Loopback stays open.** The blocked thing is leaving this machine. A test that binds a local
port to stand in for a service is still testing itself, and refusing that would push tests
toward mocks where a real socket is the better fixture.

**In-process only.** A test that runs a subprocess -- `tests/contracts/test_live_loop.py`
drives a shell script -- gets a fresh interpreter this cannot reach. Those tests block the
network their own way, by handing the child a recorder instead of a deploy command, and the
limit is written here so the next reader does not assume more cover than exists.
"""
from __future__ import annotations

import builtins
import io
import os
import re
import socket
import traceback
from pathlib import Path

import duckdb
import polars as pl
import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


class LiveCallInTheOfflineSuite(RuntimeError):
    """A unit or contract test tried to leave this machine."""


def _is_local(address: object) -> bool:
    # AF_UNIX carries a path, not a pair, and never leaves the machine.
    if isinstance(address, (str, bytes)):
        return True
    if isinstance(address, tuple) and address:
        host = address[0]
        return isinstance(host, str) and (host in _LOOPBACK or host.startswith("127."))
    return False


_THIS_FILE = os.path.abspath(__file__)


def _repo_frames() -> str:
    """The innermost three callers inside this repo, `a.py:1 <- b.py:2 <- c.py:3`.

    What a guard's message says besides the address or path it caught: the address names
    nothing a reader can act on, and the frames say which seam to stub or which default to
    redirect.

    Frames from *this file* are dropped by filename, not by position (#414). This function and
    the guard closure that called it are always on the stack and are never what a reader wants;
    a fixed `[:-2]` would be right today and wrong the day a guard grows a helper between the
    closure and here, and it would go wrong silently -- the message would just name conftest
    again. Nothing this file does is a caller worth naming, so the filter cannot over-trim.
    """
    ours = [f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_stack()
            if os.path.abspath(f.filename) != _THIS_FILE
            and (f"{os.sep}hub{os.sep}" in f.filename or "tests" in f.filename)]
    return " <- ".join(ours[-3:]) or "outside this repo"


@pytest.fixture(autouse=True)
def _the_suite_stays_offline(request, monkeypatch, tmp_path_factory):
    """Every test that is not `golden` runs with the wire cut, and the attempt is recorded.

    **Raising is not enough on its own, and that is the whole design here.** This repo
    degrades on purpose -- `hub.fetch` and every CLI meet an unreachable source with a broad
    `except Exception` and a last-good answer, which is the behaviour CLAUDE.md asks for. So
    a test that reaches the wire and has its socket refused takes the degradation path, passes,
    and says nothing. The refusal would be swallowed by the very handlers this repo is built
    out of, and the guard would look like it worked while measuring nothing.

    So the attempt is recorded and the *test* fails at teardown, where no `except` can reach
    it. A test that genuinely wants to prove its own degradation path stubs the reach, which
    is what the rest of the suite already does.
    """
    if request.node.get_closest_marker("golden"):
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    reached: list[object] = []

    def refuse(real):
        def _blocked(self, address, *a, **k):
            if _is_local(address):
                return real(self, address, *a, **k)
            # The address alone is an IP, which names nothing a reader can act on. The frames
            # inside this repo are what say which seam to stub.
            reached.append(f"{address} via {_repo_frames()}")
            raise LiveCallInTheOfflineSuite(
                f"{request.node.nodeid} opened a socket to {address!r}.")
        return _blocked

    monkeypatch.setattr(socket.socket, "connect", refuse(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse(real_connect_ex))
    # The quarterback ratings' cache (#218) is read by every rating a test builds without
    # naming a cache -- `survivor.grid_from_schedule(2026, base=tmp_path)` is the usual
    # shape -- and its default is the developer's own `data/raw/nfeloqb/`. The day a live
    # pull lands there, every such test would start rating from it and a dozen assertions
    # about the passthrough would move for a reason no test states. So the default points at
    # an empty directory here, for the reason `test_cli_surface.a_fresh_clone` redirects
    # `nflverse.RAW`: a test that wants the state writes it where its own `cache` points.
    from hub.fetch import nfeloqb
    monkeypatch.setattr(nfeloqb, "RAW", tmp_path_factory.mktemp("nfeloqb-default"))
    # A Replay a test selected is process state, and the next test must not inherit it.
    #
    # And the cache a read lands in. `load` writes a parquet and a pin for everything it serves,
    # a Replay included, and the default is the developer's own `data/raw/nflverse/`: a test
    # that read through it would be served -- or would overwrite -- the real archive.
    from hub.fetch import nflverse
    monkeypatch.setattr(nflverse, "RAW", tmp_path_factory.mktemp("nflverse-default"))
    nflverse.select(None)
    yield
    nflverse.select(None)
    if reached:
        pytest.fail(
            f"this test reached for the network: {reached!r}. tests/unit and tests/contracts "
            f"run offline -- a test that passes because the wire answered proves nothing "
            f"about its fixture, and one that passes because the wire was refused is "
            f"exercising a degradation path it did not mean to. Stub the reach, or move the "
            f"test to tests/golden and mark it `golden`.")


# What the real-data guard wraps and what it redirects, as data, so the contract that plants a
# control for each (tests/contracts/test_unit_tests_do_not_touch_real_data.py, #415) is checked
# against the guard itself and not against a second hand-kept list.
#
# `(owner, attribute, how it names a path)`; the contract's key for an arm is
# `f"{owner_name}.{attribute}"` with `owner_name` the module's `__name__` or the class's
# `module.qualname`.
#
# How a call names the path is the one thing that differs, and it is why there are two kinds:
#   "path" -- the first positional argument, or one of `_PATH_KEYWORDS`, *is* the path (or a list
#             of them, or a glob). `os.stat`, every polars reader, `duckdb.connect`, and duckdb's
#             `read_parquet`/`read_csv`/`read_json` all take it that way.
#   "sql"  -- the argument is a query. The path is a quoted literal inside it, `FROM
#             read_parquet('<path>')` or `FROM '<path>'`, which is how `hub.store` reads.
_POLARS_READERS = ("read_parquet", "scan_parquet", "read_csv", "scan_csv", "read_ndjson",
                   "scan_ndjson", "read_json", "read_ipc", "scan_ipc")
_DUCKDB_CALLS = (("sql", "sql"), ("query", "sql"), ("execute", "sql"),
                 ("read_parquet", "path"), ("read_csv", "path"), ("read_json", "path"))
_WRAPPED = (
    *((m, n, "path") for m, n in ((os, "stat"), (os, "lstat"), (os, "scandir"), (os, "listdir"),
                                  (os, "open"), (builtins, "open"), (io, "open"))),
    *((pl, n, "path") for n in _POLARS_READERS),
    (duckdb, "connect", "path"),
    *((duckdb, n, kind) for n, kind in _DUCKDB_CALLS),
    # The same calls on a connection object, which is how `hub.store` and every `con.execute`
    # reaches them. pybind11 classes accept attribute assignment, so these are wrapped in place.
    *((duckdb.DuckDBPyConnection, n, kind) for n, kind in _DUCKDB_CALLS),
)
_PATH_KEYWORDS = ("source", "path", "file", "database", "path_or_buffer", "file_name", "name")


def _owner_name(owner: object) -> str:
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__qualname__}"
    return owner.__name__  # type: ignore[attr-defined]


WRAPPED_CALLS = tuple(f"{_owner_name(o)}.{name}" for o, name, _ in _WRAPPED)

# A quoted literal that holds a path separator. The separator is what keeps `WHERE pos = 'data'`
# from resolving, against a cwd of the repo root, to the data directory itself.
_QUOTED = re.compile(r"'((?:[^']|'')*/(?:[^']|'')*)'")

# The defaults a test reaches when it names no path, pointed at a fresh clone's answer: nothing
# there. `module.attribute` -> where under the absent tmp root it now points. One redirect rather
# than a `tmp_path` threaded through every test that falls through to a default -- these are module
# constants bound at import (`from hub.paths import ROSTER_PARQUET`, `from hub.fetch.nflverse
# import RAW`), so the `nflverse.RAW` redirect above never reached the copies `inspect` took.
# Absent rather than empty-but-present, because absent is what CI and every fresh worktree see,
# and the `if not path.exists()` branches are the ones that must stay covered everywhere. A test
# that wants the state writes it where its own argument points.
#
# `store.DATA` is deliberately *not* redirected: `test_names` and `test_board_main` assert the
# production value of that constant, and a redirect would make them assert the redirect. Tests
# that reach the store's default name a base or a week instead.
REDIRECTED_DEFAULTS = {
    "hub.publish.ROSTER_PARQUET": ("processed", "roster.parquet"),
    "hub.inspect.DATA": ("processed",),
    "hub.inspect.RAW": ("raw", "nflverse"),
    "hub.draft.state.STATE": ("processed", "draft_state.json"),
}


class RealDataTouched(RuntimeError):
    """A unit or contract test reached under the repo's own `data/`."""


@pytest.fixture(autouse=True)
def _the_suite_never_touches_the_real_data_dir(request, monkeypatch, tmp_path_factory):
    """A test's result, and its coverage, must not depend on the developer's `data/` (#410).

    `publish`'s `if not src.exists()` on `ROSTER_PARQUET` and `inspect._available`'s `if not
    base.exists()` were covered on a fresh clone and uncovered on a maintainer checkout, because
    some tests resolved the real default path rather than a tmp one. The coverage ratchet then
    disagreed with CI. It is also CLAUDE.md rule 1: a test that opens a real data file is the
    thing that must never happen.

    So the calls in `_WRAPPED` are wrapped, and any whose path lies under `hub.paths.DATA` is
    **recorded, and the test fails at teardown** -- recorded and not only raised, for the reason
    the network guard above gives: this repo's broad `except Exception` handlers swallow a raise
    and the test passes. It fires whether or not `data/` exists, because the *path* is the leak,
    not the file.

    **Covered (#412).** Python's own file entry points (`os.stat`/`lstat`/`scandir`/`listdir`/
    `open`, `builtins.open`, `io.open`), and the native readers, which open files without
    Python and were once caught only if something stats the path first: polars `read_`/`scan_`
    `parquet`, `csv`, `ndjson`, `ipc` and `read_json`; `duckdb.connect(<file>)`; and duckdb's
    `sql`/`query`/`execute`/`read_parquet`/`read_csv`/`read_json`, as module functions and as
    methods of a connection. A path may be a `str`, a `Path`, a list of them, or a glob (read by
    its literal prefix, so `data/processed/**/*.parquet` is caught). In a duckdb query the path
    is found as a quoted literal containing a `/`, which is how `hub.store` and `FROM 'x'`
    write it.

    **Not covered**, and no cover is implied: (1) SQL that builds the path at runtime *inside*
    duckdb, or reads it from a table, so it never appears as a literal in the string passed in;
    (2) a glob whose wildcard sits above `data/` (`<repo>/*/x.parquet`), and a relative path
    resolved against a cwd other than the repo root only to the extent `abspath` resolves it;
    (3) readers nobody here calls -- polars `read_excel`/`read_avro`/`read_delta`/`scan_iceberg`/
    `scan_pyarrow_dataset`/`read_database`, duckdb's `from_parquet`/`from_csv_auto`/`executemany`,
    pyarrow and pandas; (4) a reference taken before the guard installs (`from polars import
    read_parquet` at import time), which `monkeypatch` cannot reach; (5) anything run in a
    subprocess. A new native reader goes in `_WRAPPED` and gets an arm in the contract, which
    `test_every_wrapped_entry_point_has_a_planted_arm` enforces.

    Fix a hit with `tmp_path`, or redirect the default as `nflverse.RAW` is redirected above.
    """
    if request.node.get_closest_marker("golden"):
        yield
        return
    from hub.paths import DATA
    root = os.fspath(DATA)
    touched: list[str] = []
    nowhere = tmp_path_factory.mktemp("no-data")
    for target, parts in REDIRECTED_DEFAULTS.items():
        monkeypatch.setattr(target, nowhere.joinpath(*parts))

    def under(path: object) -> bool:
        if isinstance(path, (int, bool)):
            return False
        # A list of paths, as `pl.scan_parquet([...])` takes. A glob needs no case of its own: it
        # is a path whose literal prefix is what the `startswith` below reads.
        if isinstance(path, (list, tuple)):
            return any(under(p) for p in path)
        try:
            text = os.fsdecode(os.fspath(path))  # type: ignore[arg-type]
        except TypeError:
            return False  # a buffer, a file object, None
        if "://" in text:
            return False  # a remote URL is not this machine's data/
        p = os.path.abspath(text)
        return p == root or p.startswith(root + os.sep)

    def named(a: tuple, k: dict) -> list[object]:
        return [*a[:1], *(k[key] for key in _PATH_KEYWORDS if key in k)]

    def sql_paths(a: tuple, k: dict) -> list[object]:
        query = a[0] if a else k.get("query", k.get("sql_query"))
        if not isinstance(query, str) or "/" not in query:
            return []
        return [m.replace("''", "'") for m in _QUOTED.findall(query)]

    def watch(real, name, kind):
        reach = named if kind == "path" else sql_paths

        def _watched(*a, **k):
            # A method is called with its receiver first (`con.execute(sql)`), a function is not.
            args = a[1:] if a and isinstance(a[0], duckdb.DuckDBPyConnection) else a
            for path in reach(args, k):
                if under(path):
                    # The path alone names a file, not the default that led there; the frames
                    # inside this repo say which constant to redirect.
                    touched.append(f"{name}({path!r}) via {_repo_frames()}")
                    raise RealDataTouched(f"{request.node.nodeid} reached {path!r} via {name}.")
            return real(*a, **k)
        return _watched

    for owner, name, kind in _WRAPPED:
        monkeypatch.setattr(owner, name,
                            watch(getattr(owner, name), f"{_owner_name(owner)}.{name}", kind))
    yield
    if touched:
        pytest.fail(
            f"this test touched the repo's real data/ directory: {sorted(set(touched))!r}. "
            f"A result that depends on the developer's `data/` is not the same result on CI. "
            f"Use `tmp_path`, or redirect the default the way `nflverse.RAW` is in tests/conftest.py.")


@pytest.fixture
def run():
    """A clean scope of nflverse reads, which the test holds.

    `with reads_of_one_run() as reads:` for a whole test. It replaces the tests that rebound the
    recorder's module global to get an empty slate -- a test depending on how `hub.fetch.nflverse`
    happens to store its state -- with the one door the module offers.
    """
    from hub.fetch import nflverse
    with nflverse.reads_of_one_run() as reads:
        yield reads


@pytest.fixture
def record_holdout_set():
    """Write one season's hold-out set the way `load` now insists on: the given values, and
    every other held-out key recorded as not refitted with a reason (#320). Fitting scripts
    record a key at a time, so tests that need a *loadable* set go through this."""
    from hub import holdout

    def record(season: int, values: dict | None = None, root: Path | None = None) -> Path:
        values = values or {}
        path = None
        for key in holdout.HELD_OUT:
            if key in values:
                path = holdout.record(season, key, values[key], command="x", root=root)
            else:
                path = holdout.record(season, key, None, command="x", root=root,
                                      why_not="not refitted in this fixture")
        assert path is not None
        return path
    return record
