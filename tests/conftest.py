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
import socket
import traceback
from pathlib import Path

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


def _repo_frames() -> str:
    """The innermost three frames inside this repo, `a.py:1 <- b.py:2 <- c.py:3`.

    What a guard's message says besides the address or path it caught: the address names
    nothing a reader can act on, and the frames say which seam to stub or which default to
    redirect.
    """
    ours = [f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_stack()
            if f"{os.sep}hub{os.sep}" in f.filename or "tests" in f.filename]
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

    So `stat`, `scandir`, `listdir`, `open` and `os.open` are wrapped, and any call whose path
    lies under `hub.paths.DATA` is **recorded, and the test fails at teardown** -- recorded and
    not only raised, for the reason the network guard above gives: this repo's broad `except
    Exception` handlers swallow a raise and the test passes. It fires whether or not `data/`
    exists, because the *path* is the leak, not the file.

    Fix a hit with `tmp_path`, or redirect the default as `nflverse.RAW` is redirected above.
    **Not covered:** native readers (polars, duckdb) open files without Python, so a real path
    handed straight to one is caught only if something stats it first, as every `.exists()`
    guard in `src/` does.
    """
    if request.node.get_closest_marker("golden"):
        yield
        return
    from hub.paths import DATA
    root = os.fspath(DATA)
    touched: list[str] = []
    # The defaults a test reaches when it names no path, pointed at a fresh clone's answer:
    # nothing there. One redirect here rather than a `tmp_path` threaded through every test
    # that falls through to a default -- these are module constants bound at import
    # (`from hub.paths import ROSTER_PARQUET`, `from hub.fetch.nflverse import RAW`), so the
    # `nflverse.RAW` redirect above never reached the copies `inspect` took. Absent rather than
    # empty-but-present, because absent is what CI and every fresh worktree see, and the
    # `if not path.exists()` branches are the ones that must stay covered everywhere. A test that
    # wants the state writes it where its own argument points.
    #
    # `store.DATA` is deliberately *not* redirected: `test_names` and `test_board_main` assert the
    # production value of that constant, and a redirect would make them assert the redirect.
    # Tests that reach the store's default name a base or a week instead.
    from hub import inspect as hub_inspect
    from hub import publish
    from hub.draft import state as draft_state
    nowhere = tmp_path_factory.mktemp("no-data")
    monkeypatch.setattr(publish, "ROSTER_PARQUET", nowhere / "processed" / "roster.parquet")
    monkeypatch.setattr(hub_inspect, "DATA", nowhere / "processed")
    monkeypatch.setattr(hub_inspect, "RAW", nowhere / "raw" / "nflverse")
    monkeypatch.setattr(draft_state, "STATE", nowhere / "processed" / "draft_state.json")

    def under(path: object) -> bool:
        if isinstance(path, int):
            return False
        try:
            p = os.path.abspath(os.fsdecode(os.fspath(path)))  # type: ignore[arg-type]
        except TypeError:
            return False
        return p == root or p.startswith(root + os.sep)

    def watch(real, name):
        def _watched(path, *a, **k):
            if under(path):
                # The path alone names a file, not the default that led there; the frames
                # inside this repo say which constant to redirect.
                touched.append(f"{name}({os.fspath(path)!r}) via {_repo_frames()}")
                raise RealDataTouched(f"{request.node.nodeid} reached {path!r} via {name}.")
            return real(path, *a, **k)
        return _watched

    for mod, name in ((os, "stat"), (os, "lstat"), (os, "scandir"), (os, "listdir"),
                      (os, "open"), (builtins, "open"), (io, "open")):
        monkeypatch.setattr(mod, name, watch(getattr(mod, name), f"{mod.__name__}.{name}"))
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
