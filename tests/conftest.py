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
            ours = [f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_stack()
                    if f"{os.sep}hub{os.sep}" in f.filename or "tests" in f.filename]
            reached.append(f"{address} via {' <- '.join(ours[-3:]) or 'outside this repo'}")
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
    yield
    if reached:
        pytest.fail(
            f"this test reached for the network: {reached!r}. tests/unit and tests/contracts "
            f"run offline -- a test that passes because the wire answered proves nothing "
            f"about its fixture, and one that passes because the wire was refused is "
            f"exercising a degradation path it did not mean to. Stub the reach, or move the "
            f"test to tests/golden and mark it `golden`.")
