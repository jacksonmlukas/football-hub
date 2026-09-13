"""One cached-file fetcher; a source of this shape is an adapter (#255).

`hub.fetch.nfeloqb` and `hub.fetch.pool` were one module written twice. Each pulls one
resource, parses it through a contract, keeps the last thing it read under a cache with a
`captured_at` stamp beside it, and serves that on the next failure with a sentence saying
why; each has the `--refresh / --status / --cache` branch tree, the refuse-the-network-
under-pytest guard, and a `report` that prints the state. Seven functions shared names and
shape, the serve-last-good path differed only in its nouns, and the guard -- a security
property, since the pool fetcher carries a session cookie -- was copied into four modules.

What lives here is everything that is the same: the guard, the stamp, the last-good policy
with its sentence, and the CLI. What a source supplies is an `Adapter`: how it pulls and
parses (`refresh`), how it reads its cache back through its contract (`read`), when that
cache was captured (`captured_at`), how a state is printed (`report`), and how a failure
is named (`describe`). The other fetchers are not this shape -- odds is a metered
append-only Snapshot archive, nflverse a pinned as-of loader, bigten a deadline capture --
and take only the guard from here.

**Degradation**, once. `CLAUDE.md`'s rule is that a failed fetch serves last-good state
rather than erroring. `serve_last_good` is that rule: the failure is said on stderr, the
cache is read back through the contract -- a cache that has drifted from the declared
shape is refused rather than served as the field, beside the failure it would have covered
-- and only a fresh clone with nothing cached exits non-zero. `--status` reads the cache
and nothing else, and refuses a drifted cache the same way; before #255 the pool adapter's
`--status` handed back a traceback there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from hub import atomic
from hub.cli import unavailable
from hub.contracts import ContractViolation

S = TypeVar("S")

# The pytest node running right now, or nothing outside a test. pytest sets this for the
# duration of each test's setup, call and teardown, and nothing else in this repo writes it.
PYTEST_NODE_ENV = "PYTEST_CURRENT_TEST"

# The one suite allowed to reach the network. `tests/golden/` exists to diff a live response
# against the frozen fixture -- it is the only thing in the repo that knows whether a
# contract written from documentation resembles reality -- and it is marked `golden` and
# deselected by default (`addopts = "-m 'not golden'"`, pyproject.toml). Node ids are
# relative to pytest's rootdir, so a run started from inside `tests/` does not match and is
# refused: erring toward refusal is the direction every fetcher errs in.
LIVE_TEST_SUITE = "tests/golden/"


class LiveCallRefused(Exception):
    """A test outside `tests/golden/` reached for the network. Refused before anything
    left the process."""


def refuse_live_call(would: str, *, patch: str, tests: str) -> None:
    """Raise `LiveCallRefused` when a test outside `tests/golden/` is running; do nothing
    otherwise. Called first thing by the one function in each fetcher that touches the
    network.

    Patching the transport per test is the right habit and every test in the four suites
    does it. It is not a guarantee, because the guarantee has to hold for the test nobody
    remembered to patch (#68: three live CFBD calls on every run of the suite for anyone
    holding a key). This is, and it costs one environment read per request. `would` is
    the sentence the refusal opens with -- what the call would have done and what it would
    have cost -- and `patch`/`tests` name the function to patch and the file that shows how.
    """
    # GUARD no-live-call-under-pytest [unit/test_fetch_cached.py]: a test outside
    # tests/golden/ never reaches the network, whichever fetcher it drives
    node = os.environ.get(PYTEST_NODE_ENV, "")
    if node and not node.startswith(LIVE_TEST_SUITE):
        raise LiveCallRefused(
            f"{node.split(' ')[0]} {would}. Patch `{patch}`, the way {tests} does; only "
            f"{LIVE_TEST_SUITE} may reach the network, and it is deselected by default.")
    # /GUARD


# --- the stamp --------------------------------------------------------------------------

def read_stamp(path: Path) -> dict[str, Any]:
    """The JSON record at `path` as a dict, or an empty one with nothing there or nothing
    readable. A stamp that will not parse is no capture time rather than a traceback: the
    fetch it sits beside still has to serve."""
    if not path.exists():
        return {}
    try:
        got = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def write_stamp(path: Path, captured_at: str, **fields: Any) -> Path:
    """The record beside a cached read: `captured_at` first, then whatever the source
    records about the read, landing whole (#258). `captured_at` is already formatted by
    the source, because the two adapters spell it differently -- naive to the second,
    and aware with its offset -- and the stamps are read back byte for byte."""
    return atomic.write_text(path, json.dumps({"captured_at": captured_at, **fields}, indent=2))


# --- the adapter ------------------------------------------------------------------------

@dataclass(frozen=True)
class Adapter(Generic[S]):
    """What one source supplies; everything else is here.

    The nouns are the source's own so the sentences a person reads are the ones they read
    before #255: `what` is what `unavailable` names ("the nfeloqb ratings"), `cached` is
    the cache as a refusal names it ("the cached file"), `kept` is how the last-good
    sentence names it ("the last-good file pulled"). `stamped_after_refresh` is whether
    the report after a live pull carries the capture time, which one adapter's did and
    the other's did not; a third source picks.

    `arguments` and `before` are how a source adds to the branch tree without the tree
    being copied: the pool adapter's `--payload FILE` and `--season` are one argument
    hook and one branch that runs first and returns an exit code when it handled the call.
    """
    prog: str
    description: str
    what: str
    cached: str
    kept: str
    cache_flag: str
    cache_help: str
    refresh_help: str
    status_help: str
    refresh_hint: str
    cache_path: Callable[[Path | None], Path]
    refresh: Callable[[argparse.Namespace], S]
    read: Callable[[Path | None], S | None]
    captured_at: Callable[[Path | None], str | None]
    report: Callable[..., list[str]]
    describe: Callable[[BaseException], str]
    stamped_after_refresh: bool = True
    arguments: Callable[[argparse.ArgumentParser], None] | None = None
    before: Callable[[argparse.Namespace], int | None] | None = None


def serve_last_good(a: Adapter[S], cache: Path | None, why: str) -> int:
    """Print why the refresh could not run, then the last-good state; or say there is none.

    `why` is already the source's sentence (`describe`), scrubbed by whoever raised it. The
    read goes through the contract, so a drifted cache is reported as unavailable beside
    the refresh's own failure rather than served.
    """
    try:
        st = a.read(cache)
    except ContractViolation as exc:
        return unavailable(a.prog, f"{a.what} ({why}; and {a.cached}", exc)
    if st is None:
        return unavailable(a.prog, a.what, RuntimeError(why))
    print(f"{a.prog}: {why}; serving {a.kept} {a.captured_at(cache)}", file=sys.stderr)
    for line in a.report(st, captured=a.captured_at(cache)):
        print(line)
    return 0


def run(a: Adapter[S], argv: Sequence[str] | None = None) -> int:
    """The branch tree: `--refresh` pulls and serves last-good on any failure; `--status`
    (the default) reads the cache and nothing else; the cache flag redirects both."""
    ap = argparse.ArgumentParser(prog=a.prog, description=a.description)
    ap.add_argument("--refresh", action="store_true", help=a.refresh_help)
    ap.add_argument("--status", action="store_true", help=a.status_help)
    if a.arguments is not None:
        a.arguments(ap)
    # `dest` is one name so the tree reads it once; the metavar keeps the flag's own, so
    # `--store STORE` and `--cache CACHE` print as they did.
    ap.add_argument(a.cache_flag, dest="cache", metavar=a.cache_flag.lstrip("-").upper(),
                    type=Path, default=None, help=a.cache_help)
    ns = ap.parse_args(argv)

    if a.before is not None:
        handled = a.before(ns)
        if handled is not None:
            return handled

    if not ns.refresh:
        try:
            st = a.read(ns.cache)
        except ContractViolation as exc:
            return unavailable(a.prog, a.cached, exc)
        if st is None:
            return unavailable(a.prog, a.what, FileNotFoundError(
                f"nothing under {a.cache_path(ns.cache)}; {a.refresh_hint}"))
        for line in a.report(st, captured=a.captured_at(ns.cache)):
            print(line)
        return 0

    try:
        st = a.refresh(ns)
    except Exception as exc:
        # `LiveCallRefused` included: under the default suite the refusal is what the CLI
        # reports as unavailable, which is how `test_cli_surface.py` drives every fetcher
        # on a fresh clone with the network absent, and no bytes have left the process.
        return serve_last_good(a, ns.cache, a.describe(exc))
    captured = a.captured_at(ns.cache) if a.stamped_after_refresh else None
    for line in a.report(st, captured=captured):
        print(line)
    return 0
