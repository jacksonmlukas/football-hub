"""Select what nflverse answers, for a test that is not driving the frozen archive.

`tests/panelarchive.py` serves the recorded panel archive. This is the other half: a test that
wants one source to answer with one frame -- or to raise, or to count its calls -- says so here,
and the production loader runs on top of it. It replaces patching the loader's private fetchers
and `nflreadpy`'s own functions, which were two more adapters built out of monkeypatches.

`tests/conftest.py` puts the network adapter back after each test.
"""
from __future__ import annotations

from typing import Any

from hub.fetch import nflverse
from hub.fetch.replay import Replay


def serve(**tables: Any) -> Replay:
    """Make each named source answer with its frame (or its callable), and nothing else.

    A source not named raises `NotRecorded` rather than reaching the wire. Called again in the
    same test it adds to the set already selected, so two sources are two calls.
    """
    current = nflverse.selected()
    if isinstance(current, Replay):
        current.add(**tables)
        return current
    rep = Replay(tables)
    nflverse.select(rep)
    return rep


def worker_load(cache: Any, results: Any) -> None:
    """The body of a worker process that tries to read nflverse, for the guard's control.

    Module-level because a spawned child imports its target by name. It reports the type of
    whatever happened rather than raising, since a child's traceback never reaches the test.
    """
    from hub.fetch import nflverse
    try:
        nflverse.load("schedules", [2025], cache=cache)
    except BaseException as e:      # the child reports; the parent asserts
        results.put(type(e).__name__)
    else:
        results.put("loaded")
