"""Serve the last good copy of a packet source when its fetch fails (CLAUDE.md, "Graceful
degradation": a fetch that fails serves last-good state from `data/processed/` rather than
erroring).

**Last-good is not an exemption from the as-of rule.** What is cached is the source object
with its *content* timestamps -- a report's publication time, a file's update time, a game's
completion time -- not just its bytes, so a cached copy is judged by `build_packet` on when the
source said it was written. Stale-but-valid passes; a copy that breaks a check is refused
exactly as a live one would be. The copy comes back marked `served_from="last_good"` with its
*original* fetch time, and the packet records both.

Pure of the network and of `data/` by construction: the caller passes the fetch and the root
directory (`DATA / "processed" / "forecasters"` in production, a temp dir in tests).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar

from hub import atomic
from hub.forecasters.inputs import SourceStamp


class Stored(Protocol):
    @property
    def stamp(self) -> SourceStamp: ...

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self: ...


T = TypeVar("T", bound=Stored)


class NoLastGood(Exception):
    """The fetch failed and there is no readable last-good copy to serve."""


@dataclass(frozen=True)
class Served:
    """What `serve` hands back: the source, and whether it is a degraded last-good copy."""
    value: Any
    degraded: bool
    why: str | None = None


def serve(name: str, fetch: Callable[[], T], cls: type[T], root: Path) -> Served:
    """The live fetch, remembered under `root`; or, if it raises, the last copy remembered.

    Raises `NoLastGood` (naming the failure) with nothing readable to fall back on. A cache
    that will not parse is no cache -- never a traceback, and never a guess.
    """
    path = root / f"{name}.json"
    try:
        value = fetch()
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
        try:
            old = cls.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as read_exc:
            raise NoLastGood(f"{name}: fetch failed ({why}) and there is no readable "
                             f"last-good copy at {path} ({type(read_exc).__name__})") from exc
        marked = replace(old, stamp=replace(old.stamp, served_from="last_good"))  # pyrefly: ignore
        return Served(marked, True, why)
    root.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, json.dumps(value.to_dict(), indent=2, sort_keys=True))
    return Served(value, False)
