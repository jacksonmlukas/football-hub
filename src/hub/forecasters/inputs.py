"""What the packet builder is handed: one object per source, each carrying the timestamp the
as-of rule is checked against, apart from when it was fetched.

**Why fetch time is a field and never the as-of time.** The pre-registration has every field
fetched *during* the run, so its fetch time is after `run_start` by construction and proves
nothing about whether the content is. Each source therefore carries the best timestamp it
actually has for its own content -- a report's publication time, a file's update time, a game's
completion time -- plus a *basis* string saying what that timestamp is, so a weak one is on the
record beside the number it qualified. `None` is a legitimate value and means "this source gave
no usable time"; the packet refuses it rather than substituting the fetch time.

Plain frozen dataclasses with `to_dict`/`from_dict` for the three that are served last-good
(`hub.forecasters.last_good`): the as-of timestamps survive the round trip, so a cached source
is judged on its own content time, not on when it was cached.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


@dataclass(frozen=True)
class SourceStamp:
    """Where a source came from: its name, when this run fetched it, and whether the bytes
    were live or the last-good copy served after a failed fetch."""
    source: str
    fetched_at: datetime
    served_from: str = "live"                     # "live" | "last_good"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "fetched_at": _iso(self.fetched_at),
                "served_from": self.served_from}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        fetched = _dt(d["fetched_at"])
        assert fetched is not None
        return cls(d["source"], fetched, d.get("served_from", "live"))


@dataclass(frozen=True)
class GameInfo:
    """The game being forecast."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    neutral_site: bool
    kickoff: datetime


@dataclass(frozen=True)
class PlayedGame:
    """One schedule row: its result if there is one, and whether the source calls it final.

    `completed_at` is when the game ended if the source says; neither nflverse's schedule nor
    ESPN's scoreboard does, so it is usually `None` and the packet then bounds it by kickoff
    plus the longest game it will credit.
    """
    game_id: str
    season: int
    game_type: str
    kickoff: datetime
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    final: bool
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"game_id": self.game_id, "season": self.season, "game_type": self.game_type,
                "kickoff": _iso(self.kickoff), "home_team": self.home_team,
                "away_team": self.away_team, "home_score": self.home_score,
                "away_score": self.away_score, "final": self.final,
                "completed_at": _iso(self.completed_at)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        kickoff = _dt(d["kickoff"])
        assert kickoff is not None
        return cls(d["game_id"], d["season"], d["game_type"], kickoff, d["home_team"],
                   d["away_team"], d["home_score"], d["away_score"], d["final"],
                   _dt(d["completed_at"]))


@dataclass(frozen=True)
class Schedule:
    """Every game of the season so far, as one source listed them."""
    stamp: SourceStamp
    games: tuple[PlayedGame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"stamp": self.stamp.to_dict(),
                "games": [g.to_dict() for g in self.games]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(SourceStamp.from_dict(d["stamp"]),
                   tuple(PlayedGame.from_dict(g) for g in d["games"]))


@dataclass(frozen=True)
class QbSource:
    """Probable starters by team, and when the source last updated them.

    `updated_at_basis` says what `updated_at` is (e.g. the committer date of the commit that
    last changed the file), because "update time" is a different claim for every source.
    """
    stamp: SourceStamp
    updated_at: datetime | None
    updated_at_basis: str
    starters: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stamp": self.stamp.to_dict(), "updated_at": _iso(self.updated_at),
                "updated_at_basis": self.updated_at_basis, "starters": dict(self.starters)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(SourceStamp.from_dict(d["stamp"]), _dt(d["updated_at"]),
                   d["updated_at_basis"], dict(d["starters"]))


@dataclass(frozen=True)
class InjuryEntry:
    team: str
    player: str
    status: str


@dataclass(frozen=True)
class InjuryReport:
    """One injury report and its own publication time (`None` when the source has none)."""
    stamp: SourceStamp
    published_at: datetime | None
    published_at_basis: str
    entries: tuple[InjuryEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"stamp": self.stamp.to_dict(), "published_at": _iso(self.published_at),
                "published_at_basis": self.published_at_basis,
                "entries": [[e.team, e.player, e.status] for e in self.entries]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(SourceStamp.from_dict(d["stamp"]), _dt(d["published_at"]),
                   d["published_at_basis"], tuple(InjuryEntry(*e) for e in d["entries"]))


@dataclass(frozen=True)
class MarketSnapshot:
    """This run's market for the game, as `market.py` (item 2) will hand it over: the de-vigged
    home win probability and the home spread, tagged with the snapshot they came from."""
    snapshot_id: str
    snapshot_time: datetime
    source: str
    fetched_at: datetime
    home_win_prob: float
    home_spread: float
