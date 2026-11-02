"""The packet's remaining source adapters: the schedule and results, the injury report, the
odds-event matcher, and the hub contestant (#447, items 1, 3 and 4).

Each is a pure function of a payload already fetched, plus one network default marked as
such, so the day pipeline can be dry-run end to end on recorded or synthetic payloads.

* **Schedule** -- nflverse's `schedules`, the hub's own source (`hub.fetch.nflverse`). nflverse
  gives `gameday`/`gametime` in Eastern wall-clock time, converted through `America/New_York`
  (never a fixed offset; the season leaves daylight time on 1 November). A game is final when
  both scores are present. Neither nflverse nor ESPN says when a game *ended*, so
  `completed_at` stays `None` and the packet bounds it by kickoff (`packet._completion`).
* **Injuries** -- ESPN's league-wide report, one call for every team (never a loop over
  games). **No injury source has a real publication time** (nflverse: the season file's
  release time only; ESPN: a minute-precision `date` per item). The basis, adopted on #448
  (2026-10-22, item 7) and restated in the prereg: `published_at` is the newest item `date`
  *strictly before run start*, and an item dated at or after run start is dropped rather than
  shown: it may carry a status the run could not have known, and omitting a line is the
  conservative direction --
  the packet can be missing news, never carry news from after it was frozen. Both rules are
  written into `published_at_basis`, so the record says what its timestamp is.
* **The hub contestant** -- `hub.models.ratings`, adopted in #446 (3): today that model is
  `ratings.forecaster()`, the betting-market baseline, which is the spread through a normal (the
  2026-10-16 review's "spread-implied market"). It is driven through its own `fit`/`predict`
  on a one-row slate built from *this run's* snapshot spread, so a replacement behind
  `forecaster()` reaches the study without an edit here; `version` is the commit.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from hub.forecasters.inputs import (
    GameInfo,
    InjuryEntry,
    InjuryReport,
    PlayedGame,
    Schedule,
    SourceStamp,
)
from hub.forecasters.packet import MarketBlock

ET = ZoneInfo("America/New_York")
SCHEDULE_SOURCE = "nflverse:schedules"
INJURY_SOURCE = "espn:site/v2/sports/football/nfl/injuries"
INJURY_TIME_BASIS = ("espn: newest per-item `date` strictly before run start (no report "
                     "publication time exists); items dated at or after run start dropped")
SCHEDULE_COLUMNS = ("game_id", "season", "game_type", "week", "gameday", "gametime",
                    "home_team", "away_team", "home_score", "away_score", "location")

# ESPN's spellings where nflverse's differ (the same two `hub.fetch.nfeloqb` maps).
ESPN_TEAM_ALIASES = {"WSH": "WAS", "LAR": "LA"}


# --- the schedule -------------------------------------------------------------------------

def kickoff(gameday: Any, gametime: Any) -> datetime:
    """nflverse's Eastern date and time as an aware UTC instant (13:00 where no time is set)."""
    wall = f"{gameday} {gametime or '13:00'}"
    return datetime.strptime(wall, "%Y-%m-%d %H:%M").replace(tzinfo=ET).astimezone(UTC)


def _score(v: Any) -> int | None:
    return None if v is None else int(v)


def schedule_from_nflverse(frame: pl.DataFrame, stamp: SourceStamp) -> Schedule:
    games = []
    for r in frame.select(SCHEDULE_COLUMNS).iter_rows(named=True):
        hs, as_ = _score(r["home_score"]), _score(r["away_score"])
        games.append(PlayedGame(game_id=r["game_id"], season=int(r["season"]),
                                game_type=r["game_type"],
                                kickoff=kickoff(r["gameday"], r["gametime"]),
                                home_team=r["home_team"], away_team=r["away_team"],
                                home_score=hs, away_score=as_,
                                final=hs is not None and as_ is not None))
    return Schedule(stamp=stamp, games=tuple(games))


def game_infos(frame: pl.DataFrame) -> dict[str, GameInfo]:
    """Every regular-season game as the packet's `GameInfo`, by game id."""
    out = {}
    for r in frame.select(SCHEDULE_COLUMNS).iter_rows(named=True):
        if r["game_type"] != "REG":
            continue
        out[r["game_id"]] = GameInfo(game_id=r["game_id"], season=int(r["season"]),
                                     week=int(r["week"]), home_team=r["home_team"],
                                     away_team=r["away_team"],
                                     neutral_site=r["location"] == "Neutral",
                                     kickoff=kickoff(r["gameday"], r["gametime"]))
    return out


def fetch_schedule(season: int) -> pl.DataFrame:                    # pragma: no cover - network
    from hub.fetch import nflverse
    return (nflverse.load("schedules", [season], refresh=True)
            .filter(pl.col("season") == season))


# --- the injury report --------------------------------------------------------------------

def _espn_time(raw: Any) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def injuries_from_espn(payload: Mapping[str, Any], *, run_start: datetime,
                       stamp: SourceStamp) -> InjuryReport:
    """ESPN's league injuries payload as one report, as of `run_start` (see the module doc)."""
    entries: list[InjuryEntry] = []
    newest: datetime | None = None
    for team in payload.get("injuries") or []:
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            abbr = (athlete.get("team") or {}).get("abbreviation") or team.get("abbreviation")
            when = _espn_time(item.get("date"))
            if not abbr or not athlete.get("displayName") or when is None or when >= run_start:
                continue
            entries.append(InjuryEntry(ESPN_TEAM_ALIASES.get(abbr, abbr),
                                       athlete["displayName"], str(item.get("status") or "")))
            newest = when if newest is None else max(newest, when)
    return InjuryReport(stamp=stamp, published_at=newest, published_at_basis=INJURY_TIME_BASIS,
                        entries=tuple(entries))


def fetch_espn_injuries() -> dict:                                  # pragma: no cover - network
    from hub.fetch import espn
    return espn._get(f"{espn.LEAGUE_PATHS['nfl']}/injuries")


# --- the odds event -> game id -------------------------------------------------------------

def matcher(games: Mapping[str, GameInfo],
            names: Mapping[str, str]) -> Callable[[Mapping[str, Any]], str | None]:
    """An Odds API event to its game id, by (home, away, Eastern date) -- `names` maps the
    feed's full team names to nflverse's abbreviations."""
    lookup = {(g.home_team, g.away_team, g.kickoff.astimezone(ET).date()): gid
              for gid, g in games.items()}

    def match(ev: Mapping[str, Any]) -> str | None:
        t = _espn_time(ev.get("commence_time"))
        if t is None:
            return None
        return lookup.get((names.get(ev.get("home_team", ""), ""),
                           names.get(ev.get("away_team", ""), ""), t.astimezone(ET).date()))
    return match


def team_names(games: Mapping[str, GameInfo]) -> dict[str, str]:  # pragma: no cover - network
    from hub.fetch import odds
    return odds._team_abbrs({t for g in games.values() for t in (g.home_team, g.away_team)})


# --- the hub contestant -------------------------------------------------------------------

def commit() -> str:
    """The commit this run is on: the workflow's `GITHUB_SHA`, else the checkout's HEAD."""
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class RatingsHub:
    """`runner.HubModel` over `hub.models.ratings.forecaster()`, fed this run's spread.

    The betting market's home spread is negative when the home team is favoured; the hub's slates
    carry `close_spread` the other way round (nflverse's `spread_line`, positive for a home
    favourite), so the sign is flipped once here.
    """

    def __init__(self, season: int, week: int, *, version: str | None = None) -> None:
        from hub.models import ratings
        from hub.models.base import FitSpec
        # Fit through the week before, as `ratings.fit` does: the leakage gap is the same.
        self._model = ratings.forecaster().fit(FitSpec("nfl", season, week - 1))
        self._season, self._week = season, week
        self.version = version or commit()

    def predict(self, game_id: str, market: MarketBlock) -> float | None:
        slate = pl.DataFrame({"game_id": [game_id], "league": ["nfl"], "season": [self._season],
                              "week": [self._week], "close_spread": [-market.home_spread]})
        out = self._model.predict(slate).filter(pl.col("game_id") == game_id)
        return float(out["home_win_prob"][0]) if out.height else None


def season_of(day: date) -> int:
    """The NFL season a date belongs to: a January or February game is last year's season."""
    return day.year if day.month >= 3 else day.year - 1
