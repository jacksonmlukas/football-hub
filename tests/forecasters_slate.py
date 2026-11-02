"""A synthetic last-weekend slate for the forecaster day-pipeline tests (#447): an nflverse
schedule frame, an ESPN league injuries payload and an Odds API payload, all invented, so the
dry run touches no network and no `data/`. A top-level module under `tests/` (an import root)
because pyrefly cannot resolve an import from one test module to another.

Sunday 2026-10-18 is the run day: four games due (two at 13:00 ET, one at 16:25, one at
20:20), plus a London game at 09:30 ET that belongs to Saturday's run, a Monday game and a
Thursday game that belong to their own days. Every team played on Sunday 2026-10-11.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from hub.forecasters.sources import kickoff

DAY = date(2026, 10, 18)
RUN_START = datetime(2026, 10, 18, 13, 31, tzinfo=UTC)          # 09:31 ET
SEASON = 2026

NAMES = {"Kansas City Chiefs": "KC", "Buffalo Bills": "BUF", "Baltimore Ravens": "BAL",
         "Cincinnati Bengals": "CIN", "Dallas Cowboys": "DAL", "Philadelphia Eagles": "PHI",
         "Los Angeles Rams": "LA", "Washington Commanders": "WAS", "Jacksonville Jaguars": "JAX",
         "Miami Dolphins": "MIA", "New York Jets": "NYJ", "New England Patriots": "NE",
         "Detroit Lions": "DET", "Green Bay Packers": "GB"}
FULL = {v: k for k, v in NAMES.items()}

# (game_id, week, gameday, gametime, home, away, home_score, away_score, location)
_ROWS: list[tuple[Any, ...]] = [
    ("2026_06_KC_BUF", 6, "2026-10-11", "16:25", "KC", "BUF", 27, 24, "Home"),
    ("2026_06_BAL_CIN", 6, "2026-10-11", "13:00", "BAL", "CIN", 20, 23, "Home"),
    ("2026_06_DAL_PHI", 6, "2026-10-11", "13:00", "DAL", "PHI", 17, 31, "Home"),
    ("2026_06_LA_WAS", 6, "2026-10-11", "20:20", "LA", "WAS", 28, 14, "Home"),
    ("2026_06_JAX_MIA", 6, "2026-10-11", "13:00", "JAX", "MIA", 10, 13, "Home"),
    ("2026_06_NYJ_NE", 6, "2026-10-11", "13:00", "NYJ", "NE", 21, 21, "Home"),
    ("2026_06_DET_GB", 6, "2026-10-11", "16:25", "DET", "GB", 34, 30, "Home"),
    ("2026_07_NE_DET", 7, "2026-10-15", "20:15", "NE", "DET", None, None, "Home"),   # Thursday
    ("2026_07_JAX_NYJ", 7, "2026-10-18", "09:30", "JAX", "NYJ", None, None, "Neutral"),
    ("2026_07_BUF_KC", 7, "2026-10-18", "13:00", "BUF", "KC", None, None, "Home"),
    ("2026_07_CIN_BAL", 7, "2026-10-18", "13:00", "CIN", "BAL", None, None, "Home"),
    ("2026_07_PHI_DAL", 7, "2026-10-18", "16:25", "PHI", "DAL", None, None, "Home"),
    ("2026_07_WAS_LA", 7, "2026-10-18", "20:20", "WAS", "LA", None, None, "Home"),
    ("2026_07_MIA_GB", 7, "2026-10-19", "20:15", "MIA", "GB", None, None, "Home"),   # Monday
]
DUE = ("2026_07_BUF_KC", "2026_07_CIN_BAL", "2026_07_PHI_DAL", "2026_07_WAS_LA")


def schedule_frame() -> pl.DataFrame:
    cols = ("game_id", "week", "gameday", "gametime", "home_team", "away_team", "home_score",
            "away_score", "location")
    rows = [dict(zip(cols, r, strict=True)) for r in _ROWS]
    return pl.DataFrame(
        [{**r, "season": SEASON, "game_type": "REG"} for r in rows],
        schema={"game_id": pl.Utf8, "week": pl.Int32, "gameday": pl.Utf8, "gametime": pl.Utf8,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "home_score": pl.Int32,
                "away_score": pl.Int32, "location": pl.Utf8, "season": pl.Int32,
                "game_type": pl.Utf8})


def injuries_payload(*, dated: datetime = RUN_START - timedelta(hours=14)) -> dict:
    def item(name: str, team: str, status: str, when: datetime) -> dict:
        return {"status": status, "date": when.strftime("%Y-%m-%dT%H:%MZ"),
                "athlete": {"displayName": name, "team": {"abbreviation": team}}}
    return {"injuries": [
        {"displayName": "Kansas City Chiefs",
         "injuries": [item("Travis Kelce", "KC", "Questionable", dated),
                      item("Chris Jones", "KC", "Out", dated - timedelta(hours=30))]},
        {"displayName": "Washington Commanders",
         "injuries": [item("Terry McLaurin", "WSH", "Doubtful", dated)]},
        {"displayName": "Los Angeles Rams",
         "injuries": [item("Puka Nacua", "LAR", "Out", dated)]},
    ]}


def _book(key: str, home: str, away: str, home_price: float, away_price: float,
          spread: float | None) -> dict:
    markets: list[dict] = [{"key": "h2h", "outcomes": [{"name": home, "price": home_price},
                                                      {"name": away, "price": away_price}]}]
    if spread is not None:
        markets.append({"key": "spreads",
                        "outcomes": [{"name": home, "price": 1.91, "point": spread},
                                     {"name": away, "price": 1.91, "point": -spread}]})
    return {"key": key, "title": key, "markets": markets}


# The due games' prices: (home decimal, away decimal, home spread).
PRICES = {"2026_07_BUF_KC": (1.80, 2.10, -1.5), "2026_07_CIN_BAL": (2.40, 1.60, 3.0),
          "2026_07_PHI_DAL": (1.50, 2.70, -4.5), "2026_07_WAS_LA": (2.00, 1.85, 1.0)}


def odds_payload(games: tuple[str, ...] = DUE) -> list[dict]:
    out = []
    for gid in games:
        row = next(r for r in _ROWS if r[0] == gid)
        _, _, gameday, gametime, home, away, *_ = row
        hp, ap, sp = PRICES[gid]
        kick = kickoff(gameday, gametime)
        out.append({"id": f"ev-{gid}", "home_team": FULL[home], "away_team": FULL[away],
                    "commence_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "bookmakers": [_book("a", FULL[home], FULL[away], hp, ap, sp),
                                   _book("b", FULL[home], FULL[away], hp + 0.02, ap - 0.02, sp)]})
    return out
