"""Fixtures shared by the forecaster packet tests: one fixed run, one fixed game, and builders
for each source. A top-level module under `tests/` (an import root) because pyrefly cannot
resolve an import from one test module to another."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from hub.forecasters.inputs import (
    GameInfo,
    InjuryEntry,
    InjuryReport,
    MarketSnapshot,
    PlayedGame,
    QbSource,
    Schedule,
    SourceStamp,
)
from hub.forecasters.packet import Condition, Packet, build_packet

RUN_START = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)          # Saturday 10:00 ET
KICKOFF = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)            # Sunday 1pm ET
FETCHED = RUN_START + timedelta(minutes=3)                     # fetched during the run
SNAPSHOT = "snap-2026-10-10T14:02Z"


def _prov(source: str, served_from: str = "live") -> SourceStamp:
    return SourceStamp(source=source, fetched_at=FETCHED, served_from=served_from)


def _game(gid: str, home: str, away: str, hs: int | None, as_: int | None, kickoff: datetime,
          *, final: bool = True, completed_at: datetime | None = None, gtype: str = "REG",
          season: int = 2026) -> PlayedGame:
    return PlayedGame(game_id=gid, season=season, game_type=gtype, kickoff=kickoff,
                      home_team=home, away_team=away, home_score=hs, away_score=as_,
                      final=final, completed_at=completed_at)


def _day(n: int, hour: int = 17) -> datetime:
    """Sundays of 2026: n=1 is Sep 13."""
    return datetime(2026, 9, 13, hour, 0, tzinfo=UTC) + timedelta(days=7 * (n - 1))


def _season_games() -> list[PlayedGame]:
    done = timedelta(hours=3)
    return [
        _game("w1_kc_lv", "KC", "LV", 27, 20, _day(1), completed_at=_day(1) + done),
        _game("w2_buf_kc", "BUF", "KC", 24, 24, _day(2), completed_at=_day(2) + done),    # tie
        _game("w3_kc_den", "KC", "DEN", 10, 17, _day(3), completed_at=_day(3) + done),
        _game("w4_kc_mia", "KC", "MIA", 31, 14, _day(4), completed_at=_day(4) + done),
        _game("w1_buf_nyj", "BUF", "NYJ", 30, 10, _day(1), completed_at=_day(1) + done),
        _game("w3_ne_buf", "NE", "BUF", 7, 21, _day(3), completed_at=_day(3) + done),
        _game("w4_buf_bal", "BUF", "BAL", 20, 23, _day(4), completed_at=_day(4) + done),
    ]


def _schedule(games: list[PlayedGame] | None = None) -> Schedule:
    return Schedule(stamp=_prov("nflverse:schedules"),
                    games=tuple(_season_games() if games is None else games))


def _qb(updated_at: datetime | None = RUN_START - timedelta(hours=20),
        starters: dict[str, str] | None = None) -> QbSource:
    return QbSource(stamp=_prov("nfeloqb:qb_elos.csv"), updated_at=updated_at,
                    updated_at_basis="github commit committer date",
                    starters=dict({"KC": "Patrick Mahomes", "BUF": "Josh Allen"}
                                  if starters is None else starters))


def _injuries(published_at: datetime | None = RUN_START - timedelta(hours=10)) -> InjuryReport:
    return InjuryReport(
        stamp=_prov("espn:injuries"), published_at=published_at,
        published_at_basis="espn item max date",
        entries=(InjuryEntry("KC", "Travis Kelce", "Questionable"),
                 InjuryEntry("KC", "Chris Jones", "Out"),
                 InjuryEntry("KC", "Marquez Valdes-Scantling", "Day-To-Day"),
                 InjuryEntry("BUF", "Von Miller", "Doubtful"),
                 InjuryEntry("BUF", "Dalton Kincaid", "Active")))


def _market(snapshot_id: str = SNAPSHOT) -> MarketSnapshot:
    return MarketSnapshot(snapshot_id=snapshot_id, snapshot_time=RUN_START + timedelta(minutes=2),
                          source="the-odds-api", fetched_at=FETCHED,
                          home_win_prob=0.6125, home_spread=-3.5)


def _info(**kw: Any) -> GameInfo:
    base: dict[str, Any] = {"game_id": "2026_05_BUF_KC", "season": 2026, "week": 5,
                            "home_team": "KC", "away_team": "BUF", "neutral_site": False,
                            "kickoff": KICKOFF}
    return GameInfo(**{**base, **kw})


def _build(condition: Condition = Condition.BLIND, **kw: Any) -> Packet:
    args: dict[str, Any] = {"run_start": RUN_START, "condition": condition,
                            "schedule": _schedule(), "qb_source": _qb(),
                            "injuries": _injuries()}
    if condition is Condition.LINE_AWARE:
        args |= {"market": _market(), "snapshot_id": SNAPSHOT}
    game = kw.pop("game", _info())
    return build_packet(game, **{**args, **kw})
