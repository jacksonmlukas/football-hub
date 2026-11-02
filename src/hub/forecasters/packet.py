"""The information packet one game's LLM forecasters see, built and validated as of a run start
(docs/prereg/llm-forecasters.md, "Information packet").

**One packet per game and condition, frozen, canonically serialised and hashed.** `blind` has no
market fields; `line_aware` adds the de-vigged home win probability and spread of *this run's*
snapshot. The packet is stored verbatim; `packet_hash` is the sha256 of `canonical_json`, which
sorts keys, writes floats to six places and every instant as UTC with microseconds, so the same
inputs hash the same in any interpreter and any field change moves the hash.

**The as-of rule is four content checks, not one fetch-time check.** Every field is fetched
during the run, so fetch time is after `run_start` by construction and certifies nothing. Each
check below is against the content's own time, and each is a strict `<` (a source stamped
exactly `run_start` could contain anything from that instant):

1. `records_as_of` / `recent_results_as_of` -- only games whose *completion* precedes
   `run_start` feed records, differentials, rest and recent results. A game still being played,
   or finished after `run_start` but already visible in the feed, is excluded by the builder.
   Neither nflverse nor ESPN says when a game ended, so absent a `completed_at` the builder
   assumes it ended no earlier than kickoff plus `_MAX_GAME_DURATION` and credits it only
   after that -- conservative: a finished game can be left out for a few hours, never let in
   early.
2. `injury_published` -- the report's own publication time precedes `run_start`.
3. `qb_source_updated` -- the starting-QB source's update time precedes `run_start`.
4. `market_snapshot` -- the betting-market fields' `snapshot_id` equals this run's.

A source with *no* usable time (`None`) fails its check; it is never replaced by fetch time.

**Two doors for each check.** The builder applies them while assembling, and `validate_packet`
re-applies them to any finished `Packet`, so a stored or hand-edited packet cannot skip them.
`build_packet` finishes by calling the validator and refuses with every failure at once.
A refused packet must not be forecast (`PacketRefused`).

**The data contract.** `PACKET_CONTRACT` is a `hub.contracts.Contract` over the packet's flat
scalar fields (`flat_frame`), so plausible-but-wrong numbers -- a percent where a proportion
belongs, a rest of 99 days -- are refused in the same words every other source's are. It lives
here, not in `hub.contracts`, because that file's discovery tests are about third-party
sources; this one describes our own artefact.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from hub.contracts import Contract, ContractViolation
from hub.forecasters.inputs import (
    GameInfo,
    InjuryReport,
    MarketSnapshot,
    PlayedGame,
    QbSource,
    Schedule,
    SourceStamp,
)

SCHEMA_VERSION = 1

# The longest a game is assumed to run when its source gives no completion time. Regulation is
# about three hours, overtime and a weather delay can double it; a game inside this window is
# treated as unfinished. Private: it bounds an as-of check and moves no prediction.
_MAX_GAME_DURATION = timedelta(hours=5)

_GAME_ZONE = ZoneInfo("America/New_York")
_STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2}
_RECENT = 3


class Condition(StrEnum):
    BLIND = "blind"
    LINE_AWARE = "line_aware"


# --- the schema ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GameBlock:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    neutral_site: bool
    kickoff: datetime


@dataclass(frozen=True)
class QbBlock:
    home: str | None
    away: str | None
    updated_at: datetime | None
    updated_at_basis: str


@dataclass(frozen=True)
class RestBlock:
    home: int | None
    away: int | None


@dataclass(frozen=True)
class TeamRecord:
    games: int
    wins: int
    losses: int
    ties: int
    points_for: int
    points_against: int
    point_diff: int
    last_completed_at: datetime | None


@dataclass(frozen=True)
class RecordsBlock:
    home: TeamRecord
    away: TeamRecord


@dataclass(frozen=True)
class Result:
    game_id: str
    kickoff: datetime
    completed_at: datetime
    completed_at_basis: str              # "completed_at" | "kickoff_plus_bound"
    opponent: str
    venue: str                           # "home" | "away"
    points_for: int
    points_against: int
    outcome: str                         # "W" | "L" | "T"


@dataclass(frozen=True)
class ResultsBlock:
    home: tuple[Result, ...]
    away: tuple[Result, ...]


@dataclass(frozen=True)
class InjuryLine:
    player: str
    status: str                          # "Out" | "Doubtful" | "Questionable"


@dataclass(frozen=True)
class InjuryBlock:
    published_at: datetime | None
    published_at_basis: str
    home: tuple[InjuryLine, ...]
    away: tuple[InjuryLine, ...]


@dataclass(frozen=True)
class MarketBlock:
    snapshot_id: str
    snapshot_time: datetime
    home_win_prob: float                 # de-vigged, proportion in (0, 1)
    home_spread: float                   # negative when the home team is favoured


@dataclass(frozen=True)
class Packet:
    schema_version: int
    condition: str
    run_start: datetime
    game: GameBlock
    starting_qbs: QbBlock
    rest_days: RestBlock
    records: RecordsBlock
    recent_results: ResultsBlock
    injuries: InjuryBlock
    market: MarketBlock | None
    stamp: dict[str, SourceStamp]


@dataclass(frozen=True)
class Failure:
    check: str
    detail: str


class PacketRefused(Exception):
    """The packet broke at least one as-of check or the contract, and must not be forecast."""

    def __init__(self, failures: Sequence[Failure]) -> None:
        self.failures = tuple(failures)
        super().__init__("; ".join(f"[{f.check}] {f.detail}" for f in self.failures))


# --- the contract --------------------------------------------------------------------------
#
# Ranges are plausibility bounds on the artefact, set wide of anything honest. Rest: Sunday to
# the next Thursday is four days and Monday to Thursday three, a bye plus a long week is under
# twenty, so [2, 30] refuses a swapped unit or a date subtracted from the wrong year. A season
# is at most 22 weeks. Probabilities are proportions, so [0, 1] is the percent-for-proportion
# swap. Spreads match the schedule contract's [-40, 40]: a sign flip stays plausible, a total
# in the spread column does not. Whole numbers only, so the declared-number scan sees no float.
PACKET_CONTRACT = Contract(
    name="forecast_packet",
    required={"game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8,
              "away_team": pl.Utf8, "neutral_site": pl.Boolean, "condition": pl.Utf8,
              "home_rest_days": pl.Int64, "away_rest_days": pl.Int64,
              "home_games": pl.Int64, "away_games": pl.Int64,
              "home_wins": pl.Int64, "away_wins": pl.Int64,
              "market_home_win_prob": pl.Float64, "market_home_spread": pl.Float64},
    non_null=("game_id", "season", "week", "home_team", "away_team", "neutral_site",
              "condition", "home_games", "away_games"),
    unique=("game_id",),
    no_nan=("market_home_win_prob", "market_home_spread"),
    ranges={"week": (1, 22), "home_rest_days": (2, 30), "away_rest_days": (2, 30),
            "home_games": (0, 22), "away_games": (0, 22),
            "home_wins": (0, 22), "away_wins": (0, 22),
            "market_home_win_prob": (0, 1), "market_home_spread": (-40, 40)},
    min_rows=1,
    verified_against_live=False,            # our own artefact: there is no live source
)

_FLAT_SCHEMA: dict[str, Any] = PACKET_CONTRACT.required


def flat_frame(p: Packet) -> pl.DataFrame:
    """The packet's scalar fields as one row, with the dtypes the contract declares (explicit,
    so an absent rest or market is a typed null and not a `Null` column)."""
    row = {
        "game_id": p.game.game_id, "season": p.game.season, "week": p.game.week,
        "home_team": p.game.home_team, "away_team": p.game.away_team,
        "neutral_site": p.game.neutral_site, "condition": p.condition,
        "home_rest_days": p.rest_days.home, "away_rest_days": p.rest_days.away,
        "home_games": p.records.home.games, "away_games": p.records.away.games,
        "home_wins": p.records.home.wins, "away_wins": p.records.away.wins,
        "market_home_win_prob": None if p.market is None else p.market.home_win_prob,
        "market_home_spread": None if p.market is None else p.market.home_spread,
    }
    return pl.DataFrame([row], schema=_FLAT_SCHEMA)


# --- canonical form and hash ---------------------------------------------------------------


def plain(obj: Any) -> Any:
    """Dataclasses, mappings and sequences as dicts and lists; leaves untouched."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, StrEnum):
        return obj.value
    return obj


def _float(x: float) -> str:
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"cannot canonicalise {x!r}")
    s = f"{x:.6f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _instant(t: datetime) -> str:
    if t.tzinfo is None:
        raise ValueError(f"naive datetime {t!r} has no place in a packet")
    return json.dumps(t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))


def _write(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _float(v)
    if isinstance(v, datetime):
        return _instant(v)
    if isinstance(v, date):
        raise ValueError("dates are written as instants; got a bare date")
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=True)
    if isinstance(v, Mapping):
        return "{" + ",".join(f"{json.dumps(str(k))}:{_write(v[k])}"
                              for k in sorted(v, key=str)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_write(x) for x in v) + "]"
    raise ValueError(f"cannot canonicalise {type(v).__name__}")


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, floats to six places, instants as UTC microsecond strings."""
    return _write(plain(obj))


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def packet_hash(p: Packet) -> str:
    return canonical_hash(p)


# --- the as-of checks ----------------------------------------------------------------------


def _before(t: datetime | None, run_start: datetime) -> bool:
    """`t` is a real instant and strictly precedes `run_start`. `None` and naive both fail."""
    return t is not None and t.tzinfo is not None and t < run_start


def validate_packet(p: Packet, *, snapshot_id: str | None) -> None:
    """Re-apply every check to a finished packet; raise `PacketRefused` with all failures."""
    out: list[Failure] = []
    rs = p.run_start
    if rs.tzinfo is None:
        raise PacketRefused([Failure("run_start", "run_start must be timezone-aware")])

    def late(check: str, what: str, t: datetime | None, basis: str = "") -> None:
        out.append(Failure(check, f"{what}={t} {basis}".rstrip()
                           + f" does not strictly precede run_start {rs.isoformat()}"))

    # 1. records, differentials and recent results: completed before the run
    for side in ("home", "away"):
        rec: TeamRecord = getattr(p.records, side)
        if rec.last_completed_at is not None and not _before(rec.last_completed_at, rs):
            late("records_as_of", f"{side} record last_completed_at", rec.last_completed_at)
        for r in getattr(p.recent_results, side):
            if not _before(r.completed_at, rs):
                late("recent_results_as_of", f"{side} result {r.game_id} completed_at",
                     r.completed_at, f"({r.completed_at_basis})")
    # 2. injury report publication time
    if not _before(p.injuries.published_at, rs):
        late("injury_published", "injury report published_at", p.injuries.published_at,
             f"({p.injuries.published_at_basis})")
    # 3. starting-QB source update time
    if not _before(p.starting_qbs.updated_at, rs):
        late("qb_source_updated", "QB source updated_at", p.starting_qbs.updated_at,
             f"({p.starting_qbs.updated_at_basis})")
    # 4. market: this run's snapshot
    if p.condition == Condition.BLIND and p.market is not None:
        out.append(Failure("condition", "a blind packet may not carry market fields"))
    if p.condition == Condition.LINE_AWARE and p.market is None:
        out.append(Failure("condition", "a line_aware packet needs market fields"))
    if p.condition not in {c.value for c in Condition}:
        out.append(Failure("condition", f"unknown condition {p.condition!r}"))
    if p.market is not None and (snapshot_id is None or p.market.snapshot_id != snapshot_id):
        out.append(Failure("market_snapshot", f"market snapshot_id={p.market.snapshot_id!r} is not "
                                              f"this run's snapshot {snapshot_id!r}"))
    # the contract on the scalar fields
    try:
        PACKET_CONTRACT.validate(flat_frame(p))
    except ContractViolation as exc:
        out.append(Failure("contract", str(exc)))
    if out:
        raise PacketRefused(out)


# --- the builder ---------------------------------------------------------------------------


def _eastern_day(t: datetime) -> date:
    return t.astimezone(_GAME_ZONE).date()


def _completion(g: PlayedGame) -> tuple[datetime, str]:
    if g.completed_at is not None:
        return g.completed_at, "completed_at"
    return g.kickoff + _MAX_GAME_DURATION, "kickoff_plus_bound"


def _completed(g: PlayedGame, info: GameInfo, run_start: datetime) -> bool:
    """A game this one's packet may know the result of: this season's regular season, not this
    game, played before it, final with a score, and completed strictly before `run_start`."""
    return (g.season == info.season and g.game_type == "REG" and g.game_id != info.game_id
            and g.kickoff < info.kickoff and g.final
            and g.home_score is not None and g.away_score is not None
            and _before(_completion(g)[0], run_start))


def _results(team: str, games: Sequence[PlayedGame]) -> list[Result]:
    out = []
    for g in games:
        if team not in (g.home_team, g.away_team):
            continue
        home = g.home_team == team
        pf, pa = (g.home_score, g.away_score) if home else (g.away_score, g.home_score)
        assert pf is not None and pa is not None
        done, basis = _completion(g)
        out.append(Result(g.game_id, g.kickoff, done, basis, g.away_team if home else g.home_team,
                          "home" if home else "away", pf, pa,
                          "W" if pf > pa else "L" if pf < pa else "T"))
    return sorted(out, key=lambda r: (r.kickoff, r.game_id), reverse=True)


def _record(rows: Sequence[Result]) -> TeamRecord:
    pf, pa = sum(r.points_for for r in rows), sum(r.points_against for r in rows)
    return TeamRecord(
        games=len(rows), wins=sum(r.outcome == "W" for r in rows),
        losses=sum(r.outcome == "L" for r in rows), ties=sum(r.outcome == "T" for r in rows),
        points_for=pf, points_against=pa, point_diff=pf - pa,
        last_completed_at=max((r.completed_at for r in rows), default=None))


def _rest(rows: Sequence[Result], kickoff: datetime) -> int | None:
    if not rows:
        return None
    return (_eastern_day(kickoff) - _eastern_day(rows[0].kickoff)).days


def _injuries(report: InjuryReport, home: str, away: str) -> InjuryBlock:
    def lines(team: str) -> tuple[InjuryLine, ...]:
        kept = [e for e in report.entries if e.team == team and e.status in _STATUS_ORDER]
        kept.sort(key=lambda e: (_STATUS_ORDER[e.status], e.player))
        return tuple(InjuryLine(e.player, e.status) for e in kept)
    return InjuryBlock(report.published_at, report.published_at_basis, lines(home), lines(away))


def build_packet(game: GameInfo, *, run_start: datetime, condition: Condition,
                 schedule: Schedule, qb_source: QbSource, injuries: InjuryReport,
                 market: MarketSnapshot | None = None,
                 snapshot_id: str | None = None) -> Packet:
    """Assemble one game's packet as of `run_start` and refuse it if any check fails.

    `snapshot_id` is this run's market snapshot; a `market` from any other is refused, and so
    is a blind packet handed a market at all (not trusted to ignore it).
    """
    if run_start.tzinfo is None:
        raise PacketRefused([Failure("run_start", "run_start must be timezone-aware")])
    done = [g for g in schedule.games if _completed(g, game, run_start)]
    home_rows, away_rows = _results(game.home_team, done), _results(game.away_team, done)
    prov = dict.fromkeys(("game", "rest_days", "records", "recent_results"),
                         schedule.stamp)
    prov["starting_qbs"] = qb_source.stamp
    prov["injuries"] = injuries.stamp
    block = None
    if market is not None:
        block = MarketBlock(market.snapshot_id, market.snapshot_time, market.home_win_prob,
                            market.home_spread)
        prov["market"] = SourceStamp(market.source, market.fetched_at)
    packet = Packet(
        schema_version=SCHEMA_VERSION, condition=str(condition), run_start=run_start,
        game=GameBlock(game.game_id, game.season, game.week, game.home_team, game.away_team,
                       game.neutral_site, game.kickoff),
        starting_qbs=QbBlock(qb_source.starters.get(game.home_team),
                             qb_source.starters.get(game.away_team),
                             qb_source.updated_at, qb_source.updated_at_basis),
        rest_days=RestBlock(_rest(home_rows, game.kickoff), _rest(away_rows, game.kickoff)),
        records=RecordsBlock(_record(home_rows), _record(away_rows)),
        recent_results=ResultsBlock(tuple(home_rows[:_RECENT]), tuple(away_rows[:_RECENT])),
        injuries=_injuries(injuries, game.home_team, game.away_team),
        market=block, stamp=prov)
    validate_packet(packet, snapshot_id=snapshot_id)
    return packet
