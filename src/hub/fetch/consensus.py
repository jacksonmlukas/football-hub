"""The weekly consensus page, captured before kickoff and kept (#432).

**Why this exists.** The 2026 forward measurement of the Weekly projection against consensus
(`docs/weekly-forward.md`) needs the incumbent's ranking *as it stood before each week's first
game*. For 2021-25 the incumbent is a dated third-party scrape in the FantasyPros archive
DynastyProcess republishes; for 2026 nothing in this repo captured it, and an archive fetched
after the fact cannot show what could have been read before kickoff. So this writes the page
down on the day, to `state/consensus/<season>/wk<NN>/cap-<ts>.json`, committed by the workflow
that ran it -- the pattern `hub.store.write_snapshot` set for the odds lines (#383), for the
same reason: a lookahead value that is not kept cannot be recovered.

**What a capture is.** The `weekly-op` rows of the newest scrape in the archive that falls in
the next week's window, with the time this repo wrote them (`captured_at`), the week's first
game day, the deadline that implies, and the archive's own digest. One scrape's rows, not the
archive: `player`, `pos`, `team`, `ecr`, a few hundred rows.

**The week capture's deadline is the first game day, at midnight Eastern.** A week capture
(`--capture`) is the one in time for every player of its week, so it must precede the day
Thursday's game is played. `LateCapture` refuses a later one at the write -- nothing lands --
and `is_late` is the same test read the other way round, so a file that reaches the tree by
another route is excluded at the read rather than trusted (method rule 18's control: a capture
written after kickoff is refused or flagged).

**And the page has to be this week's.** The newest scrape must lie inside the week's window --
before the first game day and no more than `MAX_LEAD_DAYS` ahead of it. An archive that has not
published this week's page yet would otherwise hand back last week's, filed under this week's
number, and the capture would be a lookahead ranking of the wrong games. `StaleArchive` refuses
that too, and the next scheduled run tries again.

Append-only: a second write to the same path with the same document is a no-op; with a different
one it raises, because a capture that changes is not one. Written through `hub.atomic`.

**A second, later capture: the one the archive's calendar allows.** The archive publishes the
page on Fridays (every 2026 `weekly-op` scrape seen, 09-11 through 10-16, is Friday-dated), so
the week capture's window above -- before Thursday's game day -- has never held a page (#515).
The Friday capture (`--lineup`) takes the newest `weekly-op` scrape dated after the previous
week's last game and before the week's first **Sunday** game day, captured before the start of
that Sunday, Eastern (`lineup_deadline_for`), and writes it to `state/consensus/<season>/lineup/
wk<NN>/lineup-<ts>.json` with a mandatory `purpose: "lineup"` -- the name it got when it was for
setting the lineup only, kept because it is on disk and on the page.

**In time per player (ADR-0027, 2026-10-20).** A capture is *in time* for a player when it was
taken before that player's own kickoff (`in_time_rows`, against `kickoffs`: the schedule's
Eastern date and time through `hub.schedule.kickoff_expr`, DST-aware). A Friday capture is in
time for Sunday's and Monday's players and for no one who played Thursday; a week capture is in
time for every player of its week. For anything that compares against consensus the two are one
kind: `read_every_capture` reads both and the reader leaves out, per player, whoever's game had
started (`hub.season.weekly_forward.admit`). This replaced the per-week test, under which no
2026 capture ever qualified.

`read_captures` still reads week captures only, and drops any document whose `purpose` is not a
week capture's: the lineup path prefers a capture in time for every player
(`hub.season.lineup_journal.ecr_ranks`) and labels it so, and a Friday page read there would carry
that label falsely. `hub.season.lineup_rule` reads the Friday capture after the week capture and
before last-good, labelled as taken after Thursday's games.

    uv run python -m hub.fetch.consensus --capture
    uv run python -m hub.fetch.consensus --lineup
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from hub import atomic
from hub.cli import unavailable
from hub.config import SEASON_AHEAD
from hub.declare import decision, not_an_input
from hub.paths import STATE_DIR

ROOT = STATE_DIR / "consensus"

# The cross-position weekly page the incumbent is, and the lead a scrape may have. Both are
# `hub.models.panel`'s (`CONSENSUS_PAGE`, `CONSENSUS_MAX_LEAD_DAYS`); a fetcher does not import
# a model, so they are spelled here and `tests/unit/test_fetch_consensus.py` holds the two
# spellings equal.
PAGE = "weekly-op"
MAX_LEAD_DAYS = not_an_input(
    8, "how far ahead of kickoff a scrape may be and still be this week's page: the panel's "
       "own bound, restated for a fetcher that may not import it, and a record filter")

# The columns the schedule is asked for. The `nflverse_schedules` contract refuses a frame
# missing any column it requires, so these are a superset of its required set plus what is read
# here (`game_type`, `gameday`); `tests/unit/test_fetch_consensus.py` holds the subset. The fetch
# itself is network and uncovered, which is how a four-column ask went live and failed (#437).
SCHEDULE_COLS = ("game_id", "season", "week", "game_type", "home_team", "away_team", "gameday",
                 "gametime")

# The archive's team codes that are not nflverse's, so a ranked player can be placed at his
# game (`in_time_rows`). `hub.draft.playoff_sos.TEAM_ALIASES` is the same map; a fetcher does not
# import a draft module, so `tests/unit/test_fetch_consensus.py` holds the two equal.
TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "LV": "LV", "WSH": "WAS", "ARZ": "ARI"}

EASTERN = ZoneInfo("America/New_York")
CAPTURE_COLUMNS = ("player", "pos", "team", "ecr")
_GLOB = "wk*/cap-*.json"

# What a capture is. A week capture carries no `purpose` (`build` writes none) or `"week"`; a
# Friday capture must say `"lineup"`, its name from before ADR-0027.
WEEK, LINEUP = "week", "lineup"
LINEUP_DIR = "lineup"
_LINEUP_GLOB = f"{LINEUP_DIR}/wk*/lineup-*.json"

# The CLI's exit codes for "nothing captured". With the deadline still a day or more away the
# next scheduled run can try again, and that is a notice (3); closer than this, no scheduled run
# is left to recover it before the lineup is set, and the run must go red (4).
NOTHING_YET, NOTHING_AND_LATE = 3, 4
LOUD_WITHIN = not_an_input(
    timedelta(hours=24), "how close to its deadline an empty capture run turns red instead of "
                         "a notice: a day, about one scheduled retry, by the operator's choice")


class LateCapture(RuntimeError):
    """The capture was written at or after its week's deadline, so it is not a lookahead one."""


class StaleArchive(RuntimeError):
    """The archive holds no scrape of the page inside the week's window yet."""


@dataclass(frozen=True)
class Capture:
    """One committed capture, as read back."""
    season: int
    week: int
    captured_at: datetime               # aware, UTC
    first_game_day: date
    deadline: datetime                  # aware, UTC
    scrape_date: date
    digest: str | None
    rows: pl.DataFrame
    path: Path | None = None
    purpose: str = WEEK


@dataclass(frozen=True)
class LineupCapture:
    """One committed Friday capture, as read back (filed as `lineup`; see the module doc)."""
    season: int
    week: int
    captured_at: datetime               # aware, UTC
    first_game_day: date
    first_sunday: date
    deadline: datetime                  # aware, UTC: the start of `first_sunday`, Eastern
    scrape_floor: date
    scrape_date: date
    digest: str | None
    rows: pl.DataFrame
    path: Path | None = None

    def note(self) -> str:
        """The honest label for a lineup set on this capture."""
        when = (f"taken after {self.first_game_day:%A}'s games, so in time only for players "
                "whose game had not started"
                if self.scrape_date >= self.first_game_day else
                f"captured {self.captured_at.astimezone(EASTERN):%a %Y-%m-%d} for the lineup, "
                "not before the week's first game day")
        return f"consensus ranks as of {self.scrape_date:%a %Y-%m-%d}, {when}"


def deadline_for(first_game_day: date) -> datetime:
    """Midnight at the start of the first game day, Eastern, as an aware UTC time."""
    local = datetime(first_game_day.year, first_game_day.month, first_game_day.day,
                     tzinfo=EASTERN)
    return local.astimezone(UTC)


def is_late(captured_at: datetime, first_game_day: date) -> bool:
    """At or after the deadline. The one comparison the write and the read both make."""
    return captured_at >= deadline_for(first_game_day)


def lineup_deadline_for(first_sunday: date) -> datetime:
    """The lineup capture's deadline: the start of the week's first Sunday game day, Eastern --
    before any Sunday kickoff, London's 09:30 ET included. The same clock as `deadline_for`."""
    return deadline_for(first_sunday)


def first_sundays(schedule: pl.DataFrame) -> dict[int, date]:
    """Each regular-season week's first Sunday game day. A week with no Sunday game is absent."""
    s = (schedule.filter(pl.col("game_type") == "REG")
                 .with_columns(pl.col("gameday").str.to_date().alias("_d"))
                 .filter(pl.col("_d").dt.weekday() == 7))
    out = s.group_by("week").agg(pl.col("_d").min().alias("first")).sort("week")
    return {int(w): d for w, d in zip(out["week"].to_list(), out["first"].to_list(),
                                      strict=True)}


def first_game_days(schedule: pl.DataFrame) -> dict[int, date]:
    """Each regular-season week's first game day, from a schedule with `week`, `game_type` and
    `gameday`."""
    s = schedule.filter(pl.col("game_type") == "REG")
    out = (s.group_by("week").agg(pl.col("gameday").str.to_date().min().alias("first"))
             .sort("week"))
    return {int(w): d for w, d in zip(out["week"].to_list(), out["first"].to_list(),
                                      strict=True)}


def last_game_day(schedule: pl.DataFrame, week: int) -> date | None:
    s = schedule.filter((pl.col("game_type") == "REG") & (pl.col("week") == week))
    if s.is_empty():
        return None
    last = s["gameday"].str.to_date().max()
    assert isinstance(last, date)
    return last


def scrape_floors(schedule: pl.DataFrame) -> dict[int, date]:
    """The earliest scrape date that is each week's page: the day after the previous regular-
    season week's last game, and never more than `MAX_LEAD_DAYS` before the first game.

    A scrape taken on or before the last game day of week *w-1* ranks week *w-1*'s games (the
    panel's `assign_weeks` assigns a scrape to the first week with a game still to play), so a
    run just after week 5's deadline must not file Wednesday's page, eight days ahead of week
    6, under week 6. Week 1 has no earlier week and is bounded by the lead alone.
    """
    firsts = first_game_days(schedule)
    out: dict[int, date] = {}
    for week, first in firsts.items():
        floor = first - timedelta(days=MAX_LEAD_DAYS)
        prev = last_game_day(schedule, week - 1) if week - 1 in firsts else None
        out[week] = max(floor, prev + timedelta(days=1)) if prev is not None else floor
    return out


def next_week(days: dict[int, date], now: datetime) -> tuple[int, date] | None:
    """The first week whose deadline is still ahead, or None when the season has none left."""
    for week in sorted(days):
        if now < deadline_for(days[week]):
            return week, days[week]
    return None


def window_scrape(archive: pl.DataFrame, first_game_day: date,
                  floor: date | None = None) -> tuple[date, pl.DataFrame]:
    """The newest scrape of `PAGE` inside the week's window, and its rows.

    `StaleArchive` when there is none: the scrape has to be before the first game day and no
    earlier than `floor` (`scrape_floors`; by default `MAX_LEAD_DAYS` ahead of the first game).
    """
    page = (archive.filter(pl.col("page_type") == PAGE)
                   .with_columns(pl.col("scrape_date").cast(pl.Utf8).str.to_date()
                                 .alias("_scraped"))
                   .drop_nulls(["_scraped", "ecr"]))
    lo = floor if floor is not None else first_game_day - timedelta(days=MAX_LEAD_DAYS)
    inside = page.filter((pl.col("_scraped") < first_game_day) & (pl.col("_scraped") >= lo))
    if inside.is_empty():
        newest = page["_scraped"].max() if not page.is_empty() else None
        raise StaleArchive(
            f"no `{PAGE}` scrape between {lo} and the day before {first_game_day}; the "
            f"archive's newest is {newest}. The page for this week is not published yet.")
    newest = inside["_scraped"].max()
    assert isinstance(newest, date)
    return newest, (inside.filter(pl.col("_scraped") == newest)
                          .select(*CAPTURE_COLUMNS).sort("ecr", "player"))


def build(season: int, week: int, rows: pl.DataFrame, scrape_date: date, first_game_day: date,
          captured_at: datetime, digest: str | None) -> dict:
    return {"season": season, "week": week, "page": PAGE,
            "captured_at": captured_at.astimezone(UTC).isoformat(),
            "first_game_day": first_game_day.isoformat(),
            "deadline": deadline_for(first_game_day).isoformat(),
            "scrape_date": scrape_date.isoformat(), "digest": digest,
            "rows": rows.select(*CAPTURE_COLUMNS).to_dicts()}


def capture_path(season: int, week: int, when: datetime, base: Path | None = None) -> Path:
    root = ROOT if base is None else Path(base)
    return (root / str(season) / f"wk{week:02d}"
            / f"cap-{when.astimezone(UTC):%Y%m%dT%H%M%S}.json")


@decision
def write_capture(doc: dict, base: Path | None = None) -> Path:
    """Write one capture, or refuse. Refuses a late one (`LateCapture`) before touching disk.

    This is the guard rule 18 asks for: a capture whose `captured_at` is at or after its
    week's deadline never reaches the tree through here. The committed record is also
    append-only: the same document again is a no-op, a different one at the same path raises.
    """
    when = datetime.fromisoformat(doc["captured_at"])
    first = date.fromisoformat(doc["first_game_day"])
    if is_late(when, first):
        raise LateCapture(
            f"week {doc['week']}: captured {when.isoformat()}, at or after its deadline "
            f"{deadline_for(first).isoformat()} (the start of {first}, Eastern). A page "
            f"taken once games have started is not a lookahead ranking; nothing was written.")
    path = capture_path(doc["season"], doc["week"], when, base)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return path
        raise FileExistsError(
            f"{path} already holds a different capture. Captures are append-only; a second "
            f"look is a second file with its own timestamp.")
    return atomic.write_text(path, text)


def read_capture(path: Path) -> Capture:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return Capture(
        purpose=str(doc.get("purpose") or WEEK),
        season=int(doc["season"]), week=int(doc["week"]),
        captured_at=datetime.fromisoformat(doc["captured_at"]),
        first_game_day=date.fromisoformat(doc["first_game_day"]),
        deadline=datetime.fromisoformat(doc["deadline"]),
        scrape_date=date.fromisoformat(doc["scrape_date"]), digest=doc.get("digest"),
        rows=pl.DataFrame(doc["rows"], schema={"player": pl.Utf8, "pos": pl.Utf8,
                                               "team": pl.Utf8, "ecr": pl.Float64}),
        path=path)


def read_captures(season: int, base: Path | None = None) -> list[Capture]:
    """Every week capture file for `season`, earliest first by `captured_at`.

    A document that says it is anything but a week capture is left out, wherever it is in the
    tree: a Friday capture copied into `wk<NN>/` was not taken before the week's first game day,
    and the lineup path labels what this returns as a capture in time for every player."""
    root = (ROOT if base is None else Path(base)) / str(season)
    caps = [read_capture(p) for p in root.glob(_GLOB)]
    # GUARD week-reader-skips-lineup [unit/test_fetch_consensus.py]: purpose must be a week's
    caps = [c for c in caps if c.purpose == WEEK]
    # /GUARD
    return sorted(caps, key=lambda c: c.captured_at)


def read_every_capture(season: int, base: Path | None = None) -> list[Capture]:
    """Every capture of `season` that a comparison against consensus may read, earliest first:
    the week captures (`read_captures`) and the Friday ones (`read_lineup_captures`, each as a
    `Capture` with `purpose` `"lineup"` and its own Sunday deadline).

    One list since ADR-0027: neither kind is in time for a whole week by being the kind it is,
    so the reader decides per player (`in_time_rows`). Only Friday captures that pass
    `lineup_problem` are here, as in `read_lineup_captures`."""
    caps = read_captures(season, base)
    caps += [Capture(season=c.season, week=c.week, captured_at=c.captured_at,
                     first_game_day=c.first_game_day, deadline=c.deadline,
                     scrape_date=c.scrape_date, digest=c.digest, rows=c.rows, path=c.path,
                     purpose=LINEUP)
             for c in read_lineup_captures(season, base)]
    return sorted(caps, key=lambda c: c.captured_at)


# --- in time, per player (ADR-0027) --------------------------------------------------------

def kickoffs(schedule: pl.DataFrame) -> dict[tuple[int, str], datetime]:
    """`{(week, team): kickoff}` for every regular-season game, aware UTC.

    The kickoff is `hub.schedule.kickoff_expr`'s -- nflverse's Eastern `gameday` and `gametime`,
    converted with the zone rather than a fixed offset, to naive UTC -- with UTC attached, so it
    compares with `captured_at` on one clock. A game with no `gametime` is placed at the start
    of its game day, Eastern (`deadline_for`): earlier than any real kickoff, so a capture is
    never credited with a game it could have seen. A schedule without teams and dates gives
    `{}`, and then only a capture taken before the week's first game day places anyone."""
    from hub.schedule import kickoff_expr
    if not {"week", "game_type", "home_team", "away_team", "gameday"} <= set(schedule.columns):
        return {}
    s = schedule.filter(pl.col("game_type") == "REG")
    s = s.with_columns(kickoff_expr() if "gametime" in s.columns
                       else pl.lit(None, dtype=pl.Datetime).alias("kickoff"))
    out: dict[tuple[int, str], datetime] = {}
    for r in s.iter_rows(named=True):
        k = r["kickoff"]
        at = (k.replace(tzinfo=UTC) if k is not None
              else deadline_for(date.fromisoformat(str(r["gameday"])[:10])))
        for team in (r["home_team"], r["away_team"]):
            key = (int(r["week"]), str(team))
            out[key] = min(out.get(key, at), at)
    return out


def in_time_rows(rows: pl.DataFrame, at: datetime, week: int, first_game_day: date,
                 kicks: dict[tuple[int, str], datetime]) -> tuple[pl.DataFrame, int]:
    """The rows of a capture taken at `at` that are in time for their player, and how many are
    not. In time means taken before *that player's* kickoff (ADR-0027), placed by his `team`.

    Taken before `first_game_day` (the schedule's, as it is now) it precedes every kickoff of
    the week and every row stands. Otherwise a row stands when his team's kickoff that week is
    after `at`, or his team has no game that week (a bye: there is no game to have seen). A
    team the schedule does not know -- a free agent, a code nobody maps -- cannot be placed, and
    a row that cannot be shown to be in time is left out rather than trusted."""
    if at < deadline_for(first_game_day):
        return rows, 0
    teams = {t for _, t in kicks}
    keep: list[bool] = []
    for team in rows["team"].to_list():
        code = TEAM_ALIASES.get(team, team) if team else None
        k = kicks.get((week, code)) if code else None
        ok = code in teams                           # no game that week: a bye
        # GUARD in-time-per-player [unit/test_fetch_consensus.py]: his own kickoff, not the week's
        if k is not None:
            ok = at < k
        # /GUARD
        keep.append(ok)
    kept = rows.filter(pl.Series(keep, dtype=pl.Boolean))
    return kept, rows.height - kept.height


# --- the lineup capture --------------------------------------------------------------------

def lineup_week(schedule: pl.DataFrame, now: datetime) -> tuple[int, date, date] | None:
    """`(week, first_game_day, first_sunday)` of the first week whose lineup deadline is still
    ahead, or None when the season has none left."""
    firsts, sundays = first_game_days(schedule), first_sundays(schedule)
    for week in sorted(sundays):
        if now < lineup_deadline_for(sundays[week]) and week in firsts:
            return week, firsts[week], sundays[week]
    return None


def build_lineup(season: int, week: int, rows: pl.DataFrame, scrape_date: date,
                 scrape_floor: date, first_game_day: date, first_sunday: date,
                 captured_at: datetime, digest: str | None) -> dict:
    return {"season": season, "week": week, "page": PAGE, "purpose": LINEUP,
            "captured_at": captured_at.astimezone(UTC).isoformat(),
            "first_game_day": first_game_day.isoformat(),
            "first_sunday": first_sunday.isoformat(),
            "deadline": lineup_deadline_for(first_sunday).isoformat(),
            "scrape_floor": scrape_floor.isoformat(),
            "scrape_date": scrape_date.isoformat(), "digest": digest,
            "rows": rows.select(*CAPTURE_COLUMNS).to_dicts()}


def lineup_path(season: int, week: int, when: datetime, base: Path | None = None) -> Path:
    root = ROOT if base is None else Path(base)
    return (root / str(season) / LINEUP_DIR / f"wk{week:02d}"
            / f"lineup-{when.astimezone(UTC):%Y%m%dT%H%M%S}.json")


def lineup_problem(doc: dict) -> str | None:
    """Why `doc` is not a lineup capture that may set its week's lineup, or None.

    The one test the write refuses by and the read filters by: it says it is a lineup capture,
    it was taken before the start of the week's first Sunday, and its scrape lies in the window
    -- after the previous week's last game, before that Sunday."""
    if doc.get("purpose") != LINEUP:
        return f"purpose is {doc.get('purpose')!r}, not {LINEUP!r}"
    when = datetime.fromisoformat(doc["captured_at"])
    sunday = date.fromisoformat(doc["first_sunday"])
    scraped = date.fromisoformat(doc["scrape_date"])
    floor = date.fromisoformat(doc["scrape_floor"])
    if when >= lineup_deadline_for(sunday):
        return (f"captured {when.isoformat()}, at or after the lineup deadline "
                f"{lineup_deadline_for(sunday).isoformat()} (the start of {sunday}, Eastern)")
    if not floor <= scraped < sunday:
        return f"scrape {scraped} is outside the week's window [{floor}, {sunday})"
    return None


@decision
def write_lineup_capture(doc: dict, base: Path | None = None) -> Path:
    """Write one lineup capture, or refuse (`LateCapture`) before touching disk. Append-only,
    exactly as `write_capture`."""
    if why := lineup_problem(doc):
        raise LateCapture(f"week {doc.get('week')}: not a lineup capture: {why}; nothing was "
                          "written.")
    path = lineup_path(doc["season"], doc["week"], datetime.fromisoformat(doc["captured_at"]),
                       base)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return path
        raise FileExistsError(f"{path} already holds a different lineup capture.")
    return atomic.write_text(path, text)


def read_lineup_captures(season: int, base: Path | None = None) -> list[LineupCapture]:
    """Every lineup capture for `season` that passes `lineup_problem`, earliest first."""
    root = (ROOT if base is None else Path(base)) / str(season)
    out: list[LineupCapture] = []
    for p in root.glob(_LINEUP_GLOB):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if lineup_problem(doc) is not None:
            continue
        out.append(LineupCapture(
            season=int(doc["season"]), week=int(doc["week"]),
            captured_at=datetime.fromisoformat(doc["captured_at"]),
            first_game_day=date.fromisoformat(doc["first_game_day"]),
            first_sunday=date.fromisoformat(doc["first_sunday"]),
            deadline=datetime.fromisoformat(doc["deadline"]),
            scrape_floor=date.fromisoformat(doc["scrape_floor"]),
            scrape_date=date.fromisoformat(doc["scrape_date"]), digest=doc.get("digest"),
            rows=pl.DataFrame(doc["rows"], schema={"player": pl.Utf8, "pos": pl.Utf8,
                                                   "team": pl.Utf8, "ecr": pl.Float64}),
            path=p))
    return sorted(out, key=lambda c: c.captured_at)


def capture_lineup(schedule: pl.DataFrame, archive: pl.DataFrame, season: int, now: datetime,
                   digest: str | None, base: Path | None = None) -> tuple[Path, int]:
    """The lineup capture's pure half: the upcoming week by Sunday deadline, its newest scrape
    in the lineup window, written. Raises `StaleArchive` / `LateCapture`; `ValueError` when no
    week is left."""
    nxt = lineup_week(schedule, now)
    if nxt is None:
        raise ValueError("no regular-season week has a lineup deadline still ahead")
    week, first, sunday = nxt
    floor = scrape_floors(schedule)[week]
    scrape, rows = window_scrape(archive, sunday, floor)
    doc = build_lineup(season, week, rows, scrape, floor, first, sunday, now, digest)
    return write_lineup_capture(doc, base), week


def empty_exit(deadline: datetime | None, now: datetime) -> int:
    """The exit code for a run that captured nothing: red (`NOTHING_AND_LATE`) when the deadline
    it was capturing for is under `LOUD_WITHIN` away, a notice (`NOTHING_YET`) otherwise."""
    # GUARD empty-capture-loud-near-deadline [unit/test_fetch_consensus.py]: red inside a day
    if deadline is not None and now < deadline and deadline - now < LOUD_WITHIN:
        return NOTHING_AND_LATE
    # /GUARD
    return NOTHING_YET


def capture_week(schedule: pl.DataFrame, archive: pl.DataFrame, season: int, now: datetime,
                 digest: str | None, base: Path | None = None) -> tuple[Path, int]:
    """The pure half of a run: find the next week, cut its page, write it. Returns the path and
    the week. Raises `LateCapture` / `StaleArchive`; `ValueError` when no week is left."""
    nxt = next_week(first_game_days(schedule), now)
    if nxt is None:
        raise ValueError("no regular-season week has a deadline still ahead")
    week, first = nxt
    scrape, rows = window_scrape(archive, first, scrape_floors(schedule)[week])
    return write_capture(build(season, week, rows, scrape, first, now, digest), base), week


def _fetch(
        season: int) -> tuple[pl.DataFrame, pl.DataFrame, str | None]:
    from hub.fetch import nflverse
    schedule = nflverse.load("schedules", [season], cols=SCHEDULE_COLS)
    archive = nflverse.load_rankings("all", cols=nflverse.RANKINGS_COLS, refresh=True)
    pin = nflverse.data_pin("ff_rankings", ["all"], cols=nflverse.RANKINGS_COLS, as_of=None)
    return schedule, archive, (pin.digest if pin is not None else None)


def _now() -> datetime:
    """The clock a run judges its deadline by; a seam so a test can pin it."""
    return datetime.now(UTC)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", action="store_true",
                    help="write the next week's weekly-op page to state/consensus/")
    ap.add_argument("--lineup", action="store_true",
                    help="write the upcoming week's lineup-only capture (after Thursday, before "
                         "Sunday) to state/consensus/<season>/lineup/")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    a = ap.parse_args(argv)
    if a.capture == a.lineup:
        ap.print_usage(file=sys.stderr)
        return 2
    now = _now()
    try:
        schedule, archive, digest = _fetch(a.season)
    except Exception as exc:
        return unavailable("hub.fetch.consensus", "the FantasyPros archive or the schedule", exc)
    if a.lineup:
        try:
            path, week = capture_lineup(schedule, archive, a.season, now, digest)
        except (LateCapture, StaleArchive, ValueError) as exc:
            nxt = lineup_week(schedule, now)
            deadline = lineup_deadline_for(nxt[2]) if nxt is not None else None
            left = (f"; the lineup deadline {deadline.isoformat()} is "
                    f"{(deadline - now).total_seconds() / 3600:.1f}h away"
                    if deadline is not None else "")
            print(f"hub.fetch.consensus: nothing captured for the lineup: {exc}{left}",
                  file=sys.stderr)
            return empty_exit(deadline, now)
        print(f"  consensus lineup capture: week {week}, {path}")
        return 0
    try:
        path, week = capture_week(schedule, archive, a.season, now, digest)
    except (LateCapture, StaleArchive, ValueError) as exc:
        print(f"hub.fetch.consensus: nothing captured: {exc}", file=sys.stderr)
        return 3
    print(f"  consensus capture: week {week}, {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
