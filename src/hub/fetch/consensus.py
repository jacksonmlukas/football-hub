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

**The deadline is the first game day, at midnight Eastern, and it is strict.** A capture taken
on the day Thursday's game is played contains that game; it is not a lookahead ranking for the
week. `LateCapture` refuses it at the write -- nothing lands -- and `is_late` is the same test
read the other way round, so a file that reaches the tree by another route is excluded at the
read rather than trusted (method rule 18's control: a capture written after kickoff is refused or
flagged).

**And the page has to be this week's.** The newest scrape must lie inside the week's window --
before the first game day and no more than `MAX_LEAD_DAYS` ahead of it. An archive that has not
published this week's page yet would otherwise hand back last week's, filed under this week's
number, and the capture would be a lookahead ranking of the wrong games. `StaleArchive` refuses
that too, and the next scheduled run tries again.

Append-only: a second write to the same path with the same document is a no-op; with a different
one it raises, because a capture that changes is not one. Written through `hub.atomic`.

    uv run python -m hub.fetch.consensus --capture
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
SCHEDULE_COLS = ("game_id", "season", "week", "game_type", "home_team", "away_team", "gameday")

EASTERN = ZoneInfo("America/New_York")
CAPTURE_COLUMNS = ("player", "pos", "team", "ecr")
_GLOB = "wk*/cap-*.json"


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


def deadline_for(first_game_day: date) -> datetime:
    """Midnight at the start of the first game day, Eastern, as an aware UTC time."""
    local = datetime(first_game_day.year, first_game_day.month, first_game_day.day,
                     tzinfo=EASTERN)
    return local.astimezone(UTC)


def is_late(captured_at: datetime, first_game_day: date) -> bool:
    """At or after the deadline. The one comparison the write and the read both make."""
    return captured_at >= deadline_for(first_game_day)


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
        season=int(doc["season"]), week=int(doc["week"]),
        captured_at=datetime.fromisoformat(doc["captured_at"]),
        first_game_day=date.fromisoformat(doc["first_game_day"]),
        deadline=datetime.fromisoformat(doc["deadline"]),
        scrape_date=date.fromisoformat(doc["scrape_date"]), digest=doc.get("digest"),
        rows=pl.DataFrame(doc["rows"], schema={"player": pl.Utf8, "pos": pl.Utf8,
                                               "team": pl.Utf8, "ecr": pl.Float64}),
        path=path)


def read_captures(season: int, base: Path | None = None) -> list[Capture]:
    """Every capture file for `season`, earliest first by `captured_at`."""
    root = (ROOT if base is None else Path(base)) / str(season)
    return sorted((read_capture(p) for p in root.glob(_GLOB)), key=lambda c: c.captured_at)


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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", action="store_true",
                    help="write the next week's weekly-op page to state/consensus/")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    a = ap.parse_args(argv)
    if not a.capture:
        ap.print_usage(file=sys.stderr)
        return 2
    now = datetime.now(UTC)
    try:
        schedule, archive, digest = _fetch(a.season)
    except Exception as exc:
        return unavailable("hub.fetch.consensus", "the FantasyPros archive or the schedule", exc)
    try:
        path, week = capture_week(schedule, archive, a.season, now, digest)
    except (LateCapture, StaleArchive, ValueError) as exc:
        print(f"hub.fetch.consensus: nothing captured: {exc}", file=sys.stderr)
        return 3
    print(f"  consensus capture: week {week}, {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
