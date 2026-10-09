"""Starter-change events: the one event set the quarterback gate and the line-move study both
read (#346).

A **starter change** is a team whose starter on game *g* differs from its starter on its
previous game *g - 1* of the same season. The starter is *observed*, never reported: the
source's own starter column on the row (`hub.fetch.nfeloqb`, where a played row carries who
started and the coming week's row carries who is named), or the passer of the first pass
attempt in play-by-play where the cache carries that column. The injury report is not
consulted -- #221 records why: one row per player-week, timestamped Friday, and fourteen
quarterbacks out across all of 2024, where the starter changes about fifty-three times a
season. A change across the offseason is flagged and is not an event; the betting market priced
it all summer.

**Why this is its own module.** The events are shared by construction -- one builder, two
readers -- and not merely adjacent (ADR-0021 warns against splitting the latter). The readers
are `hub.models.starter_change` (the log-loss gate, a diagnostic since #300, and the CLI) and
`hub.models.starter_study` (the line-move study and its non-inferiority verdict). Neither
builds an event, a team-game, or a price off the archive by any route but this one. Their
shared reads of the snapshot archive live here too: `priced` (the frozen price and the close
of each event game) and `results` (the home result), because both readers take the same
comparator and the same played-game filter.

**Nothing here decides anything.** There is no rule, no gate and no number a prediction reads;
what the events are *for* is in the readers' own docstrings and in `docs/gate-power.md`.
The pinned file (`hub.fetch.nfeloqb.COMMIT`, 2026-09-13) is the adjustment's and the gate's
input and holds one week of 2026; the study's events come from play-by-play (#420, ADOPTED (B)
on #421), reconciled against it here (`reconcile`).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub.config import SEASON_AHEAD
from hub.fetch import nfeloqb, nflverse

PASSER = "passer_player_id"

TEAM_GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "date": pl.Utf8,
    "team": pl.Utf8, "home": pl.Boolean, "qb": pl.Utf8, "value": pl.Float64,
    "adj": pl.Float64, "score": pl.Int64, "opp_score": pl.Int64,
    "base_prob": pl.Float64, "qb_prob": pl.Float64,
}


# --- the event construction ---------------------------------------------------------------
#
# #339: the transform below used to be hand-built here -- a private `_schedule` plus a
# duplicate of `team_games`'s own per-side unpivot -- reaching past `hub.fetch.nfeloqb`'s
# public surface six times (`nfeloqb.ABBREVIATIONS` three times over, `nfeloqb._blank`) to
# rebuild what `nfeloqb.team_games` now does once, for both this module and the fetch
# module's own state path. Everything here reads that public function; nothing computes the
# schema-level transform itself.
#
# #374: `nfeloqb.schedule`/`nfeloqb.team_games` take `regular_season_only` explicitly, with
# no default that would hide the choice. This module's question is regular-season by
# pre-registration, so every call below passes `True`; `nfeloqb.state` is the other reader,
# and passes `False` so a team's tenure run can reach back across the postseason boundary.

def team_games(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, game) off the source's rows, keyed by nflverse's game id, with the
    previous-game link (#301) `events` reads straight off. Read off
    `hub.fetch.nfeloqb.team_games`, which owns the source's schema -- the two-sided row, the
    blank convention, the abbreviation map -- and builds this frame for its own state path
    too (#339); this keeps only the columns both of this module's readers need
    (`TEAM_GAME_SCHEMA`, plus `prev_game_id`/`prev_season`/`prev_date`) and computes none of
    the schema-level transform itself. Regular season only (#374): the line-move study is
    regular-season by pre-registration."""
    return nfeloqb.team_games(rows, regular_season_only=True).select(
        *TEAM_GAME_SCHEMA, "prev_game_id", "prev_season", "prev_date")


def unreadable_games(rows: pl.DataFrame) -> int:
    """Team-games a blank side made unreadable: entries `hub.fetch.nfeloqb.schedule`'s full
    row set carries for a team that `team_games` has no row for, because that side's qb,
    value or adjustment was null. Counts both sides of a two-sided blank, one each, alongside
    every one-sided blank's single side (#328). Reported on the run line so a hole is
    counted, not silently closed over the way the previous-game link used to close it
    (#301). Regular season only (#374), matching `team_games`."""
    return nfeloqb.schedule(rows, regular_season_only=True).height - team_games(rows).height


def starters_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Who took the first pass attempt for each (game, team): the observed starter, off
    play-by-play. One row per (game_id, team) with `season`, `week` and `qb` -- the passer's
    id, so a frame built here joins the source's values through `with_values` and never by
    name. The standard `PBP_COLS` slice does not carry the passer, and a cache without it is
    refused by the column's name rather than read for a starter it cannot hold.

    **Regular season only (#420)**: the question is regular-season by pre-registration, and a
    playoff game would otherwise be a team's "previous game" for the next season's week 1.
    `season_type` is therefore required, and refused by name like the passer.

    **The rule is the passer of the first pass play, whoever he is.** Reconciled against
    nfeloqb's starter column on 2022-2025 (docs/qb-adjustment.md, 2026-10-06): 6 of 2,174
    team-games differ. It does not know a position, so a trick play that opens a game names
    its thrower as the starter and reads as a change out and a change back. One of the six
    disagreements is a brief first appearance (the first passer threw under three pass plays,
    the file's starter eight or more), and six team-games in all have a first passer who threw
    two or fewer. A guard against them would be a second definition of "starter" chosen after
    the numbers, which rule 1 forbids; the count is the standing cost of the rule.
    """
    for column in (PASSER, "season_type"):
        if column not in pbp.columns:
            raise ValueError(f"play-by-play carries no '{column}' column; the first pass "
                             f"attempt cannot name a regular-season starter without it "
                             f"(the PBP_COLS slice does not include it; the study's own is "
                             f"nflverse.STARTER_PBP_COLS)")
    passes = pbp.filter((pl.col("season_type") == "REG") & (pl.col("play_type") == "pass")
                        & pl.col(PASSER).is_not_null())
    return (passes.sort("game_id", "posteam", "play_id")
                  .group_by("game_id", "posteam", maintain_order=True).first()
                  .select(pl.col("game_id"), pl.col("season").cast(pl.Int64),
                          pl.col("week").cast(pl.Int64), pl.col("posteam").alias("team"),
                          pl.col(PASSER).alias("qb")))


def events(starters: pl.DataFrame) -> pl.DataFrame:
    """Every game where a team's starter differs from its starter on its previous known game.

    `starters` is one row per (team, game) with `game_id`, `season`, `week`, `team`, `qb` --
    `team_games` or `starters_from_pbp` -- ordered within a team by season and week. The
    departing starter is the last row with a known starter, the arriving one this row's; a
    team's first known row has nothing to differ from and is never an event. `departing_game_id`
    and `departing_season` name *that* row -- the departing starter's own last known game, not
    necessarily the team's immediately previous one -- so `in_season` and any other reader can
    be stated on it without re-deriving it. Where the frame carries `value`, `adj` and `date`
    (the source's rows do), the ex-ante values ride along: the departing starter's value off
    his last known start, the arriving starter's value and adjustment off the event row, and
    `gap`, arriving minus departing.

    `prev_game_id`, `prev_season` and `prev_date` -- the ancestor `frozen_before` prices off
    -- are the team's *actual* previous game, never the previous row that happens to survive
    a one-sided blank (#301): where the frame already carries them (`team_games` builds them
    off the full schedule, both sides, before either is filtered for blanks), they are read
    straight off it rather than re-derived from whichever rows survived. A frame with no such
    columns (`starters_from_pbp`'s, which has no one-sided blanks to lose a game to) falls
    back to the previous surviving row -- the same row `departing_game_id` names there, since
    with no full-schedule link the two coincide.

    `in_season` is whether the *departing starter's own* last known game was the same season
    as this one -- not whether the team's immediately previous game (`prev_season`) was,
    which can differ across a one-sided blank that itself spans the season boundary (#328): a
    blank week-1 row behind an offseason starter is still an ancestor of week 2 for pricing,
    but it is not evidence the change happened in-season, and the data cannot say whether it
    did. Such a change is flagged rather than counted, the same way an ordinary offseason
    change is.
    """
    has_link = {"prev_game_id", "prev_season", "prev_date"}.issubset(starters.columns)
    carried = [c for c in ("date", "value", "adj") if c in starters.columns]
    shift_cols = ["qb", "game_id", "season", *carried]
    # Every shifted column in one pass, *before* the rows that are not events are dropped:
    # shifted after the filter, "the previous row" is the previous event and not the
    # previous game, and the departing starter's value is another change's.
    prev = {c: pl.col(c).shift(1).over("team") for c in shift_cols}
    shifted = [prev["qb"].alias("departing"), pl.col("qb").alias("arriving"),
               prev["game_id"].alias("departing_game_id"),
               prev["season"].alias("departing_season")]
    if not has_link:
        shifted += [prev["game_id"].alias("prev_game_id"), prev["season"].alias("prev_season")]
        if "date" in carried:
            shifted.append(prev["date"].alias("prev_date"))
    if "value" in carried:
        shifted += [prev["value"].alias("departing_value"), pl.col("value").alias("arriving_value"),
                    (pl.col("value") - prev["value"]).alias("gap")]
    if "adj" in carried:
        shifted.append(pl.col("adj").alias("arriving_adj"))
    out = (starters.sort("team", "season", "week")
                   .with_columns(shifted)
                   .filter(pl.col("departing").is_not_null()
                           & (pl.col("arriving") != pl.col("departing")))
                   .with_columns(
                       (pl.col("season") == pl.col("departing_season")).alias("in_season")))
    keep = ["game_id", "season", "week", "team", "departing", "arriving", "departing_game_id",
            "departing_season", "prev_game_id", "prev_season", "in_season"]
    keep += [c for c in ("date", "prev_date", "departing_value", "arriving_value", "gap",
                         "arriving_adj") if c in out.columns]
    return out.select(keep)


def with_values(ev: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """Events built off play-by-play, given the source's ex-ante values by (team, game): the
    arriving starter's value and adjustment off the event row, the departing starter's
    value off the previous game's row, and the two dates. The pinned file is the one
    quality measure both readers use, so a starter identified elsewhere still prices here.

    **Dates the events already carry are kept (#420)**: events built off the schedule
    (`team_games_from_pbp`) have a date for every game the season plays, and the file has
    dates only for the games it holds -- one week of 2026 -- so overwriting them from it
    would erase the dates of exactly the games the file cannot see. Events with no dates
    (a bare starters frame) still take them from the file. A value the file does not hold is
    null, and so is the gap built from it: `event_games` refuses to read a null as zero."""
    dated = {"date", "prev_date"}.issubset(ev.columns)
    here = tg.select(pl.col("team"), pl.col("game_id"),
                     *([] if dated else [pl.col("date")]),
                     pl.col("value").alias("arriving_value"), pl.col("adj").alias("arriving_adj"))
    there = tg.select(pl.col("team"), pl.col("game_id").alias("prev_game_id"),
                      *([] if dated else [pl.col("date").alias("prev_date")]),
                      pl.col("value").alias("departing_value"))
    dropped = ["departing_value", "arriving_value", "gap", "arriving_adj",
               *([] if dated else ["date", "prev_date"])]
    return (ev.drop([c for c in dropped if c in ev.columns])
              .join(here, on=["team", "game_id"], how="left")
              .join(there, on=["team", "prev_game_id"], how="left")
              .with_columns((pl.col("arriving_value") - pl.col("departing_value")).alias("gap")))

# --- the event source for the study: play-by-play (#420, ADOPTED (B) on #421) --------------------
#
# The study's in-season events used to come from nfeloqb's starter column, and that file is
# pinned (`hub.fetch.nfeloqb.COMMIT`, 2026-09-13): it holds one week of 2026 and cannot hold a
# change made after the pin, so "2026: 0 changes" was a fact about the file (V2, #420). Moving
# the pin moves `config_digest` and the model version on every published prediction, for an
# adjustment that is off the published path; the maintainer's disposition (#421) is to leave
# it and observe the starter from play-by-play, which the slate's nflverse store carries
# through the week just played. The adjustment, the gate and the pinned input read the file
# exactly as before; only the study and the event counts read this.

SCHEDULE_COLS: tuple[str, ...] = ("game_id", "season", "week", "game_type", "gameday",
                                  "home_team", "away_team", "home_score", "away_score")


def schedule_team_games(sched: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, regular-season game) the schedule holds, played or not, in
    `TEAM_GAME_SCHEMA`'s shape plus the previous-game link `events` reads: `date` is the game
    day, `score`/`opp_score` the result (null until it is played), and `prev_game_id`,
    `prev_season`, `prev_date` the team's actual previous game on the schedule -- a bye never
    makes it the game before. `qb`, `value`, `adj` and the two probabilities are null: this
    frame names no starter, `team_games_from_pbp` does, and the file's values join on later."""
    missing = [c for c in SCHEDULE_COLS if c not in sched.columns]
    if missing:
        raise ValueError(f"the schedule carries no {missing} column(s); the study cannot "
                         f"date or settle a game without them")
    reg = sched.filter(pl.col("game_type") == "REG")
    side = {"home": ("home_team", "away_team", "home_score", "away_score"),
            "away": ("away_team", "home_team", "away_score", "home_score")}
    long = pl.concat([
        reg.select(pl.col("game_id"), pl.col("season").cast(pl.Int64),
                   pl.col("week").cast(pl.Int64), pl.col("gameday").cast(pl.Utf8).alias("date"),
                   pl.col(t).alias("team"), pl.lit(name == "home").alias("home"),
                   pl.col(s).cast(pl.Int64).alias("score"),
                   pl.col(o).cast(pl.Int64).alias("opp_score"))
        for name, (t, _, s, o) in side.items()])
    prev = {c: pl.col(c).shift(1).over("team")
            for c in ("game_id", "season", "date")}
    return (long.sort("team", "date", "game_id")
                .with_columns(prev["game_id"].alias("prev_game_id"),
                              prev["season"].alias("prev_season"),
                              prev["date"].alias("prev_date"),
                              pl.lit(None, dtype=pl.Utf8).alias("qb"),
                              *(pl.lit(None, dtype=pl.Float64).alias(c)
                                for c in ("value", "adj", "base_prob", "qb_prob")))
                .select(*TEAM_GAME_SCHEMA, "prev_game_id", "prev_season", "prev_date"))


def team_games_from_pbp(pbp: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """The schedule's team-games with each one's observed starter: the passer of the first
    pass play, null for a game with no pass play yet (not yet played, or in flight before
    its first pass) -- those rows are kept, because they are what tells a played week from
    one not yet read, and `events` is handed only the rows with a starter."""
    starters = starters_from_pbp(pbp).select("game_id", "team", pl.col("qb").alias("_qb"))
    return (schedule_team_games(sched)
            .join(starters, on=["game_id", "team"], how="left")
            .with_columns(pl.col("_qb").alias("qb")).drop("_qb")
            .select(*TEAM_GAME_SCHEMA, "prev_game_id", "prev_season", "prev_date"))


def observed(tg: pl.DataFrame) -> pl.DataFrame:
    """The team-games whose starter has been observed -- what `events` is handed."""
    return tg.filter(pl.col("qb").is_not_null())


class Reconciliation(NamedTuple):
    """The two sources' in-season changes, counted over the team-games both observed: the
    pinned file's, play-by-play's, and those both name. `compared` is those team-games."""

    compared: int
    file: int
    pbp: int
    both: int


def reconcile(pbp_tg: pl.DataFrame, file_tg: pl.DataFrame) -> Reconciliation:
    """Compare the two definitions of "starter" where both exist (#420's adopted acceptance):
    each source's in-season changes over the team-games both observed. Counts, never rows.
    A game one source does not hold is outside the comparison rather than a disagreement --
    the pinned file holds one week of 2026, and counting every later game as a miss would
    read its horizon as unreliability."""
    both_rows = observed(pbp_tg).join(file_tg.select("game_id", "team"),
                                      on=["game_id", "team"], how="semi")
    shared = both_rows.select("game_id", "team")
    file_rows = file_tg.join(shared, on=["game_id", "team"], how="semi")

    def keys(frame: pl.DataFrame) -> set[tuple[str, str]]:
        found = in_season_events(events(frame)).select("game_id", "team")
        return set(zip(found["game_id"], found["team"], strict=True))
    mine, theirs = keys(both_rows), keys(file_rows)
    return Reconciliation(compared=both_rows.height, file=len(theirs), pbp=len(mine),
                          both=len(mine & theirs))


def in_season_events(ev: pl.DataFrame) -> pl.DataFrame:
    """The events proper: a change between two games of one season."""
    return ev.filter(pl.col("in_season"))


EVENT_GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "date": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8, "home_gap": pl.Float64,
    "away_gap": pl.Float64, "net_gap": pl.Float64, "changes": pl.UInt32,
    "frozen_before": pl.Utf8,
}


def event_games(ev: pl.DataFrame) -> pl.DataFrame:
    """One row per event game: the home and away gaps (null where that side did not change),
    the net gap home minus away, how many sides changed, and `frozen_before` -- the earliest
    previous game day among the changes, before which no change could have been known.

    **`net_gap` is null when a side that changed has no gap (#420)**, not zero. A side that
    did not change contributes nothing, and that is a real zero; a side that changed to or
    from a starter the pinned file holds no value for contributes an unknown, and reading it
    as zero would enter a game as "no quality difference" -- the manufactured zero the
    study's week-mean refusal already exists to prevent. Events off the pinned file always
    have both values, so this only bites events it cannot see."""
    if ev.is_empty():
        return pl.DataFrame(schema=EVENT_GAME_SCHEMA)
    parts = pl.col("game_id").str.split("_")
    home = pl.col("team") == parts.list.get(3)
    return (ev.with_columns(home.alias("_home"))
              .group_by("game_id").agg(
                  pl.col("season").first(), pl.col("week").first(), pl.col("date").first(),
                  parts.list.get(3).first().alias("home_team"),
                  parts.list.get(2).first().alias("away_team"),
                  pl.col("gap").filter(pl.col("_home")).first().alias("home_gap"),
                  pl.col("gap").filter(~pl.col("_home")).first().alias("away_gap"),
                  pl.len().alias("changes"),
                  pl.col("gap").is_null().any().alias("_unvalued"),
                  pl.col("prev_date").min().alias("frozen_before"))
              .with_columns(pl.when(pl.col("_unvalued")).then(None)
                              .otherwise(pl.col("home_gap").fill_null(0.0)
                                         - pl.col("away_gap").fill_null(0.0)).alias("net_gap"))
              .select(*EVENT_GAME_SCHEMA)
              .sort("season", "week", "game_id"))

# --- the archive, read as the two readers need it -------------------------------------------

def poll_day(col: str = "captured_at") -> pl.Expr:
    """A naive-UTC capture as the Eastern date it fell on, ISO, comparable to the source's
    `date` column -- the same zone `hub.fetch.nfeloqb.game_day` converts through."""
    return (pl.col(col).dt.replace_time_zone("UTC")
              .dt.convert_time_zone(nfeloqb.GAME_DAY_ZONE).dt.date().cast(pl.Utf8))


def _last_before(polls: pl.DataFrame, games: pl.DataFrame, day: str, name: str) -> pl.DataFrame:
    """Per game, the last poll whose Eastern day is strictly before the game's `day` column:
    `<name>` and `<name>_at`, null where no poll precedes it."""
    joined = (games.select("game_id", day)
                   .join(polls.select("game_id", "close_spread", "captured_at", "poll_day"),
                         on="game_id", how="inner")
                   .filter(pl.col("poll_day") < pl.col(day))
                   .sort("game_id", "captured_at")
                   .group_by("game_id").agg(pl.col("close_spread").last().alias(name),
                                            pl.col("captured_at").last().alias(f"{name}_at")))
    return games.join(joined, on="game_id", how="left")


def priced(polls: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Event games with their two prices off the archive: `frozen`, the last snapshot before
    `frozen_before` (the comparator both readers use), and `close`, the last before the game
    day. A game with no snapshot before its `frozen_before` is `censored`: the archive
    could only have seen it after it moved, and the row is kept and marked rather than
    dropped, so the readers can count what they could not see.

    `censored` conflates two different facts and always has (#303): a game the archive holds
    no poll for at all, and a game it polled only after the change had already happened.
    `never_polled` tells the two apart -- true when the game's id appears nowhere in `polls`,
    false when it does (whether or not any of those polls precede `frozen_before`) -- so a
    caller can report "never polled" and "polled only after the change" as the separate
    counts they are rather than one number that could be either.

    `windowless` is a third fact (#420): a game whose frozen price is also its only price
    before the game day. It is not censored, since a snapshot predates the change; and it
    is not a move of zero, since nothing was observed moving. A back-filled lookahead from
    before the season is exactly this for every 2026 week the in-season captures have not
    reached."""
    days = polls.with_columns(poll_day().alias("poll_day"))
    out = _last_before(days, games, "frozen_before", "frozen")
    out = _last_before(days, out, "date", "close")
    polled_ids = polls["game_id"].unique().to_list()
    return out.with_columns(
        pl.col("frozen").is_null().alias("censored"),
        (~pl.col("game_id").is_in(polled_ids)).alias("never_polled"),
        # #420: the frozen poll is also the last one before the game day -- the same capture,
        # so there is no window in which a move could have been seen. Not censored (a price
        # predates the change), and not a zero move either: `study_rows` refuses it.
        (pl.col("frozen_at").is_not_null()
         & (pl.col("frozen_at") == pl.col("close_at"))).alias("windowless"))


def results(tg: pl.DataFrame) -> pl.DataFrame:
    """Per game, the home result: 1 a home win, 0 a loss, 0.5 a tie; unplayed games absent."""
    home = tg.filter(pl.col("home") & pl.col("score").is_not_null())
    return home.select(
        pl.col("game_id"),
        pl.when(pl.col("score") > pl.col("opp_score")).then(1.0)
          .when(pl.col("score") < pl.col("opp_score")).then(0.0)
          .otherwise(0.5).alias("y"),
        pl.col("base_prob"), pl.col("qb_prob"))


# --- the entry point --------------------------------------------------------------------------

class PbpGames(NamedTuple):
    """`team_games_from_pbp`'s frame, and the sources it was served from last-good rather
    than refreshed (named, so a run line can say so)."""

    tg: pl.DataFrame
    stale: tuple[str, ...]


def _last_good(source: str, seasons: Sequence[int], cols: Sequence[str] | None, *,
               fresh: bool, cache: Path | None) -> tuple[pl.DataFrame, bool]:
    """One `nflverse.load`, refreshed when `fresh` and served from the cache when the wire
    will not answer -- last-good rather than an error (CLAUDE.md, graceful degradation). The
    flag is whether last-good was what came back after a refresh was asked for."""
    if fresh:
        try:
            return nflverse.load(source, seasons, cols, refresh=True, cache=cache), False
        except Exception:
            pass
    return nflverse.load(source, seasons, cols, cache=cache), fresh


def pbp_team_games(seasons: Sequence[int], *, cache: Path | None = None) -> PbpGames:
    """The study's events source, read through `hub.fetch.nflverse.load`: play-by-play for
    each season (one per call -- the cache's own grain, and the slate's -- never a loop over
    teams or games) and the schedule for dates and results. The season in progress is
    refreshed, since it is the one that changes between runs; a completed season is a cache
    hit after its first read. Raises when neither the wire nor the cache has a source, and
    `main` then reads the pinned file instead and says so."""
    stale: list[str] = []
    frames = []
    for season in seasons:
        frame, was_stale = _last_good("pbp", [season], list(nflverse.STARTER_PBP_COLS),
                                       fresh=season >= SEASON_AHEAD, cache=cache)
        frames.append(frame)
        if was_stale:
            stale.append(f"play-by-play {season}")
    sched, was_stale = _last_good("schedules", seasons, None,
                                  fresh=max(seasons) >= SEASON_AHEAD, cache=cache)
    if was_stale:
        stale.append("schedules")
    tg = team_games_from_pbp(pl.concat(frames), sched.select(*SCHEDULE_COLS))
    return PbpGames(tg, tuple(stale))
