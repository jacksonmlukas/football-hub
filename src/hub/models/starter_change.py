"""Starter-change events, and the two readers that share them: the quarterback adjustment's
own gate (#291) and the line-move study (#221).

Both questions are asked on the same games -- the ones whose frozen price predates a change
of starting quarterback -- so the event set is built once here and read twice. A **starter
change** is a team whose starter on game *g* differs from its starter on its previous game
*g - 1* of the same season. The starter is *observed*, never reported: the source's own
starter column on the row (`hub.fetch.nfeloqb`, where a played row carries who started and
the coming week's row carries who is named), or the passer of the first pass attempt in
play-by-play where the cache carries that column. The injury report is not consulted -- #221
records why: one row per player-week, timestamped Friday, and fourteen quarterbacks out
across all of 2024, where the starter changes about fifty-three times a season. A change
across the offseason is flagged and is not an event; the betting market priced it all
summer.

**The gate** (`docs/gate-power.md`, pre-registered 2026-09-13 before this module existed):
over event games, log-loss of the frozen price moved by the shipped seam --
`nfeloqb.state(rows, as_of=<the week's first game day>)` handed to
`hub.models.quarterback.apply` on a row labelled as having no live price, exactly as
`ratings._rated_by_week` rated a played week until #299 pulled the adjustment from the
published path -- against the frozen price unmoved. Since #299 this module is the
adjustment's only reader (`tests/contracts/test_the_quarterback_adjustment_is_not_a_dependency.py`),
and the seam it replays is the one that would ship again if the gate cleared. That state
holds every team's previous game, so it prices the departing starter: the source carries a
new starter on the row of his first game and no earlier, and a replay cannot know him
before it. An **oracle** arm with the arriving starter known is reported beside it as a
diagnostic and is never read by the rule. The frozen
price is the last snapshot captured before the Eastern game day of the changed team's
previous game: the price the adjustment would actually have replaced, and not the close.
The cluster is the season; the rule is the house rule and nothing beside it; fewer than
three event-seasons is NOT-RUNNABLE ahead of every branch (ADR-0019's floor), and the
verdict sentence names the event-seasons the pilot says are needed.

**Amended 2026-09-17 (#300): this gate is a diagnostic.** No branch of it licenses ADOPT or
pulls the module any longer -- #270's pull trigger above is superseded. The ADOPT condition
is the study's coefficient, below, read against 0.132; this gate is reported beside it, its
own power requirement stated beside it (29 event-seasons at 80% power against the pilot's
target), and its NOT-RUNNABLE branch is an exemption from firing below that power, not a bar
to the coefficient's own verdict (`docs/gate-power.md`, `docs/qb-adjustment.md`).

**The study**: the home-spread move from the frozen price to the last snapshot before the
game day, less the mean move of the week's other games between the same two poll days,
regressed on the net ex-ante quality gap -- arriving starter's value minus departing, home
minus away, both off the pinned file -- against 538's 0.132 points per value unit. The
standard error is #214's noise floor per window over the gap's spread and root n, because
the event rows alone cannot resolve their own residual. Censored events (no snapshot before
the change could be known) and the change-point in days from the previous game day are
reported beside the coefficient. **Since 2026-09-17 (#300) this coefficient is the module's
ADOPT condition**: sign and magnitude against the 0.132 benchmark, season-clustered once two
seasons exist (`docs/gate-power.md`).

Nothing here fetches but one guarded call: `--study` asks `hub.fetch.odds._qb_starters` for
the depth chart #214's own floor conditions on, exactly as `odds.noise_floor_report` does,
and degrades the same way that report does when the chart is unavailable -- the floor is
still printed, over every live interval, and the line says the same-quarterback condition
was not applied rather than printing an unconditioned number under that label (#330). The
nfeloqb cache and the snapshot archive are everything else that is read, and a season either
caches does not hold is reported as not established.

    uv run python -m hub.models.starter_change --events --gate --study
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from hub import store
from hub.cli import unavailable
from hub.config import SEASON_AHEAD
from hub.declare import not_an_input
from hub.fetch import nfeloqb, odds
from hub.models import experiment, quarterback
from hub.models.experiment import SEASON_CLUSTER, run_gate
from hub.models.market import MARGIN_SD, normal_cdf

PROG = "hub.models.starter_change"

# 538's conversion, 3.3 Elo per value unit over 25 Elo per point: the coefficient the
# study's fitted slope is read against. A benchmark no prediction reads.
BENCHMARK = not_an_input(
    3.3 / 25,
    "the study's benchmark slope, 538's own construction; a comparison figure the line-move "
    "study prints, and nothing that predicts reads it")

# ADR-0019: no gate in the repo runs at fewer than three seasons, and one that did should
# say so. Pre-registered as the gate's first precondition.
EVENT_SEASONS_MINIMUM = 3

# What the gate's ceiling arm is called where it prints (#138): the betting market's own
# repricing, scored against the same frozen line.
CEILING_ARM = "the betting market's own repricing, the last snapshot before the game day"

# Amended 2026-09-17 (#300): this gate is a diagnostic. None of its three branches license
# ADOPT or pull the module any longer -- the sentences below said so until this amendment,
# kept as written per docs/method.md rule 13 at docs/qb-adjustment.md and docs/gate-power.md,
# which carry what replaced them and why. The ADOPT condition is #221's line-move coefficient.
ACTIONS = experiment.Actions(
    adopt="Diagnostic only, since #300: this branch does not license shipping the module. "
          "The ADOPT condition is #221's line-move coefficient, sign and magnitude against "
          "0.132 (docs/qb-adjustment.md, docs/gate-power.md); this log-loss picture is read "
          "beside it.",
    remove="Diagnostic only, since #300: this branch does not pull the module. The ADOPT "
           "condition is #221's line-move coefficient, sign and magnitude against 0.132 "
           "(docs/qb-adjustment.md, docs/gate-power.md); this log-loss picture is read "
           "beside it.",
    show="Diagnostic only, since #300: this branch decides nothing about the module. The "
         "ADOPT condition is #221's line-move coefficient, sign and magnitude against 0.132 "
         "(docs/qb-adjustment.md, docs/gate-power.md); this log-loss picture is read beside "
         "it.")

REGULAR_SEASON = "REG"
PASSER = "passer_player_id"

TEAM_GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "date": pl.Utf8,
    "team": pl.Utf8, "home": pl.Boolean, "qb": pl.Utf8, "value": pl.Float64,
    "adj": pl.Float64, "score": pl.Int64, "opp_score": pl.Int64,
    "base_prob": pl.Float64, "qb_prob": pl.Float64,
}


# --- the event construction ---------------------------------------------------------------

def _schedule(rows: pl.DataFrame) -> pl.DataFrame:
    """Every (team, game) the source has a row for, home and away, before either side is
    filtered for a one-sided blank -- the full row set `team_games`'s previous-game link and
    `unreadable_games`'s count are both built off, never the rows a one-sided blank drops
    (#301). Postseason rows are dropped the same way `team_games` drops them."""
    if "game_type" in rows.columns:
        rows = rows.filter(pl.col("game_type") == REGULAR_SEASON)
    week = pl.col("week").cast(pl.Utf8).cast(pl.Float64).cast(pl.Int64)
    gid = (pl.col("season").cast(pl.Utf8) + "_" + week.cast(pl.Utf8).str.zfill(2) + "_"
           + pl.col("team2").replace(nfeloqb.ABBREVIATIONS) + "_"
           + pl.col("team1").replace(nfeloqb.ABBREVIATIONS))
    return pl.concat([
        rows.select(gid.alias("game_id"), pl.col("season").cast(pl.Int64), week.alias("week"),
                    pl.col("date").cast(pl.Utf8),
                    pl.col(f"team{n}").replace(nfeloqb.ABBREVIATIONS).alias("team"))
        for n in ("1", "2")]).sort("team", "season", "week")


def team_games(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, game) off the source's rows, keyed by nflverse's game id.

    The id is rebuilt from the season, the week and the two teams in nflverse's spellings
    rather than read off the source's own `game_id`, which spells the Rams `LAR` and the
    Raiders `OAK` where the snapshot archive says `LA` and `LV` (2026-09-13: 6% of the
    source's 2022+ ids differ from the archive's, every one a spelling). `team1` is the home
    side, the source's convention. `base_prob` and `qb_prob` are the source's own home win
    probabilities, base and quarterback-adjusted, carried for the pilot and null where the
    file does not carry them. Postseason rows are dropped where the file says which they
    are; a side blank in any of the three the team layer reads is dropped alone (#283).

    `prev_game_id`, `prev_season` and `prev_date` name each row's *actual* previous game --
    the team's last entry in `_schedule`'s full row set, both sides, before either is
    filtered for blanks -- so a one-sided blank drops that side's own row above without
    moving its neighbours a game closer together or losing a game from the team's sequence
    (#301). `events` reads these three straight off this frame instead of re-deriving a
    previous game from whichever rows happen to survive.
    """
    if "week" not in rows.columns:
        raise ValueError("the rows carry no 'week' column, and the nflverse game id the "
                         "archive is keyed by cannot be rebuilt without it")
    if "game_type" in rows.columns:
        rows = rows.filter(pl.col("game_type") == REGULAR_SEASON)
    week = pl.col("week").cast(pl.Utf8).cast(pl.Float64).cast(pl.Int64)
    home = pl.col("team1").replace(nfeloqb.ABBREVIATIONS)
    away = pl.col("team2").replace(nfeloqb.ABBREVIATIONS)
    gid = (pl.col("season").cast(pl.Utf8) + "_" + week.cast(pl.Utf8).str.zfill(2)
           + "_" + away + "_" + home)
    probs = [(pl.col(c).cast(pl.Float64) if c in rows.columns
              else pl.lit(None, dtype=pl.Float64)).alias(a)
             for c, a in (("elo_prob1", "base_prob"), ("qbelo_prob1", "qb_prob"))]
    sides = []
    for n, side, own, opp in (("1", True, "score1", "score2"), ("2", False, "score2", "score1")):
        sides.append(rows.filter(~nfeloqb._blank(n)).select(
            gid.alias("game_id"), pl.col("season").cast(pl.Int64), week.alias("week"),
            pl.col("date").cast(pl.Utf8),
            pl.col(f"team{n}").replace(nfeloqb.ABBREVIATIONS).alias("team"),
            pl.lit(side).alias("home"),
            pl.col(f"qb{n}").alias("qb"),
            pl.col(f"qb{n}_value_pre").cast(pl.Float64).alias("value"),
            pl.col(f"qb{n}_adj").cast(pl.Float64).alias("adj"),
            pl.col(own).cast(pl.Int64).alias("score"),
            pl.col(opp).cast(pl.Int64).alias("opp_score"),
            *probs))
    tg = pl.concat(sides).select(*TEAM_GAME_SCHEMA).sort("team", "season", "week")
    link = _schedule(rows).with_columns(
        pl.col("game_id").shift(1).over("team").alias("prev_game_id"),
        pl.col("season").shift(1).over("team").alias("prev_season"),
        pl.col("date").shift(1).over("team").alias("prev_date"),
    ).select("team", "game_id", "prev_game_id", "prev_season", "prev_date")
    return tg.join(link, on=["team", "game_id"], how="left")


def unreadable_games(rows: pl.DataFrame) -> int:
    """Team-games a blank side made unreadable: entries `_schedule`'s full row set carries
    for a team that `team_games` has no row for, because that side's qb, value or adjustment
    was null. Counts both sides of a two-sided blank, one each, alongside every one-sided
    blank's single side (#328). Reported on the run line so a hole is counted, not silently
    closed over the way the previous-game link used to close it (#301)."""
    return _schedule(rows).height - team_games(rows).height


def starters_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Who took the first pass attempt for each (game, team): the observed starter, off
    play-by-play. One row per (game_id, team) with `season`, `week` and `qb` -- the passer's
    id, so a frame built here joins the source's values through `with_values` and never by
    name. The standard `PBP_COLS` slice does not carry the passer, and a cache without it is
    refused by the column's name rather than read for a starter it cannot hold.
    """
    if PASSER not in pbp.columns:
        raise ValueError(f"play-by-play carries no '{PASSER}' column; the first pass attempt "
                         f"cannot name a starter without it (the cached PBP_COLS slice does "
                         f"not include it)")
    passes = pbp.filter((pl.col("play_type") == "pass") & pl.col(PASSER).is_not_null())
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
    quality measure both readers use, so a starter identified elsewhere still prices here."""
    here = tg.select(pl.col("team"), pl.col("game_id"), pl.col("date"),
                     pl.col("value").alias("arriving_value"), pl.col("adj").alias("arriving_adj"))
    there = tg.select(pl.col("team"), pl.col("game_id").alias("prev_game_id"),
                      pl.col("date").alias("prev_date"),
                      pl.col("value").alias("departing_value"))
    return (ev.drop([c for c in ("date", "prev_date", "departing_value", "arriving_value",
                                 "gap", "arriving_adj") if c in ev.columns])
              .join(here, on=["team", "game_id"], how="left")
              .join(there, on=["team", "prev_game_id"], how="left")
              .with_columns((pl.col("arriving_value") - pl.col("departing_value")).alias("gap")))


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
    previous game day among the changes, before which no change could have been known."""
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
                  pl.col("prev_date").min().alias("frozen_before"))
              .with_columns((pl.col("home_gap").fill_null(0.0)
                             - pl.col("away_gap").fill_null(0.0)).alias("net_gap"))
              .select(*EVENT_GAME_SCHEMA)
              .sort("season", "week", "game_id"))


# --- the archive, read as the two readers need it -------------------------------------------

def _poll_day(col: str = "captured_at") -> pl.Expr:
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
    dropped, so the readers can count what they could not see."""
    days = polls.with_columns(_poll_day().alias("poll_day"))
    out = _last_before(days, games, "frozen_before", "frozen")
    out = _last_before(days, out, "date", "close")
    return out.with_columns(pl.col("frozen").is_null().alias("censored"))


def _log_loss(prob: float, y: float, eps: float = 1e-15) -> float:
    p = min(max(prob, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def _home_prob(spread: float) -> float:
    """`MarketBaseline`'s conversion, so the arms differ in the spread and nothing else."""
    return normal_cdf(spread / MARGIN_SD)


def _results(tg: pl.DataFrame) -> pl.DataFrame:
    """Per game, the home result: 1 a home win, 0 a loss, 0.5 a tie; unplayed games absent."""
    home = tg.filter(pl.col("home") & pl.col("score").is_not_null())
    return home.select(
        pl.col("game_id"),
        pl.when(pl.col("score") > pl.col("opp_score")).then(1.0)
          .when(pl.col("score") < pl.col("opp_score")).then(0.0)
          .otherwise(0.5).alias("y"),
        pl.col("base_prob"), pl.col("qb_prob"))


# --- the gate ---------------------------------------------------------------------------------

def _moved(part: pl.DataFrame, state: pl.DataFrame, name: str) -> pl.DataFrame:
    """`quarterback.apply` over one week's event games, labelled `stale`, priced from the
    frozen line, with `state`; the moved spread comes back as `<name>`, the points added as
    `<name>_adjustment`, and `apply`'s own marker as `<name>_by` -- null on a game `apply`
    did not touch (either side missing from `state`) and the source's name on one it did.
    `<name>` itself is never null either way: `apply` leaves it at the frozen price, unmoved,
    on a game it did not touch, which is why a reader asking "did this move" reads the
    marker and not the spread."""
    games = part.select(pl.col("game_id"), pl.col("home_team"), pl.col("away_team"),
                        pl.col("frozen").alias("close_spread"),
                        pl.lit("stale").alias("price_source"))
    moved = quarterback.apply(games, state)
    return part.join(moved.select(pl.col("game_id"), pl.col("close_spread").alias(name),
                                  pl.col("qb_adjustment").alias(f"{name}_adjustment"),
                                  pl.col("adjusted_by").alias(f"{name}_by")),
                     on="game_id", how="left")


def _adjusted(frame: pl.DataFrame, rows: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """The two arms, per week.

    **`adjusted` is the gate's arm and it is the shipped seam**: `nfeloqb.state(rows,
    as_of=<the week's first game day>)` -- rows strictly before that day (#272) -- handed
    to `quarterback.apply`, exactly as `hub.models.ratings._rated_by_week` rated a week that
    had kicked off until #299 pulled the adjustment. That state's latest row for every team
    is its *previous* game, so on an event game the arm prices the departing starter's
    adjustment: the source's file carries a new starter on the row of his first game and on
    no earlier row, and the replay cannot know him before it. Until the review of
    2026-09-13 this arm read the event game's own row -- the arriving starter known with
    certainty -- which is a better estimator than the one being gated, and a `diff` biased
    toward ADOPT.

    **`oracle` is a diagnostic and not the gate**: the same estimator with the week's own
    rows as the state, the arriving starter known. It answers what the mechanism could do
    if the state were timely, which #270's disposition needs beside the gate's answer, and
    it is never read by the rule.
    """
    out = []
    for (season, week), part in frame.group_by(["season", "week"], maintain_order=True):
        this = tg.filter((pl.col("season") == season) & (pl.col("week") == week))
        first_day = dt.date.fromisoformat(str(this["date"].min()))
        shipped = nfeloqb.state(rows, as_of=first_day)
        known = this.select(pl.col("team"), pl.col("qb"), pl.col("value").alias("qb_value"),
                            pl.col("adj").alias("qb_adj"),
                            pl.lit(0, dtype=pl.Int64).alias("tenure"),
                            pl.col("date").alias("as_of")).sort("team")
        out.append(_moved(_moved(part, shipped, "adjusted"), known, "oracle"))
    return pl.concat(out)


PAIRED_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8,
    "away_team": pl.Utf8, "net_gap": pl.Float64, "frozen": pl.Float64,
    "adjusted_adjustment": pl.Float64, "adjusted": pl.Float64,
    "oracle_adjustment": pl.Float64, "oracle": pl.Float64, "close": pl.Float64,
    "y": pl.Float64, "diff": pl.Float64, "oracle_diff": pl.Float64, "ceiling": pl.Float64,
}


def gate_rows(polls: pl.DataFrame, games: pl.DataFrame, tg: pl.DataFrame,
              rows: pl.DataFrame) -> pl.DataFrame:
    """The paired frame the gate reads: one row per scored, uncensored event game.

    `diff` is log-loss of the frozen price minus log-loss of the shipped arm's, positive
    when the adjustment helped; `oracle_diff` the same for the diagnostic arm, never read by
    the rule; `ceiling` is log-loss of the frozen price minus that of the last snapshot
    before the game day -- what the betting market's own repricing recovered, the declared
    ceiling arm. All on the home result, ties 0.5, through `MarketBaseline`'s conversion.
    `rows` are the source's rows the shipped seam builds its state from.

    **The not-null guard is on `adjusted_by` and `oracle_by`, the adjustment's own markers,
    never on `adjusted` or `oracle` themselves.** `quarterback.apply` leaves the spread
    column at the frozen price -- unmoved, and therefore never null -- on a game it did not
    touch, so a game whose team has no prior row in the state (a debut, or a name the source
    never carries before this week) used to survive this filter and enter at a manufactured
    zero difference, inflating n and shrinking both the mean and the clustered sd. The
    marker is null exactly where `apply` did not touch the row, on either arm, so excluding
    on it drops the game rather than counting it as no effect.
    """
    have = priced(polls, games).filter(~pl.col("censored") & pl.col("close").is_not_null())
    have = have.join(_results(tg).select("game_id", "y"), on="game_id", how="inner")
    if have.is_empty():
        return pl.DataFrame(schema=PAIRED_SCHEMA)
    have = _adjusted(have, rows, tg).filter(pl.col("adjusted_by").is_not_null()
                                            & pl.col("oracle_by").is_not_null())
    diff, oracle, ceiling = [], [], []
    for r in have.iter_rows(named=True):
        base = _log_loss(_home_prob(r["frozen"]), r["y"])
        diff.append(base - _log_loss(_home_prob(r["adjusted"]), r["y"]))
        oracle.append(base - _log_loss(_home_prob(r["oracle"]), r["y"]))
        ceiling.append(base - _log_loss(_home_prob(r["close"]), r["y"]))
    return (have.with_columns(pl.Series("diff", diff, dtype=pl.Float64),
                              pl.Series("oracle_diff", oracle, dtype=pl.Float64),
                              pl.Series("ceiling", ceiling, dtype=pl.Float64))
                .select(*PAIRED_SCHEMA))


def pilot(tg: pl.DataFrame, games: pl.DataFrame) -> dict[str, Any]:
    """The power input the pre-registration names: the source's own base and
    quarterback-adjusted probabilities scored on the same event games, per season. The
    target is the absolute mean of the season means -- the effect there is to find if this
    estimator reproduces the source's gain -- and `season_sd` their spread, NaN with fewer
    than two seasons. Not a verdict: the source's arm on nflfastR outcomes, not this
    estimator on a frozen line."""
    scored = (games.select("game_id", "season")
                   .join(_results(tg), on="game_id", how="inner")
                   .drop_nulls(["base_prob", "qb_prob"]))
    per = []
    for season, part in scored.group_by("season", maintain_order=True):
        diffs = [_log_loss(b, y) - _log_loss(q, y)
                 for b, q, y in zip(part["base_prob"], part["qb_prob"], part["y"], strict=True)]
        per.append({"season": int(season[0]), "n": len(diffs),
                    "mean": statistics.fmean(diffs),
                    "sd": statistics.stdev(diffs) if len(diffs) > 1 else float("nan")})
    per.sort(key=lambda d: d["season"])
    means = [d["mean"] for d in per]
    return {"seasons": len(per), "n": scored.height, "per_season": per,
            "target": abs(statistics.fmean(means)) if means else float("nan"),
            "season_sd": statistics.stdev(means) if len(means) > 1 else float("nan")}


def mde_at(k: int, season_sd: float) -> float:
    """The MDE at `k` event-seasons: `(t(0.975, k-1) + z(0.80)) * s / sqrt(k)`, on the t
    reference `docs/gate-power.md` restated on 2026-09-07 -- one function with the gate's,
    handed the season-clustered standard error `s / sqrt(k)`."""
    return experiment.minimum_detectable_effect(season_sd / math.sqrt(k), k)


def event_seasons_needed(target: float, season_sd: float, *, cap: int = 100) -> int | None:
    """The smallest number of event-seasons at which the MDE is at or below the target, or
    None when none up to `cap` reaches it -- or when either input is not a number."""
    if not (math.isfinite(target) and math.isfinite(season_sd)) or target <= 0:
        return None
    for k in range(2, cap + 1):
        if mde_at(k, season_sd) <= target:
            return k
    return None


def run(paired: pl.DataFrame, *, needed: int | None, ceiling: bool = True,
        width_path: Path | None = None, record_width: bool = False) -> experiment.GateRun:
    """The pre-registered precondition first, then -- only if it clears -- one gate run
    through `experiment.run_gate`. `ceiling` hands the declared arm's rows to the rule so
    stage 2 can fire.

    **Fewer than `EVENT_SEASONS_MINIMUM` event-seasons is NOT-RUNNABLE ahead of the summary,
    the house verdict, the width history and the rendered lines -- not a verdict string
    swapped in after they have already run.** `run_gate` bootstraps an interval, decides
    ADOPT/REMOVE/SHOW, renders the CI and the stamp, and writes this run's width into
    `width_path` as a side effect of being called at all; computing any of that from an
    underpowered frame and then only replacing the verdict sentence would publish the
    interval it exists to keep unpublished and record a width history entry for a run with
    no license to have one. So the count is read directly off `paired` and, below the
    minimum, `run_gate` is never called: the verdict is NOT-RUNNABLE, `lines` is empty, and
    nothing is written to `width_path`.
    """
    k = paired["season"].n_unique() if paired.height else 0
    if k < EVENT_SEASONS_MINIMUM:
        need = (f"the pilot says {needed} event-seasons are needed at 80% power"
                if needed is not None else "the pilot cannot say how many event-seasons are "
                                           "needed (no spread across seasons yet)")
        verdict = (
            "NOT-RUNNABLE",
            f"NOT RUNNABLE: {k} event-season(s) in the archive against a pre-registered "
            f"minimum of {EVENT_SEASONS_MINIMUM} (ADR-0019: no gate runs at fewer than three "
            f"seasons); {need}. No verdict is read. Since #300 this diagnostic's NOT-RUNNABLE "
            f"is an exemption from firing below the minimum, not a bar to the module's ADOPT "
            f"condition, which is #221's line-move coefficient (docs/qb-adjustment.md); this "
            f"is not-runnable, not a null.")
        return experiment.GateRun(summary={}, seasons=pl.DataFrame(), verdict=verdict,
                                  lines=[], stamped=paired)
    arm = (experiment.Ceiling(CEILING_ARM, paired["ceiling"].to_numpy())
           if ceiling and "ceiling" in paired.columns and paired.height else None)
    return run_gate(
        paired, cluster=SEASON_CLUSTER, actions=ACTIONS, name="quarterback_gate",
        arm_a="the frozen line", arm_b="quarterback-adjusted", unit="log-loss per event game",
        places=4, ceiling=arm,
        width_path=width_path if width_path is not None else experiment.WIDTH_STATE,
        record_width=record_width)


# --- the study --------------------------------------------------------------------------------

def _week_means(polls: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Per event game, the mean move over every *other* archived game of the same week
    between the same two poll days -- the week fixed effect as the subtraction it is; zero
    where no other game was polled on both days."""
    days = (polls.with_columns(_poll_day().alias("poll_day"))
                 .sort("game_id", "captured_at")
                 .group_by("game_id", "week", "poll_day", maintain_order=True)
                 .agg(pl.col("close_spread").last()))
    out = []
    for r in rows.iter_rows(named=True):
        f_day, c_day = r["frozen_day"], r["close_day"]
        others = days.filter((pl.col("week") == r["week"]) & (pl.col("game_id") != r["game_id"])
                             & pl.col("poll_day").is_in([f_day, c_day]))
        pivot = (others.group_by("game_id").agg(
            pl.col("close_spread").filter(pl.col("poll_day") == f_day).first().alias("a"),
            pl.col("close_spread").filter(pl.col("poll_day") == c_day).first().alias("b"))
                       .drop_nulls())
        moves = (pivot["b"] - pivot["a"]).to_list()
        out.append({"game_id": r["game_id"],
                    "week_mean": statistics.fmean(moves) if moves else 0.0,
                    "week_others": len(moves)})
    return pl.DataFrame(out, schema={"game_id": pl.Utf8, "week_mean": pl.Float64,
                                     "week_others": pl.Int64})


def _change_points(polls: pl.DataFrame, rows: pl.DataFrame, floor: float) -> list[float | None]:
    """Per event game, days from the previous game day to the first poll after the frozen
    one whose move from the frozen price clears `floor * sqrt(days since the frozen poll)`;
    None where no poll does."""
    days = polls.sort("game_id", "captured_at")
    out: list[float | None] = []
    for r in rows.iter_rows(named=True):
        later = days.filter((pl.col("game_id") == r["game_id"])
                            & (pl.col("captured_at") > r["frozen_at"]))
        seen: float | None = None
        for p in later.iter_rows(named=True):
            elapsed = (p["captured_at"] - r["frozen_at"]).total_seconds() / 86400.0
            if abs(p["close_spread"] - r["frozen"]) > floor * math.sqrt(max(elapsed, 0.0)):
                seen = float((p["captured_at"].date()
                              - dt.date.fromisoformat(r["frozen_before"])).days)
                break
        out.append(seen)
    return out


STUDY_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "net_gap": pl.Float64,
    "changes": pl.UInt32, "frozen": pl.Float64, "close": pl.Float64, "move": pl.Float64,
    "week_mean": pl.Float64, "week_others": pl.Int64, "adjusted_move": pl.Float64,
    "window_days": pl.Float64, "days_to_change": pl.Float64,
}


def study_rows(polls: pl.DataFrame, games: pl.DataFrame,
               floor_per_root_day: float | None = None) -> pl.DataFrame:
    """One row per uncensored event game: the move from the frozen price to the last
    snapshot before the game day, the week's mean move over the same two poll days, the
    week-adjusted move, the net gap, the window in days, and -- given a floor per root-day
    -- the change-point in days from the previous game day."""
    have = priced(polls, games).filter(~pl.col("censored") & pl.col("close").is_not_null())
    if have.is_empty():
        return pl.DataFrame(schema=STUDY_SCHEMA)
    have = have.with_columns(_poll_day("frozen_at").alias("frozen_day"),
                             _poll_day("close_at").alias("close_day"),
                             (pl.col("close") - pl.col("frozen")).alias("move"),
                             ((pl.col("close_at") - pl.col("frozen_at")).dt.total_seconds()
                              / 86400.0).alias("window_days"))
    have = have.join(_week_means(polls, have), on="game_id", how="left")
    changes = (_change_points(polls, have, floor_per_root_day)
               if floor_per_root_day is not None else [None] * have.height)
    return (have.with_columns((pl.col("move") - pl.col("week_mean")).alias("adjusted_move"),
                              pl.Series("days_to_change", changes, dtype=pl.Float64))
                .select(*STUDY_SCHEMA))


def study_mde(*, n: int, sd_gap: float, window_days: float, floor_per_root_day: float) -> float:
    """The coefficient's MDE before the run, as pre-registered: `(t(0.975, n-1) + z(0.80))
    * floor_window / (sd(gap) * sqrt(n))`, the floor per window `floor_per_root_day *
    sqrt(window_days)`; the cluster is the game."""
    if n < 2 or not (sd_gap > 0):
        return float("nan")
    se = floor_per_root_day * math.sqrt(window_days) / (sd_gap * math.sqrt(n))
    return experiment.minimum_detectable_effect(se, n)


def study_fit(rows: pl.DataFrame, *, floor_per_root_day: float) -> dict[str, float]:
    """The slope of the week-adjusted move on the net gap, with its error from the floor.

    Ordinary least squares with an intercept; the standard error is the noise floor per
    window over the gap's spread and root n rather than the residual's, because a season of
    events cannot resolve its own residual against a floor measured on twelve games. `t`
    against the benchmark says whether the betting market moved as the source would.
    """
    n = rows.height
    nan = float("nan")
    if n < 2:
        return {"n": float(n), "beta": nan, "se": nan, "mde": nan, "benchmark": BENCHMARK,
                "t_vs_benchmark": nan, "sd_gap": nan, "floor_window": nan}
    x = rows["net_gap"].to_numpy().astype(float)
    y = rows["adjusted_move"].to_numpy().astype(float)
    sd_gap = float(np.std(x, ddof=1))
    beta = float(np.polyfit(x, y, 1)[0]) if sd_gap > 0 else nan
    window = rows["window_days"].to_numpy().astype(float).mean()
    floor_window = floor_per_root_day * math.sqrt(float(window))
    se = floor_window / (sd_gap * math.sqrt(n)) if sd_gap > 0 else nan
    return {"n": float(n), "beta": beta, "se": se,
            "mde": experiment.minimum_detectable_effect(se, n), "benchmark": BENCHMARK,
            "t_vs_benchmark": (beta - BENCHMARK) / se if se and math.isfinite(se) else nan,
            "sd_gap": sd_gap, "floor_window": floor_window}


# --- the entry point --------------------------------------------------------------------------

def archive(season: int, base: Path | None) -> pl.DataFrame:
    """The season's polls off the store, in the shape the readers take; empty on a fresh
    clone. Read here rather than through `hub.fetch.odds`, whose reader is its own."""
    schema = {"game_id": pl.Utf8, "close_spread": pl.Float64, "captured_at": pl.Datetime("us"),
              "week": pl.Int64}
    if "lines" not in store.tables(base):
        return pl.DataFrame(schema=schema)
    got = store.sql("SELECT game_id, close_spread, captured_at, week FROM lines "
                    "WHERE league = 'nfl' AND season = ?", params=[season], base=base)
    return got.with_columns(pl.col("week").cast(pl.Utf8).cast(pl.Int64)).select(*schema)


def noise_floor_per_root_day(
        parts: Sequence[tuple[int, pl.DataFrame, pl.DataFrame | None]],
) -> tuple[float, int, tuple[int, ...]]:
    """#214's own same-quarterback floor per root-day, computed the way
    `hub.fetch.odds.noise_floor_report` computes its "floor" line -- not re-derived: frozen
    lookaheads excluded first, then intervals `odds.line_moves` marks `same_qb` false or
    unknown, off a starters frame handed to that same call.

    **One `(season, polls, starters)` triple per season, because the chart is per season
    too.** `odds._starter_at` is a backward as-of join, so a poll needs a chart *of its own
    season* before it to be conditioned at all -- a chart fetched for one season and handed
    to every season's polls (#331) leaves every other season's `same_qb` null, which
    `odds.line_moves` cannot tell apart from a genuine unknown and this function used to
    drop the same way: silently, off the floor, with the run line still calling it
    same-quarterback. `polls` is a season's own slice of the archive (`archive`'s shape);
    `starters` is the depth-chart frame `odds._qb_starters` returns for that season, or
    `None` where the chart could not be read. Each season's live intervals are filtered on
    `same_qb` only when that season has a chart; a season with none contributes its live
    intervals unconditioned -- not dropped -- and is named in the returned `not_applied`
    tuple rather than silently counted as though excluded.

    **`starters` is never this module's own team-game rows.** `apply`'s as-of join reads
    "who was listed from this moment on"; this module's `date` is *kickoff*, so a change
    dated at kickoff is invisible to every poll of that game, which by construction all
    precede its own kickoff (#330). A starters frame built off kickoff dates therefore never
    excludes the interval it exists to exclude.

    NaN and zero with nothing left to measure across every season; `not_applied` names every
    season handed no chart (or an empty one), in the order given, whether or not it had any
    live intervals to contribute.
    """
    live_parts: list[pl.DataFrame] = []
    not_applied: list[int] = []
    for season, polls, starters in parts:
        if polls.is_empty():
            continue
        have = starters is not None and not starters.is_empty()
        moves = odds.line_moves(odds.staleness(polls), starters if have else None)
        live = moves.filter(~pl.col("frozen"))
        if have:
            live = live.filter(pl.col("same_qb"))
        else:
            not_applied.append(season)
        live_parts.append(live)
    if not live_parts:
        return float("nan"), 0, tuple(not_applied)
    combined = pl.concat(live_parts)
    if combined.is_empty():
        return float("nan"), 0, tuple(not_applied)
    floor = odds.noise_floor(combined, bootstrap=1)
    return float(floor["sd_per_root_day"]), int(floor["games"]), tuple(not_applied)


def _event_lines(ev: pl.DataFrame, games: pl.DataFrame, since: int) -> list[str]:
    lines = [f"  starter changes since {since}, observed off the source's starter column "
             f"(never the injury report); an event is a change between two games of one "
             f"season:"]
    for season in sorted(set(ev["season"].to_list())):
        part = ev.filter(pl.col("season") == season)
        ins = part.filter(pl.col("in_season"))
        off = part.height - ins.height
        gaps = ins["gap"].drop_nulls().to_list()
        sd = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")
        n_games = games.filter(pl.col("season") == season).height
        lines.append(f"    {season}: {ins.height} changes on {n_games} event games, "
                     f"gap sd {sd:.1f} value units (n={len(gaps)}); {off} offseason "
                     f"change(s) not events")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Starter-change events off the pinned nfeloqb file, the quarterback "
                    "adjustment's gate over the frozen line (#291), and the line-move study "
                    "(#221). Reads the caches; fetches nothing.")
    ap.add_argument("--events", action="store_true", help="count the events per season")
    ap.add_argument("--gate", action="store_true",
                    help="run the pre-registered gate on the archive's event games")
    ap.add_argument("--ceiling", action=argparse.BooleanOptionalAction, default=True,
                    help=f"hand the rule the declared ceiling arm ({CEILING_ARM}); on by "
                         f"default, the shipped configuration -- pass --no-ceiling to turn "
                         f"stage 2 off (#302)")
    ap.add_argument("--study", action="store_true", help="run the line-move study")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD,
                    help="the last season whose snapshot archive is read; every season from "
                         "--since through this one is assembled into one run (default "
                         "SEASON_AHEAD)")
    ap.add_argument("--since", type=int, default=2022,
                    help="the first season the event counts, the pilot cover, and the "
                         "gate's archive assembles from")
    ap.add_argument("--cache", type=Path, default=None, help="the nfeloqb cache directory")
    ap.add_argument("--store", type=Path, default=None, help="the processed store")
    a = ap.parse_args(argv)
    if not (a.events or a.gate or a.study):
        ap.print_usage()
        return 2
    if a.season < a.since:
        print(f"{PROG}: --season {a.season} is before --since {a.since}, so there is no "
              f"season for the gate's archive to assemble", file=sys.stderr)
        return 2

    rows = nfeloqb.read_rows(a.cache)
    if rows is None:
        return unavailable(PROG, "the cached nfeloqb file",
                           FileNotFoundError("nothing cached; run hub.fetch.nfeloqb --refresh"))
    since_rows = rows.filter(pl.col("season") >= a.since)
    tg = team_games(since_rows)
    ev = events(tg)
    games = event_games(in_season_events(ev))
    for line in _event_lines(ev, games, a.since):
        print(line)
    unreadable = unreadable_games(since_rows)
    print(f"  {unreadable} team-game(s) since {a.since} a blank side made unreadable (that "
          f"side dropped; the previous-game link is still built off the full schedule)")

    # #302: read every season's archive from --since through --season and concatenate,
    # rather than only --season's -- with `games` filtered to match. A gate over one
    # season's polls can never reach the three-season floor no matter how many the caches
    # hold; assembling the whole asked-for range is what lets it.
    seasons = list(range(a.since, a.season + 1))
    parts = [archive(s, a.store) for s in seasons]
    have_seasons = [s for s, p in zip(seasons, parts, strict=True) if not p.is_empty()]
    polls = pl.concat(parts)
    span = str(a.season) if a.since == a.season else f"{a.since}-{a.season}"
    if polls.is_empty():
        print(f"  no snapshot archive for {span} here: the gate and the study have no "
              f"frozen price to read, and both are not established")
    else:
        print(f"  archive: {polls['game_id'].n_unique()} games, "
              f"{polls['captured_at'].n_unique()} polls, "
              f"{polls['captured_at'].min():%Y-%m-%d} to {polls['captured_at'].max():%Y-%m-%d}, "
              f"{len(have_seasons)} of {len(seasons)} season(s) asked for have polls "
              f"({', '.join(str(s) for s in have_seasons)})")
    season_games = games.filter(pl.col("season").is_in(seasons))
    seen = priced(polls, season_games)
    print(f"  {span}: {season_games.height} event games; "
          f"{int(seen['censored'].sum())} censored (no snapshot before the change could be "
          f"known), {seen.filter(~pl.col('censored') & pl.col('close').is_not_null()).height} "
          f"with a frozen price and a pre-game one")

    if a.gate:
        pil = pilot(tg, games)
        needed = event_seasons_needed(pil["target"], pil["season_sd"])
        print(f"  pilot (the source's own two columns on event games, {pil['n']} games over "
              f"{pil['seasons']} seasons): target {pil['target']:.4f} log-loss, between-season "
              f"sd {pil['season_sd']:.4f}; event-seasons needed at 80% power: "
              f"{needed if needed is not None else 'more than 100, or not computable'}")
        for d in pil["per_season"]:
            print(f"    {d['season']}: n={d['n']}, mean {d['mean']:+.4f}, sd {d['sd']:.4f}")
        paired = gate_rows(polls, season_games, tg, rows)
        # The width history is kept only when a run has an interval to keep; a run over
        # no rows would file a NaN width under the gate's name.
        got = run(paired, needed=needed, ceiling=a.ceiling, record_width=paired.height > 0)
        print(f"  gate: {paired.height} scored event games over "
              f"{paired['season'].n_unique() if paired.height else 0} event-season(s); the "
              f"arm is the shipped seam, nfeloqb.state as of the week's first game day")
        if paired.height:
            print(f"  diagnostic, not the gate -- the oracle arm (arriving starter known): "
                  f"mean {paired['oracle_diff'].mean():+.4f} log-loss against the shipped "
                  f"arm's {paired['diff'].mean():+.4f} over n={paired.height}")
        for line in got.lines:
            print(line)
        print(f"  {got.verdict[0]}: {got.verdict[1]}")

    if a.study:
        # #330: the depth chart #214's own floor conditions on -- never this module's own
        # kickoff-dated rows, which a change can never be seen through (a game's polls all
        # precede its own kickoff). #331: one chart per season in the assembled range, a
        # bounded loop over `seasons` (never over teams or games) -- `odds._starter_at`'s
        # as-of join means a chart fetched for one season cannot condition another season's
        # polls, so the last season's chart alone left every earlier season's intervals
        # silently unconditioned. Each fetch is guarded exactly as `odds.noise_floor_report`
        # guards its own, and skipped for a season with no polls to condition.
        study_parts: list[tuple[int, pl.DataFrame, pl.DataFrame | None]] = []
        qb_notes: list[str] = []
        for s, p in zip(seasons, parts, strict=True):
            if p.is_empty():
                continue
            chart: pl.DataFrame | None = None
            try:
                chart = odds._qb_starters(s)
            except Exception as exc:                         # pragma: no cover - network
                qb_notes.append(f"{s} ({type(exc).__name__}: {exc})")
            study_parts.append((s, p, chart))
        floor, floor_games, not_applied = noise_floor_per_root_day(study_parts)
        if not study_parts or not not_applied:
            label = "same-quarterback floor"
        elif len(not_applied) == len(study_parts):
            label = "all-games floor, SAME-QUARTERBACK NOT APPLIED"
        else:
            label = "same-quarterback floor, PARTIAL"
        qb_note = (f" (same-quarterback NOT applied for "
                  f"{', '.join(str(s) for s in not_applied)}: {'; '.join(qb_notes)})"
                  if not_applied else "")
        used = floor if math.isfinite(floor) else 0.40
        gaps = in_season_events(ev)["gap"].drop_nulls().to_list()
        sd_gap = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")
        per_season = games.group_by("season").len()["len"].to_list()
        n_typical = int(statistics.median(per_season)) if per_season else 0
        print(f"  study: {label} {floor:.3f} points per root-day off {floor_games} live "
              f"games of this archive{qb_note} (#214 recorded 0.40 on 12; used {used:.2f}); "
              f"gap sd {sd_gap:.1f} value units over {len(gaps)} events since {a.since}")
        mde = study_mde(n=n_typical, sd_gap=sd_gap, window_days=7.0, floor_per_root_day=used)
        print(f"  MDE before the run at a season of {n_typical} event games, a 7-day window: "
              f"{mde:.4f} points per value unit against a benchmark of {BENCHMARK:.3f}")
        rows_ = study_rows(polls, season_games,
                           floor_per_root_day=floor if math.isfinite(floor) else None)
        fit = study_fit(rows_, floor_per_root_day=used)
        if rows_.is_empty():
            print(f"  coefficient: not established -- no uncensored event game with a frozen "
                  f"and a pre-game price in the {span} archive")
        else:
            seen_change = rows_["days_to_change"].drop_nulls()
            print(f"  coefficient: {fit['beta']:+.4f} points per value unit (se {fit['se']:.4f} "
                  f"from the floor, MDE {fit['mde']:.4f}) over n={int(fit['n'])} event games "
                  f"(cluster = game); t against {BENCHMARK:.3f}: {fit['t_vs_benchmark']:+.2f}")
            print(f"  change-point: seen on {seen_change.len()} of {rows_.height} games, median "
                  f"{seen_change.median() if seen_change.len() else float('nan')} days after "
                  f"the previous game day; the depth-chart date is not cached, so timing "
                  f"against the report date is not established")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
