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
over event games, log-loss of the frozen price moved by `hub.models.quarterback.apply` --
the shipped estimator, on a row labelled as having no live price, with the state the live
path would read on the morning of the game -- against the frozen price unmoved. The frozen
price is the last snapshot captured before the Eastern game day of the changed team's
previous game: the price the adjustment would actually have replaced, and not the close.
The cluster is the season; the rule is the house rule and nothing beside it; fewer than
three event-seasons is NOT-RUNNABLE ahead of every branch (ADR-0019's floor), and the
verdict sentence names the event-seasons the pilot says are needed. Anything but ADOPT is
#270's pull trigger.

**The study** (#221) reads the same events and lands beside the gate; this commit is
the events and the gate.

Nothing here fetches. The nfeloqb cache and the snapshot archive are what is read, and a
season the caches do not hold is reported as not established.

    uv run python -m hub.models.starter_change --events --gate
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from hub import store
from hub.cli import unavailable
from hub.config import SEASON_AHEAD
from hub.fetch import nfeloqb
from hub.models import experiment, quarterback
from hub.models.experiment import SEASON_CLUSTER, run_gate
from hub.models.market import MARGIN_SD, normal_cdf

PROG = "hub.models.starter_change"

# ADR-0019: no gate in the repo runs at fewer than three seasons, and one that did should
# say so. Pre-registered as the gate's first precondition.
EVENT_SEASONS_MINIMUM = 3

# What the gate's ceiling arm is called where it prints (#138): the betting market's own
# repricing, scored against the same frozen line.
CEILING_ARM = "the betting market's own repricing, the last snapshot before the game day"

ACTIONS = experiment.Actions(
    adopt="The quarterback adjustment has earned its place: the sentence 'ungated in this "
          "repo' leaves docs/qb-adjustment.md and the -qb suffix stays for the record.",
    remove="The quarterback adjustment leaves ratings and survivor today and is reachable "
           "only from this harness (#270's pull trigger).",
    show="The quarterback adjustment leaves ratings and survivor today and is reachable "
         "only from this harness (#270's pull trigger): a null is not a pass for an arm "
         "that is in the published path on no verdict.")

REGULAR_SEASON = "REG"
PASSER = "passer_player_id"

TEAM_GAME_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "date": pl.Utf8,
    "team": pl.Utf8, "home": pl.Boolean, "qb": pl.Utf8, "value": pl.Float64,
    "adj": pl.Float64, "score": pl.Int64, "opp_score": pl.Int64,
    "base_prob": pl.Float64, "qb_prob": pl.Float64,
}


# --- the event construction ---------------------------------------------------------------

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
    return pl.concat(sides).select(*TEAM_GAME_SCHEMA).sort("team", "season", "week")


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
    """Every game where a team's starter differs from its starter on its previous game.

    `starters` is one row per (team, game) with `game_id`, `season`, `week`, `team`, `qb` --
    `team_games` or `starters_from_pbp` -- ordered within a team by season and week. The
    departing starter is the previous row's, the arriving one this row's; `in_season` is
    whether the previous game was the same season, and a change across the offseason is
    flagged rather than dropped so it can be counted as censored. A team's first row has
    nothing to differ from and is never an event. Where the frame carries `value`, `adj`
    and `date` (the source's rows do), the ex-ante values ride along: the departing
    starter's value off his last start, the arriving starter's value and adjustment off
    the event row, and `gap`, arriving minus departing.
    """
    carried = [c for c in ("date", "value", "adj") if c in starters.columns]
    prev = {c: pl.col(c).shift(1).over("team") for c in ("qb", "game_id", "season", *carried)}
    # Every shifted column in one pass, *before* the rows that are not events are dropped:
    # shifted after the filter, "the previous row" is the previous event and not the
    # previous game, and the departing starter's value is another change's.
    shifted = [prev["qb"].alias("departing"), pl.col("qb").alias("arriving"),
               prev["game_id"].alias("prev_game_id"), prev["season"].alias("prev_season")]
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
                   .with_columns((pl.col("season") == pl.col("prev_season")).alias("in_season")))
    keep = ["game_id", "season", "week", "team", "departing", "arriving", "prev_game_id",
            "prev_season", "in_season"]
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

def _adjusted(frame: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """The arm: `quarterback.apply` on each week's event games, labelled `stale`, with that
    week's state -- every team's starter and adjustment off its own row for the week, which
    is the row the live path reads on the morning of the game. One state per week because a
    team plays at most once in one."""
    out = []
    for (season, week), part in frame.group_by(["season", "week"], maintain_order=True):
        rows = tg.filter((pl.col("season") == season) & (pl.col("week") == week))
        state = rows.select(pl.col("team"), pl.col("qb"), pl.col("value").alias("qb_value"),
                            pl.col("adj").alias("qb_adj"),
                            pl.lit(0, dtype=pl.Int64).alias("tenure"),
                            pl.col("date").alias("as_of")).sort("team")
        games = part.select(pl.col("game_id"), pl.col("home_team"), pl.col("away_team"),
                            pl.col("frozen").alias("close_spread"),
                            pl.lit("stale").alias("price_source"))
        moved = quarterback.apply(games, state)
        out.append(part.join(moved.select(pl.col("game_id"),
                                          pl.col("close_spread").alias("adjusted"),
                                          pl.col("qb_adjustment")),
                             on="game_id", how="left"))
    return pl.concat(out)


PAIRED_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8,
    "away_team": pl.Utf8, "net_gap": pl.Float64, "frozen": pl.Float64,
    "qb_adjustment": pl.Float64, "adjusted": pl.Float64, "close": pl.Float64,
    "y": pl.Float64, "diff": pl.Float64, "ceiling": pl.Float64,
}


def gate_rows(polls: pl.DataFrame, games: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """The paired frame the gate reads: one row per scored, uncensored event game.

    `diff` is log-loss of the frozen price minus log-loss of the adjusted one, positive when
    the adjustment helped; `ceiling` is log-loss of the frozen price minus that of the last
    snapshot before the game day -- what the betting market's own repricing recovered, the
    declared ceiling arm. Both on the home result, ties 0.5, through `MarketBaseline`'s
    conversion.
    """
    have = priced(polls, games).filter(~pl.col("censored") & pl.col("close").is_not_null())
    have = have.join(_results(tg).select("game_id", "y"), on="game_id", how="inner")
    if have.is_empty():
        return pl.DataFrame(schema=PAIRED_SCHEMA)
    have = _adjusted(have, tg).filter(pl.col("adjusted").is_not_null())
    diff, ceiling = [], []
    for r in have.iter_rows(named=True):
        base = _log_loss(_home_prob(r["frozen"]), r["y"])
        diff.append(base - _log_loss(_home_prob(r["adjusted"]), r["y"]))
        ceiling.append(base - _log_loss(_home_prob(r["close"]), r["y"]))
    return (have.with_columns(pl.Series("diff", diff, dtype=pl.Float64),
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
    """One gate run through `experiment.run_gate`, then the pre-registered precondition:
    fewer than `EVENT_SEASONS_MINIMUM` event-seasons is NOT-RUNNABLE ahead of the house
    rule, and the sentence names the count the pilot says is needed. `ceiling` hands the
    declared arm's rows to the rule so stage 2 can fire."""
    arm = (experiment.Ceiling(CEILING_ARM, paired["ceiling"].to_numpy())
           if ceiling and "ceiling" in paired.columns and paired.height else None)
    got = run_gate(
        paired, cluster=SEASON_CLUSTER, actions=ACTIONS, name="quarterback_gate",
        arm_a="the frozen line", arm_b="quarterback-adjusted", unit="log-loss per event game",
        places=4, ceiling=arm,
        width_path=width_path if width_path is not None else experiment.WIDTH_STATE,
        record_width=record_width)
    k = paired["season"].n_unique() if paired.height else 0
    if got.verdict[0] not in ("VOID", "NOT-RUNNABLE") and k < EVENT_SEASONS_MINIMUM:
        need = (f"the pilot says {needed} event-seasons are needed at 80% power"
                if needed is not None else "the pilot cannot say how many event-seasons are "
                                           "needed (no spread across seasons yet)")
        got = got._replace(verdict=(
            "NOT-RUNNABLE",
            f"NOT RUNNABLE: {k} event-season(s) in the archive against a pre-registered "
            f"minimum of {EVENT_SEASONS_MINIMUM} (ADR-0019: no gate runs at fewer than three "
            f"seasons); {need}. No verdict is read, the mark on every -qb row stands (#270), "
            f"and this is not-runnable, not a null."))
    return got


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
    ap.add_argument("--ceiling", action="store_true",
                    help=f"hand the rule the declared ceiling arm ({CEILING_ARM})")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD,
                    help="the season whose snapshot archive is read (default SEASON_AHEAD)")
    ap.add_argument("--since", type=int, default=2022,
                    help="the first season the event counts and the pilot cover")
    ap.add_argument("--cache", type=Path, default=None, help="the nfeloqb cache directory")
    ap.add_argument("--store", type=Path, default=None, help="the processed store")
    a = ap.parse_args(argv)
    if not (a.events or a.gate):
        ap.print_usage()
        return 2

    rows = nfeloqb.read_rows(a.cache)
    if rows is None:
        return unavailable(PROG, "the cached nfeloqb file",
                           FileNotFoundError("nothing cached; run hub.fetch.nfeloqb --refresh"))
    tg = team_games(rows.filter(pl.col("season") >= a.since))
    ev = events(tg)
    games = event_games(in_season_events(ev))
    for line in _event_lines(ev, games, a.since):
        print(line)

    polls = archive(a.season, a.store)
    if polls.is_empty():
        print(f"  no snapshot archive for {a.season} here: the gate and the study have no "
              f"frozen price to read, and both are not established")
    else:
        print(f"  archive: {polls['game_id'].n_unique()} games, "
              f"{polls['captured_at'].n_unique()} polls, "
              f"{polls['captured_at'].min():%Y-%m-%d} to {polls['captured_at'].max():%Y-%m-%d}")
    season_games = games.filter(pl.col("season") == a.season)
    seen = priced(polls, season_games)
    print(f"  {a.season}: {season_games.height} event games; "
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
        paired = gate_rows(polls, season_games, tg)
        # The width history is kept only when a run has an interval to keep; a run over
        # no rows would file a NaN width under the gate's name.
        got = run(paired, needed=needed, ceiling=a.ceiling, record_width=paired.height > 0)
        print(f"  gate: {paired.height} scored event games over "
              f"{paired['season'].n_unique() if paired.height else 0} event-season(s)")
        for line in got.lines:
            print(line)
        print(f"  {got.verdict[0]}: {got.verdict[1]}")

    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
