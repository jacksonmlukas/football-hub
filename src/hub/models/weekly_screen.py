"""Does any week-level feature predict a player's week *beyond what consensus already knows*?

A **screen**, in the sense `CONTEXT.md` defines: it asks "is this real?", never "is this better
than what it replaces?". Nothing here adopts anything. The gate that decides whether a Weekly
projection sets lineups is a different question with a different incumbent, and
[ADR-0015](../../../docs/adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md) records
why it has to be.

Committed rather than left in a scratchpad because these numbers steer Phase 2 of
`docs/weekly-projection-plan.md`, and ADR-0007's trigger is citation.

THE DESIGN, PRE-REGISTERED in `docs/weekly-projection-plan.md` before the first run:

  * Outcome is player-week PPR points. Controls are season-to-date PPG measured strictly
    before the week, and that week's `weekly-op` consensus ECR.
  * **One partial correlation per (season, week) cell.** Every player appears at most once
    inside a cell, so no correlation contains repeated measures -- signal-screens.md protocol
    item 3, which turned noise into an apparent 4-sigma result once already. Pooling
    player-weeks would inflate every t here by roughly the square root of fourteen.
  * A feature clears only if its pre-stated sign holds in **every** season and the pooled
    statistic clears `MIN_SE`. A sign that flips between seasons is a bug, not a signal
    (protocol item 4), and it is the cheapest diagnostic available.

THE CONFOUND, which the first run found and which no available data removes: `weekly-op` is
FantasyPros' Monday ranking, scraped a median of six days before kickoff. Any feature carrying
Tuesday-to-Sunday news beats it for that reason alone. `LEAD_DAYS` reports the distribution so
the split is visible rather than assumed -- see `docs/weekly-screen.md`.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np
import polars as pl

from hub.config import DRAFTED_POSITIONS, SEASON_COMPLETED
from hub.models.experiment import MIN_SE, expanding_weeks
from hub.names import player_key, practice_key

# 2021 is the first season `ff_opportunity` covers in the store; weekly consensus starts 2020,
# so the overlap is what bounds this.
SEASONS: tuple[int, ...] = tuple(range(2021, SEASON_COMPLETED + 1))

# The fantasy regular season. Weeks 15-17 are the playoffs and are reported apart rather than
# pooled; week 18 is meaningless in a league that ends at 17.
GATE_WEEKS: tuple[int, ...] = tuple(range(1, 15))

# Below this a season-to-date mean is three numbers and the control is not doing its job.
MIN_GAMES_BEFORE = 3
# A cell smaller than this is a correlation on noise. 40 is roughly a tenth of a normal week.
MIN_CELL = 40
# docs/snap-trend-signal.md: anchors 4 and 6 are null and flip sign. The trend needs about
# eight weeks of snaps before it says anything, so it is dark before then rather than weak.
TREND_MIN_WEEK = 8

# The cross-position weekly page. Position pages cannot fill a flex, which is what a lineup
# needs, so the incumbent has to be the one ranking that spans positions.
CONSENSUS_PAGE = "weekly-op"
# A scrape further from kickoff than this belongs to no week. The observed distribution is
# almost entirely 6 days, which is the confound named in the module docstring.
CONSENSUS_MAX_LEAD_DAYS = 8


class Feature(NamedTuple):
    """A candidate, with the sign written down before the run."""
    name: str
    sign: str          # "+", "-", or "0" for a pre-stated null
    min_week: int


FEATURES: tuple[Feature, ...] = (
    Feature("implied_total", "+", 1),
    Feature("own_spread", "?", 1),
    Feature("dvp", "+", 1),
    Feature("wind", "-", 1),
    Feature("rest", "?", 1),
    Feature("inj_sev", "-", 1),
    Feature("td_rate_prior", "0", 1),
    Feature("snap_trend", "+", TREND_MIN_WEEK),
    Feature("tgt_trend", "+", TREND_MIN_WEEK),
)

# Screened 2026-08-29 and **not** in FEATURES, deliberately. `route_trend` clears on its own
# (+0.034 at 2.5 se, 5/5 seasons) and is a *null against `snap_trend`*: the two correlate at
# **0.917**, their underlying shares at **0.963**, and put in the same joint screen they
# annihilate each other and leave nothing. Snap share is the stronger of the two (+0.043
# against +0.034), so the literature's access-beats-presence distinction does not survive
# here. Kept in the tree with its harness per ADR-0007, out of the default screen because a
# collinear twin in the control set destroys a real signal. `--routes` reproduces it.
ROUTE_TREND = Feature("route_trend", "+", TREND_MIN_WEEK)

CONTROLS: tuple[str, ...] = ("ppg_before", "ecr")
OUTCOME = "fantasy_points_ppr"

# **Usage**, in the `CONTEXT.md` sense: the count vector a week produces. The panel carries
# each of these and its own season-to-date mean, so a feature can be screened against the
# counts and not only against the total. That is the premise of the multiplier form -- a
# feature that moves points without moving counts cannot be applied as a Usage multiplier.
USAGE: tuple[str, ...] = ("targets", "receptions", "carries", "attempts", "tds")

# Yardage is not Usage -- it is Usage times an efficiency the repo deliberately does not
# project (yards per carry persists at r = 0.108). Its prior is carried anyway because
# `hub.models.weekly` needs a per-unit rate to turn projected counts into projected yards.
YARDS: tuple[str, ...] = ("receiving_yards", "rushing_yards", "passing_yards")

# The two negative scoring components. Not Usage and not yardage, but `components.SCORING`
# prices both at -2 and a projection that omits them is not projecting fantasy points -- it
# over-projected quarterbacks by +1.44 a week until these went in, which is almost exactly an
# interception a game.
TURNOVERS: tuple[str, ...] = ("passing_interceptions", "fumbles_lost_total")

# A verdict is one of three, not two. A pre-stated null that comes back significant in every
# season is a *finding* -- it is why the prediction was written down -- and folding it in with
# the rejections would lose the most informative outcome the screen can produce.
CLEARS, KILLED, NULL_BROKEN = "clears", "killed", "null-broken"
FINDINGS = (CLEARS, NULL_BROKEN)


# --- the pure core, which is the part that has to be right ------------------

def residual(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """`y` with the controls projected out. An intercept is added here, never by the caller."""
    x = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


# A residual smaller than this fraction of the original spread is floating-point dust, not a
# signal. Exact zero is the wrong test: a feature that is an exact linear function of a control
# residualises to ~1e-16 rather than to 0, and correlating two clouds of rounding error returns
# a number that looks like a finding.
DEGENERATE = 1e-10


def partial_r(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    """Correlation of `y` and `x` after both are residualised on the same controls.

    Both sides, which is what makes it partial rather than a regression coefficient: the
    quantity is scale-free and comparable across features that have no common units.

    Returns NaN when either residual has collapsed, so the caller drops the cell instead of
    reporting the correlation of two rounding errors.
    """
    ry, rx = residual(y, controls), residual(x, controls)
    if (rx.std() <= DEGENERATE * max(float(np.std(x)), 1.0)
            or ry.std() <= DEGENERATE * max(float(np.std(y)), 1.0)):
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def cell_correlations(panel: pl.DataFrame, feature: str, *, min_week: int = 1,
                      min_cell: int = MIN_CELL, outcome: str = OUTCOME,
                      controls: Sequence[str] = CONTROLS) -> pl.DataFrame:
    """One partial correlation per (season, week). No player appears twice inside a cell.

    `outcome` is a parameter because the same screen has to run against **Usage** -- targets,
    carries, attempts -- and not only against points. That is the premise of the multiplier
    form: a feature that moves points but not counts cannot be applied as a Usage multiplier,
    whatever it does to the total.
    """
    need = [outcome, feature, *controls]
    d = panel.filter(pl.col("week") >= min_week).drop_nulls(need)
    rows = []
    for (season, week), cell in d.group_by(["season", "week"]):
        if cell.height < min_cell:
            continue
        r = partial_r(cell[outcome].to_numpy().astype(float),
                      cell[feature].to_numpy().astype(float),
                      np.column_stack([cell[c].to_numpy().astype(float) for c in controls]))
        if not np.isnan(r):
            rows.append({"season": int(season), "week": int(week), "r": r, "n": cell.height})
    return pl.DataFrame(rows, schema={"season": pl.Int64, "week": pl.Int64,
                                      "r": pl.Float64, "n": pl.Int64})


def summarise(cells: pl.DataFrame) -> dict:
    """Pooled correlation, its standard error across cells, and the per-season means.

    The standard error is of the *cell* correlations, not of the pooled player-weeks. Cells
    within a season share players, so this is not fully independent either -- but it is the
    difference between a mild overstatement and the fourteen-fold one that pooling gives.
    """
    if cells.is_empty():
        return {"r": float("nan"), "se": float("nan"), "t": float("nan"),
                "cells": 0, "n": 0, "per_season": {}}
    r = cells["r"].to_numpy().astype(float)
    se = float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
    per = {int(s): float(cast(float, cells.filter(pl.col("season") == s)["r"].mean()))
           for s in sorted(cells["season"].unique().to_list())}
    return {"r": float(r.mean()), "se": se,
            "t": float(r.mean() / se) if se > 0 else 0.0,
            "cells": len(r), "n": int(cells["n"].sum()), "per_season": per}


def verdict(summary: dict, sign: str, *, min_se: float = MIN_SE) -> tuple[str, str]:
    """The pre-registered rule, both halves of it.

    A feature clears only if the sign it was given *before the run* holds in every season and
    the pooled statistic clears `min_se`. `sign="0"` is a pre-stated null: it clears when it
    behaves like one, and a null that comes back significant is reported as a finding against
    the pre-registration rather than quietly relabelled.
    """
    per = summary["per_season"]
    if not per:
        return KILLED, "nothing measured -- no cell reached the minimum"
    t = summary["t"]
    agree = sum((v > 0) == (summary["r"] > 0) for v in per.values())
    if sign == "0":
        if abs(t) < min_se:
            return CLEARS, f"null as pre-stated ({t:+.1f} se)"
        if agree == len(per):
            return NULL_BROKEN, (f"PRE-STATED NULL BROKEN: {summary['r']:+.4f} at {t:+.1f} se, "
                                 f"consistent in {agree}/{len(per)} seasons")
        return CLEARS, f"noisy, not a signal ({agree}/{len(per)} seasons agree)"
    held = agree if sign == "?" else (
        sum(v > 0 for v in per.values()) if sign == "+" else sum(v < 0 for v in per.values()))
    if held < len(per):
        return KILLED, (f"killed: sign holds in only {held}/{len(per)} seasons "
                        f"({summary['r']:+.4f} at {t:+.1f} se)")
    if abs(t) < min_se:
        return KILLED, f"killed: {held}/{len(per)} seasons but only {t:+.1f} se"
    return CLEARS, f"clears: {summary['r']:+.4f} at {t:+.1f} se, {held}/{len(per)} seasons"


def report(rows: Sequence[dict]) -> list[str]:
    """Lines, not prints -- the reason `hub.draft.report` exists."""
    out = ["", f"  {'feature':16} {'pre':>4} {'r':>9} {'t':>7} {'cells':>6}  verdict"]
    for row in rows:
        out.append(f"  {row['feature']:16} {row['sign']:>4} {row['r']:+9.4f} "
                   f"{row['t']:+7.2f} {row['cells']:6d}  {row['note']}")
    return out


# --- assembly, which needs the network --------------------------------------

def game_context(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """One row per (team, season, week): the facts published before kickoff.

    `spread_line` is home-relative and positive means the home team is favoured, so the away
    row negates it. The implied team total is `total/2 + own_spread/2` -- the market's own
    forecast of how many points this offence scores, which is the quantity a fantasy week is
    a share of.
    """
    import nflreadpy as nfl
    s = (nfl.load_schedules()
           .filter(pl.col("season").is_in(list(seasons)) & (pl.col("game_type") == "REG")))
    sides = []
    for home in (True, False):
        team, opp = ("home_team", "away_team") if home else ("away_team", "home_team")
        rest = "home_rest" if home else "away_rest"
        spread = pl.col("spread_line") if home else -pl.col("spread_line")
        sides.append(s.select(
            pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64),
            pl.col(team).alias("team"), pl.col(opp).alias("opp"),
            spread.alias("own_spread"), pl.col("total_line"),
            pl.col(rest).alias("rest"), pl.col("roof"), pl.col("wind"),
            pl.lit(int(home)).alias("is_home")))
    return pl.concat(sides).with_columns(
        (pl.col("total_line") / 2 + pl.col("own_spread") / 2).alias("implied_total"),
        pl.col("wind").fill_null(0.0))


# The expected counterpart of each realised quantity, from `ff_opportunity`. Opportunity
# counts have no expected version and should not: a target is not an estimate, he was thrown at
# or he was not. Everything below the opportunity is an efficiency, and efficiency is what
# regresses -- which this repo measured from the other side as `td_rate_prior` at -0.040 across
# five of five seasons. See docs/what-the-field-knows.md.
EXPECTED: dict[str, str] = {
    "receptions": "receptions_exp",
    "receiving_yards": "rec_yards_gained_exp",
    "rushing_yards": "rush_yards_gained_exp",
    "passing_yards": "pass_yards_gained_exp",
}

# Required by the `ff_opportunity` contract, and carried rather than merely satisfied: it is
# weekly xFP itself, the quantity `xfp_per_game` is the season-long average of.
XFP_WEEK = "total_fantasy_points_exp"


def expected_weekly(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Weekly *expected* receptions and yardage, keyed to join onto the panel.

    `ff_opportunity` prices each opportunity by its situation -- air yards, target depth, field
    position -- so a six-target week at three yards downfield and one at fifteen stop being the
    same number. The store has had this table the whole time and the weekly model never opened
    it.
    """
    from hub.fetch import nflverse
    cols = ["season", "week", "player_id", "full_name", "position", XFP_WEEK,
            *EXPECTED.values()]
    d = nflverse.load("ff_opportunity", list(seasons), cols=cols)
    return (d.select(
                pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Float64).cast(pl.Int64),
                pl.col("full_name").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                *[pl.col(c).cast(pl.Float64) for c in (*EXPECTED.values(), XFP_WEEK)])
              .group_by(["season", "week", "key"])
              .agg([pl.col(c).sum() for c in (*EXPECTED.values(), XFP_WEEK)]))


def weekly_stats(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    import nflreadpy as nfl
    ps = nfl.load_player_stats(seasons=list(seasons), summary_level="week")
    want = ["player_id", "player_display_name", "position", "season", "week", "team",
            "opponent_team", OUTCOME, "target_share", "receiving_yards", "rushing_yards",
            "passing_yards", "receiving_tds", "rushing_tds", "passing_tds", "season_type",
            "targets", "receptions", "carries", "attempts", "completions",
            "passing_interceptions", "fumbles_lost_total"]
    ps = ps.select([c for c in want if c in ps.columns])
    if "season_type" in ps.columns:
        ps = ps.filter(pl.col("season_type") == "REG").drop("season_type")
    return (ps.filter(pl.col("position").is_in(DRAFTED_POSITIONS))
              .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)))


def prior_means(df: pl.DataFrame, keys: Sequence[str], values: Sequence[str], *,
                within_season: bool = False) -> pl.DataFrame:
    """Mean of each `value` over rows strictly earlier in (season, week), per `keys`.

    Routed through `expanding_weeks` rather than a hand-rolled cumulative sum, so the one
    statement of the leakage rule is the one that runs. Slower and worth it: this is
    `docs/method.md` rule #2 at week grain, and a cumulative sum is exactly where an
    off-by-one hides.

    `within_season` narrows the past to the current season, which is a *scope* choice and not
    a leakage one -- both are strictly prior. It is explicit at the call site rather than a
    flag on `expanding_weeks`, because a parameter that silently changes what counts as the
    past is what that function exists to prevent. The player control uses it, since the
    pre-registered control is **season-to-date** PPG and a career mean is a different and
    stronger control than the one written down. Defence-vs-position does not: a defence's
    record carries across seasons and early weeks would otherwise have no opponent term at all.
    """
    frames = []
    for season, week, past, _now in expanding_weeks(df, min_past=1):
        if within_season:
            past = past.filter(pl.col("season") == season)
            if past.is_empty():
                continue
        agg = (past.group_by(list(keys))
                   .agg([pl.col(v).mean().alias(f"{v}_prior") for v in values]
                        + [pl.len().alias("prior_n")])
                   .with_columns(pl.lit(season).cast(pl.Int64).alias("season"),
                                 pl.lit(week).cast(pl.Int64).alias("week")))
        frames.append(agg)
    if not frames:
        return df.head(0)
    # Sorted on the way out, not left to the group_by. Every caller either means these rows
    # again or joins them into something that is sorted later, and a hash-ordered frame under
    # a float aggregation moves at ~1e-15 -- improvements.md #18. One sort here fixes every
    # consumer rather than each of them separately.
    return pl.concat(frames).sort([*keys, "season", "week"])


def trend(df: pl.DataFrame, col: str, key: str, out: str, weeks: int = 18) -> pl.DataFrame:
    """mean(w-3..w-1) - mean(w-6..w-4), over *calendar weeks* rather than appearances.

    Reindexed onto a complete (player, season, week) grid first. Shifting over row order would
    make a player who missed week 5 compare weeks 6,4,3 against 2,1 -- strictly prior, so not
    leakage, but not the quantity `docs/snap-trend-signal.md` defines either, and missed games
    are exactly the interesting case for a usage trend.
    """
    if df.select(key, "season", "week").is_duplicated().any():
        raise ValueError(
            f"trend() needs one row per ({key}, season, week) and got duplicates. Called with "
            f"a team key on a player-week panel, each join fans out by the roster size and "
            f"five chained calls take the process out on memory -- which is how this guard "
            f"came to exist. Compute the trend on the unique frame, then join it on.")
    grid = (df.select(key, "season").unique()
              .join(pl.DataFrame({"week": list(range(1, weeks + 1))},
                                 schema={"week": pl.Int64}), how="cross"))
    g = (grid.join(df.select(key, "season", "week", col), on=[key, "season", "week"],
                   how="left")
             .sort([key, "season", "week"])
             .with_columns(
                 (pl.col(col).shift(1).rolling_mean(3, min_samples=2).over([key, "season"])
                  - pl.col(col).shift(4).rolling_mean(3, min_samples=2).over([key, "season"]))
                 .alias(out)))
    return df.join(g.select(key, "season", "week", out), on=[key, "season", "week"],
                   how="left")


def recent_mean(df: pl.DataFrame, col: str, key: str = "player_id", *, window: int = 3,
                weeks: int = 18) -> pl.DataFrame:
    """`<col>_recent`: the mean over the previous `window` calendar weeks, strictly prior.

    A season-to-date mean is a *lagging* control -- by week 12 it is eleven games of history
    against which three weeks of new form barely register, so a feature that is really "he has
    been busier lately" would clear against it while adding nothing a person watching could
    not see. This is the control that tests that, and `snap_trend` survives it: on carries it
    falls from +0.127 to +0.049, so most of that effect *was* recent form, and on targets from
    +0.124 to +0.087. Five of five seasons either way.

    Reindexed onto a complete calendar grid for the same reason `trend` is.
    """
    if df.select(key, "season", "week").is_duplicated().any():
        raise ValueError(
            f"trend() needs one row per ({key}, season, week) and got duplicates. Called with "
            f"a team key on a player-week panel, each join fans out by the roster size and "
            f"five chained calls take the process out on memory -- which is how this guard "
            f"came to exist. Compute the trend on the unique frame, then join it on.")
    grid = (df.select(key, "season").unique()
              .join(pl.DataFrame({"week": list(range(1, weeks + 1))},
                                 schema={"week": pl.Int64}), how="cross"))
    g = (grid.join(df.select(key, "season", "week", col), on=[key, "season", "week"], how="left")
             .sort([key, "season", "week"])
             .with_columns(pl.col(col).shift(1).rolling_mean(window, min_samples=2)
                             .over([key, "season"]).alias(f"{col}_recent")))
    return df.join(g.select(key, "season", "week", f"{col}_recent"),
                   on=[key, "season", "week"], how="left")


# Team scheme, per (team, season, week), from `ftn_charting`. Screened as trends and not as
# levels: scheme is stable within a season and a player's own recent usage already absorbs the
# level, so what would be new information is a scheme *changing*. See improvements.md #4.
SCHEME: dict[str, str] = {
    "pa_rate": "is_play_action",
    "motion_rate": "is_motion",
    "nohuddle_rate": "is_no_huddle",
    "screen_rate": "is_screen_pass",
}


def scheme_rates_from_plays(charting: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
    """Per (team, season, week) scheme rates. Pure, so it exercises without a network.

    `charting` is play-level FTN booleans; `plays` supplies the possession team and a pass-play
    marker, which FTN does not carry. That second frame is **participation**, not play-by-play:
    one narrowed `load_pbp` for four seasons was killed by the OOM reaper, and participation
    already carries `possession_team` and a non-empty `route` marks a charted pass play --
    which is the same definition `route_share` uses, so the two agree by construction.
    """
    if charting.is_empty() or plays.is_empty():
        cols = {"posteam": pl.Utf8, "season": pl.Int64, "week": pl.Int64,
                "pass_rate": pl.Float64, **dict.fromkeys(SCHEME, pl.Float64)}
        return pl.DataFrame(schema=cols)
    # FTN types the play id as an integer and nflverse's as a float, so the join needs both
    # sides cast or it raises rather than silently matching nothing.
    left = charting.with_columns(pl.col("nflverse_play_id").cast(pl.Int64))
    right = plays.with_columns(pl.col("play_id").cast(pl.Int64))
    d = (left.join(right, left_on=["nflverse_game_id", "nflverse_play_id"],
                   right_on=["nflverse_game_id", "play_id"], how="inner")
             .drop_nulls("possession_team"))
    return (d.group_by(["possession_team", "season", "week"])
             .agg(
                 pl.col("is_pass").cast(pl.Float64).mean().alias("pass_rate"),
                 *[pl.col(col).cast(pl.Float64).mean().alias(rate)
                   for rate, col in SCHEME.items()])
             .rename({"possession_team": "posteam"})
             .with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)))


# FTN charting begins in 2022, so the scheme features have four held-out seasons where
# everything else has five. That is a real narrowing of the every-season half of the bar and
# is stated rather than absorbed.
FTN_FIRST_SEASON = 2022


def scheme_rates(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """`scheme_rates_from_plays` over real seasons, skipping those FTN does not cover."""
    import nflreadpy as nfl
    frames = []
    for yr in seasons:
        if yr < FTN_FIRST_SEASON:
            continue
        c = nfl.load_ftn_charting(seasons=[yr]).select(
            "nflverse_game_id", "nflverse_play_id", *SCHEME.values())
        wk = (nfl.load_schedules().filter(pl.col("season") == yr)
                .select(pl.col("game_id").alias("nflverse_game_id"),
                        pl.col("week").cast(pl.Int64)))
        pa = (nfl.load_participation(seasons=[yr])
                .select("nflverse_game_id", "play_id", "possession_team",
                        (pl.col("route").is_not_null()
                         & (pl.col("route") != "")).alias("is_pass"))
                .join(wk, on="nflverse_game_id", how="inner")
                .with_columns(pl.lit(yr).cast(pl.Int64).alias("season")))
        frames.append(scheme_rates_from_plays(c, pa))
    return pl.concat(frames) if frames else scheme_rates_from_plays(
        pl.DataFrame(), pl.DataFrame())


SCHEME_TRENDS: tuple[Feature, ...] = tuple(
    Feature(f"{r}_trend", "?" if r != "nohuddle_rate" else "+", TREND_MIN_WEEK)
    for r in (*SCHEME, "pass_rate"))


def route_share_from_plays(plays: pl.DataFrame, season: int) -> pl.DataFrame:
    """The arithmetic, separated from the fetch so it can be exercised without a network.

    `plays` is one row per charted pass play with `week`, `possession_team` and a
    semicolon-separated `offense_players`. Out comes one row per (season, week, player_id)
    with the share of his team's charted pass plays he was on the field for.
    """
    if plays.is_empty():
        return pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64,
                                    "player_id": pl.Utf8, "route_pct": pl.Float64})
    denom = plays.group_by(["week", "possession_team"]).agg(pl.len().alias("team_pass_plays"))
    num = (plays.select("week", "possession_team",
                        pl.col("offense_players").str.split(";").alias("player_id"))
                .explode("player_id", empty_as_null=True)
                .filter(pl.col("player_id").is_not_null() & (pl.col("player_id") != ""))
                .group_by(["week", "possession_team", "player_id"])
                .agg(pl.len().alias("plays_on")))
    return (num.join(denom, on=["week", "possession_team"])
               .with_columns(pl.lit(season).cast(pl.Int64).alias("season"),
                             (pl.col("plays_on") / pl.col("team_pass_plays")).alias("route_pct"))
               .group_by(["season", "week", "player_id"])
               .agg(pl.col("route_pct").max()))


def route_share(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Share of his team's *pass* plays a player was on the field for, per game.

    Snap share tells you presence, this tells you **access**, and target share tells you
    whether the quarterback used the access. The repo's one durable in-season signal is a
    snap-share trend, and snap share counts run snaps a receiver was never going to be thrown
    to -- so this ought to be the same quantity measured better. Measured 2026-08-29, it is
    not: the two trends correlate at **0.917** and the snap version is the stronger. See
    `docs/what-the-field-knows.md` and the note on `ROUTE_TREND`.

    **Named for what it is.** `participation.route` is one value per *play*, not per player, so
    a per-player route type is not derivable and calling this "route participation" would claim
    a precision the data does not have. A non-empty `route` marks a charted pass play, and this
    is the share of those a player was on the field for. `offense_players` is populated for
    91-100% of plays across 2021-25.
    """
    import nflreadpy as nfl
    frames = []
    for yr in seasons:
        p = (nfl.load_participation(seasons=[yr])
               .filter(pl.col("route").is_not_null() & (pl.col("route") != "")
                       & pl.col("offense_players").is_not_null()
                       & (pl.col("offense_players") != "")))
        if p.is_empty():
            continue
        wk = (nfl.load_schedules().filter(pl.col("season") == yr)
                .select(pl.col("game_id").alias("nflverse_game_id"),
                        pl.col("week").cast(pl.Int64)))
        frames.append(route_share_from_plays(p.join(wk, on="nflverse_game_id", how="inner"), yr))
    if not frames:
        return route_share_from_plays(pl.DataFrame(), 0)
    return pl.concat(frames)


def snap_share(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Offensive snap share by (season, week, player key).

    Snaps key on `pfr_player_id` and everything else on `gsis_id`; joining on the normalised
    name instead is what `hub.names.player_key` is for, and `docs/snap-trend-signal.md` records
    the crosswalk at 99.8%.
    """
    import nflreadpy as nfl
    s = nfl.load_snap_counts(seasons=list(seasons)).filter(pl.col("game_type") == "REG")
    return (s.select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64),
                     pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                     pl.col("offense_pct").cast(pl.Float64))
             .group_by(["season", "week", "key"]).agg(pl.col("offense_pct").max()))


def injury_severity(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Ordinal game-status severity for the week, from the report published before it.

    Deliberately crude -- Out 3, Doubtful 2, Questionable 1, otherwise 0. The fitted
    (status, practice) retention table in `hub.models.injury` is the real instrument and it is
    already measured at this grain (+0.170 MAE at 3.8 se); this ordinal exists only so the
    screen has *something* in the injury slot, and a null here is a null about the ordinal.
    """
    import nflreadpy as nfl
    inj = nfl.load_injuries(seasons=list(seasons))
    cols = {c.lower(): c for c in inj.columns}
    status = cols.get("report_status") or cols.get("game_status")
    name = cols.get("full_name") or cols.get("player_name")
    if status is None or name is None:
        raise ValueError(
            "nflverse injuries changed shape: expected a status and a name column, got "
            f"{sorted(inj.columns)}")
    sev = (pl.when(pl.col(status) == "Out").then(3.0)
             .when(pl.col(status) == "Doubtful").then(2.0)
             .when(pl.col(status) == "Questionable").then(1.0).otherwise(0.0))
    # `status` and `practice` are carried alongside the ordinal because they are the cells
    # `hub.models.injury` fits its retention table on -- the ordinal is for the screen, the
    # pair is for the model. Practice status arrives as prose ("Did Not Participate In
    # Practice") and the first seven characters separate the three cases, which is the
    # normalisation `injury.observations` already uses.
    prac = cols.get("practice_status")
    practice = practice_key(prac) if prac else pl.lit("None")
    return (inj.select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64),
                       pl.col(name).map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                       sev.alias("inj_sev"),
                       pl.col(status).fill_null("None").alias("status"),
                       practice.alias("practice"))
               .sort("inj_sev", descending=True)
               .group_by(["season", "week", "key"])
               .agg(pl.col("inj_sev").max(), pl.col("status").first(),
                    pl.col("practice").first()))


def week_windows(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """First and **last** kickoff per (season, week). The last one is what the join uses."""
    import nflreadpy as nfl
    s = (nfl.load_schedules()
           .filter(pl.col("season").is_in(list(seasons)) & (pl.col("game_type") == "REG")))
    return (s.group_by(["season", "week"])
             .agg(pl.col("gameday").str.to_date().min().alias("first_kick"),
                  pl.col("gameday").str.to_date().max().alias("last_kick"))
             .sort("last_kick"))


def assign_weeks(scrapes: pl.DataFrame, windows: pl.DataFrame,
                 max_lead: int = CONSENSUS_MAX_LEAD_DAYS) -> pl.DataFrame:
    """Map each scrape to the first week whose games are **not all played yet**.

    On the week's *last* kickoff, not its first, and getting that wrong is an off-by-one that
    silently shifts the whole control by a week. An NFL week runs Thursday to Monday, and
    FantasyPros scrapes land mid-week: 2024-10-04 is a Friday, inside week 5 (Oct 3-7), and it
    ranks week 5's Sunday games. Joining on the next *first* kickoff sent it to week 6.

    The tell was that Saquon Barkley, CeeDee Lamb and Patrick Mahomes were each missing from
    exactly one week -- and it was the week after their team's bye, because a page that
    correctly omits a bye-week player was being attached to the following week.

    **Known and accepted:** a Friday scrape is after its own week's Thursday game, so for a
    Thursday-night player the ranking is not strictly pre-kickoff. That hands the *incumbent*
    one game of hindsight a team-week, which biases against the arm being tested, so it is
    conservative rather than dangerous.
    """
    out = (scrapes.sort("scrape_date")
                  .join_asof(windows.sort("last_kick"), left_on="scrape_date",
                             right_on="last_kick", strategy="forward")
                  .drop_nulls(["season", "week"]))
    return out.with_columns(
        (pl.col("last_kick") - pl.col("scrape_date")).dt.total_days().alias("lead_days")
    ).filter(pl.col("lead_days") <= max_lead)


def weekly_consensus(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """That week's cross-position consensus ECR, joined as-of to (season, week)."""
    import nflreadpy as nfl
    a = nfl.load_ff_rankings("all")
    page = "page_type" if "page_type" in a.columns else "page"
    w = (a.filter(pl.col(page) == CONSENSUS_PAGE)
          .select(pl.col("scrape_date").str.to_date(),
                  pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                  pl.col("ecr").cast(pl.Float64))
          .drop_nulls(["scrape_date", "ecr"]))
    joined = assign_weeks(w, week_windows(seasons))
    return (joined.group_by(["season", "week", "key"])
                  .agg(pl.col("ecr").min(), pl.col("lead_days").first())
                  .with_columns(pl.col("season").cast(pl.Int64),
                                pl.col("week").cast(pl.Int64)))


def build_panel(seasons: Sequence[int] = SEASONS, *, consensus: bool = True,
                ranks: pl.DataFrame | None = None, expected: bool = False,
                routes: bool = False, scheme: bool = False,
                ) -> pl.DataFrame:  # pragma: no cover - network
    """One row per (player, season, week), with every feature measured before its outcome.

    **This panel contains only weeks a player took a snap**, because `player_stats` has no row
    for a player who did not. That is the pre-registered Gate A treatment -- inactive weeks are
    excluded from the projection comparison and scored as zero in the lineup gate -- and it has
    one consequence worth stating where it will be read:

        the injury retention term CANNOT be fitted here.

    Of 5,473 "Out" designations across 2021-25, **six** reach this panel: 0.11%. Doubtful is
    764 against 2. The model `hub.models.injury` fitted at +0.170 MAE and 3.8 se prices an
    injury row with no stat row as *zero* -- the player who did not play is its entire subject
    -- and here he is structurally absent. Fitting retention on these rows would measure
    something much weaker, "what a Questionable player who played anyway retains", and would
    report it under the stronger result's name.

    `status` and `practice` are carried regardless, because Gate B builds a complete
    player-week grid where a missing row is a zero, and that is where the term belongs.
    """
    stats = weekly_stats(seasons).with_columns(
        pl.col("player_display_name").map_elements(player_key, return_dtype=pl.Utf8).alias("key"))
    p = stats.join(game_context(seasons), on=["season", "week", "team"], how="left")

    if expected:
        stats = stats.join(expected_weekly(seasons), on=["season", "week", "key"], how="left")
        # Where `ff_opportunity` has no row for him, the realised value stands in rather than
        # a null propagating into every prior downstream.
        stats = stats.with_columns(
            [pl.coalesce(pl.col(exp), pl.col(real)).alias(real)
             for real, exp in EXPECTED.items()])

    counted = stats.with_columns(
        (pl.col("receiving_tds") + pl.col("rushing_tds") + pl.col("passing_tds")).alias("tds"),
        (pl.col("receiving_yards") + pl.col("rushing_yards")
         + pl.col("passing_yards")).alias("yds"))
    p = p.join(counted.select("player_id", "season", "week", "tds", "yds"),
               on=["player_id", "season", "week"], how="left")
    own = prior_means(counted.sort(["player_id", "season", "week"]), ["player_id"],
                      [OUTCOME, "yds", *USAGE, *YARDS, *TURNOVERS], within_season=True)
    p = p.join(own, on=["player_id", "season", "week"], how="left").rename(
        {f"{OUTCOME}_prior": "ppg_before", "prior_n": "games_before"})
    p = p.with_columns(
        (pl.col("tds_prior") / (pl.col("yds_prior") + 1e-9)).alias("td_rate_prior"))

    # Sorted before it feeds `prior_means`, which takes a mean over these rows. A group_by
    # emits rows in a hash-dependent order that varies between calls, and floating-point
    # addition is not associative, so sum-then-mean without a sort between them moves at
    # ~1e-15 and every downstream sort can land differently. Same defect as
    # `playoff_sos._dvp_from_stats` -- improvements.md #18 -- found by auditing for the
    # pattern after diagnosing that one.
    allowed = (stats.group_by(["opponent_team", "position", "season", "week"])
                    .agg(pl.col(OUTCOME).sum().alias("allowed"))
                    .sort(["opponent_team", "position", "season", "week"]))
    d = prior_means(allowed, ["opponent_team", "position"], ["allowed"])
    lg = d.group_by(["position", "season", "week"]).agg(
        pl.col("allowed_prior").mean().alias("lg"))
    d = (d.join(lg, on=["position", "season", "week"])
          .with_columns((pl.col("allowed_prior") / pl.col("lg")).alias("dvp")))
    p = p.join(d.select("opponent_team", "position", "season", "week", "dvp"),
               on=["opponent_team", "position", "season", "week"], how="left")

    p = p.join(snap_share(seasons), on=["season", "week", "key"], how="left")
    p = trend(p, "offense_pct", "key", "snap_trend")
    if routes:
        # Joined on the gsis id rather than a normalised name: `participation` and
        # `player_stats` key on the same identifier, so this one needs no crosswalk at all.
        # Opt-in because loading five seasons of play-level participation is the slowest
        # thing in the panel and the feature it produces is a measured null.
        p = p.join(route_share(seasons), on=["season", "week", "player_id"], how="left")
        p = trend(p, "route_pct", "player_id", "route_trend")
    p = trend(p, "target_share", "player_id", "tgt_trend")
    if scheme:
        # Trended on the unique team-week frame and joined once. Trending on the panel would
        # call `trend` with a team key against thirty players a team-week -- see its guard.
        rates = scheme_rates(seasons)
        for r in (*SCHEME, "pass_rate"):
            rates = trend(rates, r, "posteam", f"{r}_trend")
        p = p.join(rates, left_on=["team", "season", "week"],
                   right_on=["posteam", "season", "week"], how="left")
    p = p.join(injury_severity(seasons), on=["season", "week", "key"], how="left")
    p = p.with_columns(pl.col("inj_sev").fill_null(0.0),
                       pl.col("status").fill_null("Healthy"),
                       pl.col("practice").fill_null("Healthy"))
    for c in USAGE:
        p = recent_mean(p, c)
    if ranks is not None:
        # Injected rather than fetched. A board lives under `draft/`, and six `draft/` modules
        # import `models/` while nothing goes the other way -- inverting that to save one
        # caller a parameter is the mistake `board_as_of` was moved out of `models/` to fix.
        p = p.join(ranks, on=["key", "season"], how="left")
    if not consensus:
        # The gate needs a projection for every rostered player, including the ones consensus
        # does not list -- being unranked is the incumbent's *answer*, not a reason to have no
        # projection. Only the screen, which measures beyond consensus, requires it to exist.
        return p
    return p.join(weekly_consensus(seasons), on=["season", "week", "key"], how="inner")


def screen(panel: pl.DataFrame, features: Sequence[Feature] = FEATURES) -> pl.DataFrame:
    """Every feature, screened, with its pre-stated sign and its verdict."""
    rows = []
    for f in features:
        s = summarise(cell_correlations(panel, f.name, min_week=f.min_week))
        status, note = verdict(s, f.sign)
        rows.append({"feature": f.name, "sign": f.sign, "r": s["r"], "se": s["se"],
                     "t": s["t"], "cells": s["cells"], "n": s["n"],
                     "status": status, "note": note,
                     "per_season": str({k: round(v, 4) for k, v in s["per_season"].items()})})
    return pl.DataFrame(rows).sort("r", descending=True)


def screen_joint(panel: pl.DataFrame, survivors: Sequence[Feature]) -> pl.DataFrame:
    """Re-screen each survivor with the other survivors added to the controls.

    Without this the screen reports collinear features as separate findings. The first run
    found `own_spread` at +0.042 across all five seasons and `implied_total` at +0.055 across
    all five -- and `implied_total = total_line/2 + own_spread/2`, so they correlate at 0.83
    and are one finding wearing two hats. Controlled for the total, the spread leaves nothing.

    A feature that clears alone and dies here is not a signal; it is another signal's shadow.
    """
    rows = []
    for f in survivors:
        # Each feature keeps its OWN week range and is controlled only for survivors that
        # exist over it. Taking the widest min_week across the set instead would drop
        # `implied_total` from 54 cells to 35 purely because `snap_trend` starts at week 8,
        # and then report the lost power as a failed control.
        others = [g.name for g in survivors
                  if g.name != f.name and g.min_week <= f.min_week]
        s = summarise(cell_correlations(panel, f.name, min_week=f.min_week,
                                        controls=(*CONTROLS, *others)))
        status, note = verdict(s, f.sign)
        rows.append({"feature": f.name, "sign": f.sign, "r": s["r"], "t": s["t"],
                     "cells": s["cells"], "status": status,
                     "note": note, "controls": ", ".join(others) or "-"})
    return pl.DataFrame(rows).sort("r", descending=True)


def screen_usage(panel: pl.DataFrame, features: Sequence[Feature],
                 components: Sequence[str] = USAGE) -> pl.DataFrame:
    """Each feature against each Usage count, controlled for that count's own recent level.

    The controls are that count's own **season-to-date** mean and its **last three weeks**,
    plus consensus ECR. Not `ppg_before`: asking whether a feature predicts this week's targets
    beyond his season-to-date *points* would let a change in role show up as a target signal.
    And not the season-to-date mean alone, which lags -- see `recent_mean`.
    """
    rows = []
    for f in features:
        for c in components:
            cells = cell_correlations(panel, f.name, min_week=f.min_week, outcome=c,
                                      controls=(f"{c}_prior", f"{c}_recent", "ecr"))
            s = summarise(cells)
            status, note = verdict(s, f.sign)
            rows.append({"feature": f.name, "component": c, "sign": f.sign,
                         "r": s["r"], "t": s["t"], "cells": s["cells"],
                         "status": status, "note": note})
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:      # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.models.weekly_screen",
        description="Screen week-level features beyond weekly consensus.")
    ap.add_argument("--run", action="store_true", help="build the panel and screen")
    ap.add_argument("--scheme", action="store_true",
                    help="add the team scheme trends -- improvements.md #4")
    ap.add_argument("--routes", action="store_true",
                    help="add route_trend -- reproduces the null against snap_trend")
    ap.add_argument("--usage", action="store_true",
                    help="screen the survivors against Usage counts, not points")
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0
    seasons = [int(x) for x in a.seasons.split(",") if x]
    panel = build_panel(seasons, routes=a.routes, scheme=a.scheme)
    sample = panel.filter(pl.col("week").is_in(list(GATE_WEEKS))
                          & (pl.col("games_before") >= MIN_GAMES_BEFORE))
    sample = sample.drop_nulls([OUTCOME, *CONTROLS])
    print(f"  {sample.height} player-weeks, {sample['player_id'].n_unique()} players, "
          f"seasons {sorted(sample['season'].unique().to_list())}")
    lead = panel["lead_days"]
    print(f"  consensus scraped a median {lead.median():.0f} days before kickoff "
          f"(the confound: see docs/weekly-screen.md)")
    extra = ((ROUTE_TREND,) if a.routes else ()) + (SCHEME_TRENDS if a.scheme else ())
    out = screen(sample, (*FEATURES, *extra))
    print("\n".join(report(out.to_dicts())))
    found = out.filter(pl.col("status").is_in(list(FINDINGS)))["feature"].to_list()
    print(f"\n  a signal on its own: {', '.join(found) if found else 'nothing'}")
    if len(found) > 1:
        pool = (*FEATURES, *extra)
        survivors = [f for f in pool if f.name in found]
        joint = screen_joint(sample, survivors)
        print("\n  each one, controlled for the others that exist over its weeks:")
        print("\n".join(report(joint.to_dicts())))
        left = joint.filter(pl.col("status").is_in(list(FINDINGS)))["feature"].to_list()
        print(f"\n  independent signals: {', '.join(left) if left else 'nothing'}")
        if a.usage and left:
            print("\n  and against Usage rather than points:")
            u = screen_usage(sample, [f for f in pool if f.name in left])
            for row in u.iter_rows(named=True):
                print(f"  {row['feature']:16} {row['component']:11} {row['r']:+7.4f} "
                      f"{row['t']:+6.2f}  {row['status']}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
