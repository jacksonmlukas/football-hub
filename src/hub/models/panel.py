"""The **Panel**: one row per player-week, every feature measured before its outcome.

The substrate three modules read -- `hub.models.weekly_screen` measures signals on it,
`hub.models.weekly` fits the Weekly projection from it, and `hub.season.weekly_gate_data`
assembles a gate from it. It used to live *inside* the screen, and the other two reached in with
**function-local imports** to get at it, which `docs/improvements.md` #17 already named as the
tell: what a caller does when an import feels wrong.

The seam is between **fetching** and **assembling**. Everything under `build_panel` is
implementation -- eleven sources, each `pragma: no cover - network`, each narrowed at its own
boundary. A caller asks for a Panel and gets one; what it costs to make is not its business.

**The one rule the whole thing exists to keep.** Every feature is measured *strictly before* its
outcome week. Pre-kickoff facts published *for* week w -- the line, the injury report, the
opponent -- count as week-w information; anything derived from play uses weeks < w only. That is
`docs/method.md` rule #2, the invariant this repo records violating at 7.4 se, and the expanding
aggregates below route through `hub.models.experiment.expanding_weeks` so there is one statement
of it rather than one per feature.

**And the Panel now says which of its columns keep it.** The realised week-level frame is what
everything is assembled on, so week w's own `targets`, `offense_pct`, `tds` and fifteen more
ride along to the return -- deliberately, because a projection is fitted *against* them. What
they were not is distinguishable: using the Panel correctly meant knowing from suffix
convention alone which of sixty columns were safe, and the one caller that got it right got it
right by being careful. `feature_columns` and `outcome_columns` answer it instead,
`require_features` refuses a raw one where a feature was meant, and `_served` refuses to hand
back a column that is on neither side. #203.
"""
from __future__ import annotations

import operator
import sys
from collections.abc import Sequence
from functools import reduce
from typing import NamedTuple

import polars as pl

from hub.config import DRAFTED_POSITIONS, SEASON_COMPLETED
from hub.contracts import INJURIES, SNAP_COUNTS, ContractViolation
from hub.fetch import nflverse
from hub.fetch.nflverse import RANKINGS_COLS, Pin, data_pin, load_rankings
from hub.models import components
from hub.models.experiment import expanding_weeks
from hub.names import player_key, practice_key


class PanelSpec(NamedTuple):
    """Which sources a caller wants on its Panel, as one declaration rather than five flags.

    `build_panel` took five separate booleans, so each of the three callers had to know which
    combination its question needed, and no call site read back as the thing it was asking for.
    Named together, with defaults, a caller states the shape it wants.

    Every field is off by default except `consensus`, because the screen -- which measures
    *beyond* consensus -- cannot run without it, while the two gate consumers must not have it.
    """
    consensus: bool = True
    # `expected` swaps the four columns of `EXPECTED` in the frame the *priors* are taken
    # over, and nothing else: the week itself stays realised, because it is the outcome. What
    # that costs and why it is the right way round is written at the coalesce in `build_panel`.
    expected: bool = False              # ff_opportunity's expected receptions and yardage
    routes: bool = False                # pass-play participation; a measured null, kept for re-runs
    scheme: bool = False                # FTN team scheme rates; six of six null
    ranks: pl.DataFrame | None = None   # preseason ECR, injected -- a board belongs to draft/


SCREEN_SPEC = PanelSpec()
"""Consensus and nothing else: what the screen asks for, and `build_panel`'s default."""


# 2021 is the first season `ff_opportunity` covers in the store; weekly consensus starts 2020,
# so the overlap is what bounds this.
SEASONS: tuple[int, ...] = tuple(range(2021, SEASON_COMPLETED + 1))


# Below this a season-to-date mean is three numbers and the control is not doing its job.
MIN_GAMES_BEFORE = 3


# The trend is dark before this. docs/snap-trend-signal.md finds anchors 4 and 6 null with the
# sign flipping between seasons: the trend needs about eight weeks of snaps before it says
# anything. So before week 8 `hub.models.weekly`'s multiplier is exactly 1 and the projection is
# the flat one -- that is not a fallback, it is the measurement.
#
# **This is the model's threshold and nothing else's** -- #178. It used to be the weekly
# screen's `min_week` as well, and that was one constant doing two jobs: 8 is the earliest of
# the anchors 4, 6, 8, 10, 12 that held when they were tested *against the outcome*
# (snap-trend-signal.md), so using it to select the screen's rows screened features on rows
# chosen by a value fitted to the outcome on the same data. The threshold keeps its measured
# value because the measurement is what it is for; what moved is that
# `hub.models.weekly_screen` no longer reads it. The screen sweeps its own anchors and
# publishes the sensitivity -- `weekly_screen.SCREEN_TREND_ANCHORS`.
TREND_MIN_WEEK = 8


# The cross-position weekly page. Position pages cannot fill a flex, which is what a lineup
# needs, so the incumbent has to be the one ranking that spans positions.
CONSENSUS_PAGE = "weekly-op"


# A scrape further from kickoff than this belongs to no week. The observed distribution is
# almost entirely 6 days, which is the confound named in the module docstring.
CONSENSUS_MAX_LEAD_DAYS = 8


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


# The three scored touchdown columns, kept apart rather than summed, because `ppg_before` is
# split along them and a touchdown is not one price. `USAGE` carries `tds`, the count, which
# is what `td_rate_prior`'s numerator wants; this tuple is what `td_ppg_before` wants, and the
# two differ by exactly the thing that matters here -- a passing touchdown is **four** points
# and the other two are six. Pricing all three at six would put an extra two points a passing
# score into the touchdown component and take the same two out of the non-touchdown remainder
# beside it, on quarterbacks only. Each name is a `SCORING` key, which `test_panel` asserts,
# so a weight that moves there moves here.
TD_COLUMNS: tuple[str, ...] = ("receiving_tds", "rushing_tds", "passing_tds")


# --- which of the sources below go through the validated, cached path (#35) --------------
#
# The path is `hub.fetch.nflverse.load`: narrowed at the boundary, checked against the source's
# `Contract` on the way out, written to a cache entry keyed by `(source, seasons, columns,
# as-of)`, with a `Pin` beside it that a run can print. Reaching `nflreadpy` from here instead
# means the frame no contract has ever seen, refetched live on every build, and a digest that
# cannot name what it read.
#
# Six nflverse sources reach this module. Five now route:
#
#   ff_rankings    `weekly_consensus`, through `load_rankings` -- #33, and the only one an
#                  as-of can filter rather than label
#   ff_opportunity `expected_weekly`
#   player_stats   `weekly_stats`
#   participation  `route_share` and `scheme_rates`
#   ftn_charting   `scheme_rates`
#
# **Three do not, and the obstacle is the frozen archive rather than this module.**
# `schedules`, `snap_counts` and `injuries` are the three captures in
# `tests/golden/fixtures/panel_archive/` that were trimmed to the columns the Panel *selects*,
# which is narrower than what their contracts require -- so routing them makes `load` refuse
# the archive, and eleven Panel tests fail on a frame that is not the Panel's fault.
# `scripts/capture_panel_archive.py` states the trim rule in code and re-takes the archive from
# a live nflverse; adopting what it produces is the rest of #35 and it needs a network run.
# `tests/unit/test_panel.py::test_the_three_unrouted_sources_are_blocked_by_the_archive_and_
# not_by_this_module` names the missing columns and goes red the day they arrive, so this
# paragraph cannot quietly outlive the block it describes.
#
# The two that route without being exercised offline -- `participation` and `ftn_charting` --
# are the two the archive deliberately refuses (they are play-level; see that directory's
# README), so nothing about them was ever driven by a test either way. Routing them costs the
# archive nothing and buys the live path the same contract every other source now gets.


def game_context(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """One row per (team, season, week): the facts published before kickoff.

    **Not routed through `hub.fetch.nflverse.load`** -- see the note above this function.
    `SCHEDULES` requires `game_id` and the frozen capture does not carry it.

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
# five of five seasons, **beyond season-to-date PPR points a game and weekly consensus ECR**.
# Re-run under #179 with that first control split into its touchdown and non-touchdown halves,
# it is -0.043 across five of five and the reading above is the one that survives; controlled
# for prior *yardage* directly it is -0.022 across four of five and does not. So this comment
# names the basis rather than the number alone -- which is the whole content of the difference
# between "efficiency regresses" and "high scorers regress", and this line depends on the
# former. See docs/weekly-screen.md and docs/what-the-field-knows.md.
# The four the screen wants, named as a subset of `components.EXPECTED` rather than restated.
# The vocabulary -- which upstream column means "expected receiving yards" -- is one thing and
# lives beside the scoring weights; *which* of them a consumer uses is that consumer's business,
# and this one deliberately leaves the touchdowns out for the reason above.
EXPECTED: dict[str, str] = {k: components.EXPECTED[k][0] for k in
                            ("receptions", "receiving_yards", "rushing_yards", "passing_yards")}


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
    cols = ["season", "week", "player_id", "full_name", "position", XFP_WEEK,
            *EXPECTED.values()]
    d = nflverse.load("ff_opportunity", list(seasons), cols=cols)
    return (d.select(
                pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Float64).cast(pl.Int64),
                pl.col("full_name").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                *[pl.col(c).cast(pl.Float64) for c in (*EXPECTED.values(), XFP_WEEK)])
              .group_by(["season", "week", "key"])
              .agg([pl.col(c).sum() for c in (*EXPECTED.values(), XFP_WEEK)]))


# What `weekly_stats` reads from `player_stats`. A module constant rather than a local list
# because it is read twice: once to narrow the fetch, and once by `column_role` below, which
# classifies every one of these as week *w*'s own realised play. Stating it in one place is
# what makes a column added here land on the outcome side of the Panel's rule by
# construction, instead of depending on whoever added it also remembering to say so.
WEEK_STATS_COLS: tuple[str, ...] = (
    "player_id", "player_display_name", "position", "season", "week", "team",
    "opponent_team", OUTCOME, "target_share", "receiving_yards", "rushing_yards",
    "passing_yards", "receiving_tds", "rushing_tds", "passing_tds", "season_type",
    "targets", "receptions", "carries", "attempts", "completions",
    "passing_interceptions", "fumbles_lost_total")


def weekly_stats(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Week *w*'s realised play, through the validated cached loader -- #35.

    `player_stats` is 150 columns wide, so `load` refuses it unnarrowed and `WEEK_STATS_COLS`
    is the narrowing: the same tuple `column_role` classifies, asked for once.

    **A column that stopped arriving is now named rather than dropped.** This used to select
    `[c for c in WEEK_STATS_COLS if c in ps.columns]` and filter on `season_type` only if it
    was there, so a rename upstream cost the Panel a feature silently -- the exact shape
    `injury_severity` below refuses three times over, and the shape `load`'s
    `column-vanished-upstream` guard exists to convert into a refusal that says which column.
    Everything narrowed here is also everything `PLAYER_STATS` requires, so the contract runs
    on the frame this function actually reads.
    """
    ps = nflverse.load("player_stats", list(seasons), cols=WEEK_STATS_COLS)
    return (ps.filter(pl.col("season_type") == "REG").drop("season_type")
              .filter(pl.col("position").is_in(DRAFTED_POSITIONS))
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


def _on_calendar_grid(df: pl.DataFrame, col: str, key: str, out: str, weeks: int,
                      expr: pl.Expr) -> pl.DataFrame:
    """Measure a window `expr` on a complete (`key`, season, week) grid, joined back onto `df`.

    The reindex both week-window features below are built on, and the reason either of them is
    more than a `shift`. Shifting over row order would make a player who missed week 5 compare
    weeks 6,4,3 against 2,1 -- strictly prior, so not leakage, but not the quantity
    `docs/snap-trend-signal.md` defines either, and missed games are exactly the interesting
    case for a usage trend. So every (`key`, season) is crossed with weeks 1..`weeks`, `df` is
    left-joined onto that, and a week a player missed becomes a null row the shift *counts*
    rather than a row that is not there.

    `expr` is written against `pl.col(col)` and carries no `.over` of its own: which rows a
    window may see is the grid's business, not the caller's. Only `out` is joined back, onto
    `df`'s own rows, so the grid weeks a caller never had do not survive into what it gets.
    """
    if df.select(key, "season", "week").is_duplicated().any():
        raise ValueError(
            f"{out} needs one row per ({key}, season, week) and got duplicates. Called with "
            f"a team key on a player-week panel, each join fans out by the roster size and "
            f"five chained calls take the process out on memory -- which is how this guard "
            f"came to exist. Compute the feature on the unique frame, then join it on.")
    grid = (df.select(key, "season").unique()
              .join(pl.DataFrame({"week": list(range(1, weeks + 1))},
                                 schema={"week": pl.Int64}), how="cross"))
    g = (grid.join(df.select(key, "season", "week", col), on=[key, "season", "week"],
                   how="left")
             .sort([key, "season", "week"])
             .with_columns(expr.over([key, "season"]).alias(out)))
    return df.join(g.select(key, "season", "week", out), on=[key, "season", "week"],
                   how="left")


def trend(df: pl.DataFrame, col: str, key: str, out: str, weeks: int = 18) -> pl.DataFrame:
    """mean(w-3..w-1) - mean(w-6..w-4), over *calendar weeks* rather than appearances.

    The calendar part is `_on_calendar_grid`, where the rule and what it costs are written
    down; what is here is the difference of two three-week means measured on that grid.
    """
    return _on_calendar_grid(
        df, col, key, out, weeks,
        pl.col(col).shift(1).rolling_mean(3, min_samples=2)
        - pl.col(col).shift(4).rolling_mean(3, min_samples=2))


def recent_mean(df: pl.DataFrame, col: str, key: str = "player_id", *, window: int = 3,
                weeks: int = 18) -> pl.DataFrame:
    """`<col>_recent`: the mean over the previous `window` calendar weeks, strictly prior.

    A season-to-date mean is a *lagging* control -- by week 12 it is eleven games of history
    against which three weeks of new form barely register, so a feature that is really "he has
    been busier lately" would clear against it while adding nothing a person watching could
    not see. This is the control that tests that, and `snap_trend` survives it: on carries it
    falls from +0.127 to +0.049, so most of that effect *was* recent form, and on targets from
    +0.124 to +0.087. Five of five seasons either way.

    Measured on the same complete calendar grid `trend` is, and for the reason written down
    with it in `_on_calendar_grid`.
    """
    return _on_calendar_grid(
        df, col, key, f"{col}_recent", weeks,
        pl.col(col).shift(1).rolling_mean(window, min_samples=2))


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
        # Both play-level sources go through the validated cached path (#35). Neither is
        # `WIDE` -- 29 and 26 columns -- so neither needs `cols`, and asking for one would
        # anyway have to be a superset of the contract's required set rather than the four
        # booleans read below. `schedules` is the one call in this function still reaching
        # nflreadpy, for the reason stated above `game_context`.
        c = nflverse.load("ftn_charting", [yr]).select(
            "nflverse_game_id", "nflverse_play_id", *SCHEME.values())
        wk = (nfl.load_schedules().filter(pl.col("season") == yr)
                .select(pl.col("game_id").alias("nflverse_game_id"),
                        pl.col("week").cast(pl.Int64)))
        pa = (nflverse.load("participation", [yr])
                .select("nflverse_game_id", "play_id", "possession_team",
                        (pl.col("route").is_not_null()
                         & (pl.col("route") != "")).alias("is_pass"))
                .join(wk, on="nflverse_game_id", how="inner")
                .with_columns(pl.lit(yr).cast(pl.Int64).alias("season")))
        frames.append(scheme_rates_from_plays(c, pa))
    return pl.concat(frames) if frames else scheme_rates_from_plays(
        pl.DataFrame(), pl.DataFrame())


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
        # Through the validated cached path (#35), as in `scheme_rates`. `schedules` below is
        # the one call left reaching nflreadpy here -- see the note above `game_context`.
        p = (nflverse.load("participation", [yr])
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

    The five columns it reads are asked of `SNAP_COUNTS` rather than taken on trust, which is
    also how `offense_pct` arrives as a fraction whichever way nflverse shipped it. This
    function loads from nflreadpy directly, so no contract had ever seen the frame, and it
    never had the private repair `hub.models.spread.snap_usage` had either -- a whole-percent
    refresh would have multiplied `snap_trend` by a hundred here while the same refresh came
    out right over there. One variation, one declaration, one answer.

    **`conform` and not `nflverse.load`, and that is the half of #35 still open.** `conform`
    checks the five columns this function reads; `load` would additionally cache the frame,
    pin it and check the other eight `SNAP_COUNTS` declares -- and the frozen capture carries
    none of those eight. See the note above `game_context`.
    """
    import nflreadpy as nfl
    s = SNAP_COUNTS.conform(nfl.load_snap_counts(seasons=list(seasons)),
                            "season", "week", "game_type", "player", "offense_pct")
    return (s.filter(pl.col("game_type") == "REG")
             .select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64),
                     pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                     pl.col("offense_pct").cast(pl.Float64))
             .group_by(["season", "week", "key"]).agg(pl.col("offense_pct").max()))


def injury_severity(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Ordinal game-status severity for the week, from the report published before it.

    Deliberately crude -- Out 3, Doubtful 2, Questionable 1, otherwise 0. The fitted
    (status, practice) retention table in `hub.models.injury` is the real instrument and it is
    already measured at this grain (+0.170 MAE at 3.8 se); this ordinal exists only so the
    screen has *something* in the injury slot, and a null here is a null about the ordinal.

    **Three shapes this once accepted and now refuses**, each of them a refusal on purpose.
    `game_status` standing in for `report_status` and `player_name` for `full_name` were read
    for years by a private lookup that took whichever name arrived: that is the Week 7 failure
    this repo names, a rename passing as a working column, and the ordinal it produces is
    computed from something `INJURIES` never promised. The third is a frame with no
    `practice_status` at all, which used to become the string "None" -- and "None" is already a
    legal designation here, meaning "on the report, nothing said about practice". A whole
    column of it is indistinguishable from a real week in which nobody was designated, so the
    (status, practice) pair `hub.models.injury` keys its retention table on would have
    narrowed to status alone with nothing saying so.

    What that refusal costs the live path is what ADR-0023 settles, one level up:
    `build_panel` degrades around it rather than propagating it, and what it degrades to is
    null rather than healthy. Refusing here and coping there is what keeps one statement of
    what this source may return.

    **`conform` and not `nflverse.load`**, for the reason `snap_share` gives above: the frozen
    capture carries five of the nine columns `INJURIES` declares, so the cached path refuses
    it. See the note above `game_context`.
    """
    import nflreadpy as nfl
    inj = INJURIES.conform(nfl.load_injuries(seasons=list(seasons)),
                           "season", "week", "full_name", "report_status", "practice_status")
    sev = (pl.when(pl.col("report_status") == "Out").then(3.0)
             .when(pl.col("report_status") == "Doubtful").then(2.0)
             .when(pl.col("report_status") == "Questionable").then(1.0).otherwise(0.0))
    # `status` and `practice` are carried alongside the ordinal because they are the cells
    # `hub.models.injury` fits its retention table on -- the ordinal is for the screen, the
    # pair is for the model. Practice status arrives as prose ("Did Not Participate In
    # Practice") and the first seven characters separate the three cases, which is the
    # normalisation `injury.observations` already uses.
    return (inj.select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64),
                       pl.col("full_name").map_elements(player_key, return_dtype=pl.Utf8)
                         .alias("key"),
                       sev.alias("inj_sev"),
                       pl.col("report_status").fill_null("None").alias("status"),
                       practice_key("practice_status").alias("practice"))
               .sort("inj_sev", descending=True)
               .group_by(["season", "week", "key"])
               .agg(pl.col("inj_sev").max(), pl.col("status").first(),
                    pl.col("practice").first()))


# The three columns the injury report contributes, with the dtypes `injury_severity` returns
# them in. Named once because the degraded path has to produce them without the report.
INJURY_COLUMNS = {"inj_sev": pl.Float64, "status": pl.Utf8, "practice": pl.Utf8}


def injury_columns(p: pl.DataFrame, seasons: Sequence[int]) -> pl.DataFrame:
    """The Panel's three injury columns, joined on -- or null, if the report was refused.

    ADR-0023. `injury_severity` refuses a report whose columns are not the ones `INJURIES`
    declares, `build_panel` calls it unconditionally, and the Sunday panel is the live path
    CLAUDE.md's degradation rule is written for: one renamed column at nflverse would
    otherwise cost the whole panel rather than the one feature that reads it.

    **Null, and not `Healthy`.** The fill below is what "this player has no row on the injury
    report" means, and a refused report is not that -- it is "nobody has one, and no one
    knows". Filling it would publish a league in perfect health on the week the source broke,
    and `inj_sev` would reach `hub.models.weekly_screen` as a measured zero on every row.
    Null is the one value in these columns that has never meant anything else -- the fill
    below leaves none on a served Panel, and `injury_severity` says a null here is a null
    about the ordinal. The screen reports it as nothing measured, and the refusal goes to
    stderr in the contract's own words, naming the column that was not there.

    Only a `ContractViolation` degrades. A source that changed shape is what the Panel can
    serve around; a `TypeError` inside the ordinal is this module being wrong, and swallowing
    that would hide a defect behind a column of nulls that looks exactly like a quiet week.
    """
    try:
        sev = injury_severity(seasons)
    except ContractViolation as e:                # graceful degradation, per CLAUDE.md
        print(f"hub.models.panel: the injury report was refused, so the Panel carries "
              f"null injury columns rather than healthy ones -- {e}", file=sys.stderr)
        return p.with_columns([pl.lit(None, dtype=dt).alias(c)
                               for c, dt in INJURY_COLUMNS.items()])
    return (p.join(sev, on=["season", "week", "key"], how="left")
             .with_columns(pl.col("inj_sev").fill_null(0.0),
                           pl.col("status").fill_null("Healthy"),
                           pl.col("practice").fill_null("Healthy")))


def week_windows(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """First and **last** kickoff per (season, week). The last one is what the join uses.

    Reads `schedules`, so it is unrouted for the same reason `game_context` is -- the note
    above that function.
    """
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


def consensus_pin(as_of: str | None = None) -> Pin | None:
    """The pin beside the rankings entry `weekly_consensus` reads, or None if it is cold.

    The screen prints a data digest and has to name the same cache entry the panel read; the
    key is `(source, page, columns, as-of)` and stating it twice is how the printed provenance
    stops describing the load it claims to describe. So it is stated once, here.
    """
    return data_pin("ff_rankings", ["all"], cols=RANKINGS_COLS, as_of=as_of)


def weekly_consensus(seasons: Sequence[int],
                     as_of: str | None = None) -> pl.DataFrame:  # pragma: no cover - network
    """That week's cross-position consensus ECR, joined as-of to (season, week).

    Routed through `hub.fetch.nflverse.load_rankings` rather than reaching nflreadpy here: this
    is the screen's largest input and the only one whose archive is append-only, so it is the
    one that can be *pinned* -- validated against `FF_RANKINGS`, cached under a key that
    carries the as-of, and recorded in a `Pin` a run can print. Two screens at one as-of read
    the same rows however much FantasyPros has published in between, which is what the repo
    means by a re-run reproducing.

    `as_of` is an ISO date and it is **inclusive** -- a scrape *on* that day counts. One
    convention, and it is the loader's: `hub.draft.board.consensus` gave up its own strictly-
    before comparison to say so, and `hub.store.lines_as_of` already meant "at or before".
    Passing None loads the whole archive, unfiltered and unpinned, which is what every caller
    that has not asked to be reproducible gets.

    Note what this bound is *not*: a lower one. Every scrape the archive holds is still
    considered, and `assign_weeks` is what decides which week a scrape belongs to.
    """
    a = load_rankings("all", as_of=as_of, cols=RANKINGS_COLS)
    w = (a.filter(pl.col("page_type") == CONSENSUS_PAGE)
          .select(pl.col("scrape_date").str.to_date(),
                  pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
                  pl.col("ecr").cast(pl.Float64))
          .drop_nulls(["scrape_date", "ecr"]))
    return best_per_week(assign_weeks(w, week_windows(seasons)))


def best_per_week(joined: pl.DataFrame) -> pl.DataFrame:
    """One ECR per (season, week, key), with the lead of the scrape that supplied it.

    Split out of `weekly_consensus` so it can be tested: everything around it is network.

    Sorted so `first()` lands on the same scrape `min()` does. 3.3% of player-weeks carry two
    scrapes (1,216 of 37,151 over 2021-25) and 1,211 of those have two different `lead_days`,
    so without the sort the reported lead came from an arbitrary row and need not be the one
    that supplied the ECR -- and `LEAD_DAYS` is what `docs/weekly-screen.md` cites for the
    confound the whole screen is read against. `injury_severity` above sorts before its own
    `first()` for exactly this reason; this one did not.
    """
    return (joined.sort("ecr")
                  .group_by(["season", "week", "key"])
                  .agg(pl.col("ecr").min(), pl.col("lead_days").first())
                  .with_columns(pl.col("season").cast(pl.Int64),
                                pl.col("week").cast(pl.Int64)))


# --- which side of the rule each column the Panel hands back is on ----------------------
#
# The rule at the top of this module is kept *mechanically* for the derived features --
# `prior_means`, `trend` and `recent_mean` all route through `expanding_weeks` and
# `.shift(1)`, so there is one statement of it -- and it was never kept for the frame those
# are built on. `build_panel` starts from the realised week-level frame and never strips it,
# so `targets`, `receptions`, `offense_pct`, `tds`, `yds` and a dozen more of week w's own
# outcome ride along to the return.
#
# **They are kept on purpose and the keeping is not the defect.** `hub.models.weekly` fits
# its multiplier with `targets` on the left-hand side, `weekly_screen.screen_usage` screens
# against **Usage**, and Gate B scores a lineup on realised points. Every one of those is
# reading a realised column *as an outcome*, which is the only thing it is.
#
# The defect was that the return said nothing about which was which, so using the Panel
# correctly meant knowing from suffix convention alone which of sixty columns were safe. A
# caller naming `offense_pct` where it meant `snap_trend` got a leak and nothing refused.
# Today's one screen is disciplined, and that is a property of the caller. #176 is what this
# species costs when nobody catches it: a coverage measurement centred on each player's own
# realised mean, published, and worth 3.7 points of headline once corrected.
#
# So the Panel names four sides, puts every column it serves on exactly one of them, and
# **refuses to hand back a column that is on none** -- the default for a column nobody has
# classified is unsafe, which is the direction a leakage guard has to fail in.


IDENTITY: tuple[str, ...] = ("player_id", "player_display_name", "key", "season", "week",
                             "team", "position")
"""Who the row is about and which week it is. The row's address, not a measurement."""


PRE_KICKOFF: tuple[str, ...] = (
    # **The line.** `game_context` is one row per (team, season, week) of what the betting
    # market and the schedule publish before the game -- the spread, the total, the derived
    # team total, and the fixture facts around it.
    "opp", "own_spread", "total_line", "implied_total", "rest", "roof", "wind", "is_home",
    # **The opponent**, under the name `player_stats` gives it. `opp` above is the same fact
    # off the schedule; the fixture is known months out either way.
    "opponent_team",
    # **The injury report.** Named once in `INJURY_COLUMNS`, because the degraded path has to
    # produce the same three without it, and spelled from there rather than beside it.
    *INJURY_COLUMNS,
    # Consensus, and the preseason board. Not among the three the module docstring names, and
    # the same species: `assign_weeks` and `CONSENSUS_MAX_LEAD_DAYS` exist to keep a scrape on
    # the near side of the kickoff it is assigned to, and `lead_days` is that distance itself
    # -- the confound `docs/weekly-screen.md` reads the whole screen against. `preseason_ecr`
    # is an August **Consensus** opinion about a September season. Both have to stay
    # reachable: `ecr` is half of `weekly_screen.CONTROLS`, and `preseason_ecr` is what
    # `hub.models.weekly`'s consensus-anchored shrinkage regresses toward.
    "ecr", "lead_days", "preseason_ecr")
"""Facts published *for* week w, which `docs/method.md` rule #2 counts as week-w information.

The exceptions the rule names, and a Panel that made them unreachable would have been broken
rather than tightened -- a screen with no line, no injury report and no opponent measures
nothing this repo asks about.
"""


DERIVED_SUFFIXES: tuple[str, ...] = ("_prior", "_recent", "_trend")
"""What `prior_means`, `recent_mean` and `trend` name their output.

A *consequence* of the rule rather than a convention beside it: all three route through
`expanding_weeks` and `.shift(1)`, so there is no way to get one of these names except by
going through the one statement of the rule. That is a claim, not a proof -- nothing stops a
future author naming a realised column `foo_trend` -- and what tests the claim is
`test_no_feature_moves_when_its_own_weeks_play_is_rewritten`, which runs its perturbation
over exactly the set `feature_columns` returns.
"""


DERIVED: tuple[str, ...] = ("ppg_before", "games_before", "dvp",
                            "td_ppg_before", "nontd_ppg_before")
"""The five derived columns that carry no suffix, because they were renamed or reshaped.

`ppg_before` and `games_before` are `prior_means`' `fantasy_points_ppr_prior` and `prior_n`
renamed at the join; `dvp` is `allowed_prior` over its positional league mean. `td_rate_prior`
needs no entry: it is one prior over another and the suffix already says so.

`td_ppg_before` and `nontd_ppg_before` are `ppg_before` split in two and sum back to it
exactly (#179). They take the same name shape as the column they partition, so they are
listed here for the same reason it is -- every ingredient is a `_prior`, and the reshape is
what loses the suffix.
"""


OUTCOMES: tuple[str, ...] = (
    # Week w's own realised play, taken from the fetch declaration rather than restated, so
    # this set grows with `weekly_stats` instead of drifting behind it. The keys and the
    # opponent come out because they are already spoken for above, and `season_type` never
    # reaches the Panel -- `weekly_stats` filters on it and drops it.
    *(c for c in WEEK_STATS_COLS
      if c not in IDENTITY and c not in PRE_KICKOFF and c != "season_type"),
    # The two totals `build_panel` forms from the columns above, on the realised frame.
    "tds", "yds",
    # The share of week w's snaps he actually took. `snap_trend` is the feature built from it.
    "offense_pct",
    # `spec.routes`: the share of week w's charted pass plays he was on the field for.
    "route_pct",
    # `spec.scheme`: how the offence actually played week w. `{r}_trend` is the feature.
    "pass_rate", *SCHEME)
"""Week w's own outcome, kept as an intermediate and never usable as a feature for week w.

Legitimate to read *as an outcome* -- `hub.models.weekly` fits against these and the Usage
screen measures against them -- which is what `require_features` refuses and
`cell_correlations`' `outcome=` parameter is for.
"""


class PanelRuleViolation(ValueError):
    """A column was about to be used on the wrong side of the before-its-outcome rule.

    Two cases, one name, because they are one mistake at two depths. `_served` raises it for a
    column that reached the return on none of the four sides -- nobody has said whether it is
    measured before week w or on it. `require_features` raises it for a column that *is*
    classified, as week w's own outcome, and was handed over where a feature was meant.

    Deliberately not a `ContractViolation`, which `injury_columns` catches and degrades
    around. That one means a source changed shape and the Panel can be served anyway; this one
    means this repo is wrong about its own frame, and serving around it would be serving a
    Panel whose one rule nobody is keeping.
    """


def column_role(name: str) -> str | None:
    """Which side of the before-its-outcome rule a column is on. None if it is on none.

    Order matters at one place only: `opponent_team` and the identity columns are also
    `player_stats` columns, so they are answered before `OUTCOMES` -- which is why `OUTCOMES`
    subtracts them at its own definition rather than relying on the order here.
    """
    if name in IDENTITY:
        return "identity"
    if name in PRE_KICKOFF:
        return "pre-kickoff"
    if name in DERIVED or name.endswith(DERIVED_SUFFIXES):
        return "derived"
    if name in OUTCOMES:
        return "outcome"
    return None


FEATURE_ROLES: tuple[str, ...] = ("pre-kickoff", "derived")
"""The two roles the rule permits as a feature for week w."""


def feature_columns(p: pl.DataFrame) -> tuple[str, ...]:
    """The Panel's columns that satisfy its rule: measured before week w, or published for it.

    What a caller asks instead of reading suffixes. In frame order, so a `select` over it
    keeps the Panel's own column order.
    """
    return tuple(c for c in p.columns if column_role(c) in FEATURE_ROLES)


def outcome_columns(p: pl.DataFrame) -> tuple[str, ...]:
    """The Panel's raw week-w columns: this week's own outcome, kept as an intermediate."""
    return tuple(c for c in p.columns if column_role(c) == "outcome")


def require_features(p: pl.DataFrame, names: Sequence[str]) -> None:
    """Raise if any name is week w's own outcome, or is not on `p` at all.

    The refusal criterion 2 of #203 asks for, at the seam where a column becomes a feature.
    A caller that means a realised column as an **outcome** says so by another route --
    `weekly_screen.cell_correlations` takes it as `outcome=`, `hub.models.weekly` puts it on
    the left-hand side of a fit -- and neither goes through here.

    **Refuses the named outcomes, not the unrecognised.** Default-deny belongs at
    `build_panel`'s own boundary, where the frame really is a Panel and an unclassified
    column is this module having grown one; here it would only refuse the hand-built frames
    the screen's own unit tests are written on, which are not Panels and do not claim to be.
    The two compose: nothing unclassified can reach a served Panel in the first place, so on
    a Panel this check is total.
    """
    bad = [c for c in names if c not in p.columns or column_role(c) == "outcome"]
    if not bad:
        return
    lines = []
    for c in sorted(set(bad)):
        if c not in p.columns:
            lines.append(f"  `{c}` is not on this frame at all")
            continue
        kin = [k for k in (f"{c}_prior", f"{c}_recent", f"{c}_trend") if k in p.columns]
        beside = f"; measured before its week it is {' or '.join(kin)}" if kin else ""
        lines.append(f"  `{c}` is week w's own outcome{beside}")
    raise PanelRuleViolation(
        "these cannot be features of the week they are measured on:\n" + "\n".join(lines)
        + "\n`hub.models.panel.feature_columns` lists what may be. A realised column is a "
          "legitimate *outcome* -- say so with `outcome=` rather than passing it here.")


def _served(p: pl.DataFrame) -> pl.DataFrame:
    """The Panel, checked column by column on the way out. The boundary the rule is kept at.

    Every source above narrows with an explicit `select`, so the way a raw outcome column
    starts surviving to the return is a change *here* -- a new join, a new total, a source
    that grew a column and was joined whole. That change now has to say which side of the
    rule its column is on, and until it does the Panel refuses to serve it.

    **Raising, on the live Sunday path.** CLAUDE.md's degradation rule is about a source that
    changed shape, which `injury_columns` serves around; this is the same distinction that
    module already draws one level down, where a `ContractViolation` degrades and a
    `TypeError` does not. An unclassified column is this repo being wrong about its own
    frame, and a Panel served with one is a Panel whose rule nobody is keeping.
    """
    unknown = [c for c in p.columns if column_role(c) is None]
    if not unknown:
        return p
    raise PanelRuleViolation(
        f"the Panel was about to hand back {unknown}, and nothing says whether they are "
        f"measured before week w or on it. Add each to `IDENTITY`, `PRE_KICKOFF`, `DERIVED` "
        f"or `OUTCOMES` in hub.models.panel. If it is week w's own play it belongs in "
        f"`OUTCOMES`, where `require_features` will refuse it as a feature and callers that "
        f"want it as an outcome still reach it -- which is the whole point of naming it.")


def build_panel(seasons: Sequence[int] = SEASONS,
                spec: PanelSpec = SCREEN_SPEC,
                as_of: str | None = None) -> pl.DataFrame:  # pragma: no cover - network
    """One row per (player, season, week), with every feature measured before its outcome.

    `as_of` is an ISO date, inclusive of the day itself, and today it bounds **one** of the
    eleven sources: the FantasyPros archive behind `weekly_consensus`. That is the one this
    panel's reproducibility turned on -- it is the largest, it grows under the harness between
    two runs of the same screen, and it is the only one of the eleven declared append-only, so
    it is the only one an as-of can *filter* rather than merely label. The other ten still
    reach nflverse live and revise in place; U2 of
    `docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md` routes them. A panel
    built at an as-of is therefore pinned in its consensus control and not yet in its outcome,
    which is worth knowing before quoting a digest as if it named the whole panel.

    A parameter rather than a `PanelSpec` field on purpose: the spec says *which sources* a
    caller wants on its panel, and the as-of says *when* -- one is the shape and the other is
    the moment, and folding the second into the first would make every existing spec a claim
    about a date it never made.

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
    player-week grid where a missing row is a zero, and that is where the term belongs. If the
    injury report itself is refused, all three of its columns are carried **null** and the
    Panel is served without them rather than not served at all -- `injury_columns` and ADR-0023.

    **What comes back, and how a caller tells one half from the other.** Both returns go
    through `_served`, which puts every column on one of the four sides named above it and
    refuses one that is on none. So the frame carries week w's own realised play *and* the
    features measured before it, as it always did, and `feature_columns` now says which are
    which -- `outcome_columns` says the rest, and `require_features` is what refuses a caller
    that reaches for one of those where a feature was meant.
    """
    stats = weekly_stats(seasons).with_columns(
        pl.col("player_display_name").map_elements(player_key, return_dtype=pl.Utf8).alias("key"))
    p = stats.join(game_context(seasons), on=["season", "week", "team"], how="left")

    # `tds` and `yds` are formed here, on the realised frame and *above* the coalesce rather
    # than below it. They are a pair -- `td_rate_prior` is the one over the other -- and below
    # the coalesce `yds` picked up the expected yardage while `tds`, which deliberately has no
    # expected version here (see `EXPECTED`), stayed realised. The rate became realised
    # touchdowns per expected yard, which is not a rate of anything. The same slip left the
    # Panel's own week disagreeing with itself: the total said one thing and the three columns
    # it is the sum of said another, on 342 of 344 rows of the frozen archive against 0 under
    # the default spec.
    counted = stats.with_columns(
        (pl.col("receiving_tds") + pl.col("rushing_tds") + pl.col("passing_tds")).alias("tds"),
        (pl.col("receiving_yards") + pl.col("rushing_yards")
         + pl.col("passing_yards")).alias("yds"))

    if spec.expected:
        counted = counted.join(expected_weekly(seasons), on=["season", "week", "key"],
                               how="left")
        # Where `ff_opportunity` has no row for him, the realised value stands in rather than
        # a null propagating into every prior downstream.
        #
        # **What this spec swaps is the four columns of `EXPECTED`, and nothing else.** Not
        # the Panel's week: `p` is bound above and keeps realised play, because the week is
        # the *outcome* the projection is fitted against and scored on, and a week whose
        # components were expected while its `fantasy_points_ppr` stayed realised would be
        # the same disagreement one level up. What the swap reaches is the frame the priors
        # are taken over, which is what the stability argument for `ff_opportunity` was ever
        # about -- a better estimate of the rate he carries into next week, not a better
        # account of the week that happened. `docs/expected-and-routes.md` measured it null.
        #
        # So under this spec the Panel carries two yardage measurements, on purpose. The
        # components' priors are expected, because `hub.models.weekly` divides each by a count
        # to hold an efficiency. `yds_prior` is realised, because it is the touchdown rate's
        # denominator and the numerator beside it is. Those two do not add up to each other
        # and are not meant to; the totals that do have to agree are the ones inside a single
        # measurement, which is what `test_panel` asserts under each spec.
        counted = counted.with_columns(
            [pl.coalesce(pl.col(exp), pl.col(real)).alias(real)
             for real, exp in EXPECTED.items()])

    p = p.join(counted.select("player_id", "season", "week", "tds", "yds"),
               on=["player_id", "season", "week"], how="left")
    own = prior_means(counted.sort(["player_id", "season", "week"]), ["player_id"],
                      [OUTCOME, "yds", *USAGE, *YARDS, *TURNOVERS, *TD_COLUMNS],
                      within_season=True)
    p = p.join(own, on=["player_id", "season", "week"], how="left").rename(
        {f"{OUTCOME}_prior": "ppg_before", "prior_n": "games_before"})
    p = p.with_columns(
        (pl.col("tds_prior") / (pl.col("yds_prior") + 1e-9)).alias("td_rate_prior"))

    # **`ppg_before`, split along the seam the feature above is built across.** The screen's
    # player control is season-to-date PPR points a game, and PPR points *contain* touchdowns
    # -- so controlling on the total holds prior scoring fixed with the feature's own numerator
    # inside it, and at a fixed total a higher touchdown rate is arithmetically fewer yards.
    # A screen run on that basis cannot tell "efficiency regresses" from "high scorers
    # regress", which is the claim two modules lean on. Issue #179.
    #
    # Priced, not counted: `TD_COLUMNS` says why the three are weighted separately. Written as
    # a chain of `+` rather than `pl.sum_horizontal`, which ignores nulls by default and would
    # turn a player with no prior of one kind into a zero contribution rather than a null row
    # the screen drops.
    #
    # The remainder is subtraction, not a second sum over the scored columns that are not
    # touchdowns. That is the property that makes the pair a *basis* for the old control
    # rather than a different one: `td_ppg_before + nontd_ppg_before == ppg_before` exactly,
    # on every row, so the two-column control set spans the one-column set and any movement in
    # a screened coefficient comes from relaxing the constraint that the two components carry
    # one shared slope -- not from having controlled for some other quantity. Reconstructing
    # the remainder from `SCORING` instead would leave the 0.6% of player-weeks whose points
    # are return and special-teams scores in neither component, and the identity would fail on
    # exactly the rows where it was doing work.
    p = p.with_columns(
        reduce(operator.add,
               (pl.col(f"{c}_prior") * components.SCORING[c] for c in TD_COLUMNS)
               ).alias("td_ppg_before"))
    p = p.with_columns(
        (pl.col("ppg_before") - pl.col("td_ppg_before")).alias("nontd_ppg_before"))
    # The three per-type priors were the ingredients and are not features anyone asked for.
    # Dropped so the Panel grows by the two columns the decomposition is, and not by five.
    p = p.drop([f"{c}_prior" for c in TD_COLUMNS])

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
    if spec.routes:
        # Joined on the gsis id rather than a normalised name: `participation` and
        # `player_stats` key on the same identifier, so this one needs no crosswalk at all.
        # Opt-in because loading five seasons of play-level participation is the slowest
        # thing in the panel and the feature it produces is a measured null.
        p = p.join(route_share(seasons), on=["season", "week", "player_id"], how="left")
        p = trend(p, "route_pct", "player_id", "route_trend")
    p = trend(p, "target_share", "player_id", "tgt_trend")
    if spec.scheme:
        # Trended on the unique team-week frame and joined once. Trending on the panel would
        # call `trend` with a team key against thirty players a team-week -- see its guard.
        rates = scheme_rates(seasons)
        for r in (*SCHEME, "pass_rate"):
            rates = trend(rates, r, "posteam", f"{r}_trend")
        p = p.join(rates, left_on=["team", "season", "week"],
                   right_on=["posteam", "season", "week"], how="left")
    p = injury_columns(p, seasons)
    for c in USAGE:
        p = recent_mean(p, c)
    if spec.ranks is not None:
        # Injected rather than fetched. A board lives under `draft/`, and six `draft/` modules
        # import `models/` while nothing goes the other way -- inverting that to save one
        # caller a parameter is the mistake `board_as_of` was moved out of `models/` to fix.
        p = p.join(spec.ranks, on=["key", "season"], how="left")
    if not spec.consensus:
        # The gate needs a projection for every rostered player, including the ones consensus
        # does not list -- being unranked is the incumbent's *answer*, not a reason to have no
        # projection. Only the screen, which measures beyond consensus, requires it to exist.
        return _served(p)
    return _served(p.join(weekly_consensus(seasons, as_of=as_of),
                          on=["season", "week", "key"], how="inner"))
