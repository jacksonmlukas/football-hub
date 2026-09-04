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
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import polars as pl

from hub.config import DRAFTED_POSITIONS, SEASON_COMPLETED
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


def build_panel(seasons: Sequence[int] = SEASONS,
                spec: PanelSpec = SCREEN_SPEC) -> pl.DataFrame:  # pragma: no cover - network
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

    if spec.expected:
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
    p = p.join(injury_severity(seasons), on=["season", "week", "key"], how="left")
    p = p.with_columns(pl.col("inj_sev").fill_null(0.0),
                       pl.col("status").fill_null("Healthy"),
                       pl.col("practice").fill_null("Healthy"))
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
        return p
    return p.join(weekly_consensus(seasons), on=["season", "week", "key"], how="inner")
