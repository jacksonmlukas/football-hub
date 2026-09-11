"""Weeks 15-17 strength of schedule.

The fantasy playoffs are three known games against known defences, and nobody drafting
off ESPN's app prices them. A season-long projection says what a player does on average
against an average defence; it says nothing about whether his week 16 is against the
softest secondary in the league or the hardest.

Two ingredients, both bulk pulls:

  * defence vs position -- PPR points each defence allowed to each position per game,
    from last season. A defence soft against WRs is not necessarily soft against RBs, so
    this is never collapsed to a single "good defence" number.
  * the weeks 15-17 schedule for the season ahead, which is already published.

The output is a ratio: 1.10 means this player's playoff opponents allowed 10% more to his
position than a league-average defence did. It is a tiebreaker, not a ranking -- it moves
players inside a tier, and last year's defence is a noisy guide to this year's.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.config import DRAFTED_POSITIONS, SEASON_AHEAD, SEASON_COMPLETED

PLAYOFF_WEEKS = (15, 16, 17)

# The ridge penalties the sensitivity reports across, fixed here before any run (#180). The
# unit is games: a penalty of 1.0 shrinks a defence's effect as if it had faced one extra
# average offence, so the range runs from "believe the data almost outright" to "eight
# games of prior". Not fitted, and deliberately -- a fitted penalty would be a constant
# needing provenance under ADR-0006, where a reported range needs none. Whether the playoff
# ranking is stable across it is the finding; which point on it is "right" is not a
# question the data can answer with a handful of games per unit.
RIDGE_PENALTIES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

# The penalty a run actually uses is `DraftConfig.sos_ridge` -- a setting, covered by
# `config_digest`, and the sensitivity above is what licenses its default. Same shape as
# `hub.season.pool`: the axis lives beside the sweep, the value beside the other choices.
NOT_FITTED_BECAUSE = (
    "nothing here is measured. RIDGE_PENALTIES is the axis `sos_sensitivity` sweeps and not "
    "a value any figure is computed at -- the penalty the board reads is "
    "DraftConfig.sos_ridge, which is a stated choice covered by `config_digest`. Moving the "
    "axis changes what the sensitivity reports across, not what any number is.")

# FantasyPros and nflverse disagree on three codes. FA is genuinely teamless.
TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "LV": "LV", "WSH": "WAS", "ARZ": "ARI"}


def canon_team(team: str | None) -> str | None:
    if not team or team in ("FA", ""):
        return None
    return TEAM_ALIASES.get(team, team)


def _offence_column(stats: pl.DataFrame) -> str:
    """nflverse has called the scoring team `team` and, before that, `recent_team`."""
    for c in ("team", "recent_team"):
        if c in stats.columns:
            return c
    raise ValueError(
        "player stats carry no offence column (`team` or `recent_team`), so points allowed "
        "cannot be adjusted for who scored them; got " + ", ".join(stats.columns))


def _ridge_defence_effects(games: pl.DataFrame, ridge: float) -> pl.DataFrame:
    """Points a defence allows against an *average* offence, per position -- issue #180.

    Unadjusted defence-vs-position is what a defence allowed to whoever it happened to play.
    A unit that drew strong offences looks porous and one that drew weak ones looks stout,
    which is the error the metric exists to avoid making about the playoff *schedule*, made
    instead about the defences on it.

    So each position is fitted as a two-way additive model on the game table:

        allowed[defence, offence] = mean + delta[defence] + omega[offence]

    with an L2 penalty `ridge` on both effect vectors and none on the mean. What this
    returns is `mean + delta`, the defence's allowance with its opponents' quality taken
    out.

    **Ridge rather than plain least squares, and the reason is the sample.** Weeks 15-17 are
    judged from a partial season, so each team's effect rests on a handful of games -- and
    early on, on one. One game per defence against one offence each is an *underdetermined*
    system: every defence effect trades off exactly against its lone opponent's, and an
    unpenalised solver returns whatever the numerics hand it. The penalty shrinks every
    effect toward zero -- toward league average -- by an amount that falls as games
    accumulate, so an early-season number sits between the raw ratio and one and grows
    toward the raw ratio as the season fills in. That is the degradation #180's third
    criterion asks for, and it is the whole reason this is a ridge and not a regression.

    Solved in closed form per position: X = [1 | D | O] over the game rows, and
    beta = (X'X + ridge * I*)^-1 X'y with I* zero on the intercept. Small enough (thirty-two
    defences, thirty-two offences) that nothing iterative is warranted.
    """
    out = []
    for pos, g in games.group_by("pos", maintain_order=True):
        d_names = sorted(g["defense"].unique().to_list())
        o_names = sorted(g["offense"].unique().to_list())
        d_ix = {n: i for i, n in enumerate(d_names)}
        o_ix = {n: i for i, n in enumerate(o_names)}
        n, nd, no = g.height, len(d_names), len(o_names)
        X = np.zeros((n, 1 + nd + no))
        X[:, 0] = 1.0
        for r, (d, o) in enumerate(zip(g["defense"].to_list(), g["offense"].to_list(),
                                        strict=True)):
            X[r, 1 + d_ix[d]] = 1.0
            X[r, 1 + nd + o_ix[o]] = 1.0
        y = g["allowed"].to_numpy().astype(float)
        penalty = np.full(1 + nd + no, float(ridge))
        penalty[0] = 0.0
        beta = np.linalg.solve(X.T @ X + np.diag(penalty), X.T @ y)
        mean, delta = beta[0], beta[1:1 + nd]
        out.append(pl.DataFrame({"defense": d_names, "pos": [str(pos[0])] * nd,
                                 "ppg_allowed": (mean + delta).tolist()}))
    return pl.concat(out).sort(["defense", "pos"])


def _dvp_from_stats(stats: pl.DataFrame, ridge: float | None = None) -> pl.DataFrame:
    """PPR points allowed per game, by defence and position, indexed to league average.

    Summed per (defence, week) first: a defence that faces a committee backfield allowed
    those points whether they came from one back or three, and averaging player-level
    rows instead would reward defences that face deeper rotations.

    With `ridge` set, the per-game allowances are adjusted for the quality of the offence
    that scored them before averaging -- see `_ridge_defence_effects`. With it `None` the
    metric is the unadjusted one every published number was measured on; `sos_sensitivity`
    reports both side by side, which is how the two are kept comparable.
    """
    # Sorted before the second aggregation, and that is not tidiness. A `group_by` emits its
    # rows in a hash-dependent order that varies between calls, the mean below sums them, and
    # floating-point addition is not associative -- so without this the same input gives
    # answers differing at 7.1e-15, the board sorted on that float lands in a different ROW
    # ORDER, and the draft indexes the board by row. Two identical `board_as_of` calls
    # returned different boards, and every measurement drafting from them wobbled by ~0.04
    # points a team-week. improvements.md #18.
    keep = stats.filter(pl.col("position").is_in(DRAFTED_POSITIONS))
    if ridge is not None:
        # The game table keeps the offence, because that is what the adjustment removes.
        # Still summed within (defence, offence, week) first, for the committee reason above.
        offence = _offence_column(keep)
        games = (keep.group_by(["opponent_team", offence, "position", "week"])
                     .agg(pl.col("fantasy_points_ppr").sum().alias("allowed"))
                     .rename({"opponent_team": "defense", offence: "offense",
                              "position": "pos"})
                     .sort(["pos", "defense", "offense", "week"]))
        dvp = _ridge_defence_effects(games, ridge)
    else:
        per_week = (keep
                    .group_by(["opponent_team", "position", "week"])
                    .agg(pl.col("fantasy_points_ppr").sum().alias("allowed"))
                    .sort(["opponent_team", "position", "week"]))
        dvp = (per_week.group_by(["opponent_team", "position"])
                       .agg(pl.col("allowed").mean().alias("ppg_allowed"))
                       .rename({"opponent_team": "defense", "position": "pos"}))
    # `.mean().over("pos")` is a third aggregation over the second's output, so it needs the
    # same treatment; and the final sort carries `defense` as a tiebreaker, because two
    # defences with an identical ratio would otherwise order arbitrarily.
    return (dvp.sort(["defense", "pos"])
               .with_columns(
                   (pl.col("ppg_allowed") / pl.col("ppg_allowed").mean().over("pos"))
                   .alias("dvp_ratio"))
               .sort(["pos", "dvp_ratio", "defense"], descending=[False, True, False]))


def _opponents_from_schedule(sched: pl.DataFrame,
                             weeks: tuple[int, ...] = PLAYOFF_WEEKS) -> pl.DataFrame:
    """One row per (team, week, opponent) -- both sides of every game."""
    g = sched.filter(pl.col("week").is_in(weeks))
    home = g.select(pl.col("home_team").alias("team"), "week",
                    pl.col("away_team").alias("opponent"))
    away = g.select(pl.col("away_team").alias("team"), "week",
                    pl.col("home_team").alias("opponent"))
    return pl.concat([home, away]).sort(["team", "week"])


def _sos_from(dvp: pl.DataFrame, opponents: pl.DataFrame) -> pl.DataFrame:
    """Mean opponent dvp_ratio over the playoff weeks, per team and position."""
    joined = opponents.join(dvp, left_on="opponent", right_on="defense", how="inner")
    return (joined.group_by(["team", "pos"])
                  .agg(pl.col("dvp_ratio").mean().alias("wk15_17_sos"),
                       pl.len().alias("sos_games"))
                  .sort(["pos", "wk15_17_sos"], descending=[False, True]))


def sos_sensitivity(stats: pl.DataFrame, sched: pl.DataFrame,
                    penalties: Sequence[float] = RIDGE_PENALTIES,
                    weeks: tuple[int, ...] = PLAYOFF_WEEKS) -> pl.DataFrame:
    """The playoff-schedule ranking at every penalty, with the unadjusted metric as ridge 0.

    The disposition on #180 was a reported range rather than a fitted penalty. What the
    range has to show is whether the *ranking* moves across it -- a penalty that reorders
    the slate is a decision, one that does not is a detail. One row per (ridge, pos, team)
    carrying the ratio and its rank within the position, so the movement is a join away.
    """
    opponents = _opponents_from_schedule(sched, weeks)
    frames = []
    for ridge in (0.0, *penalties):
        dvp = _dvp_from_stats(stats, ridge=None if ridge == 0.0 else ridge)
        sos = _sos_from(dvp, opponents)
        frames.append(sos.with_columns(
            pl.lit(float(ridge)).alias("ridge"),
            pl.col("wk15_17_sos").rank(method="min", descending=True).over("pos")
              .cast(pl.Int64).alias("rank")))
    return pl.concat(frames).sort(["pos", "team", "ridge"])


def ranking_movement(table: pl.DataFrame) -> pl.DataFrame:
    """How far each team's rank moves across the penalty range -- the one number a reader
    can act on. `spread` of 0 means the choice of penalty never reordered that team."""
    return (table.group_by(["pos", "team"])
                 .agg((pl.col("rank").max() - pl.col("rank").min()).alias("spread"),
                      pl.col("rank").filter(pl.col("ridge") == 0.0).first()
                        .alias("rank_unadjusted"),
                      pl.col("rank").filter(pl.col("ridge") == _default_ridge()).first()
                        .alias("rank_default"))
                 .sort(["pos", "spread", "team"], descending=[False, True, False]))


def attach_sos(board: pl.DataFrame, sos: pl.DataFrame) -> pl.DataFrame:
    """Add wk15_17_sos to the board. Teamless or unmatched players get null, never a default.

    A silent 1.0 would read as "average playoff schedule" for a player we simply could
    not place, which is the kind of quiet wrong answer this repo keeps finding.
    """
    keyed = board.with_columns(
        pl.col("team").map_elements(canon_team, return_dtype=pl.Utf8).alias("_team"))
    return (keyed.join(sos, left_on=["_team", "pos"], right_on=["team", "pos"], how="left")
                 .drop("_team"))


def _default_ridge() -> float | None:
    """The penalty the board reads, from the one place a choice lives."""
    from hub.config import DraftConfig
    return DraftConfig().sos_ridge


def playoff_sos(season_ahead: int = SEASON_AHEAD, dvp_season: int = SEASON_COMPLETED,
                weeks: tuple[int, ...] = PLAYOFF_WEEKS,
                ridge: float | str | None = "config") -> pl.DataFrame:
    """Fetch both inputs and build the table. Two bulk pulls, no per-team loop.

    `ridge` defaults to `DraftConfig.sos_ridge`; pass `None` for the unadjusted metric the
    published ranking was measured on, or a float to override for one call."""
    import nflreadpy as nfl

    if ridge == "config":
        ridge = _default_ridge()

    stats = nfl.load_player_stats(seasons=[dvp_season])
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == "REG")
    sched = nfl.load_schedules().filter(pl.col("season") == season_ahead)
    return _sos_from(_dvp_from_stats(stats, ridge=None if ridge is None else float(ridge)),
                     _opponents_from_schedule(sched, weeks))
