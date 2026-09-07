"""Touchdown luck: last season's actuals held against the current board.

Three separately measured facts make this worth computing:

1. **Touchdown rate per yard does not persist.** Year over year it is +0.004 receiving and
   -0.030 rushing -- indistinguishable from zero (`docs/component-projection.md`). A
   player's touchdown rate says nothing about his next one.
2. **Fully regressing it beats carrying his points forward**, and the fitted optimal shrink
   is 1.0 rather than a partial one (`docs/volume-model.md`).
3. **The draft market does not fully regress it.** Weighting prior touchdown points against
   prior yardage points, the room prices touchdowns at 1.02 relative to volume for
   quarterbacks where their true predictive weight is -0.05. See `docs/td-luck.md` for the
   per-position table and, more importantly, for how much of this is established and how
   much is directional.

So a player whose prior season carried more touchdowns than his yardage supports is being
priced on something that will not repeat, and one whose touchdowns lagged his yardage is
being marked down for the same reason in reverse.

This is not the board's existing `fp_over_expected`. That measures realised points against
expected points from opportunity; this measures realised touchdowns against the yardage that
produced them. On the live board they correlate at +0.16 -- and a genuine overlap would show
as a strong *negative*, since the two are signed in opposite directions.
"""
from __future__ import annotations

import polars as pl

from hub.config import drafted_positions
from hub.draft import prior_signal
from hub.models.components import td_rate

# Fewer games than this and the number is noise wearing a number's clothes: a two-game
# sample of touchdown luck is one red-zone target either way.
MIN_GAMES = 6

# Points of next-season scoring lost per point of prior-season touchdown luck, from
# `ppg_next ~ proj_ppg + td_luck` on historical ESPN projections. A projection that already
# regressed touchdowns would leave nothing for td_luck to explain and this would be zero.
#
# QB is the clear case: -0.540, 95% CI [-1.057, -0.125], 99.5%.
#
# WR is applied at Jackson's direction and is a judgment call, which is worth stating
# plainly rather than letting a future reader assume it cleared the same bar. It comes back
# at -0.286 with a 95% interval of [-0.797, +0.170] -- 89% of the bootstrap on the right
# side, but the interval contains zero. What makes it defensible rather than fishing is that
# the mechanism was measured first and independently (touchdown rate has no year-over-year
# persistence) and predicts this sign for every position before any of it was fitted.
#
# RB stays out, and that is not a threshold call: it comes back at +0.253, the wrong sign
# entirely, so ESPN is if anything conservative about running back touchdowns. Correcting it
# would move the projection the wrong way.
#
# The level of the projections is fine -- the coefficient on proj_ppg is 0.95-1.04 across
# positions. This is specifically a touchdown bias, not a calibration problem.
TD_LUCK_BETA: dict[str, float] = {"QB": -0.540, "WR": -0.286}

_PHASES = (("receiving_yards", "receiving_tds", "rec", 6.0),
           ("rushing_yards", "rushing_tds", "rush", 6.0),
           ("passing_yards", "passing_tds", "pass", 4.0))


def prior_season(season: int, cache=None) -> pl.DataFrame:
    """Per-player season totals for the phases touchdown luck needs.

    Regular season only: a playoff run inflates totals for a handful of players and none of
    it is in the fantasy season anyway.
    """
    from hub.fetch import nflverse
    cols = ("player_id", "player_display_name", "position", "season", "week",
            "season_type", "fantasy_points_ppr", "receiving_yards", "receiving_tds",
            "rushing_yards", "rushing_tds", "passing_yards", "passing_tds")
    w = nflverse.load("player_stats", seasons=[season], cols=cols, cache=cache).filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(list(drafted_positions())))
    return (w.group_by(["player_display_name", "position"])
             .agg(pl.len().alias("g"),
                  *[pl.col(c).sum().alias(c) for c in
                    ("receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds",
                     "passing_yards", "passing_tds")])
             .rename({"player_display_name": "player", "position": "pos"}))


def td_luck(season: pl.DataFrame) -> pl.DataFrame:
    """Fantasy points per game a player scored above the touchdowns his yardage supports.

    `season` needs player, pos, g and the per-phase yardage and touchdown totals. Positive
    means he outscored his own yardage -- the part least likely to repeat.
    """
    df = season.filter(pl.col("g") >= MIN_GAMES)
    if df.is_empty():
        return df.select("player", "pos").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("td_luck"))

    # Scored in fantasy points, not touchdown counts: a passing touchdown is worth four and
    # a rushing one six, so counting touchdowns understates quarterbacks by a third -- and
    # quarterbacks are where the market misprices most.
    actual = sum((pts * pl.col(td) for _, td, _, pts in _PHASES), start=pl.lit(0.0))
    expected = sum(
        (pts * pl.col(yd) * pl.col("pos").replace_strict(
            {p: td_rate(p, phase) for p in drafted_positions()},
            default=0.0, return_dtype=pl.Float64)
         for yd, _, phase, pts in _PHASES), start=pl.lit(0.0))

    return df.with_columns(((actual - expected) / pl.col("g")).alias("td_luck"))


def attach(board: pl.DataFrame, season: pl.DataFrame) -> pl.DataFrame:
    """Add a `td_luck` column to a draft board.

    The join itself is `prior_signal.join_by_player`, which `hub.draft.durability` also uses:
    it was the same twelve lines in both files -- improvements.md #15.
    """
    return prior_signal.join_by_player(board, td_luck(season), "td_luck")


def correct_projection(board: pl.DataFrame, column: str = "proj_blend") -> pl.DataFrame:
    """Mark a projection down for touchdown luck, where the projection is known to carry it.

    This is what connects the signal to the objective. `hub.draft.optimize` scores seasons
    against `proj_blend`, so a bias left in that column is a bias in every P(win) it
    reports -- and a signal that is only printed is decoration.

    **Silent when `td_luck` is absent, on purpose, and that silence has a price.** It is paid
    one caller away and caught there; the comment on the early return says by what.
    """
    if column not in board.columns or "td_luck" not in board.columns:
        # Two absences, one return, and they do not cost the same thing.
        #
        # `column` is `proj_blend`, which only the ADP stage leaves. Without it there is no
        # Corrected ADP for this term to be missing *from*, and the build report says so by
        # naming nothing at all when its `adp` flag is false.
        #
        # `td_luck` gone from a board that does carry `proj_blend` is the expensive one, and
        # it is the whole of issue #121. `board.build` absorbs the touchdown-luck stage
        # rather than refusing to build -- correctly; a board that will not build for one
        # advisory column is the operator-dependence CLAUDE.md warns about -- and
        # `_attach_market` then computes Corrected ADP from the terms that did apply and
        # reports it as having run. What comes out is a different ranking rather than a
        # thinner board: 346 of 457 players moved by up to 28.1 picks, 110 of the first 192
        # changing rank by up to 19 places. Wider than the quarterbacks and receivers this
        # term touches, because `optimize.corrected_adp` re-fits the curve every player is
        # priced against, not only the ones corrected.
        #
        # Nothing is recorded from here and nothing should be: this runs inside `_stage`, and
        # the report is what records a stage rather than what the stage writes to. The catch
        # is `BuildReport.corrections_missing`, which derives the missing terms from flags the
        # build already set. `hub.draft.report` prints them beside the ranking for an operator
        # on the clock, and `hub.models.experiment.require_corrections` refuses the board
        # outright for a Gate, where a season short a term is a second arm and not a thin one.
        return board
    adjustment = prior_signal.priced("td_luck", TD_LUCK_BETA)
    return board.with_columns(
        (pl.col(column) + adjustment).clip(0.0).alias(column))
