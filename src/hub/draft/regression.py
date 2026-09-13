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

**Shown, and since 2026-09-10 never ranked on.** The three facts above are unchanged and the
column is still computed, attached and printed. What this module no longer has is a
`correct_projection`: #186 asked whether the *price* it put on the signal earned its place,
and held out it did not, so #48 emptied `TD_LUCK_BETA` and took touchdown luck out of the
board's Corrections. The comment on that constant carries the evidence, and it is the same
shape as ADR-0013 and ADR-0016 -- a real signal this repo cannot yet size.
"""
from __future__ import annotations

import polars as pl

from hub.config import drafted_positions
from hub.declare import chosen, fitted
from hub.draft import prior_signal
from hub.models.components import td_rate

# Fewer games than this and the number is noise wearing a number's clothes: a two-game
# sample of touchdown luck is one red-zone target either way.
MIN_GAMES = chosen(6)

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
#
# **REFITTED 2026-09-07 AND BOTH ARE DISPUTED. The constants below are unchanged; #48 is
# where they move or do not.** `hub.draft.fit_corrections` is the first committed harness for
# them, and it fits against `proj_blend` -- the column `correct_projection` actually writes to
# -- rather than `proj_ppg`, which is one of the two halves that column is built from. The two
# halves disagree about touchdowns by construction: `xfp_per_game` is an expectation and has
# already regressed them, `proj_ppg` carries them forward. Over 2018-2025:
#
#   QB  +0.504  [-0.150, +0.939]  sign-reversed, and positive against every baseline that can
#               be reconstructed, so no ESPN projection could produce -0.540
#   WR  -0.117  [-0.323, +0.095]  reproduced but underpowered, and it changes sign inside the
#               baseline bracket -- not resolvable without a projection that no longer exists
#
# Held out, applying either constant scores worse than applying no correction at all.
# See docs/fitted-corrections.md for the population, the bracket and what it does not say.
#
# **WITHDRAWN 2026-09-10. #186 decided it, #48 applied it, and this is the empty dict.** The
# ticket was re-scoped from "how should this correction be refined" to "does it earn its place
# at all", which is docs/method.md rule 8 -- the ceiling is below zero, so a refinement is
# chasing a gap that is not there. The three options were remove it, keep it at a measured
# shrink, or keep it per position where the sign is defensible; held out, all three land on
# removal:
#
#   * no position has a defensible sign. QB is positive at *both* ends of the bracket, and WR
#     changes sign inside it.
#   * the measured shrink is zero. `fit_corrections.shrink_curve` scores the shipped constant
#     at every factor from nothing to whole on the same held-out rows, and nothing wins.
#
# Empty rather than deleted, and the dict rather than a flag, because `prior_signal.priced`
# already gives an absent position the only honest reading there is: "a position absent from a
# `BETA` is one the fit found nothing for, and inventing a coefficient for it would ship an
# effect nobody measured". Every drafted position is now absent, which is the finding. The
# name survives so `fitted_digest` keeps covering it -- a correction that stopped applying is
# a model change, and ADR-0006 wants the version to move for it.
#
# **The signal is not withdrawn, only the price.** `td_luck` stays on the board, stays in the
# report and stays a stage; what stops is `proj_blend` moving by it, so touchdown luck is no
# longer a Correction in `hub.draft.board`'s sense and `correct_projection` is gone from this
# module. docs/td-luck.md and docs/fitted-corrections.md carry the restatement.
TD_LUCK_BETA: dict[str, float] = fitted({})

_PHASES = chosen((("receiving_yards", "receiving_tds", "rec", 6.0),
           ("rushing_yards", "rushing_tds", "rush", 6.0),
           ("passing_yards", "passing_tds", "pass", 4.0)))


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
