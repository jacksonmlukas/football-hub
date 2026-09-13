"""Availability as a per-player trait, and today's injury news as a separate thing.

`TALENT_CV` already carried availability, but only as a population average -- it was fitted
on points per *team* game, so missed time sits inside it at the positional level. Every
running back therefore carried identical durability risk, and a back with a long injury
history looked to the simulation exactly like one who has never missed a snap.

Two measured facts make a per-player version worth having (`docs/durability.md`):

**Availability persists.** Games missed correlates +0.407 year over year, which is stronger
than the folklore that injury proneness is hindsight. A player who missed six or more games
last season misses three or more this season 76% of the time against a 55% base rate; one
who missed none does it 41% of the time.

**The projection does not price it.** Regressing next-season points per team game on the
projection *plus* prior games missed leaves -0.186 per game, P(<0) = 100%. ESPN projects
per-game scoring for a healthy player and discounts expected absence by less than it should.

The per-position cut is the surprising part and the reason this is not applied uniformly.
Running backs come back at **-0.065 (71%, nothing)** -- the market already discounts running
back durability, because everyone knows running backs break. The inefficiency is at
quarterback and receiver, not at the position people worry about.

One season is enough. A two-year history scores R2 0.4597 against 0.4614 for last season
alone, and the second year's coefficient is -0.040 against -0.159, so almost all of the
signal is in the most recent season.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hub.config import drafted_positions
from hub.declare import chosen, fitted
from hub.draft import prior_signal
from hub.names import player_key

TEAM_GAMES = chosen(17)

# How strongly games missed persists year over year: Pearson r across 1,531 player-season
# pairs with a real prior role (`docs/durability.md`). Spearman is +0.344; the linear form is
# the one used, so the linear coefficient is the one carried.
#
# **This number was measured in August and used for one thing only: shifting a projected
# mean through `BETA`.** Issue #183 is that the simulator had no concept of absence at all --
# `season.simulate_weeks` drew every player every week and expressed a bust by shrinking his
# mean and, through `sd_eff = sd * sqrt(ratio)`, his spread with it. That makes a player who
# misses the season a low-mean *low-variance* player, which is the opposite of an injury. The
# data and its provenance already existed and were wired to the wrong place.
#
# `next_season_absence` below is what wires them to the right one. It is not a new fit: given
# r and the observed spread of `missed`, the conditional distribution of next season's missed
# games is determined, which is why this ticket adds one constant rather than a mixture
# weight nothing derives -- ADR-0006's line, and the reason the disposition took the
# games-played draw over the other two options the ticket offered.
MISSED_YOY_R = fitted(0.407)

# Points per team game lost per prior-season game missed, beyond what the projection already
# prices. From `ppg_next ~ proj_ppg + missed` on historical ESPN projections.
#
#   QB  -0.457  [-0.645, -0.257]  100.0%   applied
#   WR  -0.151  [-0.266, -0.045]   99.6%   applied
#   TE  -0.097  [-0.242, +0.062]   89.0%   not applied
#   RB  -0.065  [-0.293, +0.167]   71.2%   not applied -- the market already prices it
#
# **REFITTED 2026-09-07. The constants below are unchanged; #48 is where they move or do
# not.** `hub.draft.fit_corrections` fits against `proj_blend`, the column
# `correct_projection` actually writes to, rather than `proj_ppg`. Unlike touchdown luck, this
# signal barely moves with the baseline -- so both halves of the bracket say the same thing.
# Over 2018-2025, season-clustered:
#
#   QB  -0.486  [-0.591, -0.377]  reproduced; the gap to -0.457 is under the MDE, so this is
#               agreement rather than evidence of agreement
#   WR  -0.327  [-0.394, -0.261]  UNREPRODUCED and resolved: about twice the shipped -0.151,
#               and the only one of the five corrections that beats no-correction out of
#               sample -- 7/7 held-out seasons
#
# **#48 CLOSED 2026-09-10. Both keep their shipped values, and WR ships behind a flag.** The
# ticket's rule for an unreproduced coefficient is that it goes on applying: withdrawing a
# correction mid-season is itself an unmeasured change to the board, and the run's finding is
# that -0.151 is too *small*, so applying it is the conservative half of the disagreement
# rather than the reckless one. Neither is moved to the refit -- a refit against a stand-in
# for half of `proj_blend` is a finding and not a licence, which is what the harness's own
# docstring says of itself.
#
# What #48 does add is that the disagreement is visible on the night. `hub.draft.board`'s
# `CORRECTION_FLAG` carries the WR line and `hub.draft.report` prints it beside the ranking,
# so an operator reading a corrected board is told which of its numbers is disputed and how.
# The flag lives there rather than here because this module's constants are declared
# `fitted` beside their provenance, and a disposition string is not one of them.
#
# QB is reproduced and carries no flag. Its caveat is the one above -- agreement by width.
#
# See docs/fitted-corrections.md.
BETA: dict[str, float] = fitted({"QB": -0.457, "WR": -0.151})

# Below this a player was not a starter, and his missed games are being a backup rather than
# being hurt. Without it the signal is mostly roster churn.
MIN_PPG = chosen(5.0)

# Designations worth putting in front of a drafter. ACTIVE is not news.
FLAG_STATUS = chosen(frozenset({"OUT", "DOUBTFUL", "QUESTIONABLE", "INJURY_RESERVE"}))

# Points per team game lost by carrying a designation *now*, beyond what the projection
# prices. Fitted against week-1 injury reports, which are the closest historical analogue to
# an August designation: `total_next/17 ~ proj_ppg + designation`, 1,263 player-seasons.
#
#   Out / Doubtful  -1.631  [-2.554, -0.736]  100.0%   applied
#   Questionable    -0.949  [-2.495, +0.459]   90.2%   NOT applied
#
# Questionable is left out for two reasons, and the second is the stronger one. It does not
# clear significance at n=36. And an August QUESTIONABLE is a different population from the
# week-1 one this would be fitted on: 12.6% of the August board carries it against 2.9% at
# week 1, **4.4x more common**. Pricing an eighth of the board on a coefficient estimated
# from a much sicker group would be worse than leaving it for judgment.
#
# Out, Doubtful and IR transfer far better -- 4.1% of the August board against 2.1% at week
# 1. IR has no coefficient of its own, since nobody on IR appears on a practice report;
# starting a season there means missing at least four games by rule, so it is at least as
# severe as Out and borrowing that number understates it.
#
# **REFITTED 2026-09-07. Unchanged here; #48 decides.** `hub.draft.fit_corrections` returns
# **-0.833, 95% CI [-1.475, -0.211]** season-clustered over 2018-2025 -- about half the
# shipped size, and -1.631 is outside the interval. Read it with its two sample counts, which
# this comment and docs/durability.md previously published without reconciling: the regression
# runs on 4,595 player-seasons and **68 of them carry the designation**. The MDE is 1.051,
# larger than the point estimate itself, so the run can say the shipped value is outside the
# interval and cannot pin the level. Held out, applying it beats applying nothing in 1 season
# of 7.
#
# The refit cannot speak to INJURY_RESERVE at all, for the same reason the original could not:
# a week-1 practice report has no IR rows. One number serving three keys is the part of this
# that no amount of refitting fixes. See docs/fitted-corrections.md.
#
# **#48 CLOSED 2026-09-10: unchanged, and shipping behind a flag.** Unreproduced and
# *underpowered* -- the MDE of 1.051 is larger than the refit's own point estimate, so the
# run can say -1.631 sits outside the interval and cannot say where the level is. Moving a
# constant onto an estimate the same run declines to pin would be trading a number with
# provenance for one without. It applies, and `hub.draft.board.CORRECTION_FLAG` carries the
# disposition so `hub.draft.report` prints it beside the ranking rather than leaving an
# operator to find it here.
INJURY_BETA: dict[str, float] = fitted({
    "OUT": -1.631, "DOUBTFUL": -1.631, "INJURY_RESERVE": -1.631,
})


def is_flagworthy(status: str | None) -> bool:
    """Whether a current injury designation is worth surfacing."""
    return bool(status) and str(status).upper() in FLAG_STATUS


def games_missed(season: pl.DataFrame) -> pl.DataFrame:
    """Games a player missed last season, for players who had a real role.

    `season` needs player, pos, g and ppg.
    """
    return (season.filter(pl.col("ppg") > MIN_PPG)
                  .with_columns((pl.lit(TEAM_GAMES) - pl.col("g")).clip(0).alias("missed")))


def next_season_absence(missed) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread of *next* season's missed games, per player, from last season's.

    `missed` is one entry per player on the prior-season scale (`TEAM_GAMES`), with a
    non-finite entry -- `None`, a NaN from a null column -- for a player who had no prior
    role. Returns two arrays of the same length: the conditional mean and the conditional
    standard deviation of what he will miss this season.

    **Nothing here is a new fit, and that is the whole reason this ticket takes a
    games-played draw.** Treat (missed last season, missed this season) as one bivariate
    quantity with correlation `MISSED_YOY_R` and a common marginal. Its conditional
    distribution is then fixed by three numbers, two of which are properties of the vector
    handed in:

        E[next | prior] = mbar + r * (prior - mbar)
        SD[next | prior] = s * sqrt(1 - r^2)

    -- the textbook regression to the mean, with `mbar` and `s` estimated from `missed`
    itself rather than carried as constants. That is deliberate twice over. It keeps the
    fitted surface of this change to the single published correlation, where a mixture
    component or a zero-inflated week model would each have needed a weight nothing in this
    repo derives. And the population is right by construction: `games_missed` applies
    `MIN_PPG`, which is the same "real prior role" filter the +0.407 was measured under.

    A player with no prior role gets the marginal rather than the conditional -- mean `mbar`,
    spread `s`, no shrinkage -- because knowing nothing about him is not the same as knowing
    he is average, and the difference is exactly the variance a rookie should carry.

    Degrades to "no absence" when nothing is known: an empty vector, or one with no finite
    entry, returns zeros rather than raising. A simulator that refuses to run because an
    advisory column is missing is the operator-dependence CLAUDE.md warns about, and the
    caller has a `None` path for the case where the column is absent entirely.
    """
    m = np.asarray(missed, dtype=float)
    known = np.isfinite(m)
    if not known.any():
        return np.zeros(m.shape), np.zeros(m.shape)
    mbar = float(m[known].mean())
    # ddof=1 because `missed` is a sample of players, not the population of them. With one
    # known player there is no spread to estimate and `np.std` would return a NaN that
    # propagated into every draw, so say zero and let the mean carry it.
    s = float(m[known].std(ddof=1)) if int(known.sum()) > 1 else 0.0
    mu = np.where(known, mbar + MISSED_YOY_R * (m - mbar), mbar)
    sd = np.where(known, s * float(np.sqrt(1.0 - MISSED_YOY_R**2)), s)
    return np.clip(mu, 0.0, float(TEAM_GAMES)), sd


def prior_season(season: int, cache=None) -> pl.DataFrame:
    """Games played and scoring rate per player, last season."""
    from hub.fetch import nflverse
    cols = ("player_id", "player_display_name", "position", "season", "week",
            "season_type", "fantasy_points_ppr")
    w = nflverse.load("player_stats", seasons=[season], cols=cols, cache=cache).filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(list(drafted_positions())))
    return (w.group_by(["player_display_name", "position"])
             .agg(pl.len().alias("g"),
                  pl.col("fantasy_points_ppr").mean().alias("ppg"))
             .rename({"player_display_name": "player", "position": "pos"}))


def appearances(season: int, cache=None) -> pl.DataFrame:
    """Everyone with a regular-season stats row last season, at any position.

    Presence, not scoring: `prior_season` filters to the drafted positions because it needs
    a scoring rate, and that filter is wrong for this question -- a receiver nflverse lists
    as a cornerback, a back it lists as a fullback, played (#86).
    """
    from hub.fetch import nflverse
    # The contract's required set, and the name; presence needs nothing else.
    cols = ("player_id", "player_display_name", "position", "season", "week", "season_type",
            "fantasy_points_ppr")
    w = nflverse.load("player_stats", seasons=[season], cols=cols, cache=cache)
    return (w.filter(pl.col("season_type") == "REG")
             .select(pl.col("player_display_name").alias("player")).unique())


def sat_out(prior: pl.DataFrame, appeared: pl.DataFrame) -> pl.DataFrame:
    """The players who were in the league last season and played nothing (#86).

    In last season's preseason consensus and in no stats row of any position. A rookie is in
    neither, and the two used to be indistinguishable because the stats source gives no row
    to a player who recorded nothing. Returns one row per such player with `sat_out` true;
    matched on the player key like every other prior signal.
    """
    key = pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("_k")
    seen = appeared.with_columns(key).select("_k").unique() if appeared.height else \
        pl.DataFrame({"_k": []}, schema={"_k": pl.Utf8})
    return (prior.with_columns(key).join(seen, on="_k", how="anti")
                 .select("player", pl.lit(True).alias("sat_out")))


def attach(board: pl.DataFrame, season: pl.DataFrame,
           sat_out: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add `missed`, and `sat_out`. Players with no prior role keep a null, never a zero.

    **`missed` stays the measured column.** A player who sat the whole season out gets the
    flag and not `missed = 17`: `next_season_absence` estimates its population moments from
    `missed` itself, and the +0.407 persistence was measured on players with a real prior
    role, which a whole season on the shelf is not. So the price carries the full-season
    markdown (`correct_projection`) and the simulator draws him as unknown -- the marginal,
    like a rookie -- rather than extrapolating a linear persistence to seventeen (#86).

    The join itself is `prior_signal.join_by_player`, which `hub.draft.regression` also uses:
    it was the same twelve lines in both files -- improvements.md #15.
    """
    out = prior_signal.join_by_player(board, games_missed(season), "missed")
    if sat_out is None or sat_out.is_empty():
        return out.with_columns(pl.lit(False).alias("sat_out"))
    flagged = prior_signal.join_by_player(out, sat_out, "sat_out")
    return flagged.with_columns(pl.col("sat_out").fill_null(False))


def correct_projection(board: pl.DataFrame, column: str = "proj_blend") -> pl.DataFrame:
    """Mark a projection down for a player's own availability history.

    Applied where the market leaves a residual and nowhere else. `hub.draft.optimize` scores
    seasons against this column, so this is where a durability signal has to land to reach a
    pick -- one that is only printed does not change a decision.

    Today's designation is priced separately and only where it transfers -- Out, Doubtful
    and IR. QUESTIONABLE is carried for judgment and not priced; see `INJURY_BETA`.

    **Silent when `missed` is absent, on purpose, and that silence has a price.** It is paid
    one caller away and caught there; the comments on the two returns say by what.
    """
    if column not in board.columns:
        # The cheap absence, and the reason the expensive one is the branch below rather than
        # this line. `column` is `proj_blend`, which only the ADP stage leaves, so this fires
        # on a board that has no Corrected ADP at all -- nothing to be short a term, and
        # `hub.draft.board.BuildReport.corrections_missing` names nothing when `adp` is false.
        return board
    adjustment = pl.lit(0.0)
    # **Both column reads below survive issue #199, and the reason is that this is a
    # producer.** They look like provenance questions and are not askable as ones: this runs
    # inside `board._stage`, which is what *sets* the flag a report would answer from, so at
    # this moment no report describing this frame exists yet. The report is what records a
    # stage; it is not something the stage reads. `#199` moved the eleven *consumer* sites
    # onto the report and left the four producer sites where they are, annotated -- the same
    # split #164 drew between reading a column for provenance and reading it to work on.
    if "missed" in board.columns:
        # The expensive absence is this condition being false, and it is the whole of issue
        # #121. `board.build` absorbs the durability stage rather than refusing to build --
        # correctly; a board that will not build for one advisory column is the
        # operator-dependence CLAUDE.md warns about -- and `_attach_market` then computes
        # Corrected ADP from the terms that did apply and reports it as having run. What
        # comes out is a different ranking rather than a thinner board: 350 of 457 players
        # moved by up to 36.8 picks, 134 of the first 192 changing rank by up to 36 places.
        # Wider than the quarterbacks and receivers this term touches, because
        # `optimize.corrected_adp` re-fits the curve every player is priced against.
        #
        # Nothing is recorded from here and nothing should be: this runs inside `_stage`, and
        # the report is what records a stage rather than what the stage writes to. The catch
        # is `BuildReport.corrections_missing`, which derives the missing terms from flags the
        # build already set. `hub.draft.report` prints them beside the ranking for an operator
        # on the clock, and `hub.models.experiment.require_corrections` refuses the board
        # outright for a Gate, where a season short a term is a second arm and not a thin one.
        # A whole season sat out is priced as the full seventeen (#86) -- the flag rather
        # than the column, for the reason `attach` gives. Only ever a linear coefficient
        # extrapolated to seventeen; `BETA` was fitted on players who missed fewer.
        missed = (pl.when(pl.col("sat_out")).then(pl.lit(float(TEAM_GAMES)))
                    .otherwise(pl.col("missed")) if "sat_out" in board.columns
                  else pl.col("missed"))
        adjustment = adjustment + prior_signal.priced(missed, BETA)
    if "injury_status" in board.columns:
        # Applied at every position, unlike the durability trait: being ruled out is news,
        # not a trait the market has had years to discount.
        adjustment = adjustment + pl.col("injury_status").fill_null("").str.to_uppercase(
        ).replace_strict(INJURY_BETA, default=0.0, return_dtype=pl.Float64)
    return board.with_columns((pl.col(column) + adjustment).clip(0.0).alias(column))
