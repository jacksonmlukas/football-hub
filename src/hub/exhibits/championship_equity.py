"""Championship equity: the pick that most raises P(win the league). Removed, kept re-runnable.

**This is an exhibit.** ADR-0009 gated championship equity against following the draft
market on realised outcomes -- P0b, 2026-08-24, n=80 -- and it lost by 19.66 points a
team-game, four seasons of four, re-measured at -19.13 and then at -17.30 on the fixed
bracket. It does not pick. Nothing the product ships imports this module; `hub.draft.backtest`
does, because ADR-0007 requires the measurement that removed it to stay re-runnable, and
ADR-0009 says reopening the question means re-running that harness rather than re-arguing
from first principles. The package docstring says why it lives here and not in `hub.draft`
(#198), and `tests/contracts/test_the_exhibit_is_not_a_dependency.py` keeps it that way.

Until #198 `win_probability`, `_lift_frame` and `rank_tiers` lived in `hub.draft.optimize`
and `champion_probability` in `hub.draft.season`, each moved here verbatim. `tag_for`, which
labelled the equity column the board no longer prints, had no caller and no measurement to
re-run and was deleted with that move. What stays behind is machinery the live Gates run on:
`optimize.simulate_remaining_draft` (the room, which `hub.draft.cohort` draws two other
Gates' rosters through) and `season.simulate_weeks` / `seed_table` / `champion` (the season,
which `hub.exhibits.leverage` measures with).

The question this asks is a different one from every other ranking in the draft package,
and it disagrees with them in specific places. VOR prices talent against a positional
baseline. `cost_of_waiting` prices talent you will not get back. Neither knows that you can
only start three WRs, that a league is won in a 6-team playoff, or that a boom/bust roster
and a steady one with identical projected points do not win at the same rate.

The method is nested simulation:

  1. take the candidate,
  2. play out the rest of the draft -- opponents off a noisy blended board, you greedily
     filling your own lineup,
  3. simulate the season to a champion many times,
  4. read off how often you win.

The output is a probability, so it is directly comparable across candidates and directly
interpretable: "this pick is worth 1.4 points of championship equity".

READ THE LIFT, NOT THE LEVEL.
---------------------------
The absolute P(win) this reports is far too high -- around 30-50% for a 12-team league
where the baseline is 8.3% -- and the reason is structural, not a tuning problem.

The season is scored against `proj_blend`, and the greedy ranks candidates on
`proj_blend`. The simulation therefore hands this drafter the exact quantity that
defines its own truth, while opponents navigate by ADP, a proxy for it. Measured that
way a projection-follower cannot lose. A null control confirms the machinery is
otherwise sound: probabilities sum to one, and the eleven ADP-following seats show a
sensible snake gradient from 7.4% down to 2.9%. But the level is an artefact of assuming
ESPN's projection IS the season, and in reality ADP aggregates the whole market and is
plausibly the better estimator of the two.

The out-of-sample test that settles it is the one ADR-0009 ran: replay past seasons and
score both arms against what actually happened. The level never became credible; the
comparison did, and the comparison is what lost.

What survives is the COMPARISON. Every candidate is evaluated against identical simulated
futures (common random numbers), so the paired lift between two candidates is a real
within-model quantity with a usable standard error, and the leak above applies equally to
both sides of it. Rank on `lift`, check it clears two standard errors, and ignore `p_win`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from hub.draft.availability import DEFAULT_ESPN_WEIGHT
from hub.draft.optimize import (
    DEFAULT_ROUNDS,
    ROLLOUT,
    SEASON_SIM,
    prepare_room,
    root_seed,
    simulate_remaining_draft,
    stream,
)
from hub.draft.season import (
    PLAYOFF_ROUNDS,
    REG_SEASON_WEEKS,
    champion,
    seed_table,
    simulate_weeks,
    talent_cv_for,
)
from hub.draft.state import DraftState
from hub.models.predict import WEEKLY_SKEW_POOLED, CorrelationReport, moments

if TYPE_CHECKING:                  # `board` reaches this module function-locally, both ways
    from hub.draft.board import BuildReport


def champion_probability(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                         pos: np.ndarray, n_sims: int = 400,
                         rng: np.random.Generator | None = None,
                         talent_cv: float | np.ndarray | None = None,
                         nfl_team: np.ndarray | None = None,
                         skew: np.ndarray | None = None,
                         missed: np.ndarray | None = None,
                         report: CorrelationReport | None = None,
                         bye_week: np.ndarray | None = None) -> np.ndarray:
    """P(each team wins the league). Returns (teams,) summing to 1.

    14-week H2H regular season, top 6 seeds, two byes, then single elimination on one-week
    matchups -- read off the live league, not assumed. Three further weeks are simulated so
    the bracket has draws of its own.

    The two byes here are playoff seeding; `bye_week` is the NFL bye #226 is about, one
    per player, handed straight to `simulate_weeks` like `missed`, which is the absence
    model -- see `_absence_factor`. `leverage.py` uses the word in the seeding sense.
    """
    rng = rng or np.random.default_rng(0)
    teams = len(rosters)
    pts = simulate_weeks(rosters, mu, sd, pos, n_sims,
                         REG_SEASON_WEEKS + PLAYOFF_ROUNDS, rng, talent_cv, nfl_team, skew,
                         missed, report, bye_week=bye_week)
    _, seeds = seed_table(pts)
    champs = np.array([champion(pts, seeds, s) for s in range(n_sims)])
    return np.bincount(champs, minlength=teams) / n_sims




def win_probability(board: pl.DataFrame, state: DraftState, candidates: list[str], *,
                    my_slot: int, teams: int = 12, rounds: int = DEFAULT_ROUNDS,
                    n_draft_sims: int = 24, n_season_sims: int = 300,
                    w: float = DEFAULT_ESPN_WEIGHT,
                    seed: int | np.random.SeedSequence = 0,
                    report: BuildReport | None = None,
                    correlation: CorrelationReport | None = None,
                    opp_noise: float = 1.0) -> pl.DataFrame:
    """P(you win the league) for each candidate, averaged over simulated drafts.

    `opp_noise` is the room's noise scale, handed down to every rollout unchanged (#49). The
    rollouts are the room *as this objective imagines it*, and `backtest.compare` plays the
    same objective inside a room at the same scale -- so a sweep over the scale moves the
    knob and not the gap between what arm B believes about the room and what the room is.
    1.0 is the fitted law itself, the default `simulate_remaining_draft` carries.

    Scored over the *whole board*, and rosters include the players each seat already holds.
    They used to include only picks made during the simulation, which made the objective
    blind to your own roster: holding a quarterback, it ranked a second one above a
    startable back.

    **Two reports, about two different runs, which is why they are two parameters** -- the
    naming `backtest.diagnose` already uses, and which this function was the last holdout
    from. `report` is the `BuildReport` describing the board handed in: which stages built
    this frame, and therefore which currency the room below ranks in (issue #199). It is
    resolved once here and handed down, rather than re-derived inside every one of the
    candidates x draft-sims rollouts -- and since #259 so is everything else a rollout
    reads off the Board and does not change, as one `optimize.Room` prepared here and
    handed to each rollout. Byte-identical output: the #197 pin did not move.

    `correlation` is a `CorrelationReport` the caller owns, and every one of those
    simulations writes into it. Passing one is how a run learns that some team's players were
    simulated independently -- a correlation block that will not factor falls back to the
    model the structure exists to replace, and the count is the only evidence of it, since
    the draw it produces has exactly the shape a correlated one has.

    **`seed` may be a `SeedSequence`, and a caller that is itself a level of a larger
    experiment should pass one.** An integer is a root of its own, which is right for the
    live board -- there is one draft and it is this one. `backtest.compare` is the other
    case: it plays a room and then evaluates candidates *inside* that room, so its room and
    this function's rollouts have to be two coordinates of one tree rather than two integers
    that happen to differ. Handing down the root is what makes them so; handing down an
    integer is how rollout 0 came to be the room itself (issue #195).
    """
    # The room's board-invariant state, once (#259). A call makes candidates x draft-sims
    # rollouts, and each used to rebuild the blended ADP, the sigma over it, the currency,
    # and a `player_key` index of every row -- none of which a rollout changes. The report
    # is resolved inside `prepare_room`, through the same `report_for` seam as before, and
    # the frame the season is scored on below is the room's pool: the same `blended_adp`
    # call this function used to make itself.
    room = prepare_room(board, w, report=report)
    report = room.report
    pool = room.pool
    pred = moments(pool)
    mu = pred["mu"].fill_null(0.0).to_numpy()
    sd = pred["sd"].fill_null(2.0).to_numpy()
    # Read the skew off the same frame that produced mu and sd. The simulator used to
    # recompute it from `pos`, which agreed only because both routes read one table --
    # a per-player skew (from components, say) would have been computed here and silently
    # dropped on the way in.
    skew = pred["skew"].fill_null(WEEKLY_SKEW_POOLED).to_numpy()
    pos = pool["pos"].fill_null("NA").to_numpy()
    # NFL team, so the simulator can correlate a quarterback with his own pass catchers.
    # Without it a stacked roster is drawn independent and comes out less volatile than it
    # is -- see docs/correlation.md, where independence gives a nominal 80% interval that
    # covers 72.9%.
    nfl_team = (pool["team"].to_numpy() if "team" in pool.columns else None)
    # Prior-season missed games, so the simulator can model absence as absence rather than as
    # a shrunken mean -- issue #183 and `season._absence_factor`. Cast to float first: the
    # column is integral and nullable, and `to_numpy()` on a null-carrying integer Series does
    # not give a NaN to test for.
    #
    # **Whether the durability stage ran is a provenance question, so it goes to the report
    # and not to the frame** -- #199, and this is a consumer, unlike the two producer sites in
    # `durability.correct_projection`. `missed` is written inside `board._stage`, which
    # `build` absorbs on failure, so a board reaching here may legitimately not carry it; a
    # board read back off disk gets the same answer through `BuildReport.of_served`, which
    # derives the flag once rather than letting every consumer sniff for it. `None` then means
    # "not modelled", which is what `simulate_weeks` reads it as.
    #
    # The flag alone, with no membership test beside it. Adding one would be the
    # belt-and-braces version of what #146 removed from `report.injuries`: a second answer to
    # a question that has an owner, silently preferring the frame whenever the two disagree,
    # which is the failure rather than the mitigation.
    #
    # Nothing is reported from here. `BuildReport.corrections_missing` already names the gap
    # for an operator on the clock, and `experiment.require_corrections` refuses such a board
    # outright for a Gate, where a season short a term is a second arm and not a thin one.
    missed = pool["missed"].cast(pl.Float64).to_numpy() if report.durability else None
    # The same provenance rule for byes (#226): the stage decides. A null is a player the
    # schedule could not place, and 0 is what the simulator reads as "no week".
    bye_week = (pool["bye_week"].fill_null(0).cast(pl.Int64).to_numpy()
                if report.bye else None)
    # An invented projection is less certain than a measured one (#87). Not a stage and not
    # provenance: `_impute_xfp` is part of every build, and the flag is the column itself.
    talent_cv = (talent_cv_for(pos, pool["xfp_imputed"].fill_null(False).to_numpy())
                 if "xfp_imputed" in pool.columns else None)

    # Common random numbers. Every candidate is evaluated against the SAME simulated
    # futures -- same draft rollouts, same talent draws, same weekly scores -- so the
    # only thing that differs between two columns is the player at this pick. The levels
    # still carry the full noise of the simulation, but the *difference* between two
    # candidates is paired, and the paired difference is what the decision needs.
    #
    # `seed` names the experiment, and the two levels below hang off it as coordinates rather
    # than as offsets -- see the seeding tree above. Future k is still the same future for
    # every candidate, which is the whole point of the paired design; what it is no longer is
    # the room this evaluation is about to be scored in, or the future the next draft in the
    # sweep evaluates against (issue #195).
    root = root_seed(seed)
    mat = np.empty((len(candidates), n_draft_sims))
    for i, c in enumerate(candidates):
        for k in range(n_draft_sims):
            rosters = simulate_remaining_draft(board, state, my_slot=my_slot, teams=teams,
                                               rounds=rounds, forced=c, w=w,
                                               rng=stream(root, ROLLOUT, k),
                                               report=report, opp_noise=opp_noise,
                                               room=room)
            p = champion_probability(rosters, mu, sd, pos, n_sims=n_season_sims,
                                     rng=stream(root, SEASON_SIM, k),
                                     nfl_team=nfl_team, skew=skew, missed=missed,
                                     report=correlation, bye_week=bye_week,
                                     talent_cv=talent_cv)
            mat[i, k] = p[my_slot - 1]

    return _lift_frame(candidates, mat)


def _lift_frame(candidates: list[str], mat: np.ndarray) -> pl.DataFrame:
    """The candidate table, with both error bars taken across the shared futures.

    `mat` is one row per candidate and one column per simulated future, and column k is the
    SAME future for every candidate -- common random numbers. That is what lets every
    subtraction below be paired.

    Two error bars come out of it and they are not interchangeable:

    * `lift_se` -- the error on one candidate's own distance from the field.
    * `lead_gap_se` -- the error on the *gap* between the leader and this candidate, which is
      the spread of their per-future difference. `rank_tiers` reads this one and nothing else.

    They are separate columns because the second cannot be recovered from the first. Adding
    two `lift_se` values in quadrature is the formula for a difference between two unrelated
    estimates, and two lifts read off the same futures are not unrelated: where they rise and
    fall together it is too wide, and where they trade off against each other it is too
    narrow. Only the per-future difference knows which, so it is measured here rather than
    reconstructed later.
    """
    n = mat.shape[1]
    field = mat.mean(axis=0)                       # the field, per simulated future
    diff = mat - field[None, :]                    # paired lift, same futures
    lift = diff.mean(axis=1)
    # The leader by lift, which is the row `rank_tiers` measures every gap against -- it
    # sorts on the same column, so the two agree on who the leader is.
    lead = int(np.argmax(lift))
    if n > 1:
        se = diff.std(axis=1, ddof=1) / np.sqrt(n)
        # The field term cancels out of the subtraction, so this is the spread of
        # `mat[lead] - mat[i]`: one number per future, differenced before it is averaged.
        gap_se = (diff[lead][None, :] - diff).std(axis=1, ddof=1) / np.sqrt(n)
    else:
        se = np.zeros(len(candidates))
        gap_se = np.zeros(len(candidates))

    return pl.DataFrame({
        "player": candidates,
        "p_win": mat.mean(axis=1),
        "lift": lift,
        "lift_se": se,
        "lead_gap_se": gap_se,
    }).sort("lift", descending=True)




def rank_tiers(wp: pl.DataFrame) -> pl.DataFrame:
    """Mark which candidates the simulation cannot separate from the best one.

    The objective is the player with the highest chance of winning the league, and sometimes
    that is two players. Printing a strict order where the simulation cannot support one
    asserts a distinction it cannot make, and re-running names a different leader.

    Gaps are in standard errors of the *difference*, taken across the same simulated seasons
    that produced both lifts: `lead_gap_se`, which `win_probability` measures from the
    per-future difference itself. Anything inside two of them is reported as tied for the
    lead.

    This used to add the two `lift_se` values in quadrature, which is the standard error of a
    difference between two *unrelated* estimates. These two are drawn from one set of
    simulated seasons and move together, so the quadrature figure was wrong in both
    directions and by no fixed factor: too wide where two lifts rise and fall together --
    which is the case a tie tier exists for, and there it held players in a tier the
    simulation can in fact separate -- and too narrow where they trade off against each other.

    **The illustration is a fixture, on purpose.** `CORRELATED_FUTURES` in
    `tests/unit/test_optimize.py` is re-measured on every run of the suite, where a board is
    not: A and B's lifts rise and fall together (+0.80 across ten futures) and their gap is
    0.008 of championship equity. Paired, the error on that gap is 0.0034421 and the gap is
    2.32 of them; in quadrature it was 0.0064602 and 1.24, so B leaves the tier. On
    `UNCORRELATED_FUTURES` (+0.00004) the two agree to five figures -- 0.0099086 against
    0.0099087, 1.008 either way -- and B stays in it, which is what makes this a correction
    rather than a rescale.

    This docstring used to illustrate the tie with a pick-3 board instead: *"the top two
    differ by 0.17 points of championship equity with standard errors near 0.5"*. The 0.5 is a
    `lift_se` and `_lift_frame` still computes it the same way, but the *tie* it illustrated
    was decided by the quadrature expression above, and the board it was read off cannot be
    rebuilt -- so the illustration is superseded rather than wrong at the time. Issue #189,
    and `docs/decisions.md` carries the restatement.
    """
    if "lead_gap_se" not in wp.columns:
        raise ValueError(
            "rank_tiers needs `lead_gap_se`, the standard error of the paired difference, "
            "which `win_probability` measures across the futures both lifts came from. "
            "Combining two `lift_se` values in quadrature is not the same number.")
    top = wp.sort("lift", descending=True)
    lead_lift = float(top["lift"][0])
    paired = pl.col("lead_gap_se").fill_null(0.0)
    gap = (pl.lit(lead_lift) - pl.col("lift"))
    return top.with_columns(
        pl.when(paired > 0).then(gap / paired).otherwise(
            pl.when(gap.abs() > 0).then(pl.lit(float("inf"))).otherwise(pl.lit(0.0))
        ).alias("gap_se")
    ).with_columns((pl.col("gap_se") < 2.0).alias("co_leader"))


