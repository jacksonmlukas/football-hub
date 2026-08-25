"""Weekly lineup optimisation against the matchup you actually have.

`docs/foundation-plan.md` 5.2 asks for "the ceiling preference from
`docs/championship-leverage.md`". That doc's ceiling argument rests on a premise that is
false for this league, and the correction is the whole design of this module.

The doc says *"12 teams, 8 make playoffs, 3 weeks (15-17), no byes"*, concludes
*"dP(champ)/d(regular-season win) is close to zero"*, and lands on *"never sacrifice
ceiling for a marginal regular-season win."* The live league, from raw
`mSettings.scheduleSettings` and recorded in `docs/decisions.md`, is `playoffTeamCount: 6`
with two byes. Half the league misses and the top two seeds skip a single-elimination
round, so regular-season wins are worth a great deal and the blanket rule does not follow.

The fix is not to invert it into a blanket floor preference. The doc's own point 3 is
right: variance preference is state-dependent and sign-flips. So the objective here is
P(win this matchup), and the ceiling-or-floor question answers itself from whether you are
favoured. A big underdog needs variance because the mean is not enough; a big favourite
wants none because he is already past the line.

Head-to-head is what makes this legitimate rather than cute: surplus points above your
opponent's score are worth nothing, so expected points is the wrong objective even before
any playoff structure is considered.
"""
import numpy as np
import polars as pl
import pytest

from hub.season import lineup


def _players(rows):
    """rows: (player, pos, mu, sd)"""
    return pl.DataFrame({"player": [r[0] for r in rows], "pos": [r[1] for r in rows],
                         "mu": [float(r[2]) for r in rows],
                         "sd": [float(r[3]) for r in rows]})


# A minimal legal roster: one of each slot filled, nothing to decide.
BASE = [("qb", "QB", 18.0, 6.0), ("rb1", "RB", 12.0, 5.0), ("rb2", "RB", 11.0, 5.0),
        ("wr1", "WR", 13.0, 5.0), ("wr2", "WR", 10.0, 5.0), ("wr3", "WR", 9.0, 5.0),
        ("te", "TE", 8.0, 4.0), ("flex", "RB", 7.0, 3.0)]

SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}


# --- legality -------------------------------------------------------------

def test_it_fills_every_slot_exactly_once():
    got = lineup.optimize(_players(BASE), opp_mu=90.0, opp_sd=25.0)
    assert got["starters"].height == sum(SLOTS.values()) + 1  # + flex


def test_no_player_starts_twice():
    got = lineup.optimize(_players(BASE), opp_mu=90.0, opp_sd=25.0)
    names = got["starters"]["player"].to_list()
    assert len(set(names)) == len(names)


def test_positions_are_respected():
    got = lineup.optimize(_players(BASE), opp_mu=90.0, opp_sd=25.0)
    counts = dict(got["starters"].group_by("pos").len().iter_rows())
    assert counts["QB"] == 1 and counts["TE"] == 1
    assert counts["RB"] >= 2 and counts["WR"] >= 3


def test_a_quarterback_cannot_fill_the_flex():
    """FLEX is RB/WR/TE. A second QB on the roster must stay on the bench however good."""
    roster = [*BASE, ("qb2", "QB", 40.0, 2.0)]
    got = lineup.optimize(_players(roster), opp_mu=90.0, opp_sd=25.0)
    assert got["starters"].filter(pl.col("pos") == "QB").height == 1


def test_a_roster_that_cannot_field_a_legal_lineup_says_so():
    """Byes and injuries do this. Silently starting seven is how you lose a week."""
    with pytest.raises(lineup.NoLegalLineup):
        lineup.optimize(_players(BASE[:3]), opp_mu=90.0, opp_sd=25.0)


# --- the objective, and the sign flip -------------------------------------

def _matchup_roster():
    """Identical rosters but for one slot, contested by two very different players.

    `safe` has the higher mean and almost no spread. `boom` is worse on average and wildly
    volatile. Which one starts is the entire question this module exists to answer.
    """
    return [*BASE[:-1], ("safe", "RB", 10.0, 1.0), ("boom", "RB", 8.0, 14.0)]


def test_a_heavy_underdog_starts_the_boom_player():
    """Down 30 points on projection, the safe floor loses with near-certainty. The only
    lineups that win are the ones that can get lucky, so variance is worth paying mean for."""
    got = lineup.optimize(_players(_matchup_roster()), opp_mu=125.0, opp_sd=5.0)
    assert "boom" in got["starters"]["player"].to_list()


def _favourite_roster():
    """The tradeoff pointed the other way: `risky` has the *higher* projection as well as
    the higher spread. A max-points optimizer starts him unconditionally, so this is the
    roster that tells the two objectives apart when you are ahead.
    """
    return [*BASE[:-1], ("steady", "RB", 10.0, 1.0), ("risky", "RB", 13.0, 16.0)]


def test_a_heavy_favourite_declines_projected_points_for_a_floor():
    """Up 35, the mean is already past the line and every extra point of spread is a
    chance to give the week away -- so three projected points are worth giving up. This is
    the case the blanket ceiling rule gets wrong, and note it cannot be passed by accident
    by an optimizer that just maximises points."""
    got = lineup.optimize(_players(_favourite_roster()), opp_mu=55.0, opp_sd=5.0)
    assert "steady" in got["starters"]["player"].to_list()
    assert got["mu"] < lineup.best_by_points(_players(_favourite_roster()))["mu"]


def test_the_same_favourite_roster_takes_the_risk_when_behind():
    """Same roster, same two candidates, opponent moved. Now the extra projection *and*
    the extra spread both help, so the choice reverses."""
    got = lineup.optimize(_players(_favourite_roster()), opp_mu=130.0, opp_sd=5.0)
    assert "risky" in got["starters"]["player"].to_list()


def test_the_same_player_is_a_start_or_a_sit_depending_only_on_the_opponent():
    """The headline. Nothing about the roster changes between these two calls -- only who
    it is playing. A blanket "start the boom-bust player" rule cannot express this, and
    `docs/championship-leverage.md` asserts exactly that rule from a false premise."""
    roster = _players(_matchup_roster())
    underdog = lineup.optimize(roster, opp_mu=125.0, opp_sd=5.0)["starters"]["player"].to_list()
    favourite = lineup.optimize(roster, opp_mu=60.0, opp_sd=5.0)["starters"]["player"].to_list()
    assert ("boom" in underdog) and ("boom" not in favourite)


def test_an_even_matchup_falls_back_to_points():
    """With the line right at your projection, the extra variance is symmetric -- it wins
    and loses the week equally often -- so the mean is all that is left to prefer."""
    roster = _players(_matchup_roster())
    even = lineup.optimize(roster, opp_mu=lineup.best_by_points(roster)["mu"], opp_sd=25.0)
    assert "safe" in even["starters"]["player"].to_list()


def test_maximising_points_is_not_the_same_answer():
    """If it always were, this module would be pointless and should be deleted."""
    roster = _players(_matchup_roster())
    assert (lineup.best_by_points(roster)["starters"]["player"].to_list()
            != lineup.optimize(roster, opp_mu=125.0, opp_sd=5.0)["starters"]["player"].to_list())


def test_it_never_scores_worse_than_the_points_lineup_on_win_probability():
    """The property that makes it safe to follow: whatever it does to the projected total,
    it cannot be beaten on the thing being optimised."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        roster = _players([(f"p{i}", p, float(rng.uniform(4, 20)), float(rng.uniform(1, 12)))
                           for i, p in enumerate(["QB", "QB", "RB", "RB", "RB", "RB",
                                                  "WR", "WR", "WR", "WR", "WR", "TE", "TE"])])
        opp_mu = float(rng.uniform(70, 140))
        got = lineup.optimize(roster, opp_mu=opp_mu, opp_sd=25.0)
        pts = lineup.best_by_points(roster)
        assert got["win_prob"] + 1e-9 >= lineup.win_probability(
            pts["mu"], pts["sd"], opp_mu, 25.0)


# --- the probability itself -----------------------------------------------

def test_win_probability_matches_simulation():
    """Closed form against brute force. The normal approximation is an assumption, but the
    arithmetic on top of it should not also be wrong."""
    rng = np.random.default_rng(1)
    me = rng.normal(110.0, 22.0, 400_000)
    opp = rng.normal(100.0, 25.0, 400_000)
    assert lineup.win_probability(110.0, 22.0, 100.0, 25.0) == pytest.approx(
        float((me > opp).mean()), abs=0.003)


def test_an_exact_tie_on_the_mean_is_a_coin_flip():
    assert lineup.win_probability(100.0, 20.0, 100.0, 20.0) == pytest.approx(0.5)


def test_variance_is_irrelevant_when_the_means_are_equal():
    """Symmetric, so more spread cannot help. Guards against a sign error that would make
    the optimizer chase variance in every matchup."""
    assert (lineup.win_probability(100.0, 40.0, 100.0, 5.0)
            == pytest.approx(lineup.win_probability(100.0, 5.0, 100.0, 40.0)))


# --- the opponent ---------------------------------------------------------

def test_the_opponent_is_modelled_as_starting_their_best_projection():
    """`docs/championship-leverage.md`: start from "they start their highest ESPN
    projection". Assuming they optimise against *you* would be modelling an opponent
    nobody in this league is."""
    opp = lineup.opponent_moments(_players([*BASE, ("bench", "WR", 1.0, 1.0)]))
    assert opp["mu"] == pytest.approx(lineup.best_by_points(_players(BASE))["mu"])


def test_a_bigger_roster_never_lowers_the_opponents_projection():
    small = lineup.opponent_moments(_players(BASE))
    big = lineup.opponent_moments(_players([*BASE, ("star", "WR", 25.0, 6.0)]))
    assert big["mu"] >= small["mu"]


# --- refusing rather than guessing ----------------------------------------

def test_an_absurd_roster_is_refused_rather_than_silently_sampled():
    """Exhaustive enumeration is exact and fast for a real roster. If it ever stops being
    either, that must be an error and not a quietly approximate answer."""
    roster = _players([(f"p{i}", "WR", 10.0, 5.0) for i in range(60)]
                      + [("qb", "QB", 18.0, 6.0), ("rb1", "RB", 12.0, 5.0),
                         ("rb2", "RB", 11.0, 5.0), ("te", "TE", 8.0, 4.0)])
    with pytest.raises(lineup.TooManyLineups):
        lineup.optimize(roster, opp_mu=90.0, opp_sd=25.0, max_lineups=1000)


def test_a_player_on_bye_does_not_start_over_a_playable_one():
    roster = [*BASE[:-1], ("bye", "RB", 0.0, 0.0), ("plays", "RB", 6.0, 3.0)]
    got = lineup.optimize(_players(roster), opp_mu=90.0, opp_sd=25.0)
    assert "bye" not in got["starters"]["player"].to_list()


def test_a_matchup_with_no_uncertainty_at_all_is_decided_by_the_means():
    """Degenerate but reachable: everyone on a bye except one player, opp_sd of zero.
    Dividing by a zero spread would be a crash on a Sunday morning."""
    assert lineup.win_probability(100.0, 0.0, 90.0, 0.0) == 1.0
    assert lineup.win_probability(90.0, 0.0, 100.0, 0.0) == 0.0
    assert lineup.win_probability(90.0, 0.0, 90.0, 0.0) == 0.5


# --- the CLI --------------------------------------------------------------

def _roster_file(tmp_path, rows):
    p = tmp_path / "roster.parquet"
    _players(rows).write_parquet(p)
    return str(p)


def test_the_cli_names_the_swap_and_prices_it(tmp_path, capsys):
    """The output has to justify itself. Being told to bench a player projected for more
    points, with no number attached, is advice nobody should follow."""
    path = _roster_file(tmp_path, _favourite_roster())
    assert lineup.main(["--roster", path, "--opp-mu", "55", "--opp-sd", "5"]) == 0
    out = capsys.readouterr().out
    assert "risky" in out and "win" in out


def test_the_cli_says_when_the_matchup_changes_nothing(tmp_path, capsys):
    """Most weeks it will not, and silence would leave you wondering whether it ran."""
    path = _roster_file(tmp_path, BASE)
    assert lineup.main(["--roster", path, "--opp-mu", "95"]) == 0
    assert "does not change it" in capsys.readouterr().out


def test_the_cli_fails_cleanly_when_the_roster_cannot_field_a_lineup(tmp_path, capsys):
    path = _roster_file(tmp_path, BASE[:3])
    assert lineup.main(["--roster", path, "--opp-mu", "95"]) == 1
    assert "cannot fill" in capsys.readouterr().err


# --- teammates are not independent ----------------------------------------

def _stack():
    """A lineup where the quarterback and one receiver play for the same NFL team."""
    rows = [("qb", "QB", 18.0, 8.0, "KC"), ("rb1", "RB", 12.0, 6.0, "SF"),
            ("rb2", "RB", 11.0, 6.0, "DAL"), ("wr1", "WR", 13.0, 7.0, "KC"),
            ("wr2", "WR", 10.0, 6.0, "BUF"), ("wr3", "WR", 9.0, 6.0, "MIA"),
            ("te", "TE", 8.0, 5.0, "NYJ"), ("flex", "RB", 7.0, 4.0, "LA")]
    return pl.DataFrame({"player": [r[0] for r in rows], "pos": [r[1] for r in rows],
                         "mu": [r[2] for r in rows], "sd": [r[3] for r in rows],
                         "nfl_team": [r[4] for r in rows]})


def test_a_stacked_lineup_is_more_volatile_than_independence_implies():
    """Measured at +0.232 between a quarterback and a receiving teammate, and the L1 gate
    in docs/correlation.md shows what ignoring it costs: an 80% interval that covers 72.9%
    of the time. Treating a stack as independent is straightforwardly overconfident."""
    stacked = lineup.optimize(_stack(), opp_mu=110.0, opp_sd=25.0)
    apart = _stack().with_columns(
        pl.when(pl.col("player") == "wr1").then(pl.lit("DEN"))
          .otherwise(pl.col("nfl_team")).alias("nfl_team"))
    assert stacked["sd"] > lineup.optimize(apart, opp_mu=110.0, opp_sd=25.0)["sd"]


def test_two_receivers_on_one_team_are_treated_as_independent():
    """Measured at +0.014. Only the quarterback edges carry real correlation, and the gate
    shows a no-quarterback group is already well calibrated without it."""
    # Move the quarterback off KC first, or putting wr2 there creates a QB pairing and the
    # comparison stops being about two receivers at all. The first version of this test did
    # exactly that and failed for the right reason.
    no_qb_stack = _stack().with_columns(
        pl.when(pl.col("player") == "qb").then(pl.lit("DEN"))
          .otherwise(pl.col("nfl_team")).alias("nfl_team"))
    together = no_qb_stack.with_columns(
        pl.when(pl.col("player") == "wr2").then(pl.lit("KC"))
          .otherwise(pl.col("nfl_team")).alias("nfl_team"))
    assert (lineup.optimize(together, opp_mu=110.0, opp_sd=25.0)["sd"]
            == pytest.approx(lineup.optimize(no_qb_stack, opp_mu=110.0,
                                             opp_sd=25.0)["sd"], rel=1e-9))


def test_a_roster_without_team_information_still_works():
    """Back-compat, and the common case before the board carries NFL teams."""
    got = lineup.optimize(_players(BASE), opp_mu=95.0, opp_sd=25.0)
    assert got["sd"] > 0


def test_the_underdog_prefers_the_stack_and_the_favourite_does_not():
    """The payoff, and it is the same state-dependence as everywhere else: correlation is
    volatility, so it helps when you need variance and hurts when you do not."""
    stack = _stack()
    apart = stack.with_columns(
        pl.when(pl.col("player") == "wr1").then(pl.lit("DEN"))
          .otherwise(pl.col("nfl_team")).alias("nfl_team"))
    behind = (lineup.optimize(stack, opp_mu=145.0, opp_sd=20.0)["win_prob"]
              > lineup.optimize(apart, opp_mu=145.0, opp_sd=20.0)["win_prob"])
    ahead = (lineup.optimize(stack, opp_mu=60.0, opp_sd=20.0)["win_prob"]
             < lineup.optimize(apart, opp_mu=60.0, opp_sd=20.0)["win_prob"])
    assert behind and ahead


# --- the CLI's missing input ----------------------------------------------

def test_an_absent_roster_is_a_sentence_not_a_traceback(tmp_path, capsys):
    """Nothing in this repo writes a roster, so the default path is absent until after the
    draft. It used to surface as a bare polars FileNotFoundError."""
    from hub.season import lineup as L
    assert L.main(["--opp-mu", "110", "--roster", str(tmp_path / "nope.parquet")]) == 1
    err = capsys.readouterr().err
    assert "no roster at" in err
    assert "player, pos, mu, sd" in err, "it has to say what the file should contain"


def test_a_roster_missing_columns_names_them(tmp_path, capsys):
    import polars as pl

    from hub.season import lineup as L
    p = tmp_path / "r.parquet"
    pl.DataFrame({"player": ["a"], "pos": ["RB"]}).write_parquet(p)
    assert L.main(["--opp-mu", "110", "--roster", str(p)]) == 1
    assert "'mu'" in capsys.readouterr().err
