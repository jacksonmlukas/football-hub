"""The Panel: one row per player-week, every feature measured before its outcome.

Split out of `test_weekly_screen.py` alongside the module itself. These tests are about
*assembly* -- what a source contributes, what survives a join, what a window belongs to. The
statistic that gets measured on the result is tested next door.
"""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from hub.models import panel as pnl


def _windows():
    """Two NFL weeks: Thursday to Monday, a week apart."""
    return pl.DataFrame({
        "season": [2024, 2024],
        "week": [5, 6],
        "first_kick": [dt.date(2024, 10, 3), dt.date(2024, 10, 10)],
        "last_kick": [dt.date(2024, 10, 7), dt.date(2024, 10, 14)]})


def _scrape(day):
    return pl.DataFrame({"scrape_date": [dt.date(*day)], "key": ["x"], "ecr": [1.0]})


def test_recent_mean_is_strictly_prior():
    """Week w's control may not contain week w. The whole screen rests on it."""
    p = pl.DataFrame({"player_id": ["a"] * 5, "season": [2024] * 5,
                      "week": [1, 2, 3, 4, 5],
                      "targets": [10.0, 0.0, 0.0, 0.0, 99.0]})
    out = pnl.recent_mean(p, "targets").sort("week")
    assert out["targets_recent"].to_list()[4] == pytest.approx(0.0), \
        "weeks 2-4 are zeros; week 5's own 99 must not reach its own control"
    assert out["targets_recent"].to_list()[0] is None, "week 1 has no history"


def test_recent_mean_counts_calendar_weeks_not_appearances():
    """A player who missed week 3 has a two-week window, not one padded from week 1."""
    p = pl.DataFrame({"player_id": ["a"] * 3, "season": [2024] * 3,
                      "week": [1, 2, 4], "targets": [10.0, 10.0, 4.0]})
    out = pnl.recent_mean(p, "targets").sort("week")
    assert out["targets_recent"].to_list()[2] == pytest.approx(10.0), \
        "week 4 averages weeks 1-3, of which only 1 and 2 exist"


def test_the_panel_carries_the_designation_but_cannot_fit_retention_on_it():
    """A pin on a structural fact, so nobody quietly fits the injury term on this panel.

    `player_stats` has no row for a player who did not play, so of 5,473 "Out" designations
    across 2021-25 exactly six reach the panel. `hub.models.injury` scores an injury row with
    no stat row as zero -- the player who did not play is its whole subject -- so retention
    fitted here would measure "what a Questionable player who played anyway retains" and
    report it under the stronger result's name. The term belongs in Gate B, which builds a
    complete grid where a missing row is a zero.
    """
    import inspect
    src = inspect.getsource(pnl.build_panel)
    assert "CANNOT be fitted here" in src, "the constraint must stay stated where it is read"
    assert "status" in src and "practice" in src, "and the columns are carried for Gate B"


def test_a_midweek_scrape_belongs_to_the_week_it_is_inside():
    """The bug that shifted the whole control by a week. An NFL week runs Thursday to Monday
    and FantasyPros scrapes land mid-week: 2024-10-04 is a Friday, *inside* week 5 (Oct 3-7),
    and it ranks week 5's Sunday games. Joining on the next *first* kickoff sent it to week 6.

    The tell was that Saquon Barkley, CeeDee Lamb and Patrick Mahomes were each missing from
    exactly one week -- and it was the week after their team's bye, because a page that
    correctly omits a bye-week player was being attached to the following week.
    """
    got = pnl.assign_weeks(_scrape((2024, 10, 4)), _windows())
    assert got["week"].to_list() == [5], "a Friday inside week 5 is week 5's ranking"


def test_a_scrape_before_a_week_opens_still_belongs_to_it():
    assert pnl.assign_weeks(_scrape((2024, 10, 1)), _windows())["week"].to_list() == [5]


def test_a_scrape_after_a_week_closes_belongs_to_the_next():
    assert pnl.assign_weeks(_scrape((2024, 10, 8)), _windows())["week"].to_list() == [6]


def test_a_scrape_too_far_from_any_week_is_dropped():
    """Pre-season and post-season scrapes rank no week that exists here."""
    assert pnl.assign_weeks(_scrape((2024, 9, 1)), _windows()).is_empty()


def test_lead_days_is_measured_to_the_week_being_ranked():
    got = pnl.assign_weeks(_scrape((2024, 10, 4)), _windows())
    assert got["lead_days"].to_list() == [3], "Friday to the following Monday"


def test_route_share_is_plays_on_over_team_pass_plays():
    plays = pl.DataFrame({
        "week": [1, 1, 1, 1],
        "possession_team": ["PHI"] * 4,
        "offense_players": ["a;b;c", "a;b", "a;c", "a"]})
    got = pnl.route_share_from_plays(plays, 2024).sort("player_id")
    share = dict(zip(got["player_id"], got["route_pct"], strict=True))
    assert share["a"] == pytest.approx(1.0), "on every pass play"
    assert share["b"] == pytest.approx(0.5)
    assert share["c"] == pytest.approx(0.5)


def test_the_denominator_is_per_team_per_week():
    """A player's share is of *his own* team's pass plays, not the league's."""
    plays = pl.DataFrame({
        "week": [1, 1, 1],
        "possession_team": ["PHI", "DAL", "DAL"],
        "offense_players": ["a", "b", "b"]})
    got = pnl.route_share_from_plays(plays, 2024)
    assert set(got["route_pct"].to_list()) == {1.0}, "each is on all of his own team's plays"


def test_an_empty_slate_yields_an_empty_frame_with_the_right_shape():
    got = pnl.route_share_from_plays(pl.DataFrame(), 2024)
    assert got.is_empty()
    assert set(got.columns) == {"season", "week", "player_id", "route_pct"}


def test_blank_ids_in_the_player_list_are_dropped():
    plays = pl.DataFrame({"week": [1], "possession_team": ["PHI"],
                          "offense_players": ["a;;b;"]})
    assert sorted(pnl.route_share_from_plays(plays, 2024)["player_id"].to_list()) == ["a", "b"]


def test_only_efficiency_quantities_get_an_expected_variant():
    """A target is not an estimate -- he was thrown at or he was not. Everything below the
    opportunity is an efficiency, and efficiency is what regresses."""
    assert set(pnl.EXPECTED) == {"receptions", "receiving_yards", "rushing_yards",
                                "passing_yards"}
    for opportunity in ("targets", "carries", "attempts"):
        assert opportunity not in pnl.EXPECTED


def test_the_expected_columns_are_the_ff_opportunity_names():
    assert pnl.EXPECTED["receptions"] == "receptions_exp"
    assert pnl.EXPECTED["receiving_yards"] == "rec_yards_gained_exp"
    assert pnl.XFP_WEEK == "total_fantasy_points_exp"


def test_scheme_rates_are_per_team_week():
    charting = pl.DataFrame({
        "nflverse_game_id": ["g1"] * 4,
        "nflverse_play_id": [1, 2, 3, 4],
        "is_play_action": [True, False, True, False],
        "is_motion": [True, True, True, True],
        "is_no_huddle": [False, False, False, False],
        "is_screen_pass": [True, False, False, False]})
    plays = pl.DataFrame({
        "nflverse_game_id": ["g1"] * 4, "play_id": [1, 2, 3, 4],
        "possession_team": ["PHI"] * 4, "is_pass": [True, True, False, False],
        "week": [3] * 4, "season": [2024] * 4})
    got = pnl.scheme_rates_from_plays(charting, plays).to_dicts()[0]
    assert got["posteam"] == "PHI" and got["week"] == 3
    assert got["pa_rate"] == pytest.approx(0.5)
    assert got["motion_rate"] == pytest.approx(1.0)
    assert got["nohuddle_rate"] == pytest.approx(0.0)
    assert got["pass_rate"] == pytest.approx(0.5)


def test_a_play_charted_but_not_participated_is_dropped():
    """An inner join, so a play FTN charted and participation did not is not a null team."""
    charting = pl.DataFrame({"nflverse_game_id": ["g1", "g1"], "nflverse_play_id": [1, 99],
                             "is_play_action": [True, True], "is_motion": [True, True],
                             "is_no_huddle": [False, False], "is_screen_pass": [False, False]})
    plays = pl.DataFrame({"nflverse_game_id": ["g1"], "play_id": [1],
                          "possession_team": ["PHI"], "is_pass": [True],
                          "week": [3], "season": [2024]})
    assert pnl.scheme_rates_from_plays(charting, plays).height == 1


def test_the_play_id_dtypes_are_reconciled():
    """FTN types it as an integer and nflverse as a float; unreconciled the join raises."""
    charting = pl.DataFrame({"nflverse_game_id": ["g1"], "nflverse_play_id": [1],
                             "is_play_action": [True], "is_motion": [True],
                             "is_no_huddle": [False], "is_screen_pass": [False]})
    plays = pl.DataFrame({"nflverse_game_id": ["g1"], "play_id": [1.0],
                          "possession_team": ["PHI"], "is_pass": [True],
                          "week": [3], "season": [2024]})
    assert pnl.scheme_rates_from_plays(charting, plays).height == 1


def test_an_empty_side_yields_the_right_empty_shape():
    got = pnl.scheme_rates_from_plays(pl.DataFrame(), pl.DataFrame())
    assert got.is_empty() and "pass_rate" in got.columns and "pa_rate" in got.columns


def test_trend_refuses_a_key_that_is_not_unique_per_week():
    """The guard that came out of the scheme build. `trend` fans out on a left join, so a team
    key against a player-week panel multiplies by the roster size -- and five chained calls
    took the process out on memory before this existed."""
    dup = pl.DataFrame({"team": ["PHI", "PHI"], "season": [2024, 2024],
                        "week": [3, 3], "pa_rate": [0.3, 0.4]})
    with pytest.raises(ValueError, match="one row per"):
        pnl.trend(dup, "pa_rate", "team", "pa_rate_trend")


def test_trend_accepts_the_unique_frame_it_asks_for():
    ok = pl.DataFrame({"team": ["PHI"] * 3, "season": [2024] * 3,
                       "week": [1, 2, 3], "pa_rate": [0.3, 0.4, 0.5]})
    assert "pa_rate_trend" in pnl.trend(ok, "pa_rate", "team", "pa_rate_trend").columns


def test_prior_means_returns_a_deterministically_ordered_frame():
    """A group_by emits rows in a hash-dependent order, and every caller of this means them
    again. Floating-point addition is not associative, so an unsorted hand-off moves the answer
    at ~1e-15 and every downstream sort can land differently -- improvements.md #18, which was
    found in `playoff_sos` and then found here by auditing for the pattern.
    """
    df = pl.DataFrame({"player_id": ["b", "a", "b", "a", "c"],
                       "season": [2024] * 5, "week": [1, 1, 2, 2, 1],
                       "pts": [1.0, 2.0, 3.0, 4.0, 5.0]})
    got = pnl.prior_means(df, ["player_id"], ["pts"])
    assert got.equals(got.sort(["player_id", "season", "week"])), \
        "sorted on the way out, not left to the group_by"


def test_the_designation_columns_survive_a_player_with_no_injury_row():
    """A healthy player must come through as Healthy, not as a null that a group_by drops."""
    inj = pl.DataFrame({"season": [2024], "week": [5], "key": ["hurt"],
                        "inj_sev": [1.0], "status": ["Questionable"], "practice": ["Limited"]})
    stats = pl.DataFrame({"season": [2024, 2024], "week": [5, 5],
                          "key": ["hurt", "fine"], "x": [1.0, 2.0]})
    out = (stats.join(inj, on=["season", "week", "key"], how="left")
                .with_columns(pl.col("status").fill_null("Healthy"),
                              pl.col("practice").fill_null("Healthy")))
    assert out.filter(pl.col("key") == "fine")["status"][0] == "Healthy"
    assert out.height == 2


def test_a_two_stage_aggregation_is_stable_when_the_hand_off_is_sorted():
    """The property itself, on a frame big enough for the group order to actually vary."""
    rng = np.random.default_rng(0)
    n = 4000
    df = pl.DataFrame({
        "team": [f"T{i % 32}" for i in range(n)],
        "pos": [["QB", "RB", "WR", "TE"][i % 4] for i in range(n)],
        "week": [(i % 14) + 1 for i in range(n)],
        "pts": rng.normal(10, 6, n)})

    def two_stage(sort_between):
        a = df.group_by(["team", "pos", "week"]).agg(pl.col("pts").sum().alias("s"))
        if sort_between:
            a = a.sort(["team", "pos", "week"])
        return (a.group_by(["team", "pos"]).agg(pl.col("s").mean().alias("m"))
                 .sort(["team", "pos"])["m"])

    unsorted = [two_stage(False) for _ in range(4)]
    sorted_ = [two_stage(True) for _ in range(4)]
    assert all(r.equals(sorted_[0]) for r in sorted_), "sorted: bit-identical every run"
    # the unsorted version is *usually* unstable; assert only that sorting cannot hurt
    assert sorted_[0].len() == unsorted[0].len()


def test_the_reported_lead_comes_from_the_scrape_whose_ecr_was_used():
    """`ecr.min()` was paired with `lead_days.first()` over an unsorted group_by, so the two
    could come from different scrapes. 3.3% of player-weeks carry two scrapes and 1,211 of
    those 1,216 have two different leads, and LEAD_DAYS is what docs/weekly-screen.md cites
    for the confound the whole screen is read against."""
    joined = pl.DataFrame({
        "season": [2024, 2024], "week": [5, 5], "key": ["chase", "chase"],
        # the better rank is the *later* scrape, so an unsorted first() could take the other
        "ecr": [12.0, 3.0], "lead_days": [7, 2],
    })
    r = pnl.best_per_week(joined).row(0, named=True)
    assert (r["ecr"], r["lead_days"]) == (3.0, 2), "the lead must match the ECR that won"


def test_one_scrape_a_week_is_unaffected():
    joined = pl.DataFrame({"season": [2024, 2024], "week": [5, 6], "key": ["x", "x"],
                           "ecr": [10.0, 11.0], "lead_days": [6, 5]})
    got = pnl.best_per_week(joined).sort("week")
    assert got["ecr"].to_list() == [10.0, 11.0]
    assert got["lead_days"].to_list() == [6, 5]
