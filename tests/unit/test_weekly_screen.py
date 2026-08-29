"""The Phase 1 screen for week-level features.

Everything here is offline. The network functions are `# pragma: no cover` by design -- what
has to be right is the statistics and the leakage discipline, and neither needs nflverse.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import weekly_screen as ws

# --- the statistic ----------------------------------------------------------

def test_residual_removes_the_control_entirely():
    rng = np.random.default_rng(0)
    c = rng.normal(size=200)
    y = 3.0 + 2.0 * c
    assert np.allclose(ws.residual(y, c.reshape(-1, 1)), 0.0, atol=1e-9)


def test_an_intercept_is_added_here_not_by_the_caller():
    """A caller who passed their own column of ones would get a singular design."""
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ws.residual(y, np.zeros((3, 1))), y - y.mean())


def test_partial_r_is_the_correlation_of_two_residuals():
    rng = np.random.default_rng(1)
    c = rng.normal(size=500)
    x = c + rng.normal(size=500)
    y = c + rng.normal(size=500)
    raw = np.corrcoef(y, x)[0, 1]
    partial = ws.partial_r(y, x, c.reshape(-1, 1))
    assert raw > 0.3, "the shared control makes them look correlated"
    assert abs(partial) < 0.15, "and controlling for it takes that away"


def test_a_constant_feature_has_no_partial_correlation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(ws.partial_r(y, np.ones(4), np.zeros((4, 1))))


# --- the cell structure, which is what stops repeated measures --------------

def _panel(seasons=(2023, 2024), weeks=(1, 2), players=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "fantasy_points_ppr": float(rng.normal(12, 6)),
                             "ppg_before": float(rng.normal(12, 4)),
                             "ecr": float(i + 1), "feat": float(rng.normal())})
    return pl.DataFrame(rows)


def test_one_correlation_per_season_week_cell():
    cells = ws.cell_correlations(_panel(), "feat")
    assert cells.height == 4
    assert set(zip(cells["season"], cells["week"], strict=True)) == {(2023, 1), (2023, 2),
                                                        (2024, 1), (2024, 2)}


def test_no_player_appears_twice_inside_a_cell():
    """The property the design rests on -- protocol item 3. Pooling player-weeks would
    inflate every t here by roughly the square root of fourteen."""
    p = _panel()
    for (_s, _w), cell in p.group_by(["season", "week"]):
        assert cell["player_id"].n_unique() == cell.height


def test_a_thin_cell_is_dropped_not_correlated():
    p = _panel(players=10)
    assert ws.cell_correlations(p, "feat").is_empty()
    assert ws.cell_correlations(p, "feat", min_cell=5).height == 4


def test_a_feature_missing_before_its_week_is_excluded_not_zero_filled():
    p = _panel().with_columns(
        pl.when(pl.col("week") >= 2).then(pl.col("feat")).otherwise(None).alias("feat"))
    cells = ws.cell_correlations(p, "feat", min_week=2)
    assert cells["week"].unique().to_list() == [2]


# --- the standard error is of the cells, not of the player-weeks ------------

def test_the_standard_error_is_across_cells():
    cells = pl.DataFrame({"season": [2023, 2023, 2024, 2024], "week": [1, 2, 1, 2],
                          "r": [0.1, 0.2, 0.3, 0.4], "n": [100, 100, 100, 100]})
    s = ws.summarise(cells)
    assert s["r"] == pytest.approx(0.25)
    expected = float(np.std([0.1, 0.2, 0.3, 0.4], ddof=1) / np.sqrt(4))
    assert s["se"] == pytest.approx(expected)
    assert s["n"] == 400, "n is reported, but it is not what the se is built from"


def test_per_season_means_are_over_that_season_s_cells():
    cells = pl.DataFrame({"season": [2023, 2023, 2024], "week": [1, 2, 1],
                          "r": [0.1, 0.3, -0.5], "n": [50, 50, 50]})
    assert ws.summarise(cells)["per_season"] == {2023: pytest.approx(0.2), 2024: -0.5}


def test_nothing_measured_is_reported_rather_than_crashing():
    s = ws.summarise(pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64,
                                          "r": pl.Float64, "n": pl.Int64}))
    assert s["cells"] == 0
    assert ws.verdict(s, "+")[0] == ws.KILLED


# --- the pre-registered rule, including every losing branch -----------------

def _summary(per, t):
    return {"r": float(np.mean(list(per.values()))), "se": 1.0, "t": t,
            "cells": len(per) * 5, "n": 1000, "per_season": per}


def test_a_feature_clears_only_with_both_halves():
    per = {2021: 0.05, 2022: 0.04, 2023: 0.06, 2024: 0.05, 2025: 0.07}
    assert ws.verdict(_summary(per, 4.6), "+")[0] == ws.CLEARS


def test_one_season_against_it_kills_a_feature_however_significant():
    """The half that killed defence-vs-position at t = 4.9 on a single negative season."""
    per = {2021: 0.05, 2022: 0.04, 2023: 0.04, 2024: -0.004, 2025: 0.06}
    status, note = ws.verdict(_summary(per, 4.9), "+")
    assert status == ws.KILLED and "4/5" in note


def test_every_season_is_not_enough_without_significance():
    per = {2021: 0.01, 2022: 0.01, 2023: 0.01, 2024: 0.01, 2025: 0.01}
    status, note = ws.verdict(_summary(per, 1.2), "+")
    assert status == ws.KILLED and "se" in note


def test_the_wrong_sign_does_not_clear_however_consistent():
    """A feature pre-stated `+` that comes back consistently negative has failed, not won."""
    per = dict.fromkeys(range(2021, 2026), -0.05)
    assert ws.verdict(_summary(per, -6.0), "+")[0] == ws.KILLED


def test_a_pre_stated_null_clears_by_being_null():
    per = {2021: 0.001, 2022: -0.002, 2023: 0.0, 2024: 0.001, 2025: -0.001}
    status, note = ws.verdict(_summary(per, 0.3), "0")
    assert status == ws.CLEARS and "as pre-stated" in note


def test_a_broken_null_is_a_finding_not_a_rejection():
    """The most informative outcome a screen can produce, and the reason the prediction was
    written down. Folding it in with the rejections would lose it."""
    per = dict.fromkeys(range(2021, 2026), -0.045)
    status, note = ws.verdict(_summary(per, -6.0), "0")
    assert status == ws.NULL_BROKEN
    assert "PRE-STATED NULL BROKEN" in note
    assert status in ws.FINDINGS, "so it goes on to the joint screen"


def test_a_null_that_is_merely_noisy_is_still_a_null():
    per = {2021: 0.05, 2022: -0.05, 2023: 0.05, 2024: -0.05, 2025: 0.05}
    assert ws.verdict(_summary(per, 2.5), "0")[0] == ws.CLEARS


def test_an_unsigned_feature_only_needs_consistency():
    per = dict.fromkeys(range(2021, 2026), -0.04)
    assert ws.verdict(_summary(per, -4.0), "?")[0] == ws.CLEARS


# --- the joint screen, and the bug it was written with ----------------------

def _collinear_panel(n=80, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for s in (2023, 2024):
        for w in range(1, 11):
            common = rng.normal(size=n)
            for i in range(n):
                # Two noisy readings of one underlying quantity, which is what drives the
                # outcome. Symmetric on purpose: neither is the truth and the other a copy,
                # so neither *residual* carries signal once the other is controlled for.
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "ppg_before": 12.0, "ecr": float(i + 1),
                             "a": float(common[i] + 0.4 * rng.normal()),
                             "b": float(common[i] + 0.4 * rng.normal()),
                             "late": float(rng.normal()),
                             "fantasy_points_ppr": float(10 + 4 * common[i]
                                                         + rng.normal())})
    return pl.DataFrame(rows)


def test_two_readings_of_one_quantity_both_clear_on_their_own():
    """Which is the reason the joint screen exists. `own_spread` and `implied_total` each
    cleared at 5/5 seasons, and one is a linear function of the other."""
    p = _collinear_panel()
    alone = ws.screen(p, [ws.Feature("a", "+", 1), ws.Feature("b", "+", 1)])
    assert set(alone.filter(pl.col("status") == ws.CLEARS)["feature"]) == {"a", "b"}


def test_a_feature_that_is_another_one_rescaled_leaves_nothing():
    """The mechanism, tested where it is deterministic rather than noise-dependent: an exact
    linear duplicate residualises to zero, so there is no correlation left to report and the
    verdict says nothing was measured rather than inventing one."""
    p = _collinear_panel().with_columns((2.0 * pl.col("a") + 1.0).alias("dup"))
    joint = ws.screen_joint(p, [ws.Feature("a", "+", 1), ws.Feature("dup", "+", 1)])
    dup = joint.filter(pl.col("feature") == "dup").to_dicts()[0]
    assert dup["status"] == ws.KILLED
    assert dup["cells"] == 0 and "nothing measured" in dup["note"]
    assert dup["controls"] == "a"


def test_a_late_starting_feature_does_not_shrink_the_others_sample():
    """The bug this was written with. Taking the widest min_week across the survivor set
    dropped `implied_total` from 54 cells to 35 purely because `snap_trend` starts at week 8,
    then reported the lost power as a failed control."""
    p = _collinear_panel()
    joint = ws.screen_joint(p, [ws.Feature("a", "+", 1), ws.Feature("late", "+", 8)])
    early = joint.filter(pl.col("feature") == "a").to_dicts()[0]
    late = joint.filter(pl.col("feature") == "late").to_dicts()[0]
    assert early["cells"] == 20, "week 1 feature keeps all its cells"
    assert late["cells"] == 6, "and the week-8 feature keeps only its own"
    assert early["controls"] == "-", "a week-1 feature cannot be controlled for a week-8 one"
    assert late["controls"] == "a"


# --- Usage, and the lagging-control problem ---------------------------------

def _usage_panel(seasons=(2023, 2024), weeks=range(1, 13), players=60, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "ecr": float(i + 1), "ppg_before": 12.0,
                             "targets": float(rng.poisson(5)),
                             "fantasy_points_ppr": float(rng.normal(12, 6)),
                             "feat": float(rng.normal())})
    return pl.DataFrame(rows)


def test_recent_mean_is_strictly_prior():
    """Week w's control may not contain week w. The whole screen rests on it."""
    p = pl.DataFrame({"player_id": ["a"] * 5, "season": [2024] * 5,
                      "week": [1, 2, 3, 4, 5],
                      "targets": [10.0, 0.0, 0.0, 0.0, 99.0]})
    out = ws.recent_mean(p, "targets").sort("week")
    assert out["targets_recent"].to_list()[4] == pytest.approx(0.0), \
        "weeks 2-4 are zeros; week 5's own 99 must not reach its own control"
    assert out["targets_recent"].to_list()[0] is None, "week 1 has no history"


def test_recent_mean_counts_calendar_weeks_not_appearances():
    """A player who missed week 3 has a two-week window, not one padded from week 1."""
    p = pl.DataFrame({"player_id": ["a"] * 3, "season": [2024] * 3,
                      "week": [1, 2, 4], "targets": [10.0, 10.0, 4.0]})
    out = ws.recent_mean(p, "targets").sort("week")
    assert out["targets_recent"].to_list()[2] == pytest.approx(10.0), \
        "week 4 averages weeks 1-3, of which only 1 and 2 exist"


def test_the_usage_screen_controls_for_the_count_not_for_points():
    """Controlling this week's targets on season-to-date *points* would let a change in role
    show up as a target signal."""
    p = _usage_panel()
    p = ws.recent_mean(ws.prior_means(p, ["player_id"], ["targets"], within_season=True)
                       .join(p, on=["player_id", "season", "week"], how="right"), "targets")
    out = ws.screen_usage(p, [ws.Feature("feat", "+", 1)], components=("targets",))
    assert out.height == 1 and out["component"][0] == "targets"


def test_a_feature_that_is_recent_form_leaves_nothing():
    """Why `recent_mean` is in the control set at all. Tested where it is deterministic: a
    feature that IS the last three weeks' level residualises away entirely, so the screen
    reports nothing rather than re-discovering its own control.

    The noisy version of this is the real finding and lives in docs/weekly-screen.md --
    `snap_trend` on carries falls from +0.127 to +0.049 once recent form is controlled for,
    so most of that effect was form, and what is left is still 5/5 seasons.
    """
    rng = np.random.default_rng(11)
    rows = [{"season": 2024, "week": w, "player_id": f"p{i}", "ecr": float(i + 1),
             "targets": float(rng.poisson(4 + (i % 5)))}
            for w in range(1, 13) for i in range(80)]
    p = pl.DataFrame(rows)
    p = p.join(ws.prior_means(p, ["player_id"], ["targets"], within_season=True),
               on=["player_id", "season", "week"], how="left")
    p = (ws.recent_mean(p, "targets")
           .drop_nulls(["targets_prior", "targets_recent"])
           .with_columns(pl.col("targets_recent").alias("form")))
    cells = ws.cell_correlations(p, "form", outcome="targets",
                                 controls=("targets_prior", "targets_recent", "ecr"))
    assert cells.is_empty(), "a feature identical to a control has no residual left"
    alone = ws.cell_correlations(p, "form", outcome="targets",
                                 controls=("targets_prior", "ecr"))
    assert not alone.is_empty(), "and against the lagging control alone it is measurable"


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
    src = inspect.getsource(ws.build_panel)
    assert "CANNOT be fitted here" in src, "the constraint must stay stated where it is read"
    assert "status" in src and "practice" in src, "and the columns are carried for Gate B"


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


# --- the as-of week assignment, and the off-by-one that shifted everything ---

def _windows():
    """Two NFL weeks: Thursday to Monday, a week apart."""
    import datetime as dt
    return pl.DataFrame({
        "season": [2024, 2024],
        "week": [5, 6],
        "first_kick": [dt.date(2024, 10, 3), dt.date(2024, 10, 10)],
        "last_kick": [dt.date(2024, 10, 7), dt.date(2024, 10, 14)]})


def _scrape(day):
    import datetime as dt
    return pl.DataFrame({"scrape_date": [dt.date(*day)], "key": ["x"], "ecr": [1.0]})


def test_a_midweek_scrape_belongs_to_the_week_it_is_inside():
    """The bug that shifted the whole control by a week. An NFL week runs Thursday to Monday
    and FantasyPros scrapes land mid-week: 2024-10-04 is a Friday, *inside* week 5 (Oct 3-7),
    and it ranks week 5's Sunday games. Joining on the next *first* kickoff sent it to week 6.

    The tell was that Saquon Barkley, CeeDee Lamb and Patrick Mahomes were each missing from
    exactly one week -- and it was the week after their team's bye, because a page that
    correctly omits a bye-week player was being attached to the following week.
    """
    got = ws.assign_weeks(_scrape((2024, 10, 4)), _windows())
    assert got["week"].to_list() == [5], "a Friday inside week 5 is week 5's ranking"


def test_a_scrape_before_a_week_opens_still_belongs_to_it():
    assert ws.assign_weeks(_scrape((2024, 10, 1)), _windows())["week"].to_list() == [5]


def test_a_scrape_after_a_week_closes_belongs_to_the_next():
    assert ws.assign_weeks(_scrape((2024, 10, 8)), _windows())["week"].to_list() == [6]


def test_a_scrape_too_far_from_any_week_is_dropped():
    """Pre-season and post-season scrapes rank no week that exists here."""
    assert ws.assign_weeks(_scrape((2024, 9, 1)), _windows()).is_empty()


def test_lead_days_is_measured_to_the_week_being_ranked():
    got = ws.assign_weeks(_scrape((2024, 10, 4)), _windows())
    assert got["lead_days"].to_list() == [3], "Friday to the following Monday"


# --- pass-play participation, and the twin it turned out to be --------------

def test_route_share_is_plays_on_over_team_pass_plays():
    plays = pl.DataFrame({
        "week": [1, 1, 1, 1],
        "possession_team": ["PHI"] * 4,
        "offense_players": ["a;b;c", "a;b", "a;c", "a"]})
    got = ws.route_share_from_plays(plays, 2024).sort("player_id")
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
    got = ws.route_share_from_plays(plays, 2024)
    assert set(got["route_pct"].to_list()) == {1.0}, "each is on all of his own team's plays"


def test_an_empty_slate_yields_an_empty_frame_with_the_right_shape():
    got = ws.route_share_from_plays(pl.DataFrame(), 2024)
    assert got.is_empty()
    assert set(got.columns) == {"season", "week", "player_id", "route_pct"}


def test_blank_ids_in_the_player_list_are_dropped():
    plays = pl.DataFrame({"week": [1], "possession_team": ["PHI"],
                          "offense_players": ["a;;b;"]})
    assert sorted(ws.route_share_from_plays(plays, 2024)["player_id"].to_list()) == ["a", "b"]


def test_route_trend_is_not_in_the_default_screen():
    """A pin on a decision, not on a preference. `route_trend` clears alone (+0.034 at 2.5 se,
    5/5 seasons) and correlates with `snap_trend` at **0.917** -- put in the same joint screen
    the two annihilate each other and leave nothing, which is a fact about collinearity and not
    about either signal. Snap share is the stronger, so it is the one that stays."""
    assert ws.ROUTE_TREND.name == "route_trend"
    assert ws.ROUTE_TREND not in ws.FEATURES
    assert "snap_trend" in [f.name for f in ws.FEATURES]


# --- expected values replace efficiency, never opportunity ------------------

def test_only_efficiency_quantities_get_an_expected_variant():
    """A target is not an estimate -- he was thrown at or he was not. Everything below the
    opportunity is an efficiency, and efficiency is what regresses."""
    assert set(ws.EXPECTED) == {"receptions", "receiving_yards", "rushing_yards",
                                "passing_yards"}
    for opportunity in ("targets", "carries", "attempts"):
        assert opportunity not in ws.EXPECTED


def test_the_expected_columns_are_the_ff_opportunity_names():
    assert ws.EXPECTED["receptions"] == "receptions_exp"
    assert ws.EXPECTED["receiving_yards"] == "rec_yards_gained_exp"
    assert ws.XFP_WEEK == "total_fantasy_points_exp"
