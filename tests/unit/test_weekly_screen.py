"""The Phase 1 screen for week-level features.

Everything here is offline. The network functions are `# pragma: no cover` by design -- what
has to be right is the statistics and the leakage discipline, and neither needs nflverse.
"""
import ast
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.models import panel as pnl
from hub.models import weekly_screen as ws


def _panel(seasons=(2023, 2024), weeks=(1, 2), players=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "fantasy_points_ppr": float(rng.normal(12, 6)),
                             "yds_prior": float(rng.normal(12, 4)),
                             "ecr": float(i + 1), "feat": float(rng.normal())})
    return pl.DataFrame(rows)


def _summary(per, t):
    return {"r": float(np.mean(list(per.values()))), "se": 1.0, "t": t,
            "cells": len(per) * 5, "n": 1000, "per_season": per}


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
                             "yds_prior": 12.0, "ecr": float(i + 1),
                             "a": float(common[i] + 0.4 * rng.normal()),
                             "b": float(common[i] + 0.4 * rng.normal()),
                             "late": float(rng.normal()),
                             "fantasy_points_ppr": float(10 + 4 * common[i]
                                                         + rng.normal())})
    return pl.DataFrame(rows)


def _usage_panel(seasons=(2023, 2024), weeks=range(1, 13), players=60, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "ecr": float(i + 1), "yds_prior": 12.0,
                             "targets": float(rng.poisson(5)),
                             "fantasy_points_ppr": float(rng.normal(12, 6)),
                             "feat": float(rng.normal())})
    return pl.DataFrame(rows)


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


def test_one_correlation_per_season_week_cell():
    cells = ws.cell_correlations(_panel(), "feat")
    assert cells.height == 4
    assert set(zip(cells["season"], cells["week"], strict=True)) == {(2023, 1), (2023, 2),
                                                        (2024, 1), (2024, 2)}


def test_a_thin_cell_is_dropped_not_correlated():
    p = _panel(players=10)
    assert ws.cell_correlations(p, "feat").is_empty()
    assert ws.cell_correlations(p, "feat", min_cell=5).height == 4


def test_a_feature_missing_before_its_week_is_excluded_not_zero_filled():
    p = _panel().with_columns(
        pl.when(pl.col("week") >= 2).then(pl.col("feat")).otherwise(None).alias("feat"))
    cells = ws.cell_correlations(p, "feat", min_week=2)
    assert cells["week"].unique().to_list() == [2]


def test_the_screen_refuses_a_raw_outcome_column_as_a_feature_or_as_a_control():
    """#203, at the seam where a column becomes a feature.

    `FEATURES` above names derived features and pre-kickoff facts only, and it always did --
    which made the screen correct by the care of whoever last edited a tuple rather than by
    anything refusing. A caller naming `targets` where it meant `targets_recent` correlated
    this week's points against this week's own targets and got a number back.

    Asked once in `cell_correlations`, which `screen`, `screen_joint` and `screen_usage` all
    route through, so the three are covered by the one question.
    """
    p = _usage_panel()
    with pytest.raises(pnl.PanelRuleViolation, match="targets"):
        ws.cell_correlations(p, "targets")
    with pytest.raises(pnl.PanelRuleViolation, match="targets"):
        ws.cell_correlations(p, "feat", controls=("yds_prior", "targets"))
    with pytest.raises(pnl.PanelRuleViolation, match="targets"):
        ws.screen(p, [ws.Feature("targets", "+", 1)])


def test_the_same_raw_column_is_still_reachable_as_the_outcome_it_is():
    """The other side of the refusal, and the reason it is not simply a ban.

    **Usage** is the premise of the multiplier form: a feature that moves points without
    moving counts cannot be applied as a Usage multiplier, so the screen has to run *against*
    `targets`. Naming it as `outcome=` is a caller saying it means the realised column as an
    outcome, which is what it is. A refusal with no way through would have broken
    `screen_usage` rather than tightened it.
    """
    p = _usage_panel()
    p = pnl.recent_mean(pnl.prior_means(p, ["player_id"], ["targets"], within_season=True)
                        .join(p, on=["player_id", "season", "week"], how="right"), "targets")
    cells = ws.cell_correlations(p, "feat", outcome="targets",
                                 controls=("targets_prior", "targets_recent", "ecr"))
    assert not cells.is_empty(), "the Usage screen must still be able to run"


# --- the se is over the unit the verdict reads (issue #169) ------------------
#
# The se used to be taken across the season-week cells while `verdict` requires the sign to
# hold across the per-season means. The precision came from dozens of correlated cells and
# the decision from four or five seasons, so the interval was too narrow for the rule
# reading it. That is docs/method.md rule 3 -- repeated measures are not independent
# observations -- broken by the screen that exists to enforce it.


def test_the_standard_error_is_across_seasons():
    cells = pl.DataFrame({"season": [2023, 2023, 2024, 2024], "week": [1, 2, 1, 2],
                          "r": [0.1, 0.2, 0.3, 0.4], "n": [100, 100, 100, 100]})
    s = ws.summarise(cells)
    assert s["r"] == pytest.approx(0.25)
    expected = float(np.std([0.15, 0.35], ddof=1) / np.sqrt(2))
    assert s["se"] == pytest.approx(expected)
    assert s["seasons"] == 2 and s["cells"] == 4
    assert s["n"] == 400, "n is reported, but it is not what the se is built from"


def test_the_season_clustered_se_is_wider_on_within_season_correlated_cells():
    """The load-bearing case, built so the two answers genuinely differ.

    Cells inside a season agree closely and the seasons disagree, which is what
    within-season correlation looks like: almost all the variance is *between* the units
    the verdict reads and almost none within them. Treating the cells as independent
    divides that between-season spread by sqrt(20) instead of sqrt(4).

    A fixture where the two agree would prove nothing -- with one cell per season they are
    identical by construction, and `test_a_single_cell_per_season_is_the_same_either_way`
    below pins that as the boundary rather than passing it off as the test.
    """
    per_season = [0.30, 0.10, -0.05, 0.25]
    rows = [{"season": 2022 + s, "week": w, "r": mu + 0.001 * w, "n": 100}
            for s, mu in enumerate(per_season) for w in range(1, 6)]
    cells = pl.DataFrame(rows)

    s = ws.summarise(cells)
    flat = cells["r"].to_numpy().astype(float)
    by_cell = float(flat.std(ddof=1) / np.sqrt(len(flat)))

    assert s["cells"] == 20 and s["seasons"] == 4
    assert s["se"] > by_cell, (
        f"season-clustered se {s['se']:.5f} must exceed the cell se {by_cell:.5f}")
    assert s["se"] > 1.9 * by_cell, "and by roughly sqrt(cells per season), not marginally"
    assert abs(s["t"]) < abs(float(np.mean(flat)) / by_cell), "so the t shrinks with it"


def test_a_single_cell_per_season_is_the_same_either_way():
    """The boundary the test above must not be standing on: with one cell per season the
    cell vector *is* the season vector, so the two standard errors coincide. Asserted so a
    future fixture that quietly collapses to this case is visible rather than reassuring."""
    cells = pl.DataFrame({"season": [2022, 2023, 2024, 2025], "week": [1, 1, 1, 1],
                          "r": [0.30, 0.10, -0.05, 0.25], "n": [100] * 4})
    s = ws.summarise(cells)
    flat = cells["r"].to_numpy().astype(float)
    assert s["se"] == pytest.approx(float(flat.std(ddof=1) / np.sqrt(4)))


def test_r_is_the_mean_of_the_season_means_when_the_seasons_are_unbalanced():
    """`r` is the season vector's mean, not the cell vector's, and the two differ exactly
    when a season contributes fewer cells than the others.

    This is not a fixture detail. `wind` was real and unbalanced -- 7 cells in 2022 against
    11 in every other season, because a cell under `MIN_CELL` is dropped -- and its `r` was
    the one figure on `docs/weekly-screen.md` that the #169 correction moved, from -0.024 to
    -0.020. Every balanced fixture in this file agrees under both definitions and so pins
    nothing here; a mutation putting the cell mean back survived all of them.

    **Wind itself left the screen under #170** and both those figures are superseded and
    unestablished -- the feature was screened with a null-filled column and cannot be re-run.
    What it demonstrated about unbalanced cells did not go with it, which is why this test is
    still here and this fixture is still the thing that pins it.

    A t whose numerator comes from the cells and whose denominator comes from the seasons is
    the same defect the se half fixed, wearing the other hat.
    """
    thin, fat = 0.60, 0.00
    cells = pl.DataFrame({
        "season": [2023] * 2 + [2024] * 8,
        "week": list(range(1, 3)) + list(range(1, 9)),
        "r": [thin] * 2 + [fat] * 8,
        "n": [100] * 10})
    s = ws.summarise(cells)

    assert s["per_season"] == {2023: pytest.approx(thin), 2024: pytest.approx(fat)}
    assert s["r"] == pytest.approx(0.30), "the two season means, weighted equally"
    assert s["r"] != pytest.approx(float(cells["r"].to_numpy().mean())), (
        "and not the cell mean, which the thin season pulls to 0.12")
    assert s["t"] == pytest.approx(s["r"] / s["se"]), "numerator and denominator agree"


def test_the_report_names_the_unit_the_t_is_built_from():
    """55 cells printed beside a t of 5.7 reads as a 55-unit statistic unless the header
    says otherwise, which is the misreading #169 corrected. Also the only test on `report`,
    which `main` calls and `main` is `pragma: no cover` -- so a missing key here would
    reach an operator as a KeyError from a command that had run for minutes."""
    rows = ws.screen(_panel(), [ws.Feature("feat", "+", 1)]).to_dicts()
    lines = ws.report(rows)
    assert lines[0] == "", "a blank line before the block, as every report here opens"
    assert "szn" in lines[1] and "cells" in lines[1]
    assert "t is over the seasons" in lines[2]
    assert lines[-1].split()[0] == "feat"


def test_the_snap_share_trend_is_still_screened_and_reported_as_not_licensed():
    """#248, implementing #233. The licence is revoked and the feature stays in the family:
    the screen is the record of what was tried and why it lost (ADR-0007), so its cell is
    still computed, its verdict is still whatever the numbers say at that anchor -- it
    clears at 10 on the settled basis, and the record has to be able to say so -- and the
    line it prints on says the licence is gone, so a run that finds it clearing somewhere
    cannot be read as the licence coming back. Both halves: the verdict machinery does not
    know about the licence, and the report does."""
    assert "snap_trend" in [f.name for f in ws.FEATURES], "still in the family"
    assert "snap_trend" in ws.UNLICENSED and "#233" in ws.UNLICENSED["snap_trend"]
    panel = _panel().rename({"feat": "snap_trend"})
    rows = ws.screen(panel, [ws.Feature("snap_trend", "+", 1), ws.Feature("ecr", "?", 1)])
    by_name = {r["feature"]: r for r in rows.to_dicts()}
    assert by_name["snap_trend"]["status"] in (ws.CLEARS, ws.KILLED), (
        "the verdict is the numbers' verdict, not the licence's")
    lines = {ln.split()[0]: ln for ln in ws.report(rows.to_dicts())[3:]}
    assert "not licensed" in lines["snap_trend"] and "#233" in lines["snap_trend"]
    assert "not licensed" not in lines["ecr"], "the note is the revoked licence's alone"
    assert lines["snap_trend"].index(by_name["snap_trend"]["note"]) < \
        lines["snap_trend"].index("not licensed"), "the verdict first, then the licence"


def test_the_sweep_report_carries_the_revoked_licence_at_every_anchor():
    """The sweep is what `--run` prints by default, and it is where the trend clears at one
    anchor and not the others; the sensitivity line is the one a reader would take a
    licence from, so it is the one that says there is none."""
    feats = (*_SWEEP_FEATURES, ws.Feature("snap_trend", "+", ws.TREND_ANCHOR_UNSET))
    panel = _sweep_panel().with_columns(pl.col("late_trend").alias("snap_trend"))
    swept = ws.sweep(panel, feats, (4, 12))
    lines = ws.sweep_report(swept, ws.sensitivity(swept))
    trend = [ln for ln in lines if ln.split()[:1] == ["snap_trend"]]
    assert len(trend) == 1 and "not licensed" in trend[0] and "#233" in trend[0]
    assert not [ln for ln in lines if ln.split()[:1] == ["late_trend"]
                and "not licensed" in ln]


def test_the_reported_t_and_the_verdict_read_the_same_seasons():
    """`verdict` counts sign agreement over `per_season`; `t` is now built from the same
    vector. The two used to be a season-level rule beside a cell-level precision."""
    rows = [{"season": 2022 + s, "week": w, "r": mu, "n": 100}
            for s, mu in enumerate([0.30, 0.10, -0.05, 0.25]) for w in range(1, 6)]
    s = ws.summarise(pl.DataFrame(rows))
    seasons = np.array([s["per_season"][k] for k in sorted(s["per_season"])])

    assert s["r"] == pytest.approx(float(seasons.mean()))
    assert s["se"] == pytest.approx(float(seasons.std(ddof=1) / np.sqrt(len(seasons))))
    assert s["t"] == pytest.approx(s["r"] / s["se"])


def test_per_season_means_are_over_that_season_s_cells():
    cells = pl.DataFrame({"season": [2023, 2023, 2024], "week": [1, 2, 1],
                          "r": [0.1, 0.3, -0.5], "n": [50, 50, 50]})
    assert ws.summarise(cells)["per_season"] == {2023: pytest.approx(0.2), 2024: -0.5}


def test_nothing_measured_is_reported_rather_than_crashing():
    s = ws.summarise(pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64,
                                          "r": pl.Float64, "n": pl.Int64}))
    assert s["cells"] == 0
    assert ws.verdict(s, "+")[0] == ws.KILLED


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
    assert ws.is_signal(status, "0"), "so it goes on to the joint screen"


def test_a_null_that_is_merely_noisy_is_still_a_null():
    per = {2021: 0.05, 2022: -0.05, 2023: 0.05, 2024: -0.05, 2025: 0.05}
    assert ws.verdict(_summary(per, 2.5), "0")[0] == ws.CLEARS


def test_an_unsigned_feature_only_needs_consistency():
    per = dict.fromkeys(range(2021, 2026), -0.04)
    assert ws.verdict(_summary(per, -4.0), "?")[0] == ws.CLEARS


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


def test_route_trend_is_not_in_the_default_screen():
    """A pin on a decision, not on a preference. `route_trend` clears alone (+0.034 at 2.5 se,
    5/5 seasons) and correlates with `snap_trend` at **0.917** -- put in the same joint screen
    the two annihilate each other and leave nothing, which is a fact about collinearity and not
    about either signal. Snap share is the stronger, so it is the one that stays."""
    assert ws.ROUTE_TREND.name == "route_trend"
    assert ws.ROUTE_TREND not in ws.FEATURES
    assert "snap_trend" in [f.name for f in ws.FEATURES]


def test_wind_left_the_family_rather_than_being_quietly_retained():
    """#170's third criterion, and the refusal that backs it, where the family is.

    Wind was screened as a week-w pre-kickoff feature with a pre-stated negative sign, and the
    reading is taken *at* kickoff -- `docs/method.md` rule 2, broken by the screen that exists
    to enforce it. Reclassifying it in `hub.models.panel` is what makes it unscreenable; this
    is the other half, and the half a ticket criterion had to ask for: **the count of features
    tried moves with it.** A screen reporting eight verdicts while nine features were run is a
    multiple-comparison family that understates its own size, and the number is quoted in
    `hub.models.weekly` and on two pages.

    Both halves are asserted, because either alone is satisfiable without the other. Dropping
    the tuple entry while the Panel still called wind a feature would leave it screenable by
    anyone who typed the name; reclassifying it while it stayed in `FEATURES` would make every
    run raise instead of reporting a shorter family.
    """
    assert "wind" not in [f.name for f in ws.FEATURES], (
        "wind is back in the default screen; it is observed at kickoff and cannot be a "
        "predictor of the week it was measured on")
    assert len(ws.FEATURES) == 8, (
        f"the screened family is {len(ws.FEATURES)}. If that is deliberate, the count moves "
        f"in `hub.models.weekly`'s docstring and on docs/weekly-projection.md with it -- "
        f"which is what #170's third criterion is about")

    # Offline, like everything else here: `require_features` refuses a *classified* non-feature
    # role on any frame, so a hand-built one carrying a `wind` column is refused exactly as a
    # Panel would be. That is the property under test -- the refusal follows the column's name
    # and role, not the provenance of the frame it arrived on.
    p = _panel().with_columns(pl.lit(7.0).alias("wind"))
    assert pnl.column_role("wind") == "recorded"
    with pytest.raises(pnl.PanelRuleViolation, match="observed during week w"):
        ws.cell_correlations(p, "wind")
    with pytest.raises(pnl.PanelRuleViolation, match="observed during week w"):
        ws.screen(p, [ws.Feature("wind", "-", 1)])
    with pytest.raises(pnl.PanelRuleViolation, match="observed during week w"):
        ws.cell_correlations(p, "feat", controls=("yds_prior", "wind"))


def test_the_scheme_trends_are_not_in_the_default_screen():
    assert all(f not in ws.FEATURES for f in ws.SCHEME_TRENDS)
    assert {f.name for f in ws.SCHEME_TRENDS} == {
        "pa_rate_trend", "motion_rate_trend", "nohuddle_rate_trend",
        "screen_rate_trend", "pass_rate_trend"}


def test_no_player_appears_twice_inside_a_cell():
    """The property the design rests on -- protocol item 3. Pooling player-weeks would
    inflate every t here by roughly the square root of fourteen."""
    p = _panel()
    for (_s, _w), cell in p.group_by(["season", "week"]):
        assert cell["player_id"].n_unique() == cell.height


def test_the_usage_screen_controls_for_the_count_not_for_points():
    """Controlling this week's targets on season-to-date *points* would let a change in role
    show up as a target signal."""
    p = _usage_panel()
    p = pnl.recent_mean(pnl.prior_means(p, ["player_id"], ["targets"], within_season=True)
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
    p = p.join(pnl.prior_means(p, ["player_id"], ["targets"], within_season=True),
               on=["player_id", "season", "week"], how="left")
    p = (pnl.recent_mean(p, "targets")
           .drop_nulls(["targets_prior", "targets_recent"])
           .with_columns(pl.col("targets_recent").alias("form")))
    cells = ws.cell_correlations(p, "form", outcome="targets",
                                 controls=("targets_prior", "targets_recent", "ecr"))
    assert cells.is_empty(), "a feature identical to a control has no residual left"
    alone = ws.cell_correlations(p, "form", outcome="targets",
                                 controls=("targets_prior", "ecr"))
    assert not alone.is_empty(), "and against the lagging control alone it is measurable"


def _panel_with_every_feature(seasons=(2023, 2024), weeks=tuple(range(1, 15)),
                              players=80, seed=11):
    """A panel carrying every column `FEATURES` screens, so a re-run covers all of them.

    Weeks run past the **last** anchor of `SCREEN_TREND_ANCHORS` deliberately. At four weeks
    the two trend features have no cell that qualifies, `summarise` returns NaN for each, and
    NaN compares unequal to itself -- so the comparison below would have reported a difference
    for exactly the two features it had failed to screen. Deep enough that all eight produce a
    number at every anchor, and the equality means what it says."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                row = {"season": s, "week": w, "player_id": f"p{i}",
                       "fantasy_points_ppr": float(rng.normal(12, 6)),
                       "yds_prior": float(rng.normal(12, 4)),
                       "ecr": float(i + 1)}
                for f in ws.FEATURES:
                    row[f.name] = float(rng.normal())
                rows.append(row)
    return pl.DataFrame(rows)


def test_the_screen_returns_the_same_correlation_for_every_feature_when_rerun():
    """Issue #34: "the screen re-run at one as-of returns the same correlation for every
    feature twice".

    That criterion is two claims joined, and they are tested in the two places they live.
    `test_panel.py::test_two_as_ofs_are_two_pins_and_one_as_of_reproduces` holds the input
    half -- one as-of reads back one pin over the same rows, however much FantasyPros has
    published since. This holds the other half: given that input, the screen is a function of
    it and nothing else.

    Worth asserting rather than assuming. `cell_correlations` groups by cell and
    `summarise` pools across them, and a pooled float that depended on group order would
    reproduce under a fixed as-of only by luck -- which is the shape of reproducibility claim
    this repo has already had to withdraw once. Equality is exact, not approximate: two runs
    of the same arithmetic on the same rows have no licence to differ in the last bit.
    """
    panel = _panel_with_every_feature()
    # At the deepest anchor, so the trend features screen on the fewest weeks this sweep ever
    # gives them. A run that reproduces there reproduces at every shallower anchor.
    features = ws.at_anchor(ws.FEATURES, max(ws.SCREEN_TREND_ANCHORS))
    runs = []
    for _ in range(2):
        runs.append({f.name: ws.summarise(
            ws.cell_correlations(panel, f.name, min_week=f.min_week)) for f in features})

    assert set(runs[0]) == {f.name for f in ws.FEATURES}, "a feature went unscreened"
    empty = [n for n, v in runs[0].items() if v["cells"] == 0]
    assert not empty, (
        f"{empty} produced no cell, so this compares NaN with NaN and proves nothing. "
        f"The panel has to run past the deepest of `SCREEN_TREND_ANCHORS` for the trend "
        f"features to screen.")
    for name in runs[0]:
        assert runs[0][name] == runs[1][name], (
            f"{name} moved between two runs over one panel: {runs[0][name]} then "
            f"{runs[1][name]}. The screen has an input the as-of does not pin.")


# --- The control bases: #179's alternative, and #229's decision ---------------------------
#
# `ppg_before` contains prior touchdowns and `td_rate_prior`'s numerator is prior touchdowns,
# so the pre-registered control set contained the feature's own numerator. What these hold is
# the *property* the decomposition was chosen for -- that the new set spans the old one, so a
# coefficient that moves moved because of the constraint that was relaxed and not because
# something else stopped being controlled for -- that the basis a run is taken on actually
# reaches the arithmetic, and that the basis the screen *defaults* to is the one #229 decided
# on and not the pre-registration it replaced.


def _split_panel(n=80, seed=11):
    """A panel whose `ppg_before` is genuinely two components that sum to it.

    The outcome is driven by the touchdown half **alone**, which is what lets a test tell the
    two bases apart: pooling the halves into one control leaves signal in that holding them
    apart removes. `yds_prior` is carried beside them, unrelated to either, so the yardage
    basis is a third distinguishable set on the same frame.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in (2023, 2024):
        for w in range(1, 11):
            for i in range(n):
                td = float(abs(rng.normal(2.0, 1.0)))
                nontd = float(abs(rng.normal(6.5, 2.0)))
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "td_ppg_before": td, "nontd_ppg_before": nontd,
                             "ppg_before": td + nontd, "ecr": float(i + 1),
                             "yds_prior": float(abs(rng.normal(60.0, 20.0))),
                             "feat": float(rng.normal()),
                             "fantasy_points_ppr": float(2.0 * td + rng.normal())})
    return pl.DataFrame(rows)


def test_the_default_basis_is_prior_yardage_and_consensus_rank():
    """#229, as a fact the suite holds rather than a sentence in a docstring.

    Three bases were run on one panel and two of them contained or pinned `td_rate_prior`'s
    numerator: `ppg_before` is PPR points and PPR points contain touchdowns, and holding the
    touchdown half of it fixed pins the count outright. `(yds_prior, ecr)` is the one that
    does neither, it is the set #179's issue body pre-registered, and the decision on #229
    made it what every surviving claim is conditional on. A default that drifted back to the
    pre-registration would restate every figure on `docs/weekly-screen.md` silently.
    """
    assert ws.CONTROLS == ("yds_prior", "ecr")
    assert ws.BASES[ws.DEFAULT_BASIS] is ws.CONTROLS
    for c in ws.CONTROLS:
        assert "ppg" not in c and "td" not in c, (
            f"the default basis controls on `{c}`, which carries prior touchdowns -- the "
            f"feature's own numerator, which is the confound #229 moved the basis to escape.")


def test_the_screen_defaults_to_the_decided_basis_not_the_pre_registration():
    """`screen` and `screen_joint` called without `controls` run on `CONTROLS`, and `CONTROLS`
    is not the pooled set. The two halves are asked separately: a default that still bound to
    `CONTROLS_POOLED` would pass the first test above on the constant and fail here on the
    arithmetic, which is where a run actually reads it.
    """
    p = _split_panel()
    feature = [ws.Feature("feat", "0", 1)]
    default = ws.screen(p, feature).to_dicts()[0]
    yardage = ws.screen(p, feature, ws.CONTROLS).to_dicts()[0]
    pooled = ws.screen(p, feature, ws.CONTROLS_POOLED).to_dicts()[0]
    assert default["r"] == yardage["r"]
    assert default["r"] != pooled["r"], \
        "`screen` with no basis named is still running on the pre-registration"

    both = [*feature, ws.Feature("ppg_before", "?", 1)]
    j_default = {d["feature"]: d for d in ws.screen_joint(p, both).to_dicts()}
    j_yardage = {d["feature"]: d for d in ws.screen_joint(p, both, ws.CONTROLS).to_dicts()}
    j_pooled = {d["feature"]: d
                for d in ws.screen_joint(p, both, ws.CONTROLS_POOLED).to_dicts()}
    assert j_default["feat"]["r"] == j_yardage["feat"]["r"]
    assert j_default["feat"]["r"] != j_pooled["feat"]["r"], \
        "`screen_joint` with no basis named is still running on the pre-registration"


def test_the_decomposed_basis_spans_the_pooled_one():
    """`ppg_before` residualised on the two halves is nothing, because it is their sum.

    This is what makes #179 a re-run on a *basis* rather than on a different control set. If
    the halves did not span the pooled control, a coefficient that moved between them could
    have moved because the new set stopped controlling for something -- and the whole point
    was to isolate one constraint, that a point of touchdown scoring and a point of everything
    else carry the same slope.
    """
    p = _split_panel()
    controls = np.column_stack([p[c].to_numpy() for c in ws.CONTROLS_DECOMPOSED])
    left = ws.residual(p["ppg_before"].to_numpy(), controls)
    assert np.allclose(left, 0.0, atol=1e-9), (
        f"`ppg_before` left a residual of {np.abs(left).max():.3e} on the decomposed basis. "
        f"The two sets are then not nested, and #179's comparison is not attributable to the "
        f"one constraint it relaxed.")


def test_the_pooled_control_does_not_span_the_decomposed_one():
    """The premise of the test above: the nesting goes one way only.

    Without this, "spans" is satisfied by the two bases being the *same* basis and the re-run
    measures nothing -- a re-run that re-runs nothing being the failure `docs/method.md` rule
    13 is a record of. A half of `ppg_before` is not recoverable from the total, which is
    exactly why the pooled basis could not tell the two stories apart.

    Asked over **every** column of the decomposed basis rather than over `td_ppg_before` by
    name. The first draft of this named the column, which made it a fact about the fixture's
    columns and not about the two bases: it passed unchanged with `CONTROLS_DECOMPOSED` set
    to `CONTROLS_POOLED`, the one mutation it exists to catch.
    """
    p = _split_panel()
    pooled = np.column_stack([p[c].to_numpy() for c in ws.CONTROLS_POOLED])
    left = max(float(np.abs(ws.residual(p[c].to_numpy(), pooled)).max())
               for c in ws.CONTROLS_DECOMPOSED)
    assert left > 1e-3, (
        "every column of the decomposed basis is recoverable from the pooled control, so the "
        "two bases are one basis and #179 re-ran the published screen against itself.")


def test_the_screen_takes_the_basis_it_is_given():
    """`screen` and `screen_joint` route the basis through rather than accepting and ignoring.

    A `controls` parameter that were accepted and dropped would return the same number twice,
    and every figure reported under `--basis decomposed` would silently be the published one
    -- a re-run that re-runs nothing, which `docs/method.md` rule 13 is the record of this
    repo doing three times.
    """
    p = _split_panel()
    feature = [ws.Feature("feat", "0", 1)]
    pooled = ws.screen(p, feature, ws.CONTROLS_POOLED).to_dicts()[0]
    split = ws.screen(p, feature, ws.CONTROLS_DECOMPOSED).to_dicts()[0]
    assert pooled["cells"] == split["cells"] == 20, "same rows, so only the basis differs"
    assert pooled["r"] != split["r"], (
        "the two bases returned an identical correlation on a panel built to separate them; "
        "`controls` is not reaching `cell_correlations`.")

    both = [*feature, ws.Feature("ppg_before", "?", 1)]
    j_pooled = {d["feature"]: d for d in ws.screen_joint(p, both, ws.CONTROLS_POOLED).to_dicts()}
    j_split = {d["feature"]: d
               for d in ws.screen_joint(p, both, ws.CONTROLS_DECOMPOSED).to_dicts()}
    assert j_pooled["feat"]["r"] != j_split["feat"]["r"], \
        "`screen_joint` is not routing the basis either"


def test_every_named_basis_is_one_the_panel_would_serve_as_features():
    """`--basis` chooses between named sets, and every column on one has to pass the rule.

    A control is residualised out of **both** sides, so a week-w column in a control set leaks
    into every feature at once -- and it is the one door `require_features` does not watch,
    because it refuses a bad *feature* and a control arrives by the other parameter. Checked
    here against `column_role` so a basis added later cannot introduce one quietly.
    """
    assert ws.BASES["yardage"] == ws.CONTROLS
    assert ws.BASES["pooled"] == ws.CONTROLS_POOLED
    assert ws.BASES["decomposed"] == ws.CONTROLS_DECOMPOSED
    for name, controls in ws.BASES.items():
        for c in controls:
            assert pnl.column_role(c) in pnl.FEATURE_ROLES, (
                f"basis {name!r} controls on `{c}`, which the Panel classifies as "
                f"{pnl.column_role(c)!r}.")


# --- #178: the screen's minimum week is swept, not borrowed from the model ------------------


def test_the_screen_does_not_read_the_model_s_trend_threshold():
    """The whole of #178, as a source-level guard.

    `panel.TREND_MIN_WEEK` is 8 because 8 is the earliest of the anchors 4, 6, 8, 10, 12 that
    held when they were tested **against the outcome** (`docs/snap-trend-signal.md`). That is
    the right way to set a model threshold and the wrong way to choose which rows a screen
    reads: rows selected by a value fitted to the outcome are not a neutral sample to screen
    other features on, and `cell_correlations` filters on exactly that value.

    A comment saying the two are separate is not a control -- the import is. This asserts the
    screen cannot reach the constant at all, so re-coupling them means deleting this test
    rather than editing a line and moving on.

    Over the **AST**, not over the text, so that prose may still name the constant it is
    explaining. The declaration of `SCREEN_TREND_ANCHORS` has to be able to say what it is not
    doing and why, and a substring check would make that comment illegal.
    """
    tree = ast.parse(Path(ws.__file__).read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             for a in n.names}
    assert "TREND_MIN_WEEK" not in used, (
        "`hub.models.weekly_screen` reads `TREND_MIN_WEEK` again. The model threshold and the "
        "screen's minimum week are two constants with two justifications (#178); the screen's "
        "is `SCREEN_TREND_ANCHORS`, and it is swept rather than chosen.")
    assert not hasattr(ws, "TREND_MIN_WEEK")
    assert pnl.TREND_MIN_WEEK == 8, "the model threshold keeps its measured value"


def test_the_sweep_range_and_step_are_the_five_anchors_that_were_measured():
    """Fixed in the commit that runs the sweep, per the pre-registration on #178 -- not chosen
    after seeing which range flatters the answer. These are the anchors
    `docs/snap-trend-signal.md` measured the trend at."""
    assert ws.SCREEN_TREND_ANCHORS == (4, 6, 8, 10, 12)
    assert ws.PUBLISHED_ANCHOR in ws.SCREEN_TREND_ANCHORS


def test_a_trend_feature_carries_no_anchor_until_one_is_given():
    trends = [f for f in ws.FEATURES if f.name.endswith("_trend")]
    assert {f.name for f in trends} == {"snap_trend", "tgt_trend"}
    assert all(f.min_week == ws.TREND_ANCHOR_UNSET for f in trends), \
        "a trend feature declares its sign, not its week -- the sweep supplies the week"
    assert all(f.min_week == 1 for f in ws.FEATURES if not f.name.endswith("_trend")), \
        "a pre-kickoff fact is the same quantity in week 2 as in week 12"
    assert ws.ROUTE_TREND.min_week == ws.TREND_ANCHOR_UNSET
    assert all(f.min_week == ws.TREND_ANCHOR_UNSET for f in ws.SCHEME_TRENDS)


def test_at_anchor_moves_the_trend_features_and_nothing_else():
    moved = ws.at_anchor(ws.FEATURES, 10)
    by_name = {f.name: f for f in moved}
    assert by_name["snap_trend"].min_week == 10
    assert by_name["tgt_trend"].min_week == 10
    assert by_name["implied_total"].min_week == 1
    assert by_name["td_rate_prior"].min_week == 1
    assert [f.sign for f in moved] == [f.sign for f in ws.FEATURES], \
        "the pre-stated sign is pre-registered and the anchor does not touch it"


@pytest.mark.parametrize("run", [
    lambda p, fs: ws.screen(p, fs),
    lambda p, fs: ws.screen_joint(p, fs),
    lambda p, fs: ws.screen_usage(p, fs, ("targets",)),
])
def test_the_screen_refuses_a_feature_set_that_carries_no_anchor(run):
    """`min_week=0` filters no rows, so a forgotten `at_anchor` would not crash -- it would
    screen the trend features from week 1 and print a number, taken on rows the trend does not
    exist over, in the same column as everything else. All three entry points ask."""
    p = _usage_panel()
    with pytest.raises(ValueError, match="carry no anchor"):
        run(p, [ws.Feature("feat", "+", 1), ws.Feature("x_trend", "+", ws.TREND_ANCHOR_UNSET)])


def test_an_anchored_feature_set_runs():
    """The other half of the guard: it refuses the unset anchor and nothing else. Without
    this, a `require_anchor` that raised unconditionally would pass every test above."""
    p = _sweep_panel()
    anchored = ws.screen(p, ws.at_anchor(
        [ws.Feature("late_trend", "+", ws.TREND_ANCHOR_UNSET)], 8))
    direct = ws.screen(p, [ws.Feature("late_trend", "+", 8)])
    assert anchored["cells"].item() == direct["cells"].item()
    assert anchored["r"].item() == direct["r"].item()
    assert ws.is_signal(anchored["status"].item(), "+"), \
        "and the anchored run reaches a verdict rather than merely not raising"


def _sweep_panel(seasons=(2021, 2022, 2023, 2024, 2025), weeks=range(1, 15),
                 players=60, seed=17, late_from=8, late=4.0, early=6.0):
    """A panel whose `late_trend` carries the real effect only from week `late_from` on.

    Which is the shape the sensitivity has to be able to see: a feature that is a finding at
    the deep anchors and noise at the shallow ones, so the verdict genuinely depends on where
    the row filter is put. `flat` is signal at every week and `dud` at none, so the two stable
    cases are present in the same run.

    Before `late_from` the effect is real within a season and its **sign alternates between
    seasons**, which is how `docs/snap-trend-signal.md` describes anchors 4 and 6: not absent,
    but flipping. That is the failure mode the every-season half of the rule exists to catch
    (`docs/method.md` rule 4), so it is the one worth building the fixture out of -- a shallow
    anchor mixes those weeks in and the season means stop agreeing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k, s in enumerate(seasons):
        sign = 1.0 if k % 2 == 0 else -1.0
        for w in weeks:
            for i in range(players):
                lt = float(rng.normal())
                flat = float(rng.normal())
                y = 12.0 + 3.0 * flat + rng.normal(0, 2)
                y += late * lt if w >= late_from else sign * early * lt
                rows.append({"season": s, "week": w, "player_id": f"p{i}",
                             "yds_prior": 12.0, "ecr": float(i + 1),
                             "late_trend": lt, "flat": flat,
                             "dud": float(rng.normal()),
                             "fantasy_points_ppr": y})
    return pl.DataFrame(rows)


_SWEEP_FEATURES = (ws.Feature("flat", "+", 1), ws.Feature("dud", "+", 1),
                   ws.Feature("late_trend", "+", ws.TREND_ANCHOR_UNSET))


def test_the_sweep_reports_every_feature_at_every_anchor():
    swept = ws.sweep(_sweep_panel(), _SWEEP_FEATURES, (4, 6, 8, 10, 12))
    assert swept.height == 3 * 5, "one row per (anchor, feature), and none missing"
    assert sorted(swept["anchor"].unique().to_list()) == [4, 6, 8, 10, 12]
    trend = swept.filter(pl.col("feature") == "late_trend").sort("anchor")
    assert trend["min_week"].to_list() == [4, 6, 8, 10, 12], \
        "the anchor is what the trend feature's row filter became"
    assert swept.filter(pl.col("feature") == "flat")["min_week"].unique().to_list() == [1]


def test_the_anchor_moves_only_the_trend_features_numbers():
    """The structural fact worth pinning, because it bounds what the sweep can possibly find.

    The anchor is a floor on the trend features' weeks. A week-1 feature keeps all its cells
    at every anchor, and it is never controlled for a trend feature either -- `screen_joint`
    admits a survivor as a control only where `g.min_week <= f.min_week`, and every anchor in
    the sweep is above 1. So if any week-1 feature's verdict moved across the sweep, the cause
    would be a bug and not the anchor.
    """
    swept = ws.sweep(_sweep_panel(), _SWEEP_FEATURES, (4, 6, 8, 10, 12))
    for name in ("flat", "dud"):
        d = swept.filter(pl.col("feature") == name)
        assert d["alone_r"].n_unique() == 1, f"{name} moved with the anchor"
        assert d["cells"].n_unique() == 1, f"{name} lost cells to the anchor"
    trend = swept.filter(pl.col("feature") == "late_trend").sort("anchor")
    assert trend["cells"].to_list() == sorted(trend["cells"].to_list(), reverse=True), \
        "a deeper anchor has to leave the trend feature fewer cells, not more"


def test_a_verdict_that_changes_across_the_sweep_is_named_as_conditional():
    """The half of #178 that matters if the answer comes back unstable: the screen reports the
    range over which the finding holds rather than a single verdict taken at one anchor."""
    swept = ws.sweep(_sweep_panel(late_from=12), _SWEEP_FEATURES, (4, 6, 8, 10, 12))
    sens = {r["feature"]: r for r in ws.sensitivity(swept).iter_rows(named=True)}

    assert not sens["late_trend"]["stable"]
    assert sens["late_trend"]["verdict"] == "DEPENDS ON THE ANCHOR"
    assert sens["late_trend"]["finding_at"] == "10, 12", (
        "the anchors where it is a finding, not a single verdict: the effect is consistent "
        "only from week 12 on, so a shallower anchor mixes in the weeks where its sign "
        "alternates between seasons and the every-season half fails")
    assert sens["late_trend"]["by_anchor"].startswith("4:killed 6:killed 8:killed")
    assert sens["flat"]["stable"], "the week-1 features cannot move with the anchor"
    assert ws.surviving(swept, 4) == ["flat"]
    assert ws.surviving(swept, 12) == ["flat", "late_trend"]

    lines = "\n".join(ws.sweep_report(swept, ws.sensitivity(swept)))
    assert "CONDITIONAL ON THE ANCHOR: late_trend" in lines
    assert "changed nothing" not in lines


def test_a_verdict_that_holds_everywhere_is_reported_as_it_stands():
    swept = ws.sweep(_sweep_panel(late_from=1), _SWEEP_FEATURES, (4, 6, 8, 10, 12))
    sens = {r["feature"]: r for r in ws.sensitivity(swept).iter_rows(named=True)}
    assert sens["late_trend"]["stable"], "signal at every week cannot depend on the anchor"
    assert ws.is_signal(sens["late_trend"]["verdict"], "+")
    assert sens["late_trend"]["finding_at"] == "4, 6, 8, 10, 12"
    assert sens["dud"]["stable"] and sens["dud"]["verdict"] == ws.KILLED
    lines = "\n".join(ws.sweep_report(swept, ws.sensitivity(swept)))
    assert "changed nothing" in lines
    assert "CONDITIONAL ON THE ANCHOR" not in lines


def test_a_feature_killed_alone_is_reported_on_the_screen_it_actually_reached():
    """`final` falls back to the alone verdict, because a feature killed alone never enters
    the joint screen and printing a joint verdict for it would read as a stronger rejection
    than the run performed."""
    swept = ws.sweep(_sweep_panel(), _SWEEP_FEATURES, (8,))
    dud = swept.filter(pl.col("feature") == "dud").to_dicts()[0]
    assert dud["alone"] == ws.KILLED
    assert dud["joint"] is None and dud["joint_r"] is None
    assert dud["final"] == dud["alone"]
    flat = swept.filter(pl.col("feature") == "flat").to_dicts()[0]
    assert flat["joint"] is not None, "a survivor does reach the joint screen"
    assert flat["final"] == flat["joint"]


def test_a_null_that_clears_by_being_null_is_not_a_survivor():
    """A pre-stated null that behaves as a null has not found anything -- #229.

    `dud` is noise with a pre-registered null. It clears, as it should: the prediction held.
    But a null that held is the *absence* of a signal, and before this the survivor filter
    was `status in (CLEARS, NULL_BROKEN)`, which carried it into the joint screen as a
    finding, printed it under "independent signals" beside a note reading "noisy, not a
    signal", and controlled every real survivor for it. Latent while `td_rate_prior` was
    `NULL_BROKEN` on every basis; the yardage basis is the first on which a null cleared.
    """
    features = (ws.Feature("flat", "+", 1), ws.Feature("dud", "0", 1),
                ws.Feature("late_trend", "+", ws.TREND_ANCHOR_UNSET))
    swept = ws.sweep(_sweep_panel(), features, (8,))
    dud = swept.filter(pl.col("feature") == "dud").to_dicts()[0]
    assert dud["alone"] == ws.CLEARS, "the null held -- that is the premise"
    assert not ws.is_signal(dud["alone"], "0")
    assert dud["joint"] is None, "and a held null does not enter the joint screen"
    assert "dud" not in ws.surviving(swept, 8)
    flat = swept.filter(pl.col("feature") == "flat").to_dicts()[0]
    assert "dud" not in flat["joint_controls"], \
        "nor is a real survivor controlled for a quantity the screen said carries nothing"
    sens = {r["feature"]: r for r in ws.sensitivity(swept).to_dicts()}
    assert sens["dud"]["finding_at"] == "-"
    surviving_set = "\n".join(ws.sweep_report(swept, ws.sensitivity(swept))
                              ).split("verdict across the sweep")[0]
    assert "dud" not in surviving_set, \
        "the surviving set the report prints does not carry it either"


# --- #238: the every-season half under the null ---------------------------------------------


def test_under_the_null_one_season_crosses_zero_almost_always():
    """Five seasons of noise: at least one has the wrong sign in about 1 - 2^-5 of draws.

    That is the fact the anchor question turns on. If the figure were anchor-dependent, one
    season crossing zero at week 12 would carry information about week 12; it is not, so the
    every-season half's false-positive rate is the same ~3% at every anchor and the question
    is about power. Held at two cell counts per season so the independence from the count is
    a thing the suite checks rather than a sentence.
    """
    p = _sweep_panel()
    dud = ws.Feature("dud", "+", 1)
    wide = ws.every_season_null(p, dud, draws=600, seed=1)
    narrow = ws.every_season_null(p, dud._replace(min_week=12), draws=600, seed=1)
    assert wide["per_season_cells"] == dict.fromkeys(range(2021, 2026), 14)
    assert narrow["per_season_cells"] == dict.fromkeys(range(2021, 2026), 3)
    for n in (wide, narrow):
        assert abs(n["p_any_wrong_sign"] - (1 - 2 ** -5)) < 0.03, n
        assert abs(n["p_every_season"] + n["p_any_wrong_sign"] - 1.0) < 1e-12
        assert 0.0 <= n["p_value"] <= 1.0


def test_a_real_effect_crosses_zero_more_often_on_fewer_cells():
    """The other half: the same true effect fails the every-season half more often when each
    season mean is three cells than when it is fourteen, because the mean is noisier. This is
    what "weak evidence against the trend" means, made a number."""
    p = _sweep_panel()
    dud = ws.Feature("dud", "+", 1)
    wide = ws.every_season_null(p, dud, draws=600, seed=2, effects=(0.04,))
    narrow = ws.every_season_null(p, dud._replace(min_week=12), draws=600, seed=2,
                                  effects=(0.04,))
    assert narrow["alternatives"][0.04] > wide["alternatives"][0.04] + 0.1, (narrow, wide)
    assert wide["alternatives"][0.04] < wide["p_any_wrong_sign"], \
        "a true effect crosses zero less often than no effect does"


def test_the_null_permutes_within_the_cell_and_not_across_it():
    """`flat` is a real signal at every week. Permuted within its cell it has to look like
    nothing -- that is the placebo the page reports -- so the permutation p on the observed r
    is small and the null r's sit near zero. A permutation that reached across cells would
    also break the link and pass this; what would not pass is no permutation at all, which
    is the failure a placebo can have silently."""
    p = _sweep_panel()
    n = ws.every_season_null(p, ws.Feature("flat", "+", 1), draws=300, seed=3)
    assert n["r"] > 0.3, "the fixture's signal is large"
    assert n["p_value"] == 0.0, "and no permutation draw reaches it"
    assert n["cell_sd"] < 0.2


def test_a_pre_stated_null_has_no_every_season_half():
    with pytest.raises(ValueError, match="pre-stated null"):
        ws.every_season_null(_sweep_panel(), ws.Feature("dud", "0", 1), draws=5)


# --- the null's own report, and its two degenerate inputs (#238 coverage) ------

def test_the_null_report_carries_every_figure_including_the_alternatives():
    """`null_report` is what a reader sees; a figure computed and not rendered is a claim
    nobody can check (#57's rule). Held on a result with an alternative effect, so the
    power line renders too."""
    p = _sweep_panel()
    dud = ws.Feature("dud", "+", 1)
    n = ws.every_season_null(p, dud, draws=200, seed=4, effects=(0.04,))
    lines = ws.null_report("dud", dud.min_week, n)
    text = "\n".join(lines)
    for needle in ("cells, per season", "permutation", "P(at least one season has the wrong",
                   "P(every season holds)", "sd of one cell", "if the true effect were +0.0400"):
        assert needle in text, f"the report does not carry {needle!r}:\n{text}"


def test_a_feature_with_no_qualifying_cells_reports_zero_rather_than_raising():
    """An anchor past every week in the panel leaves nothing to permute. The answer is a
    result that says so -- zero cells, NaN r -- not an exception, because the sweep calls
    this for every anchor and one empty anchor must not take the others down."""
    p = _sweep_panel()
    late = ws.Feature("dud", "+", 1)._replace(min_week=99)
    n = ws.every_season_null(p, late, draws=50, seed=5)
    assert n["cells"] == 0 and n["seasons"] == 0
    assert n["r"] != n["r"], "an empty result should carry NaN, not a number"
    assert ws.null_report("dud", 99, n)          # renders without raising


def test_a_pre_stated_null_is_refused_and_a_thin_cell_is_skipped():
    """Two branches on the way in, and they are different acts. A `"0"`-signed feature has
    no sign for the every-season half to hold, so asking is a category error and it raises.
    A cell under `min_cell` rows is skipped rather than contributing a noisy r, so a panel
    made entirely of thin cells comes back as an empty result rather than an error."""
    p = _sweep_panel()
    with pytest.raises(ValueError, match="pre-stated null"):
        ws.every_season_null(p, ws.Feature("dud", "0", 1), draws=50, seed=6)
    n = ws.every_season_null(p, ws.Feature("dud", "+", 1), draws=50, seed=6, min_cell=10_000)
    assert n["cells"] == 0, "thin cells were scored rather than skipped"


# --- the screen names its own reads, however it is invoked (issue #192) -----------------
#
# `main` is network-bound and `# pragma: no cover` for it; what is driven here is the whole of
# it with the Panel build replaced by a synthetic Panel that records the one read it made. The
# seam under test is what the `data` line names, and that it is the screen's own reads rather
# than a run's around it -- or, as it was until #192, one named cache entry standing in for
# everything the Panel loaded.

def _screenable_panel(seasons=(2023, 2024), weeks=range(1, 15), players=60, seed=11):
    """A Panel with every column `main` reads, so the real sweep runs on it."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                td, non = float(rng.normal(3, 1)), float(rng.normal(9, 3))
                rows.append({
                    "season": s, "week": w, "player_id": f"p{i}", "games_before": 5,
                    "lead_days": 3.0, "ecr": float(i + 1),
                    "fantasy_points_ppr": float(rng.normal(12, 6)),
                    "yds_prior": float(rng.normal(60, 20)), "ppg_before": td + non,
                    "td_ppg_before": td, "nontd_ppg_before": non,
                    "implied_total": float(rng.normal(23, 3)),
                    "own_spread": float(rng.normal(0, 5)), "dvp": float(rng.normal(1, .2)),
                    "rest": 7.0, "inj_sev": float(rng.integers(0, 3)),
                    "td_rate_prior": float(rng.normal(0.05, 0.02)),
                    "snap_trend": float(rng.normal()), "tgt_trend": float(rng.normal()),
                })
    return pl.DataFrame(rows)


def test_the_screen_called_in_process_names_only_its_own_reads(monkeypatch, capsys, tmp_path):
    """Called from inside a run that has already read something, the `data` line the screen
    prints covers the screen's reads and not the enclosing run's -- and the enclosing run
    still ends up holding both, because the scope narrows what a component reports and is
    not a way for a run to lose a read."""
    from hub.config import data_digest
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    outer = nv.Pin(source="player_stats", as_of=None, digest="0ut51de0", rows=1,
                   pinned_at=None)
    inner = nv.Pin(source="ff_rankings", as_of="2024-09-01", digest="1n51de01", rows=1,
                   pinned_at=None)
    nv._remember(tmp_path / "the-enclosing-runs-entry.parquet", outer)

    def builds(seasons, spec, as_of=None):
        nv._remember(tmp_path / "the-screens-own-entry.parquet", inner)
        return _screenable_panel(seasons)

    monkeypatch.setattr(ws, "build_panel", builds)
    assert ws.main(["--run", "--seasons", "2023,2024", "--trend-min-week", "8"]) == 0
    said = capsys.readouterr().out
    line = next(ln for ln in said.splitlines() if ln.lstrip().startswith("cfg "))
    assert f"data {data_digest([inner])}" in line, (
        "the screen's data line is not a digest over the screen's own read")
    assert data_digest([outer, inner]) not in line, (
        "the screen printed a digest over the enclosing run's reads as well as its own")
    assert sorted(p.source for p in nv.pins_this_run()) == ["ff_rankings", "player_stats"], (
        "the screen's read did not reach the run around it: scoping lost a read")
