"""The Phase 1 screen for week-level features.

Everything here is offline. The network functions are `# pragma: no cover` by design -- what
has to be right is the statistics and the leakage discipline, and neither needs nflverse.
"""
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
                             "ppg_before": float(rng.normal(12, 4)),
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
                             "ppg_before": 12.0, "ecr": float(i + 1),
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
                             "ecr": float(i + 1), "ppg_before": 12.0,
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

    This is not a fixture detail. `wind` is real and unbalanced -- 7 cells in 2022 against
    11 in every other season, because a cell under `MIN_CELL` is dropped -- and its `r` is
    the one figure on `docs/weekly-screen.md` that the #169 correction moves, from -0.024 to
    -0.020. Every balanced fixture in this file agrees under both definitions and so pins
    nothing here; a mutation putting the cell mean back survived all of them.

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
    assert status in ws.FINDINGS, "so it goes on to the joint screen"


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

    Weeks run past `TREND_MIN_WEEK` deliberately. At four weeks the two trend features
    have no cell that qualifies, `summarise` returns NaN for each, and NaN compares
    unequal to itself -- so the comparison below would have reported a difference for
    exactly the two features it had failed to screen. Deep enough that all nine produce
    a number, and the equality means what it says."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for i in range(players):
                row = {"season": s, "week": w, "player_id": f"p{i}",
                       "fantasy_points_ppr": float(rng.normal(12, 6)),
                       "ppg_before": float(rng.normal(12, 4)),
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
    runs = []
    for _ in range(2):
        runs.append({f.name: ws.summarise(
            ws.cell_correlations(panel, f.name, min_week=f.min_week)) for f in ws.FEATURES})

    assert set(runs[0]) == {f.name for f in ws.FEATURES}, "a feature went unscreened"
    empty = [n for n, v in runs[0].items() if v["cells"] == 0]
    assert not empty, (
        f"{empty} produced no cell, so this compares NaN with NaN and proves nothing. "
        f"The panel has to run past `TREND_MIN_WEEK` for the trend features to screen.")
    for name in runs[0]:
        assert runs[0][name] == runs[1][name], (
            f"{name} moved between two runs over one panel: {runs[0][name]} then "
            f"{runs[1][name]}. The screen has an input the as-of does not pin.")
