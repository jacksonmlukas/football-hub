"""The line-move study and its verdict (#221, #329, #346), held to its pre-registration in
`docs/gate-power.md`: the week's mean move is subtracted, the regressor is the net ex-ante gap,
the floor sets the standard error, and the verdict is non-inferiority against DELTA on
`experiment`'s interval and MDE. The events it reads are held in `test_starter_events.py`.

Every fixture is synthetic and every test runs with no `data/` and no network.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics

import polars as pl
import pytest
from starter_world import (
    ARCHIVE,
    PRIOR,
    SEASON,
    pbp_events,
    polls,
    serve_world,
    source_rows,
    utc,
)

from hub.models import experiment
from hub.models import starter_events as se
from hub.models import starter_study as ss


@pytest.fixture
def rows():
    return source_rows(PRIOR + SEASON)


@pytest.fixture(autouse=True)
def _the_events_source_is_served_offline(rows):
    """Every test here that drives `main` reads its events through `nflverse.load`, which on
    the network adapter would leave the machine; the offline suite fails a test that tries.
    The default world is the fixture's own rows read as play-by-play."""
    serve_world(rows)




# --- the study ---------------------------------------------------------------------------


def test_the_study_subtracts_the_weeks_mean_move_and_regresses_on_the_net_gap(rows):
    """Week 3: KC-LA moved 3 -> -2 (-5), the other week-3 game moved 1 -> 2 (+1) over the
    same two poll days, so the week-adjusted move is -6 on a net gap of -145 (KC's change
    alone; LA's is offseason and not an event). Week 4 (KC-DEN) has no result yet -- the
    fixture's own unplayed game (#303) -- and does not enter the study at all."""
    tg = se.team_games(rows)
    games = se.event_games(se.in_season_events(se.events(tg)))
    study = ss.study_rows(ARCHIVE, games, tg).sort("week")
    assert study["game_id"].to_list() == ["2026_03_LA_KC"]
    assert study["move"].to_list() == [-5.0]
    assert study["week_mean"].to_list() == [1.0]
    assert study["adjusted_move"].to_list() == [-6.0]
    assert study["net_gap"].to_list() == [-145.0]


def test_an_in_flight_game_with_polls_after_kickoff_is_excluded_and_counted(rows):
    """#303, defect 1: week 4 (KC-DEN) is a genuine in-season event -- Mahomes returns -- but
    the fixture gives it no score, an in-flight game whose archive polls (utc(24), utc(30))
    straddle its own kickoff. Before the fix, `study_rows` had never joined results, so its
    'last snapshot before the game day' was just the latest snapshot of a game still being
    played, and the move it fed the regression was truncated with nothing marking the row.
    Excluded here, and `unplayed_study_games` reports the one game the exclusion dropped,
    off the same uncensored/priced frame `study_rows` itself filters."""
    tg = se.team_games(rows)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games.height == 2                                    # both KC changes are events
    study = ss.study_rows(ARCHIVE, games, tg)
    assert "2026_04_DEN_KC" not in study["game_id"].to_list()
    assert ss.unplayed_study_games(ARCHIVE, games, tg) == 1


def test_the_change_point_compares_eastern_days_not_utc_ones():
    """#303, defect 2: every other date comparison in this module goes through the poll-day
    conversion; the change-point's own arithmetic didn't, comparing a poll's raw UTC calendar
    date against `frozen_before`, an Eastern one. A capture at 2026-09-08T02:00 UTC is
    2026-09-07, 22:00 Eastern -- the previous *Eastern* day -- so the correct change-point is
    one day after the previous game day (09-06), not two: the UTC date alone would have
    counted a day the poll never saw."""
    rows_ = source_rows([
        ("2026-09-06", 2026, "1.0", "AA", "BB", "a1", "b1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "AA", "CC", "a2", "c1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
        ("2026-09-06", 2026, "1.0", "DD", "EE", "d1", "e1", 100.0, 100.0, 0.0, 0.0,
         14, 14, .5, .5),
        ("2026-09-13", 2026, "2.0", "DD", "EE", "d1", "e1", 100.0, 100.0, 0.0, 0.0,
         21, 21, .5, .5),
    ])
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games["game_id"].to_list() == ["2026_02_CC_AA"]
    archive = polls([
        # AA-CC: frozen before the previous game day (09-06); the 02:00 UTC capture on 09-08
        # is 09-07 Eastern; close before the game day (09-13).
        ("2026_02_CC_AA", 0.0, dt.datetime(2026, 9, 4, 20, 0), 2),
        ("2026_02_CC_AA", 5.0, dt.datetime(2026, 9, 8, 2, 0), 2),
        ("2026_02_CC_AA", 6.0, dt.datetime(2026, 9, 12, 16, 0), 2),
        # DD-EE: not an event, polled the same two poll days as the control this week.
        ("2026_02_EE_DD", 2.0, dt.datetime(2026, 9, 4, 20, 0), 2),
        ("2026_02_EE_DD", 2.5, dt.datetime(2026, 9, 12, 16, 0), 2),
    ])
    study = ss.study_rows(archive, games, tg, floor_per_root_day=0.4)
    assert study["days_to_change"].to_list() == pytest.approx([1.0])


def test_the_change_point_scans_poll_days_and_is_bounded_at_the_game_day():
    """#303, defect 3: two sub-cases of one defect.

    **Several captures in one day** -- 09-09 carries an early spike (10.0, clears the floor
    on its own) and a later same-day poll back near the frozen price (0.5, does not); the
    Eastern-date convention (`hub.fetch.odds`: the last poll of a date stands for it) means
    the day's own value is 0.5 and the threshold does not fire on 09-09 at all -- it fires
    two days later, on 09-11, the first day whose own representative value clears it.

    **Bounded at the game day** -- FF-HH's game day is 09-13; a poll on 09-15, after kickoff,
    would clear the floor on its own, but no poll before the game day for GG-HH does, so the
    scan -- which `priced`'s own `close` never reads past the game day either -- reports no
    change-point rather than reading a lookahead nothing before kickoff could have seen.
    """
    rows_ = source_rows([
        ("2026-09-06", 2026, "1.0", "FF", "GG", "f1", "g1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "FF", "HH", "f2", "h1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
        ("2026-09-06", 2026, "1.0", "II", "JJ", "i1", "j1", 100.0, 100.0, 0.0, 0.0,
         14, 14, .5, .5),
        ("2026-09-13", 2026, "2.0", "II", "JJ", "i1", "j1", 100.0, 100.0, 0.0, 0.0,
         21, 21, .5, .5),
    ])
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games["game_id"].to_list() == ["2026_02_HH_FF"]
    archive = polls([
        ("2026_02_HH_FF", 0.0, dt.datetime(2026, 9, 4, 16, 0), 2),     # frozen
        ("2026_02_HH_FF", 10.0, dt.datetime(2026, 9, 9, 14, 0), 2),    # 09-09 early: a spike
        ("2026_02_HH_FF", 0.5, dt.datetime(2026, 9, 9, 22, 0), 2),     # 09-09 late: the day's value
        ("2026_02_HH_FF", 6.0, dt.datetime(2026, 9, 11, 16, 0), 2),    # 09-11: really clears it
        ("2026_02_HH_FF", 6.5, dt.datetime(2026, 9, 12, 16, 0), 2),    # close, before the game day
        ("2026_02_HH_FF", 50.0, dt.datetime(2026, 9, 15, 16, 0), 2),   # after kickoff: unseen
        ("2026_02_JJ_II", 1.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_JJ_II", 1.5, dt.datetime(2026, 9, 12, 16, 0), 2),
    ])
    study = ss.study_rows(archive, games, tg, floor_per_root_day=0.4)
    # frozen_before is FF's previous game day, 09-06; the change is seen 09-11, 5 days later.
    assert study["days_to_change"].to_list() == pytest.approx([5.0])

    # A second game whose only poll clearing the floor falls after its own game day: bounded
    # scan finds nothing, where an unbounded one would have read the lookahead.
    late_only = source_rows([
        ("2026-09-06", 2026, "1.0", "KK", "LL", "k1", "l1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "KK", "MM", "k2", "m1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
        ("2026-09-06", 2026, "1.0", "NN", "OO", "n1", "o1", 100.0, 100.0, 0.0, 0.0,
         14, 14, .5, .5),
        ("2026-09-13", 2026, "2.0", "NN", "OO", "n1", "o1", 100.0, 100.0, 0.0, 0.0,
         21, 21, .5, .5),
    ])
    tg2 = se.team_games(late_only)
    games2 = se.event_games(se.in_season_events(se.events(tg2)))
    assert games2["game_id"].to_list() == ["2026_02_MM_KK"]
    archive2 = polls([
        ("2026_02_MM_KK", 0.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_MM_KK", 0.1, dt.datetime(2026, 9, 12, 16, 0), 2),    # close: never clears
        ("2026_02_MM_KK", 50.0, dt.datetime(2026, 9, 15, 16, 0), 2),   # after kickoff: unseen
        ("2026_02_OO_NN", 1.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_OO_NN", 1.2, dt.datetime(2026, 9, 12, 16, 0), 2),
    ])
    study2 = ss.study_rows(archive2, games2, tg2, floor_per_root_day=0.4)
    assert study2["days_to_change"].is_null().all()


def test_a_weeks_control_set_excludes_every_other_event_game(rows):
    """#303, defect 4, part one: a week with two starter changes never uses one treated game
    as the other's control. AA and BB both change starters in week 2, and CC-DD, the week's
    only other game, is the sole legitimate control -- if BB were left in AA's control set
    (the pre-fix behaviour, which excluded only the row's own game_id), AA's week_mean would
    read BB's own treated move instead."""
    rows_ = source_rows([
        ("2026-09-06", 2026, "1.0", "AA", "PP", "a1", "p1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "AA", "QQ", "a2", "q1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
        ("2026-09-06", 2026, "1.0", "BB", "RR", "b1", "r1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "BB", "SS", "b2", "s1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
        ("2026-09-06", 2026, "1.0", "CC", "DD", "c1", "d1", 100.0, 100.0, 0.0, 0.0,
         14, 14, .5, .5),
        ("2026-09-13", 2026, "2.0", "CC", "DD", "c1", "d1", 100.0, 100.0, 0.0, 0.0,
         21, 21, .5, .5),
    ])
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    # CC-DD keeps the same starters both weeks -- not an event, and the week's only control.
    assert sorted(games["game_id"].to_list()) == ["2026_02_QQ_AA", "2026_02_SS_BB"]
    archive = polls([
        ("2026_02_QQ_AA", 0.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_QQ_AA", 7.0, dt.datetime(2026, 9, 12, 16, 0), 2),      # AA moves 7
        ("2026_02_SS_BB", 0.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_SS_BB", -20.0, dt.datetime(2026, 9, 12, 16, 0), 2),    # BB moves -20
        ("2026_02_DD_CC", 1.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_DD_CC", 2.0, dt.datetime(2026, 9, 12, 16, 0), 2),      # CC-DD moves 1
    ])
    study = ss.study_rows(archive, games, tg).sort("game_id")
    aa = study.filter(pl.col("game_id") == "2026_02_QQ_AA").row(0, named=True)
    assert aa["week_mean"] == pytest.approx(1.0)                # CC-DD only, BB excluded
    assert aa["adjusted_move"] == pytest.approx(6.0)


def test_a_row_with_no_control_left_is_refused_not_fitted_as_a_zero_week_mean(rows):
    """#303, defect 4, part two: a week where the only two archived games are both event
    games has no control for either -- `week_mean` cannot be a genuine zero, since there was
    nothing to average, and the row is dropped from the study rather than fitted as though
    the week moved by nothing."""
    rows_ = source_rows([
        ("2026-09-06", 2026, "1.0", "AA", "PP", "a1", "p1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "AA", "QQ", "a2", "q1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
        ("2026-09-06", 2026, "1.0", "BB", "RR", "b1", "r1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "BB", "SS", "b2", "s1", 150.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
    ])
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert sorted(games["game_id"].to_list()) == ["2026_02_QQ_AA", "2026_02_SS_BB"]
    archive = polls([
        ("2026_02_QQ_AA", 0.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_QQ_AA", 7.0, dt.datetime(2026, 9, 12, 16, 0), 2),
        ("2026_02_SS_BB", 0.0, dt.datetime(2026, 9, 4, 16, 0), 2),
        ("2026_02_SS_BB", -20.0, dt.datetime(2026, 9, 12, 16, 0), 2),
    ])
    study = ss.study_rows(archive, games, tg)
    assert study.is_empty()


def test_the_fit_recovers_the_slope_and_takes_its_error_from_the_floor():
    """Moves manufactured at 0.132 points per unit of gap plus a week effect; after the
    week's mean is subtracted the slope is the benchmark, and the standard error is the
    floor per window over the gap's spread and root n - 1 (OLS's own denominator, #303),
    not the residual's."""
    gaps = [-150.0, -80.0, -20.0, 30.0, 90.0, 140.0]
    rows_ = pl.DataFrame({
        "season": [2026] * 6, "week": [1, 1, 2, 2, 3, 3], "game_id": [f"g{i}" for i in range(6)],
        "net_gap": gaps, "adjusted_move": [0.132 * g for g in gaps],
        "window_days": [7.0] * 6})
    fit = ss.study_fit(rows_, floor_per_root_day=0.4)
    assert fit["beta"] == pytest.approx(0.132)
    assert fit["n"] == 6
    floor_window = 0.4 * math.sqrt(7.0)
    sd_gap = statistics.stdev(gaps)
    assert fit["se"] == pytest.approx(floor_window / (sd_gap * math.sqrt(5)))
    assert fit["mde"] == pytest.approx(
        (experiment.t_quantile(0.975, 5) + 0.8416) * fit["se"], abs=1e-3)
    assert fit["benchmark"] == ss.BENCHMARK == pytest.approx(3.3 / 25)
    assert fit["t_vs_benchmark"] == pytest.approx(0.0)


def test_the_slope_se_at_n_equals_2_differs_from_the_old_root_n_formula_by_41_percent():
    """#303, defect 6, the n=2 case named in the ticket: dividing by root n instead of root
    (n - 1) understates the standard error, and at n=2 the gap between the two denominators
    (root 1 vs root 2) is its largest relative size, about 41%."""
    gaps = [-60.0, 60.0]
    rows_ = pl.DataFrame({
        "season": [2026, 2026], "week": [1, 1], "game_id": ["g0", "g1"],
        "net_gap": gaps, "adjusted_move": [0.132 * g for g in gaps], "window_days": [7.0, 7.0]})
    fit = ss.study_fit(rows_, floor_per_root_day=0.4)
    floor_window = 0.4 * math.sqrt(7.0)
    sd_gap = statistics.stdev(gaps)
    correct_se = floor_window / (sd_gap * math.sqrt(1))          # n - 1 = 1
    old_wrong_se = floor_window / (sd_gap * math.sqrt(2))        # the bug's n
    assert fit["se"] == pytest.approx(correct_se)
    relative_gap = (correct_se - old_wrong_se) / old_wrong_se
    assert relative_gap == pytest.approx(math.sqrt(2) - 1.0, rel=1e-9)
    assert relative_gap > 0.4                                    # "41% at 2" (#303's ticket)


def test_the_slope_se_at_n_equals_53_is_immaterially_different_from_root_n():
    """#303, defect 6, the n=53 case named in the ticket: at a season's worth of events the
    root n and root (n - 1) denominators are close, so the fix moves `se` by about 1%."""
    gaps = [float(i - 26) * 5.0 for i in range(53)]               # 53 distinct, mean-zero-ish
    rows_ = pl.DataFrame({
        "season": [2026] * 53, "week": [1] * 53, "game_id": [f"g{i}" for i in range(53)],
        "net_gap": gaps, "adjusted_move": [0.132 * g for g in gaps], "window_days": [7.0] * 53})
    fit = ss.study_fit(rows_, floor_per_root_day=0.4)
    floor_window = 0.4 * math.sqrt(7.0)
    sd_gap = statistics.stdev(gaps)
    correct_se = floor_window / (sd_gap * math.sqrt(52))
    old_wrong_se = floor_window / (sd_gap * math.sqrt(53))
    assert fit["se"] == pytest.approx(correct_se)
    relative_gap = abs(correct_se - old_wrong_se) / old_wrong_se
    assert relative_gap < 0.02                                    # "immaterial at 53"


def test_the_study_mde_before_the_run_is_stated_from_the_events_gap_spread():
    """With no archived event the MDE line is still stated: the pinned file's gap spread,
    the noise floor per window, and the event count a season carries -- over root (n - 1),
    matching `study_fit`'s own denominator (#303)."""
    line = ss.study_mde(n=53, sd_gap=70.0, window_days=7.0, floor_per_root_day=0.4)
    assert line == pytest.approx((experiment.t_quantile(0.975, 52) + 0.8416)
                                 * 0.4 * math.sqrt(7.0) / (70.0 * math.sqrt(52)), abs=1e-4)


def _fit(*, beta, se, n, sd_gap=float("nan"), floor_window=float("nan")):
    """A `study_fit`-shaped summary built directly -- rule 15's fixture for a function whose
    own input is a summary dict, not a frame `study_fit` would have to be driven through."""
    return {"n": float(n), "beta": beta, "se": se,
            "mde": experiment.minimum_detectable_effect(se, n), "benchmark": ss.BENCHMARK,
            "t_vs_benchmark": (beta - ss.BENCHMARK) / se if se else float("nan"),
            "sd_gap": sd_gap, "floor_window": floor_window}


def _interval(fit):
    """The coefficient's interval as the verdict reads it: `experiment.t_interval` on the fit's
    `se`, with the event games as the cluster count (#346: `starter_study` has no interval of
    its own any longer)."""
    return experiment.t_interval(fit["beta"], fit["se"], int(fit["n"]))


def test_verdict_adopts_when_the_lower_bound_clears_delta():
    """n=10, se=0.002: the MDE clears delta (about 0.0062 against 0.0075) and beta=0.02 puts
    the interval's lower bound, not the point estimate, above delta -- ADOPT reads the bound."""
    fit = _fit(beta=0.02, se=0.002, n=10)
    assert fit["mde"] < ss.DELTA
    lo, _hi = _interval(fit)
    assert lo > ss.DELTA
    label, sentence = ss.verdict(fit)
    assert label == "ADOPT"
    assert f"{lo:+.4f}" in sentence
    assert "route back opens" in sentence and "-qb mark" in sentence


def test_verdict_shows_when_positive_but_the_lower_bound_does_not_clear_delta():
    """Same n and se as the ADOPT fixture, beta lowered so the lower bound is positive but at
    or below delta -- 'a real effect too small to price', the ADOPTED comment's own words."""
    fit = _fit(beta=0.009, se=0.002, n=10)
    assert fit["mde"] < ss.DELTA
    lo, _hi = _interval(fit)
    assert 0 < lo <= ss.DELTA
    label, sentence = ss.verdict(fit)
    assert label == "SHOW"
    assert "too small to price" in sentence and "harness-only" in sentence


def test_verdict_removes_when_the_interval_excludes_zero_negatively():
    """Same n and se, beta negative enough that the whole interval sits below zero -- the
    market moving the wrong way on a downgrade, REMOVE's own condition."""
    fit = _fit(beta=-0.02, se=0.002, n=10)
    assert fit["mde"] < ss.DELTA
    _lo, hi = _interval(fit)
    assert hi < 0
    label, sentence = ss.verdict(fit)
    assert label == "REMOVE"
    assert "Exhibit" in sentence and "ADR-0007" in sentence and "hub.models" in sentence


def test_verdict_is_not_runnable_when_the_mde_exceeds_delta_and_names_the_events_needed():
    """A wide se (0.01 against 0.002 above) pushes the MDE above delta regardless of the point
    estimate -- no branch reads the interval, and the sentence names the event games delta
    needs, off the same sd_gap/floor_window `study_events_needed` itself reads."""
    fit = _fit(beta=0.132, se=0.01, n=10, sd_gap=66.3, floor_window=0.4 * math.sqrt(7.0))
    assert fit["mde"] > ss.DELTA
    needed = ss.study_events_needed(fit["sd_gap"], fit["floor_window"])
    assert needed is not None
    label, sentence = ss.verdict(fit)
    assert label == "NOT-RUNNABLE"
    assert "No branch below is read" in sentence
    assert str(needed) in sentence


def test_verdict_is_not_runnable_with_no_usable_spread_and_says_so():
    """No rows at all -- `study_fit`'s own n<2 shape, sd_gap and floor_window both NaN. The
    sentence says the event count cannot be stated rather than printing a stale or NaN one."""
    fit = _fit(beta=float("nan"), se=float("nan"), n=0)
    label, sentence = ss.verdict(fit)
    assert label == "NOT-RUNNABLE"
    assert "cannot be stated from these inputs" in sentence


def test_the_gap_sd_restatement_flag_is_silent_inside_the_band_and_fires_outside_it():
    """50-85 inclusive, the band the per-season gap sds (50.3-80.4) DELTA was derived from
    round out to; a value the pre-registration's own worked example would have flagged."""
    assert ss.gap_sd_restatement_flag(66.3) is None
    assert ss.gap_sd_restatement_flag(50.0) is None
    assert ss.gap_sd_restatement_flag(85.0) is None
    below = ss.gap_sd_restatement_flag(49.9)
    above = ss.gap_sd_restatement_flag(85.1)
    assert below is not None and "derivation flagged for restatement" in below
    assert above is not None and "derivation flagged for restatement" in above


def test_the_benchmark_reading_is_replication_below_or_above():
    """Three fitted intervals against 0.132 -- containing it, entirely below, entirely above.
    None of the three verdict sentences above (ADOPT/SHOW/REMOVE) mention 0.132: the reading
    is informational and never part of the rule."""
    assert (ss.benchmark_reading(0.132, 0.10, 0.16)
            == "replication (the interval contains 0.132)")
    assert ss.benchmark_reading(0.05, 0.02, 0.09) == "below 0.132"
    assert ss.benchmark_reading(0.20, 0.15, 0.25) == "above 0.132"


def test_study_events_needed_matches_the_mde_search_it_runs():
    """Constructed against the search itself rather than a hardcoded n: the returned count's
    own MDE clears delta and one fewer event game's does not. The denominator is n - 1,
    matching `study_fit`'s own (#303)."""
    sd_gap, floor_window = 66.3, 0.4 * math.sqrt(7.0)
    n = ss.study_events_needed(sd_gap, floor_window)
    assert n is not None
    se_n = floor_window / (sd_gap * math.sqrt(n - 1))
    se_prev = floor_window / (sd_gap * math.sqrt(n - 2))
    assert experiment.minimum_detectable_effect(se_n, n) <= ss.DELTA
    assert experiment.minimum_detectable_effect(se_prev, n - 1) > ss.DELTA


def test_study_events_needed_is_none_with_no_usable_spread():
    assert ss.study_events_needed(float("nan"), float("nan")) is None
    assert ss.study_events_needed(0.0, 1.0) is None


def test_the_noise_floor_excludes_a_change_the_chart_dates_between_the_polls():
    """#330: `starters` is the depth-chart frame `odds.load_qb_starters` returns -- `dt` the
    chart's own timestamp, published day by day through the week -- and never this module's
    own team-game rows, whose `date` is *kickoff*. A real archive's polls of a game all
    precede that game's own kickoff, so a change dated at kickoff is invisible to every one
    of them and a starters frame built off kickoff dates can never exclude the interval it
    exists to exclude -- which is exactly the shape the superseded fixture missed, dating a
    team's *other* game between two polls of this one, a shape no real archive has.

    Same three pre-kickoff polls of DAL-PHI, KC-LAC and SF-SEA throughout: a chart entry
    dated *between* the DAL-PHI polls excludes that interval (KC-LAC and SF-SEA are the two
    left); the identical chart dating the same change *at* kickoff -- after both polls --
    does not, because neither poll's as-of lookup ever sees it."""
    g1, g2, g3 = "2026_01_DAL_PHI", "2026_01_KC_LAC", "2026_01_SF_SEA"
    tue = dt.datetime(2026, 8, 25, 4, 58)
    thu = dt.datetime(2026, 8, 28, 2, 1)
    kickoff = dt.datetime(2026, 9, 13, 17, 0)                    # well after both polls
    polls_ = pl.DataFrame({
        "game_id": [g1, g1, g2, g2, g3, g3],
        "close_spread": [-3.0, -3.5, -3.0, -3.5, 1.0, 1.5],
        "captured_at": [tue, thu, tue, thu, tue, thu],
        "week": [1, 1, 1, 1, 1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    base = [(dt.datetime(2026, 8, 1), "PHI", "qb-phi"), (dt.datetime(2026, 8, 1), "DAL", "qb-dal-1"),
            (dt.datetime(2026, 8, 1), "KC", "qb-kc"), (dt.datetime(2026, 8, 1), "LAC", "qb-lac"),
            (dt.datetime(2026, 8, 1), "SF", "qb-sf"), (dt.datetime(2026, 8, 1), "SEA", "qb-sea")]

    def _chart(extra):
        rows_ = [*base, extra]
        return pl.DataFrame({"dt": [r[0] for r in rows_], "team": [r[1] for r in rows_],
                             "qb": [r[2] for r in rows_]})

    mid_week = _chart((dt.datetime(2026, 8, 26), "DAL", "qb-dal-2"))
    floor, floor_games, not_applied = ss.noise_floor_per_root_day([(2026, polls_, mid_week)])
    assert not_applied == ()
    assert floor_games == 2 and math.isfinite(floor)      # KC-LAC and SF-SEA survive

    at_kickoff = _chart((kickoff, "DAL", "qb-dal-2"))
    floor2, floor_games2, not_applied2 = ss.noise_floor_per_root_day([(2026, polls_, at_kickoff)])
    assert not_applied2 == ()
    assert floor_games2 == 3 and math.isfinite(floor2)    # neither poll sees the change


def test_the_noise_floor_with_no_starters_names_the_season_not_applied():
    """The degradation `noise_floor_per_root_day` shares with `noise_floor_report`: with no
    chart for a season, that season's live intervals are still counted -- not dropped as
    unknown -- and the season is named in the returned `not_applied` tuple rather than
    letting a caller print an unconditioned number under the same-quarterback label."""
    polls_ = pl.DataFrame({
        "game_id": ["2026_01_DAL_PHI", "2026_01_DAL_PHI"],
        "close_spread": [-3.0, -3.5],
        "captured_at": [dt.datetime(2026, 8, 25, 4, 58), dt.datetime(2026, 8, 28, 2, 1)],
        "week": [1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    _floor, floor_games, not_applied = ss.noise_floor_per_root_day([(2026, polls_, None)])
    assert not_applied == (2026,)
    assert floor_games == 1


def test_the_noise_floor_covers_every_season_with_its_own_chart():
    """#331: `odds._starter_at` is a backward as-of join, so a chart fetched for one season
    cannot condition another season's polls -- fetching once for the last season only left
    every earlier season's intervals with `same_qb` null, dropped as unknown, and the run
    line still called the result same-quarterback. One `(season, polls, starters)` triple
    per season: with a chart for both 2025 and 2026, both seasons' games reach `floor_games`
    -- neither silently drops out for lacking the other season's chart; with a chart for
    2026 only, 2025's game is still counted (unconditioned rather than dropped) and 2025 is
    the one named in `not_applied`."""
    g2025, g2026_stable, g2026_change = "2025_01_AA_BB", "2026_01_KC_LAC", "2026_01_DAL_PHI"
    tue, thu = dt.datetime(2026, 8, 25, 4, 58), dt.datetime(2026, 8, 28, 2, 1)
    polls_2025 = pl.DataFrame({
        "game_id": [g2025, g2025], "close_spread": [1.0, 1.5],
        "captured_at": [dt.datetime(2025, 8, 25, 4, 58), dt.datetime(2025, 8, 28, 2, 1)],
        "week": [1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    polls_2026 = pl.DataFrame({
        "game_id": [g2026_stable, g2026_stable, g2026_change, g2026_change],
        "close_spread": [-3.0, -3.5, 2.0, 2.5], "captured_at": [tue, thu, tue, thu],
        "week": [1, 1, 1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    chart_2025 = pl.DataFrame({
        "dt": [dt.datetime(2025, 8, 1), dt.datetime(2025, 8, 1)], "team": ["AA", "BB"],
        "qb": ["qb-aa", "qb-bb"]})
    chart_2026 = pl.DataFrame({
        "dt": [dt.datetime(2026, 8, 1)] * 4 + [dt.datetime(2026, 8, 26)],
        "team": ["KC", "LAC", "PHI", "DAL", "DAL"],
        "qb": ["qb-kc", "qb-lac", "qb-phi", "qb-dal-1", "qb-dal-2"]})

    both = ss.noise_floor_per_root_day(
        [(2025, polls_2025, chart_2025), (2026, polls_2026, chart_2026)])
    floor, floor_games, not_applied = both
    assert not_applied == ()
    assert floor_games == 2 and math.isfinite(floor)      # 2025's game and KC-LAC both count

    one = ss.noise_floor_per_root_day(
        [(2025, polls_2025, None), (2026, polls_2026, chart_2026)])
    floor2, floor_games2, not_applied2 = one
    assert not_applied2 == (2025,)
    assert floor_games2 == 2 and math.isfinite(floor2)    # 2025's game still counted


# --- typed results (#345): the CLI's own computation, extracted so a test can assert on
# values instead of driving `main` and parsing sentences.


def test_same_quarterback_floor_label_applied_when_every_season_with_polls_has_a_chart():
    label, note = ss.same_quarterback_floor_label(2, (), [])
    assert label == "same-quarterback floor" and note == ""


def test_same_quarterback_floor_label_applied_when_no_season_has_polls_to_condition():
    """`n_seasons == 0` reads the same as `not not_applied`: nothing to try is not a failure
    to try, and the label says nothing failed rather than naming an empty list."""
    label, note = ss.same_quarterback_floor_label(0, (), [])
    assert label == "same-quarterback floor" and note == ""


def test_same_quarterback_floor_label_is_partial_when_some_seasons_charts_fail():
    label, note = ss.same_quarterback_floor_label(
        2, (2025,), ["2025 (ConnectionError: no network)"])
    assert label == "same-quarterback floor, PARTIAL"
    assert "same-quarterback NOT applied for 2025" in note and "ConnectionError" in note


def test_same_quarterback_floor_label_is_not_applied_when_every_seasons_chart_fails():
    label, note = ss.same_quarterback_floor_label(
        2, (2025, 2026), ["2025 (ConnectionError: a)", "2026 (ConnectionError: b)"])
    assert label == "all-games floor, SAME-QUARTERBACK NOT APPLIED"
    assert "2025" in note and "2026" in note


def test_study_report_returns_typed_results_matching_its_own_components(rows):
    """Built the same way `main`'s `--study` block builds its inputs, with no depth chart for
    the fixture's one season. Calling `study_report`'s own components by hand on the same
    inputs is the check that the typed result computes nothing differently from what `main`
    used to compute inline before #345."""
    tg = se.team_games(rows)
    ev = se.events(tg)
    games = se.event_games(se.in_season_events(ev))
    season_games = games
    study_parts: list[tuple[int, pl.DataFrame, pl.DataFrame | None]] = [(2026, ARCHIVE, None)]
    qb_notes = ["2026 (ConnectionError: no chart)"]

    rep = ss.study_report(study_parts, qb_notes, ev, games, ARCHIVE, season_games, tg)
    assert isinstance(rep, ss.StudyReport)

    floor, floor_games, not_applied = ss.noise_floor_per_root_day(study_parts)
    label, qb_note = ss.same_quarterback_floor_label(len(study_parts), not_applied, qb_notes)
    assert rep.floor == floor and rep.floor_games == floor_games
    assert rep.label == label and rep.qb_note == qb_note
    assert rep.unplayed == ss.unplayed_study_games(ARCHIVE, season_games, tg)

    rows_ = ss.study_rows(ARCHIVE, season_games, tg,
                          floor_per_root_day=floor if math.isfinite(floor) else None)
    assert rep.established == (not rows_.is_empty())
    assert rep.n_events == rows_.height
    fit = ss.study_fit(rows_, floor_per_root_day=rep.used)
    for key, want in fit.items():
        got = rep.fit[key]
        assert got == pytest.approx(want, nan_ok=True), key
    assert (rep.verdict_label, rep.verdict_sentence) == ss.verdict(fit)
    if rep.established:
        lo, hi = _interval(fit)
        assert rep.benchmark_sentence == ss.benchmark_reading(fit["beta"], lo, hi)
    else:
        assert rep.benchmark_sentence is None


def test_a_frozen_price_that_is_also_the_only_pre_game_price_has_no_window(rows):
    """The 2026 shape the maintainer's note ("weeks 1-5 stay censored") did not name: the
    back-filled lookahead predates the season, so a week-2 event *has* a frozen price -- and
    the same capture is its last one before the game day. It is not censored, and it is not
    a move of zero; `priced` marks it `windowless` and `study_rows` refuses it. The positive
    control is the same game with a later poll, which has a window. Mutation: dropping the
    `windowless` term from `_priced_both_ways` leaves the first block red."""
    tg, ev = pbp_events(rows)
    games = se.event_games(se.with_values(ev, se.team_games(rows)))
    lone = polls([("2026_03_LA_KC", 3.0, utc(15), 3)])
    got = se.priced(lone, games).filter(pl.col("game_id") == "2026_03_LA_KC")
    assert got["censored"].to_list() == [False] and got["windowless"].to_list() == [True]
    assert ss._priced_both_ways(lone, games).is_empty()
    assert ss.study_rows(lone, games, tg).is_empty()

    later = polls([("2026_03_LA_KC", 3.0, utc(15), 3), ("2026_03_LA_KC", 2.0, utc(24), 3)])
    got = se.priced(later, games).filter(pl.col("game_id") == "2026_03_LA_KC")
    assert got["windowless"].to_list() == [False]
    assert ss._priced_both_ways(later, games)["game_id"].to_list() == ["2026_03_LA_KC"]


def test_an_event_whose_starter_the_pinned_file_cannot_value_has_no_gap_not_a_zero(rows):
    """A change whose arriving starter the pinned file holds no value for -- every 2026 event
    past the pin -- joins to a null gap, and its game has a null net gap: unknown, never zero.
    The study refuses it and counts it (`unvalued_study_games`); the game where the file does
    value both starters is untouched. Mutation: restoring `fill_null(0.0)` over the changed
    side's gap (the old `net_gap`) makes `net_gap` 0.0 and the first assertion red."""
    tg, ev = pbp_events(rows)
    file_tg = se.team_games(rows)
    blind = file_tg.filter(pl.col("game_id") != "2026_03_LA_KC")        # no values for week 3
    games = se.event_games(se.with_values(ev, blind))
    by_game = dict(zip(games["game_id"], games["net_gap"], strict=True))
    assert by_game["2026_03_LA_KC"] is None
    assert by_game["2026_04_DEN_KC"] is None            # its departing starter is the week-3 one
    valued = se.event_games(se.with_values(ev, file_tg))
    assert valued["net_gap"].null_count() == 0
    assert ss.unvalued_study_games(ARCHIVE, games) == 2
    assert ss.unvalued_study_games(ARCHIVE, valued) == 0
    assert ss.study_rows(ARCHIVE, games, tg).is_empty()


def test_the_verdict_reads_experiments_interval_at_its_edge():
    """#346: the verdict has no interval of its own. A fit whose lower bound -- on
    `experiment.t_interval`, the t reference at n - 1 degrees of freedom -- sits just above
    DELTA is ADOPT, and one just below is SHOW. The edge is where a different reference (the
    normal's 1.96 in place of t(0.975, 9) = 2.26) would flip the answer, so this fails if the
    study ever reads an interval other than the Gate's."""
    se, n = 0.002, 10
    half = experiment.t_interval(0.0, se, n)[1]
    assert half == pytest.approx(experiment.t_quantile(0.975, n - 1) * se)
    just_over = _fit(beta=ss.DELTA + half + 1e-6, se=se, n=n)
    just_under = _fit(beta=ss.DELTA + half - 1e-6, se=se, n=n)
    assert ss.verdict(just_over)[0] == "ADOPT"
    assert ss.verdict(just_under)[0] == "SHOW"
    normal_half = 1.959964 * se
    assert normal_half < half
    assert ss.verdict(_fit(beta=ss.DELTA + normal_half + 1e-6, se=se, n=n))[0] == "SHOW"
