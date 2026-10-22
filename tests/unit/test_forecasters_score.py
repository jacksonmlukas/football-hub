"""`hub.forecasters.score`: H1's encompassing test, H2's non-inferiority test, the estimates.

Each statistic is held to a number computed here by hand (`math.log`, `statistics.stdev`, the
published t table) rather than to the module's own arithmetic, and each test plants the
condition its check exists to detect (`docs/method.md` rules 15 and 18): deviations that are
pure noise, a model that holds information the betting market lacks, a contestant missing
games the betting market got badly wrong, a probability outside the clip, and a margin on
either side of the bound.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.forecasters import config as study_config
from hub.forecasters import score
from hub.forecasters.runner import NotPinned
from hub.forecasters.score import Analysis, Game

CFG = study_config.load()                      # the committed file: four models, llm1..llm4
AWARE = [f"{m.name}:line_aware" for m in CFG.llms]
PINNED = Analysis(delta=0.02, h1_min_deviation_sd=0.05,
                  h2_implementation="hub.forecasters.score@abcdef0")
# A burn-in that found every line-aware model estimable at PINNED's threshold.
ALL_IN = score.BurnIn(season=2026, week=8, threshold=0.05,
                      deviation_sd=dict.fromkeys(AWARE, 0.3))

# Student's t, 0.95 quantile, from the published table -- not from the module's bisection.
T95 = {2: 2.919986, 11: 1.795885}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _world(rng, *, weeks, per_week, kind, sd):
    """Games and one model's forecasts.

    `noise`: the betting market is the truth and the model adds N(0, sd) to its log-odds --
    deviations that carry nothing, so c = 0. `knows`: the truth is the betting market's
    log-odds plus N(0, sd) the betting market missed, and the model forecasts the truth --
    deviations that deserve full weight, so c = 1.
    """
    games, fc = [], {}
    for w in range(1, weeks + 1):
        for i in range(per_week):
            lm = rng.normal(0.1, 0.9)
            if kind == "noise":
                lq, lp = lm, lm + rng.normal(0.0, sd)
            else:
                lq = lm + rng.normal(0.0, sd)
                lp = lq
            y = int(rng.random() < _sig(lq))
            gid = f"w{w}g{i}"
            games.append(Game(gid, w, 20 + 3 * y, 20 + 3 * (1 - y), float(_sig(lm))))
            fc[gid] = float(_sig(lp))
    return games, fc


# --- refusal -----------------------------------------------------------------------------

def test_the_committed_analysis_settings_are_the_adopted_ones_and_wait_for_the_h2_pin():
    """#448 (adopted 2026-10-22) items 5 and 6: delta 0.02 nats and the H1 threshold 0.25;
    the H2 implementation pin waits for this lane's merge commit, so scoring still refuses
    -- naming only that."""
    a = score.load_analysis()
    assert a == Analysis(0.02, 0.25, None) == CFG.analysis
    with pytest.raises(NotPinned) as exc:
        score.require_pinned(a, CFG)
    assert "analysis.h2_implementation" in str(exc.value)
    assert "analysis.delta" not in str(exc.value)
    assert "h1_min_deviation_sd" not in str(exc.value)


def test_each_unset_setting_is_named_on_its_own(tmp_path):
    p = tmp_path / "f.toml"
    p.write_text('[analysis]\ndelta = 0.02\nh1_min_deviation_sd = "TBD"\n'
                 'h2_implementation = "hub.forecasters.score@abc1234"\n')
    a = score.load_analysis(p)
    assert a == Analysis(0.02, None, "hub.forecasters.score@abc1234")
    with pytest.raises(NotPinned, match="h1_min_deviation_sd") as exc:
        score.require_pinned(a, CFG)
    assert "analysis.delta" not in str(exc.value)
    assert "h2_implementation" not in str(exc.value)


@pytest.mark.parametrize("pin", ["1.4.2", "parity@abc1234", "hub.forecasters.score@main",
                                 "hub.forecasters.score@abc12", "hub.forecasters.score"])
def test_the_h2_pin_must_name_the_scorer_at_a_commit(pin):
    """#448 item 1: H2 runs on this module pinned at a commit, not at a Parity version or a
    branch name."""
    with pytest.raises(NotPinned, match="h2_implementation"):
        score.require_pinned(Analysis(0.02, 0.25, pin), CFG)
    score.require_pinned(Analysis(0.02, 0.25, "hub.forecasters.score@" + "a" * 40), CFG)


def test_a_non_positive_margin_and_a_missing_table_are_refused(tmp_path):
    with pytest.raises(NotPinned, match=r"analysis\.delta"):
        score.require_pinned(Analysis(0.0, 0.05, PINNED.h2_implementation), CFG)
    p = tmp_path / "f.toml"
    p.write_text("[study]\n")
    assert score.load_analysis(p) == Analysis()


def test_a_study_without_four_models_is_refused():
    three = study_config.StudyConfig(llms=CFG.llms[:3], season_cap_usd=1.0, home_rate=0.55,
                                     home_rate_seasons="x", est_input_tokens=1)
    with pytest.raises(NotPinned, match="four"):
        score.require_pinned(PINNED, three)
    score.require_pinned(PINNED, CFG)                       # four, all set: no refusal


def test_the_cli_refuses_in_one_sentence_while_unpinned(tmp_path):
    lines: list[str] = []
    rc = score.main(["--forecasts", str(tmp_path), "--games", str(tmp_path / "g.jsonl")],
                    echo=lines.append)
    assert rc == 3
    assert len(lines) == 1 and lines[0].startswith("refused: ")


def test_analyse_itself_refuses_while_unpinned():
    with pytest.raises(NotPinned):
        score.analyse([], {}, analysis=Analysis(), config=CFG, burn_in=ALL_IN)


# --- inputs: ties, the week played, the clip, records ------------------------------------

def test_a_tie_and_an_unplayed_game_are_not_scored():
    assert Game("t", 1, 17, 17, 0.5).outcome is None
    assert Game("u", 1, None, None, 0.5).outcome is None
    assert Game("h", 1, 21, 17, 0.5).outcome == 1 and Game("a", 1, 17, 21, 0.5).outcome == 0
    games = [Game("t", 1, 17, 17, 0.6), Game("h", 1, 21, 17, 0.6)]
    s = score.describe(games, {"x": {"t": 0.9, "h": 0.7}}, "x")
    assert s.log_loss.games == 1 and s.scorable == 1


def test_a_postponed_game_counts_in_the_week_it_was_played():
    """`Game.week` is the week played: a game forecast in week 5's run and played in week 6
    is a week-6 value, so week 6 is the only week."""
    s = score.describe([Game("g", 6, 21, 17, 0.6)], {"x": {"g": 0.7}}, "x")
    assert s.log_loss.weeks == 1
    assert s.log_loss.estimate == pytest.approx(-math.log(0.7) + math.log(0.6))


def test_the_contestant_is_clipped_and_the_betting_market_is_not():
    """*Forecast procedure*, step 4, restated 2026-10-22 (#448 item 2). A contestant at 0.0 on
    a home win scores as 0.01; the betting market at 0.995 is the benchmark and scores as
    0.995: Delta = ln(100) + ln(0.995) -- not ln(100) + ln(0.99) (the market clipped too), and
    not the unclipped 34.5."""
    s = score.describe([Game("g", 1, 21, 17, 0.995)], {"x": {"g": 0.0}}, "x")
    assert s.log_loss.estimate == pytest.approx(math.log(100) + math.log(0.995), abs=1e-12)
    assert s.abs_gap.estimate == pytest.approx(0.985)
    assert list(score.clip([0.0, 0.5, 1.0])) == [0.01, 0.5, 0.99]
    lost = score.describe([Game("g", 1, 17, 21, 0.995)], {"x": {"g": 0.0}}, "x")
    assert lost.log_loss.estimate == pytest.approx(math.log(0.005) - math.log(0.99), abs=1e-12)


def test_records_are_read_as_the_runner_writes_them(tmp_path):
    d = tmp_path / "2026"
    d.mkdir()
    rows = [{"game_id": "g1", "contestant": "llm1:line_aware", "p": 0.6, "excluded": None},
            {"game_id": "g2", "contestant": "llm1:line_aware", "p": None, "excluded": None},
            {"game_id": "g3", "contestant": "llm1:line_aware", "p": 0.4,
             "excluded": "finished_within_60min_of_kickoff"},
            {"game_id": "g1", "contestant": "coin_flip", "p": 0.5, "excluded": None}]
    (d / "2026-10-25.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    (tmp_path / "exclusions.jsonl").write_text(json.dumps({"game_id": "g9"}) + "\n")
    assert score.load_forecasts(tmp_path) == {"llm1:line_aware": {"g1": 0.6},
                                              "coin_flip": {"g1": 0.5}}
    (d / "2026-10-26.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="two forecasts"):
        score.load_forecasts(tmp_path)


# --- H2: a hand computation on a fixed dataset ------------------------------------------

# Three scored weeks for "x". g6 is a game the betting market got badly wrong and "x" did
# not forecast; g7 has no moneyline; week 4 has no forecast from "x" at all. A paired
# comparison uses none of them.
FIXED = [Game("g1", 1, 24, 17, 0.6), Game("g2", 1, 10, 13, 0.5),
         Game("g3", 2, 30, 3, 0.7), Game("g6", 2, 0, 35, 0.9), Game("g7", 2, 20, 10, None),
         Game("g4", 3, 7, 14, 0.3), Game("g5", 3, 28, 27, 0.55),
         Game("g8", 4, 14, 10, 0.5)]
FIXED_P = {"g1": 0.7, "g2": 0.4, "g3": 0.6, "g7": 0.8, "g4": 0.2, "g5": 0.5}


def _ll(p, y):
    return -math.log(p) if y else -math.log(1 - p)


def _hand():
    w1 = ((_ll(0.7, 1) - _ll(0.6, 1)) + (_ll(0.4, 0) - _ll(0.5, 0))) / 2
    w2 = _ll(0.6, 1) - _ll(0.7, 1)
    w3 = ((_ll(0.2, 0) - _ll(0.3, 0)) + (_ll(0.5, 1) - _ll(0.55, 1))) / 2
    weekly = [w1, w2, w3]
    return statistics.fmean(weekly), statistics.stdev(weekly) / math.sqrt(3)


def test_delta_is_the_mean_of_paired_weekly_means_by_hand():
    delta_hat, se = _hand()
    s = score.describe(FIXED, {"x": FIXED_P}, "x")
    assert s.log_loss.weeks == 3 and s.log_loss.games == 5
    assert s.log_loss.estimate == pytest.approx(delta_hat, abs=1e-12)
    assert s.log_loss.se == pytest.approx(se, abs=1e-12)
    half = 4.302653 * se                                     # t(0.975, 2), the table
    assert (s.log_loss.lo, s.log_loss.hi) == pytest.approx((delta_hat - half,
                                                            delta_hat + half), abs=1e-5)
    assert s.missing == 2 and s.scorable == 8                # g6 and g8 of eight


def test_the_90_percent_interval_is_reported_beside_the_95_and_matches_the_test():
    """#448 item 4: H2 is one-sided at 0.05, so the matching two-sided interval is the 90%
    one, t(0.95, W - 1) wide -- from the table, not the 95%'s t(0.975). Its lower end is the
    bound the test rejects against: non-inferior at delta iff lo_90 > -delta."""
    delta_hat, se = _hand()
    s = score.describe(FIXED, {"x": FIXED_P}, "x")
    half = T95[2] * se
    assert (s.log_loss.lo90, s.log_loss.hi90) == pytest.approx((delta_hat - half,
                                                                delta_hat + half), abs=1e-5)
    assert s.log_loss.lo < s.log_loss.lo90 < s.log_loss.hi90 < s.log_loss.hi
    for delta in (-s.log_loss.lo90 + 1e-4, -s.log_loss.lo90 - 1e-4):
        assert score.noninferiority(s, delta).noninferior == (s.log_loss.lo90 > -delta)
    one = score.describe([Game("g", 1, 21, 17, 0.6)], {"x": {"g": 0.7}}, "x")
    assert math.isnan(one.log_loss.lo90) and math.isnan(one.log_loss.hi90)


# --- the inputs without a games file: the records' market, nflverse's results -------------

def _rec(gid, contestant, *, p=0.6, market=None, snap="s1", start="2026-11-01 14:00:00+00:00"):
    return {"game_id": gid, "contestant": contestant, "p": p, "excluded": None,
            "market_home_win_prob": market, "market_snapshot_id": snap, "run_start": start}


def test_the_benchmark_is_read_off_the_records_and_the_first_snapshot_wins(tmp_path):
    """A game's market is the forecast-time value its records carry (#448 item 10), blind
    records with none (a partial run) notwithstanding. A crash between snapshot and records
    can leave a game on two snapshots: the earliest run's is the benchmark, and it is
    counted."""
    rows = [_rec("g1", "llm1:blind"),                                     # partial: no line
            _rec("g1", "llm1:line_aware", market=0.62),
            _rec("g1", "hub", market=0.62),
            _rec("g2", "llm1:line_aware", market=0.55, snap="late",
                 start="2026-11-01 14:30:00+00:00"),
            _rec("g2", "llm2:line_aware", market=0.40, snap="early"),
            _rec("g3", "coin_flip")]                                       # no line at all
    (tmp_path / "day.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = score.load_markets(tmp_path)
    assert m.home == {"g1": 0.62, "g2": 0.40} and m.conflicts == 1


SCHEDULE = pl.DataFrame({
    "game_id": ["b8", "s9", "s10", "tie", "unplayed", "post", "refused", "p1", "old", "w19"],
    "season": [2026] * 8 + [2025, 2026],
    "game_type": ["REG"] * 7 + ["WC", "REG", "REG"],
    "week": [8, 9, 10, 10, 11, 12, 9, 19, 9, 19],
    "home_score": [21, 24, 10, 17, None, 30, 20, 21, 21, 21],
    "away_score": [17, 20, 13, 17, None, 3, 10, 17, 17, 17],
})


def test_the_scored_games_are_nflverse_results_from_the_week_after_the_burn_in(tmp_path):
    """Regular season, the burn-in's season, weeks 9 to 18 by the week played, less the games
    the runner refused; each carries the records' market (None where there was no line)."""
    games = score.games_from_schedule(SCHEDULE, {"s9": 0.6, "post": 0.7}, season=2026,
                                      first_week=9, refused={"refused"})
    by = {g.game_id: g for g in games}
    assert set(by) == {"s9", "s10", "tie", "unplayed", "post"}
    assert by["s9"].market == 0.6 and by["s10"].market is None
    assert by["s9"].outcome == 1 and by["s10"].outcome == 0
    assert by["tie"].outcome is None and by["unplayed"].outcome is None
    assert by["post"].week == 12


def test_brier_and_the_absolute_gap_are_reported_the_same_way():
    s = score.describe(FIXED, {"x": FIXED_P}, "x")
    gaps = [0.1, 0.1, 0.1, 0.1, 0.05]
    weekly_gap = [0.1, 0.1, 0.075]
    assert s.abs_gap.estimate == pytest.approx(statistics.fmean(weekly_gap))
    assert s.abs_gap.games == len(gaps)
    b1 = (((0.7 - 1) ** 2 - (0.6 - 1) ** 2) + (0.4 ** 2 - 0.5 ** 2)) / 2
    b2 = (0.6 - 1) ** 2 - (0.7 - 1) ** 2
    b3 = ((0.2 ** 2 - 0.3 ** 2) + ((0.5 - 1) ** 2 - (0.55 - 1) ** 2)) / 2
    assert s.brier.estimate == pytest.approx(statistics.fmean([b1, b2, b3]))


def test_non_inferiority_rejects_just_inside_the_bound_and_not_just_outside():
    """Reject Delta <= -delta iff (Delta + delta) / se > t(0.95, W - 1), one-sided at 0.05."""
    delta_hat, se = _hand()
    bound = T95[2] * se - delta_hat                          # the delta at which t = crit
    s = score.describe(FIXED, {"x": FIXED_P}, "x")
    inside = score.noninferiority(s, bound + 1e-4)
    outside = score.noninferiority(s, bound - 1e-4)
    assert inside.noninferior and not outside.noninferior
    assert inside.t == pytest.approx((delta_hat + bound + 1e-4) / se, rel=1e-9)
    assert inside.p < score.ALPHA < outside.p


def test_equivalence_needs_both_one_sided_halves():
    delta_hat, se = _hand()
    s = score.describe(FIXED, {"x": FIXED_P}, "x")
    wide = abs(delta_hat) + T95[2] * se + 1e-3
    assert score.noninferiority(s, wide).equivalent
    # Delta-hat < 0 here, so at delta = t * se the upper half rejects and the lower does not.
    upper_only = score.noninferiority(s, T95[2] * se)
    assert upper_only.p_upper < score.ALPHA and not upper_only.noninferior
    assert not upper_only.equivalent


def test_one_week_cannot_be_tested():
    s = score.describe([Game("g", 1, 21, 17, 0.6)], {"x": {"g": 0.7}}, "x")
    r = score.noninferiority(s, 1.0)
    assert not r.noninferior and not r.equivalent and math.isnan(r.p)


def test_h2_claims_only_when_every_model_rejects():
    s = score.describe(FIXED, {"x": FIXED_P}, "x")
    passing = score.noninferiority(s, 1.0)
    failing = score.noninferiority(s, 1e-6)
    assert score.h2_verdict([passing] * 4).claim
    assert not score.h2_verdict([passing] * 3 + [failing]).claim
    assert not score.h2_verdict([]).claim


def test_the_week_block_bootstrap_brackets_the_estimate():
    rng = np.random.default_rng(3)
    games, fc = _world(rng, weeks=12, per_week=10, kind="noise", sd=0.3)
    s = score.describe(games, {"x": fc}, "x", seed=1)
    assert s.log_loss.boot_lo < s.log_loss.estimate < s.log_loss.boot_hi
    assert score.describe(games, {"x": fc}, "x", seed=1) == s           # seeded


# --- H1 ----------------------------------------------------------------------------------

def test_noise_deviations_cover_zero_at_roughly_the_nominal_rate():
    """c = 0 by construction. The one-sided 95% upper bound covers 0 on at least ~95% of
    worlds (the jackknife is conservative), and the test rejects on at most ~5%."""
    rng = np.random.default_rng(452)
    covered = rejected = 0
    reps = 200
    for _ in range(reps):
        games, fc = _world(rng, weeks=12, per_week=10, kind="noise", sd=0.4)
        f = score.encompassing(games, {"x": fc}, "x", in_family=True)
        assert f.estimable and f.weeks == 12
        covered += f.upper >= 0
        rejected += f.p <= score.ALPHA
    assert covered / reps >= 0.90
    assert rejected / reps <= 0.10


def test_a_model_that_knows_what_the_betting_market_missed_gets_full_weight():
    rng = np.random.default_rng(7)
    games, fc = _world(rng, weeks=12, per_week=40, kind="knows", sd=1.0)
    f = score.encompassing(games, {"x": fc}, "x", in_family=True)
    assert f.estimable
    assert f.c == pytest.approx(1.0, abs=0.3)
    assert f.b == pytest.approx(1.0, abs=0.3)
    assert f.p < 0.001 and f.upper > f.c


def test_h1_detects_the_informed_model_after_holm_and_not_the_noisy_ones():
    rng = np.random.default_rng(11)
    games, knows = _world(rng, weeks=12, per_week=40, kind="knows", sd=1.0)
    m = np.array([g.market for g in games])
    fc = {AWARE[0]: knows}
    for j, name in enumerate(AWARE[1:]):
        noise = _sig(np.log(m / (1 - m)) + np.random.default_rng(j).normal(0, 0.4, m.size))
        fc[name] = {g.game_id: float(q) for g, q in zip(games, noise, strict=True)}
    fits = [score.encompassing(games, fc, c, in_family=True) for c in AWARE]
    r = score.h1_verdict(fits)
    assert [x.detected for x in r.rows] == [True, False, False, False]
    assert r.rows[0].holm_p == pytest.approx(min(1.0, 4 * fits[0].p))
    assert "deviations carry information" in r.sentence


def test_a_model_the_burn_in_found_copying_leaves_h1_and_the_holm_family():
    rng = np.random.default_rng(5)
    games, knows = _world(rng, weeks=12, per_week=40, kind="knows", sd=1.0)
    fits = [score.encompassing(games, {"k": knows, "c": knows}, "k", in_family=True),
            score.encompassing(games, {"k": knows, "c": knows}, "c", in_family=False)]
    # Out of the family on the burn-in's word, however far its scored forecasts stray.
    assert fits[1].deviation_sd > 0.5
    assert fits[1].status == score.NOT_ESTIMABLE and math.isnan(fits[1].c)
    r = score.h1_verdict(fits)
    assert r.rows[0].holm_p == pytest.approx(fits[0].p)                # a family of one
    assert math.isnan(r.rows[1].holm_p) and not r.rows[1].detected


# --- the H1 family is the burn-in's ------------------------------------------------------

def _burn_in(tmp_path, **kw):
    raw = {"season": 2026, "burn_in_week": 8, "h1_min_deviation_sd": 0.05,
           "deviation_sd": dict.fromkeys(AWARE, 0.3)} | kw
    p = tmp_path / "burn-in.json"
    p.write_text(json.dumps(raw))
    return p


def test_the_family_is_the_burn_in_models_at_or_above_the_pinned_threshold(tmp_path):
    b = score.load_burn_in(_burn_in(tmp_path, deviation_sd={
        AWARE[0]: 0.30, AWARE[1]: 0.05, AWARE[2]: 0.0499, AWARE[3]: 0.0}))
    assert b.first_scored_week == 9 and b.season == 2026
    assert score.h1_family(b, AWARE, 0.05) == {AWARE[0], AWARE[1]}


def test_a_burn_in_record_that_is_absent_or_does_not_fit_the_pin_is_refused(tmp_path):
    with pytest.raises(score.BurnInRefused, match="no burn-in estimability record"):
        score.load_burn_in(tmp_path / "absent.json")
    off = score.load_burn_in(_burn_in(tmp_path, h1_min_deviation_sd=0.2))
    with pytest.raises(score.BurnInRefused, match=r"measured against threshold 0\.2"):
        score.h1_family(off, AWARE, 0.05)
    short = score.load_burn_in(_burn_in(tmp_path, deviation_sd={AWARE[0]: 0.3}))
    with pytest.raises(score.BurnInRefused, match=AWARE[1]):
        score.h1_family(short, AWARE, 0.05)


def test_holm_runs_over_the_burn_in_family_not_over_what_the_scored_games_show():
    """#448 item 3. The burn-in found all four estimable; on the scored games three of them
    copy the betting market exactly (deviation SD 0, so their fits cannot exist). They stay
    in the family with p = 1: the informed model's Holm p is 4 p, not the p a family of one
    -- measured on the scored games -- would give it."""
    rng = np.random.default_rng(11)
    games, knows = _world(rng, weeks=12, per_week=40, kind="knows", sd=1.0)
    copy = {g.game_id: float(g.market) for g in games}
    fc = {AWARE[0]: knows, **dict.fromkeys(AWARE[1:], copy)}
    r = score.analyse(games, fc, analysis=PINNED, config=CFG, burn_in=ALL_IN)
    first = r.h1.rows[0]
    assert first.fit.estimable and first.fit.p > 0
    assert first.holm_p == pytest.approx(min(1.0, 4 * first.fit.p))
    assert [x.fit.status == score.NOT_ESTIMABLE for x in r.h1.rows] == [False] * 4
    assert all(x.fit.deviation_sd == 0 for x in r.h1.rows[1:])
    # And the other way: the burn-in left AWARE[1] out, so it is out whatever it does now.
    fc[AWARE[1]] = knows
    out1 = score.BurnIn(2026, 8, 0.05, {**ALL_IN.deviation_sd, AWARE[1]: 0.01})
    r = score.analyse(games, fc, analysis=PINNED, config=CFG, burn_in=out1)
    assert r.h1.rows[1].fit.status == score.NOT_ESTIMABLE and not r.h1.rows[1].detected


def test_the_jackknife_se_is_the_leave_one_week_out_formula_by_hand():
    rng = np.random.default_rng(13)
    games, fc = _world(rng, weeks=6, per_week=30, kind="knows", sd=0.8)
    f = score.encompassing(games, {"x": fc}, "x", in_family=True)
    cj = list(f.jackknife)
    w = len(cj)
    bar = statistics.fmean(cj)
    assert w == 6
    assert f.se == pytest.approx(math.sqrt((w - 1) / w * sum((x - bar) ** 2 for x in cj)))
    assert f.t == pytest.approx(f.c / f.se)


def test_a_null_reads_as_an_upper_bound_on_the_weight():
    rng = np.random.default_rng(17)
    games, fc = _world(rng, weeks=12, per_week=10, kind="noise", sd=0.4)
    f = score.encompassing(games, {"x": fc}, "x", in_family=True)
    r = score.h1_verdict([f])
    if not r.rows[0].detected:
        assert f"c <= {f.upper:.2f}" in r.sentence
    assert f.upper == pytest.approx(f.c + T95[11] * f.se, abs=1e-5)


def test_a_fit_that_cannot_exist_is_named_not_estimated():
    games = [Game(f"g{i}", 1 + i % 3, 21 if i % 2 else 17, 17 if i % 2 else 21, 0.5)
             for i in range(12)]
    fc = {g.game_id: (0.9 if g.outcome else 0.1) for g in games}       # separates perfectly
    f = score.encompassing(games, {"x": fc}, "x", in_family=True)
    assert not f.estimable and "did not converge" in f.status
    assert score.h1_verdict([f]).sentence.startswith("H1: no line-aware model is estimable")
    one = score.encompassing(games[:1], {"x": fc}, "x", in_family=True)
    assert "fewer than two games" in one.status
    wk = [g for g in games if g.week == 1]
    assert "two weeks" in score.encompassing(wk, {"x": {g.game_id: 0.3 + 0.04 * i
                                                       for i, g in enumerate(wk)}},
                                             "x", in_family=True).status


def test_holm_by_hand():
    assert score.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert score.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert score.holm([]) == []


# --- the whole analysis ------------------------------------------------------------------

def test_the_cli_scores_a_pinned_study_end_to_end(tmp_path: Path):
    rng = np.random.default_rng(21)
    games, fc = _world(rng, weeks=4, per_week=8, kind="noise", sd=0.3)
    games.append(Game("tie", 2, 20, 20, 0.5))
    (tmp_path / "games.jsonl").write_text("".join(json.dumps(
        {"game_id": g.game_id, "week": g.week, "home_points": g.home_points,
         "away_points": g.away_points, "market": g.market}) + "\n" for g in games))
    out = tmp_path / "forecasts"
    out.mkdir()
    rows = [{"game_id": gid, "contestant": c, "p": p, "excluded": None}
            for c in [*AWARE, "coin_flip"] for gid, p in fc.items()]
    (out / "day.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines: list[str] = []
    burn = ["--burn-in", str(_burn_in(tmp_path))]
    rc = score.main(["--forecasts", str(out), "--games", str(tmp_path / "games.jsonl"), *burn],
                    echo=lines.append, config=CFG, analysis=PINNED)
    assert rc == 0
    s = json.loads(lines[0])
    assert [m["contestant"] for m in s["h1"]["models"]] == AWARE
    assert len(s["h2"]["models"]) == 4 and s["excluded"]["tie"] == 1
    assert {e["contestant"] for e in s["estimates"]} == {*AWARE, "coin_flip"}
    assert s["h2_implementation"] == PINNED.h2_implementation
    assert s["burn_in"] == {"season": 2026, "week": 8, "threshold": 0.05}
    ll = s["h2"]["models"][0]["log_loss_diff"]
    assert ll["lo"] < ll["lo_90"] < ll["hi_90"] < ll["hi"]
    assert score.main(["--forecasts", str(tmp_path / "nope"), "--games", "x", *burn],
                      echo=lines.append, config=CFG, analysis=PINNED) == 2


def test_the_cli_scores_from_the_records_and_the_schedule_with_no_games_file(tmp_path):
    """#448 item 10: no hand-built games file. The records carry the market, nflverse's
    schedule (injected; no network) the results, the burn-in record the window."""
    rng = np.random.default_rng(23)
    games, fc = _world(rng, weeks=4, per_week=8, kind="noise", sd=0.3)
    games = [Game(g.game_id, g.week + 8, g.home_points, g.away_points, g.market)
             for g in games]                                             # weeks 9..12
    frame = pl.DataFrame({
        "game_id": [g.game_id for g in games] + ["burnin", "lost", "refused"],
        "season": [2026] * (len(games) + 3), "game_type": ["REG"] * (len(games) + 3),
        "week": [g.week for g in games] + [8, 12, 12],
        "home_score": [g.home_points for g in games] + [21, 21, 21],
        "away_score": [g.away_points for g in games] + [17, 17, 17]})
    out = tmp_path / "forecasts"
    out.mkdir()
    market = {g.game_id: g.market for g in games}
    rows = [_rec(gid, c, p=p, market=market[gid]) for c in [*AWARE, "coin_flip"]
            for gid, p in fc.items()]
    rows += [_rec("burnin", c, p=0.99, market=0.5) for c in AWARE]      # never scored
    (out / "day.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "exclusions.jsonl").write_text(json.dumps({"game_id": "refused"}) + "\n")
    seasons: list[int] = []

    def schedule(season):
        seasons.append(season)
        return frame
    lines: list[str] = []
    rc = score.main(["--forecasts", str(out), "--burn-in", str(_burn_in(tmp_path))],
                    echo=lines.append, config=CFG, analysis=PINNED, schedule=schedule)
    assert rc == 0 and seasons == [2026]
    s = json.loads(lines[0])
    want = score.describe(games, {AWARE[0]: fc}, AWARE[0])
    got = next(e for e in s["estimates"] if e["contestant"] == AWARE[0])
    assert got["log_loss_diff"]["estimate"] == pytest.approx(want.log_loss.estimate, abs=1e-6)
    assert got["log_loss_diff"]["weeks"] == 4 and got["log_loss_diff"]["games"] == len(games)
    assert got["missing"] == 1 and got["scorable"] == len(games) + 1    # "lost": no record
    assert s["excluded"]["packet_refused"] == 1 and s["market_snapshot_conflicts"] == 0


def test_the_cli_refuses_in_one_sentence_without_the_burn_in_record(tmp_path):
    lines: list[str] = []
    rc = score.main(["--forecasts", str(tmp_path), "--burn-in", str(tmp_path / "absent.json")],
                    echo=lines.append, config=CFG, analysis=PINNED)
    assert rc == 4
    assert len(lines) == 1 and lines[0].startswith("refused: no burn-in estimability record")
    off = _burn_in(tmp_path, h1_min_deviation_sd=0.25)
    assert score.main(["--forecasts", str(tmp_path), "--burn-in", str(off)],
                      echo=lines.append, config=CFG, analysis=PINNED) == 4
    assert "measured against threshold 0.25" in lines[-1]
