"""The committed coverage harness, and the lookahead it exists to remove.

`docs/weekly-coverage.md` measured this once, in a document, on a centre built from the very
weeks it then scored. Two properties matter more than any number the harness prints, and
both get a test rather than a paragraph:

  * **The prior centre cannot see the week it is scoring.** This is `docs/track-record.md`
    rule 1 and the same leak `hub.models.conformal` tests for its calibration window. The
    test drives it the hard way -- a week whose points are absurd -- and asserts the centre
    does not move.
  * **The harness grades the deployed moment function.** `hub.models.predict.moments` is what
    the draft optimiser and the roster path call, and a grader holding its own copy of
    `WEEKLY_K` would agree with it today and quietly stop agreeing later. The test moves the
    constant and requires the harness to move with it.

The counting itself is checked against a distribution whose coverage is known by
construction: points drawn *through* `predict.skewed` must be covered at nominal. That is
the one case where a fixture agreeing with the code under test is the point -- if the
counter is wrong, a sample drawn from the model's own distribution is what exposes it.
"""
import json
import math
from typing import cast

import numpy as np
import polars as pl
import pytest

from hub.models import coverage, predict


def _stats(rows):
    """nflverse-shaped weekly stats: what `player_weeks` is handed."""
    return pl.DataFrame(rows, schema={
        "player_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32,
        "week": pl.Int32, "season_type": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _player(pid, pos, season, points, season_type="REG"):
    return [{"player_id": pid, "position": pos, "season": season, "week": w + 1,
             "season_type": season_type, "fantasy_points_ppr": float(p)}
            for w, p in enumerate(points)]


def _drawn(n_players=400, weeks=17, pos="WR", mu=12.0, seed=0, spread=1.0):
    """Weeks drawn from the model's own distribution, so nominal coverage is the truth.

    The centre is the same for every week of a player-season, which is what makes the
    realised mean an (almost) unbiased estimate of it -- the sample is built so that the
    lookahead centre is nearly right and the prior centre is nearly right, and the two are
    then separated by what each is allowed to see.

    `spread` above 1 draws weeks wider than the model believes, which is a model whose
    intervals are too narrow -- the failure the gate exists to catch.
    """
    rng = np.random.default_rng(seed)
    sd = predict.WEEKLY_K[pos] * math.sqrt(mu) * spread
    sk = predict.WEEKLY_SKEW[pos]
    rows = []
    for p in range(n_players):
        z = rng.standard_normal(weeks)
        rows += _player(f"p{p}", pos, 2024, predict.skewed(mu, sd, sk, z))
    return _stats(rows)


# --- the lookahead --------------------------------------------------------

def test_the_prior_centre_cannot_see_the_week_it_scores():
    """The whole reason this harness exists. Week 9 is a 300-point week and must not move
    the centre that week 9 is graded against."""
    normal = _player("p1", "WR", 2024, [10.0] * 8 + [300.0] + [10.0] * 8)
    g = coverage.centred(coverage.player_weeks(_stats(normal)), "prior", min_prior=4)
    at9 = g.filter(pl.col("week") == 9)["centre"].to_list()
    assert at9 == [pytest.approx(10.0)], (
        "the centre for week 9 was built from week 9's own points; that is the lookahead "
        "this module was written to remove")
    # And the weeks *after* it do move, which is what says the centre is running at all.
    assert g.filter(pl.col("week") == 10)["centre"].item() > 10.0


def test_the_realised_centre_is_the_documents_and_is_marked_as_lookahead():
    stats = _drawn(n_players=60)
    got = coverage.measure(stats, "realised")
    assert got["lookahead"] is True and got["centre"] == "realised"
    assert coverage.measure(stats, "prior")["lookahead"] is False


def test_the_lookahead_centre_reports_the_better_coverage_of_the_two():
    """Not a claim about which is right -- a claim about which flatters. On one sample, with
    one filter, the centre that saw the outcome covers more of it."""
    stats = _drawn(n_players=300, seed=3)
    look = coverage.measure(stats, "realised")
    honest = coverage.measure(stats, "prior")
    assert look["gate_cov80"] > honest["gate_cov80"]


def test_a_week_is_not_scored_until_enough_earlier_weeks_exist():
    rows = _player("p1", "WR", 2024, [10.0] * 17)
    g = coverage.centred(coverage.player_weeks(_stats(rows)), "prior", min_prior=4)
    assert min(g["week"].to_list()) == 5, "week 5 is the first with four weeks behind it"
    assert min(g["n_prior"].to_list()) == 4


# --- grading the deployed function ---------------------------------------

def test_the_harness_moves_when_the_deployed_spread_law_moves(monkeypatch):
    """`graded` must read `predict.moments`, not a copy of `WEEKLY_K` living here.

    Doubling the shipped constant doubles every interval, so coverage must rise. A harness
    that restated the law would return the identical table and this test would fail.
    """
    stats = _drawn(n_players=120, seed=1)
    before = coverage.measure(stats, "prior")["gate_cov80"]
    monkeypatch.setattr(predict, "WEEKLY_K", {k: v * 2 for k, v in predict.WEEKLY_K.items()})
    after = coverage.measure(stats, "prior")["gate_cov80"]
    assert after > before + 0.05, (
        "the shipped spread law doubled and the graded coverage did not move, so the "
        "harness is grading its own copy of it")


def test_the_moments_it_grades_are_the_ones_moments_returns():
    stats = _drawn(n_players=20, seed=2)
    sample = coverage.centred(coverage.player_weeks(stats), "prior")
    g = coverage.graded(sample)
    deployed = predict.moments(sample.with_columns(pl.col("centre").alias("proj_ppg")))
    for col in ("mu", "sd", "skew"):
        assert g[col].to_list() == deployed[col].to_list()


def test_the_bounds_are_the_shipped_transform_at_the_normal_quantile():
    sample = coverage.centred(coverage.player_weeks(_drawn(n_players=5, seed=4)), "prior")
    g = coverage.graded(sample)
    want = predict.skewed(g["mu"].to_numpy(), g["sd"].to_numpy(), g["skew"].to_numpy(),
                          coverage._z(0.10))
    assert g["q10"].to_numpy() == pytest.approx(want)


# --- the counting ---------------------------------------------------------

def test_points_drawn_through_the_model_are_covered_at_nominal():
    """The counter's own correctness check. Draw from the shipped distribution with the
    centre handed over rather than estimated, and 80% must come back."""
    rng = np.random.default_rng(7)
    mu, pos = 12.0, "WR"
    sd = predict.WEEKLY_K[pos] * math.sqrt(mu)
    sk = predict.WEEKLY_SKEW[pos]
    n = 40_000
    y = predict.skewed(mu, sd, sk, rng.standard_normal(n))
    frame = pl.DataFrame({"player_id": ["p"] * n, "season": [2024] * n,
                          "week": list(range(n)), "position": [pos] * n,
                          "points": y, "centre": [mu] * n})
    row = coverage._row("all", coverage.graded(frame))
    assert row["cov80"] == pytest.approx(0.80, abs=0.01)
    assert row["cov68"] == pytest.approx(0.68, abs=0.01)
    assert row["below_p10"] == pytest.approx(0.10, abs=0.01)
    assert row["above_p90"] == pytest.approx(0.10, abs=0.01)


def test_the_floor_split_finds_the_weeks_whose_lower_bound_was_clipped():
    """A 2.5-point player's p10 lands below zero and is clipped; a 20-point player's does
    not. The split has to separate them, because the clipped ones cannot be fallen below."""
    rows = _player("low", "TE", 2024, [2.5] * 17) + _player("high", "TE", 2024, [20.0] * 17)
    g = coverage.graded(coverage.centred(coverage.player_weeks(_stats(rows)), "prior"))
    clipped = g.filter(pl.col("p10_raw") <= 0.0)["player_id"].unique().to_list()
    positive = g.filter(pl.col("p10_raw") > 0.0)["player_id"].unique().to_list()
    assert clipped == ["low"] and positive == ["high"]
    assert (g.filter(pl.col("p10_raw") <= 0.0)["q10"] == 0.0).all(), (
        "a clipped p10 is zero, which is the reason nothing can fall below it")


def test_the_no_skew_column_is_a_plain_normal_on_the_same_moments():
    g = coverage.graded(coverage.centred(coverage.player_weeks(_drawn(n_players=5)), "prior"))
    want = g["mu"].to_numpy() + g["sd"].to_numpy() * coverage._z(0.90)
    assert g["n90"].to_numpy() == pytest.approx(want)


def test_the_table_carries_every_position_present_and_then_the_pool():
    rows = (_player("a", "QB", 2024, [18.0] * 17) + _player("b", "RB", 2024, [12.0] * 17))
    g = coverage.graded(coverage.centred(coverage.player_weeks(_stats(rows)), "prior"))
    got = coverage.table(g)
    assert [r["group"] for r in got] == ["QB", "RB", "all"]
    assert got[-1]["n"] == got[0]["n"] + got[1]["n"]


def test_post_season_weeks_and_other_positions_are_out_of_the_sample():
    rows = (_player("a", "WR", 2024, [12.0] * 17)
            + _player("a", "WR", 2024, [12.0] * 3, season_type="POST")
            + _player("k", "K", 2024, [8.0] * 17))
    got = coverage.player_weeks(_stats(rows))
    assert got.height == 17
    assert got["position"].unique().to_list() == ["WR"]


def test_stats_without_the_columns_it_needs_says_which():
    with pytest.raises(ValueError, match="fantasy_points_ppr"):
        coverage.player_weeks(pl.DataFrame({"player_id": ["a"]}))


def test_a_sample_nothing_survives_is_reported_not_returned_empty():
    rows = _player("a", "WR", 2024, [0.5] * 17)          # under MIN_MU
    with pytest.raises(coverage.NotEnoughWeeks):
        coverage.centred(coverage.player_weeks(_stats(rows)), "prior")


def test_an_unknown_centre_is_refused():
    with pytest.raises(ValueError, match="centre must be"):
        coverage.centred(coverage.player_weeks(_drawn(n_players=2)),
                         cast(coverage.Centre, "realised-ish"))


# --- the verdict, and what reads it --------------------------------------

@pytest.mark.parametrize("cov,want", [
    (0.80, "COVERS"), (0.815, "COVERS"), (0.774, "UNDER-COVERS"), (0.893, "OVER-COVERS")])
def test_the_verdict_reads_the_pre_registered_band(cov, want):
    assert coverage.verdict({"cov80": cov}) == want


def test_the_gate_reads_the_unclipped_weeks_not_the_pool():
    """The clipped weeks report a coverage the interval did not earn, so pooling them in is
    how a miss hides. The gate takes the strictly-positive split."""
    stats = _drawn(n_players=200, seed=5)
    got = coverage.measure(stats, "prior")
    unclipped = next(r for r in got["floor_split"] if r["group"] == "strictly positive")
    assert got["gate_cov80"] == unclipped["cov80"] and got["gate_n"] == unclipped["n"]
    assert got["gate_n"] < got["n"]


def test_the_summary_round_trips_for_the_publisher(tmp_path):
    got = coverage.measure(_drawn(n_players=40), "prior")
    path = coverage.write_summary(got, tmp_path / "interval_coverage.json")
    back = coverage.published_summary(path)
    assert back is not None
    assert back["verdict"] == got["verdict"]
    assert back["gate_cov80"] == pytest.approx(got["gate_cov80"])
    assert "by_position" not in back, "the published field is a summary, not the table"


def test_a_missing_measurement_is_no_field_rather_than_a_traceback(tmp_path):
    assert coverage.published_summary(tmp_path / "nope.json") is None


def test_an_unreadable_measurement_is_the_same_nothing_as_an_absent_one(tmp_path):
    p = tmp_path / "interval_coverage.json"
    p.write_text("{not json")
    assert coverage.published_summary(p) is None
    p.write_text(json.dumps({"name": "interval_coverage"}))       # no verdict in it
    assert coverage.published_summary(p) is None


# --- the survivor price ---------------------------------------------------

def _schedule(spreads_and_results):
    return pl.DataFrame(
        {"spread_line": [float(s) for s, _ in spreads_and_results],
         "result": [float(r) for _, r in spreads_and_results]})


def test_a_favourite_that_wins_more_often_than_priced_is_under_confident():
    """Ten-point home favourites priced at about 0.78. Win them 97% of the time and the
    price is under-confident, which is the direction that matters to survivor: the plan is
    safer than the number it prints."""
    games = [(10.0, 7.0)] * 97 + [(10.0, -7.0)] * 3
    got = coverage.survivor_price(_schedule(games))
    assert got["verdict"] == "UNDER-CONFIDENT"
    assert got["favourite_gap"] > 0

    # And the other way, on the same spread.
    flipped = [(10.0, 7.0)] * 50 + [(10.0, -7.0)] * 50
    assert coverage.survivor_price(_schedule(flipped))["verdict"] == "OVER-CONFIDENT"


def test_a_price_that_holds_is_not_reported_as_a_miss():
    """Games generated at the model's own margin dispersion. The verdict must be HOLDS, or
    the grader would condemn a calibrated price."""
    rng = np.random.default_rng(11)
    from hub.models.market import MARGIN_SD
    spreads = rng.choice([7.0, 9.0, 10.5, 13.0], size=1200)
    results = spreads + rng.normal(0.0, MARGIN_SD, spreads.size)
    results[results == 0.0] = 1.0
    got = coverage.survivor_price(_schedule(list(zip(spreads, results, strict=True))))
    assert got["verdict"] == "HOLDS", got


def test_both_sides_of_every_game_are_graded():
    got = coverage.survivor_price(_schedule([(3.0, 1.0), (3.0, -1.0)]))
    assert got["n_games"] == 2 and got["n_sides"] == 4


def test_a_tied_game_is_not_scored_as_a_home_loss():
    """`hub.models.margin.home_won` is the repo's one tie convention and this reads it
    rather than restating it -- issue #64."""
    got = coverage.survivor_price(_schedule([(3.0, 1.0), (3.0, 0.0)]))
    assert got["n_games"] == 1


def test_the_buckets_are_spread_ranges_not_probability_ranges():
    got = coverage.survivor_price(_schedule([(1.0, 1.0), (8.0, 1.0), (20.0, 1.0)]))
    filled = [b["bin"] for b in got["buckets"] if b["n"]]
    assert filled == ["0.0-3.0", "6.0-9.0", "14.0-30.0"]


def test_schedules_without_a_spread_says_so():
    with pytest.raises(ValueError, match="spread_line"):
        coverage.survivor_price(pl.DataFrame({"result": [1.0]}))


def test_no_completed_priced_game_is_reported_not_divided_by_zero():
    with pytest.raises(coverage.NotEnoughWeeks):
        coverage.survivor_price(pl.DataFrame({"spread_line": [3.0], "result": [None]},
                                             schema={"spread_line": pl.Float64,
                                                     "result": pl.Float64}))


# --- the CLI --------------------------------------------------------------

def test_the_cli_prints_the_table_and_the_verdict(capsys, monkeypatch):
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(n_players=80))
    assert coverage.main(["--measure"]) == 0
    out = capsys.readouterr().out
    assert "centre=prior" in out and "gate reads the unclipped weeks" in out
    assert "LOOKAHEAD" not in out


def test_the_cli_says_so_when_the_centre_is_the_lookahead_one(capsys, monkeypatch):
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(n_players=80))
    assert coverage.main(["--measure", "--centre", "realised"]) == 0
    assert "LOOKAHEAD CENTRE" in capsys.readouterr().out


def test_the_gate_refuses_when_the_interval_leaves_the_band(capsys, monkeypatch):
    """The consumption. A measurement whose answer nothing acts on is the thing issue #176
    was opened about, so the exit code is part of the contract."""
    monkeypatch.setattr(coverage, "_stats",
                        lambda seasons, cache: _drawn(n_players=300, spread=1.3, seed=9))
    assert coverage.main(["--gate"]) == 1, (
        "weeks drawn 30% wider than the model believes must not clear the band")
    assert "does not cover" in capsys.readouterr().err


def test_the_gate_passes_a_calibrated_interval(monkeypatch):
    """Same command, same band, weeks drawn at the width the model claims. A gate that only
    ever refuses is not reading anything.

    The seasons here are absurdly long on purpose. Even a perfectly specified model
    under-covers once the centre is *estimated*, because the residual carries the centre's
    own error on top of the week's -- which is the whole finding on real data. Give the
    centre a hundred weeks to settle and that term goes away, and what is left is the gate
    reading a model that is right.
    """
    monkeypatch.setattr(coverage, "_stats",
                        lambda seasons, cache: _drawn(n_players=40, weeks=200, seed=9))
    assert coverage.main(["--gate", "--min-prior", "100"]) == 0


def test_the_cli_writes_the_artifact_the_publisher_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(n_players=40))
    monkeypatch.setattr(coverage, "ARTIFACT", tmp_path / "interval_coverage.json")
    assert coverage.main(["--measure", "--write"]) == 0
    assert coverage.published_summary(tmp_path / "interval_coverage.json") is not None


def test_the_survivor_cli_reports_by_bucket(capsys, monkeypatch):
    monkeypatch.setattr(coverage, "_schedules",
                        lambda seasons, cache: _schedule([(10.0, 7.0)] * 40
                                                         + [(10.0, -7.0)] * 10))
    assert coverage.main(["--survivor", "--seasons", "2024"]) == 0
    out = capsys.readouterr().out
    assert "survivor price" in out and "favourites of 7+" in out


def _raises(*_a, **_k):
    raise OSError("no such cache entry and nflverse is unreachable")


def test_unreachable_schedules_are_a_sentence_not_a_traceback(capsys, monkeypatch):
    monkeypatch.setattr(coverage, "_schedules", _raises)
    assert coverage.main(["--survivor"]) == 1
    err = capsys.readouterr().err
    assert "nflverse schedules unavailable" in err and "Traceback" not in err


def test_unreachable_player_stats_are_a_sentence_not_a_traceback(capsys, monkeypatch):
    monkeypatch.setattr(coverage, "_stats", _raises)
    assert coverage.main(["--measure"]) == 1
    assert "nflverse player_stats unavailable" in capsys.readouterr().err


def test_a_window_nothing_survives_is_reported_by_the_cli(capsys, monkeypatch):
    """The filters can empty the sample -- a one-week season, a season of blanks -- and the
    operator gets the reason rather than a `NotEnoughWeeks` up the stack."""
    monkeypatch.setattr(coverage, "_stats",
                        lambda seasons, cache: _stats(_player("a", "WR", 2024, [0.1] * 17)))
    assert coverage.main(["--measure"]) == 1
    assert "no player-week survived" in capsys.readouterr().err


def test_a_survivor_window_with_no_completed_game_is_reported_by_the_cli(capsys,
                                                                        monkeypatch):
    monkeypatch.setattr(coverage, "_schedules", lambda seasons, cache: pl.DataFrame(
        {"spread_line": [3.0], "result": [None]},
        schema={"spread_line": pl.Float64, "result": pl.Float64}))
    assert coverage.main(["--survivor"]) == 1
    assert "no completed game" in capsys.readouterr().err


def test_no_flag_prints_help_rather_than_measuring(capsys):
    assert coverage.main([]) == 0
    assert "usage:" in capsys.readouterr().out


# --- the gate is wired (#273) -------------------------------------------------------------

def test_the_artifact_lives_where_a_commit_can_carry_it():
    """`data/processed/` is gitignored, so a file written there never reaches the runner
    and the publisher read nothing, by design. `state/` is committed."""
    from hub.paths import STATE_DIR
    assert coverage.ARTIFACT.parent == STATE_DIR


def test_the_survivor_verdict_is_written_beside_the_weekly_one(tmp_path, monkeypatch):
    """One file, two verdicts, one reader. `--survivor --write` after `--measure --write`
    adds the survivor block without losing the weekly one, and the publisher's summary
    carries both."""
    art = tmp_path / "interval_coverage.json"
    monkeypatch.setattr(coverage, "ARTIFACT", art)
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(n_players=40))
    monkeypatch.setattr(coverage, "_schedules",
                        lambda seasons, cache: _schedule([(10.0, 7.0)] * 40 + [(10.0, -7.0)] * 10))
    assert coverage.main(["--measure", "--write"]) == 0
    assert coverage.main(["--survivor", "--write", "--seasons", "2024"]) == 0
    got = coverage.published_summary(art)
    assert got is not None and got["verdict"] in ("COVERS", "DOES NOT COVER")
    assert got["survivor"]["verdict"] and "favourite_gap" in got["survivor"]
    # the order does not matter either
    assert coverage.main(["--measure", "--write"]) == 0
    again = coverage.published_summary(art)
    assert again is not None and again["survivor"]["verdict"]
