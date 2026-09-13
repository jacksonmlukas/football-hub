"""Rolling conformal calibration.

`docs/foundation-plan.md` 3.3. `Conformalized` already existed in `hub.models.base` and
could calibrate against a window handed to it; nothing decided what that window was, so it
was never actually used.

What conformal buys is a coverage guarantee that does not depend on the model being right.
A model can be badly overconfident about its own intervals and the conformal ones still
cover at the nominal rate, because they are built from that model's *observed* errors rather
than its beliefs. That is the property worth testing, and it is the one that makes this worth
wiring at all.

The calibration window is strictly past weeks. Calibrating on the week being predicted is
the same leak `docs/track-record.md` rule 1 forbids, and it would manufacture perfect
coverage out of nothing -- so it gets its own test rather than a comment.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import conformal


def _frame(n_weeks=12, per_week=16, bias=0.0, noise=13.0, seed=0, season=None):
    """Predictions and outcomes where the true error spread is known. With `season`, the
    frame carries the column the store hands back; without, it is the one-season shape
    every test before #262 built."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        mean = rng.normal(0.0, 7.0, per_week)
        actual = mean + bias + rng.normal(0.0, noise, per_week)
        for m, a in zip(mean, actual, strict=True):
            row = {"week": w, "margin_mean": float(m), "margin_actual": float(a)}
            if season is not None:
                row = {"season": season, **row}
            rows.append(row)
    return pl.DataFrame(rows)


# --- the guarantee --------------------------------------------------------

def test_empirical_coverage_lands_near_nominal():
    got = conformal.rolling_coverage(_frame(), alpha=0.2, min_calibration=32)
    assert got["nominal"] == pytest.approx(0.8)
    assert abs(got["empirical"] - 0.8) < 0.07


def test_a_tighter_alpha_gives_wider_intervals():
    wide = conformal.rolling_coverage(_frame(), alpha=0.05, min_calibration=32)
    narrow = conformal.rolling_coverage(_frame(), alpha=0.3, min_calibration=32)
    assert wide["mean_width"] > narrow["mean_width"]
    assert wide["empirical"] > narrow["empirical"]


def test_coverage_survives_a_model_that_is_wrong_about_its_own_spread():
    """The whole point. Conformal intervals are built from observed errors, so a model that
    badly understates its own uncertainty still gets covered at the nominal rate."""
    got = conformal.rolling_coverage(_frame(noise=25.0), alpha=0.2, min_calibration=32)
    assert abs(got["empirical"] - 0.8) < 0.08


def test_a_biased_model_is_still_covered_though_the_intervals_grow():
    """Conformal fixes coverage, not bias -- it pays for the bias in width. Worth knowing
    before reading a wide interval as a broken model."""
    fair = conformal.rolling_coverage(_frame(bias=0.0), alpha=0.2, min_calibration=32)
    biased = conformal.rolling_coverage(_frame(bias=9.0), alpha=0.2, min_calibration=32)
    assert abs(biased["empirical"] - 0.8) < 0.09
    assert biased["mean_width"] > fair["mean_width"]


# --- no leakage -----------------------------------------------------------

def test_calibration_uses_only_earlier_weeks():
    """Calibrating on the week being predicted manufactures coverage out of nothing. Same
    rule as docs/track-record.md rule 1, and the exact bug this repo caught in its own
    depth-chart screen."""
    seen = []
    original = conformal.interval

    def spy(residuals, alpha):
        seen.append(len(residuals))
        return original(residuals, alpha)

    conformal.interval = spy
    try:
        conformal.rolling_coverage(_frame(n_weeks=6, per_week=10), alpha=0.2,
                                   min_calibration=10)
    finally:
        conformal.interval = original
    # weeks 2..6 are scored; calibration sets grow 10, 20, 30, 40, 50 -- never including
    # the week being predicted
    assert seen == [10, 20, 30, 40, 50]


# --- within a season (#262) -------------------------------------------------------------
#
# `rolling_coverage` walked week *numbers* with no season on the row, so week 6 of 2025 was
# calibrated on weeks 1-5 of every season in the store, 2026 included -- the same
# within-season split wearing a temporal split's name that `eval._holdout_window` fixed
# for its sibling (#168). The walk is over (season, week) in time order now.

def _two_seasons(early_noise=13.0, late_noise=40.0):
    """Two seasons that differ sharply in calibration: the later one's errors are three
    times wider, so any leak of it into the earlier season's window shows up as width."""
    return pl.concat([_frame(n_weeks=8, per_week=12, noise=early_noise, seed=3, season=2025),
                      _frame(n_weeks=8, per_week=12, noise=late_noise, seed=4, season=2026)])


def test_a_later_season_does_not_move_an_earlier_seasons_coverage():
    alone = conformal.rolling_coverage(_frame(n_weeks=8, per_week=12, seed=3, season=2025),
                                       alpha=0.2, min_calibration=24)
    both = conformal.rolling_coverage(_two_seasons(), alpha=0.2, min_calibration=24)
    early = [w for w in both["by_week"] if w["season"] == 2025]
    assert [w["half_width"] for w in early] == [w["half_width"] for w in alone["by_week"]]
    assert [w["coverage"] for w in early] == [w["coverage"] for w in alone["by_week"]]


def test_the_walk_is_in_time_order_across_seasons():
    """Every calibration set is strictly earlier than the cell it scores, season first."""
    seen = []
    original = conformal.interval

    def spy(residuals, alpha):
        seen.append(len(residuals))
        return original(residuals, alpha)

    conformal.interval = spy
    try:
        got = conformal.rolling_coverage(_two_seasons(), alpha=0.2, min_calibration=24)
    finally:
        conformal.interval = original
    cells = [(w["season"], w["week"]) for w in got["by_week"]]
    assert cells == sorted(cells) and cells[0] == (2025, 3)
    # 2026's first week calibrates on all of 2025 (96 rows), and the set only grows
    assert seen == [24, 36, 48, 60, 72, 84] + [96 + 12 * i for i in range(8)]
    assert got["first_scored_season"] == 2025 and got["first_scored_week"] == 3


def test_a_season_filter_scores_that_season_alone():
    """`--season` is the other half of the ticket: the walk keyed on the pair is right, and a
    reader asking about one season should not have to subtract the other from the report."""
    got = conformal.rolling_coverage(_two_seasons(), alpha=0.2, min_calibration=24,
                                     season=2026)
    assert {w["season"] for w in got["by_week"]} == {2026}
    assert got["first_scored_week"] == 3, "no earlier season to lean on once filtered"


def test_the_window_counts_cells_across_the_season_boundary():
    """A window of two calibrates 2026 week 1 on 2025 weeks 7 and 8 -- the two most recent
    cells in time, not the two most recent week numbers."""
    seen = []
    original = conformal.interval

    def spy(residuals, alpha):
        seen.append(len(residuals))
        return original(residuals, alpha)

    conformal.interval = spy
    try:
        conformal.rolling_coverage(_two_seasons(), alpha=0.2, min_calibration=24, window=2)
    finally:
        conformal.interval = original
    # fourteen cells scored -- 2025 weeks 3-8 and all of 2026 -- each on the two before it;
    # a walk that restarted at the season boundary would skip 2026 weeks 1 and 2
    assert seen == [24] * 14


def test_the_single_season_result_is_unchanged_with_or_without_the_column():
    """The one-season case is the case the week-number walk got right, and a frame that
    carries `season` must score exactly as one that does not."""
    bare = conformal.rolling_coverage(_frame(), alpha=0.2, min_calibration=32)
    keyed = conformal.rolling_coverage(_frame(season=2026), alpha=0.2, min_calibration=32)
    assert bare["empirical"] == keyed["empirical"] and bare["mean_width"] == keyed["mean_width"]
    assert [w["half_width"] for w in bare["by_week"]] == [
        w["half_width"] for w in keyed["by_week"]]


def test_the_first_weeks_are_skipped_not_calibrated_on_themselves():
    got = conformal.rolling_coverage(_frame(n_weeks=8, per_week=10), alpha=0.2,
                                     min_calibration=25)
    assert got["n_scored"] == 50, "weeks 1-3 build the window, 4-8 are scored"
    assert got["first_scored_week"] == 4


def test_never_reaching_the_minimum_is_reported_not_silently_empty():
    with pytest.raises(conformal.NotEnoughCalibration):
        conformal.rolling_coverage(_frame(n_weeks=3, per_week=5), alpha=0.2,
                                   min_calibration=100)


# --- the window -----------------------------------------------------------

def test_a_rolling_window_forgets_old_weeks():
    """A season is not stationary. An unbounded window drags January's errors into
    September, which is what `conf/` exists to make a choice about rather than a default."""
    everything = conformal.rolling_coverage(_frame(n_weeks=14), alpha=0.2,
                                            min_calibration=32, window=None)
    recent = conformal.rolling_coverage(_frame(n_weeks=14), alpha=0.2,
                                        min_calibration=32, window=4)
    assert recent["mean_calibration_n"] < everything["mean_calibration_n"]


def test_the_interval_quantile_carries_the_finite_sample_correction():
    """Split conformal needs (1-alpha)(n+1)/n, not the plain quantile, or coverage sits
    just under nominal for small windows -- which is exactly where this runs."""
    r = pl.Series([float(i) for i in range(1, 21)])
    plain = r.quantile(0.8)
    assert plain is not None
    assert conformal.interval(r, 0.2) >= float(plain)


# --- reporting ------------------------------------------------------------

def test_it_reports_per_week_coverage_too():
    got = conformal.rolling_coverage(_frame(), alpha=0.2, min_calibration=32)
    assert len(got["by_week"]) == got["n_weeks_scored"]
    assert all(0.0 <= w["coverage"] <= 1.0 for w in got["by_week"])


def test_the_cli_reports_empirical_against_nominal(capsys, monkeypatch):
    monkeypatch.setattr(conformal, "load_scored",
                        lambda model, base=None, season=None: _frame())
    assert conformal.main(["--recalibrate", "--model", "market_baseline"]) == 0
    out = capsys.readouterr().out.lower()
    assert "empirical" in out and "nominal" in out


def test_the_cli_season_reaches_the_reader_and_the_report_names_it(capsys, monkeypatch):
    asked = []

    def fake(model, base=None, season=None):
        asked.append(season)
        return _two_seasons().filter(pl.col("season") == season)

    monkeypatch.setattr(conformal, "load_scored", fake)
    assert conformal.main(["--recalibrate", "--season", "2026", "--min-calibration", "24"]) == 0
    assert asked == [2026]
    assert "2026 week 3+" in capsys.readouterr().out


# --- scoring a prediction is a join ----------------------------------------
#
# `load_scored` used to `SELECT margin_actual FROM preds`, and no such column exists. The
# CLI died on a DuckDB binder error on every real invocation, which is the practical reason
# nothing in the repo consumes a conformal interval.

def _preds(rows):
    """(game_id, week, margin_mean), all in season 2026 -- the store always carries the
    season, and since #262 `load_scored` carries it through."""
    return pl.DataFrame({"game_id": [r[0] for r in rows],
                         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
                         "week": [r[1] for r in rows],
                         "margin_mean": [r[2] for r in rows]})


def _sched(rows):
    """(game_id, result) -- nflverse's realised margin, home minus away."""
    return pl.DataFrame({"game_id": [r[0] for r in rows],
                         "result": [r[1] for r in rows]},
                        schema={"game_id": pl.Utf8, "result": pl.Float64})


def test_a_played_game_gets_its_realised_margin(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("g1", 1, 3.0)]))
    got = conformal.load_scored("m", schedules=_sched([("g1", 7.0)]))
    assert got["margin_actual"].to_list() == [7.0]
    assert got.columns == ["season", "week", "margin_mean", "margin_actual"]
    assert got["season"].to_list() == [2026]


def test_an_unplayed_game_does_not_survive_the_join(monkeypatch):
    """The whole 2026 board is unplayed games. They must not arrive as zeros -- a zero
    margin is a tie, not a missing result, and it would calibrate against fiction."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("g1", 1, 3.0)]))
    assert conformal.load_scored("m", schedules=_sched([("g1", None)])).is_empty()


def test_a_prediction_with_no_matching_game_is_dropped(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("ghost", 1, 3.0)]))
    assert conformal.load_scored("m", schedules=_sched([("g1", 7.0)])).is_empty()


def test_no_predictions_returns_the_right_shape_not_a_crash(monkeypatch):
    """Before any game is published this is the normal state, and it has to flow through to
    the `not enough calibration` message rather than blowing up on a missing column."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([]))
    got = conformal.load_scored("m", schedules=_sched([]))
    assert got.is_empty() and got.columns == ["season", "week", "margin_mean", "margin_actual"]


def test_schedules_without_a_result_column_says_so(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("g1", 1, 3.0)]))
    with pytest.raises(ValueError, match="result"):
        conformal.load_scored("m", schedules=pl.DataFrame({"game_id": ["g1"]}))


def test_a_store_with_no_preds_at_all_is_empty_not_a_catalog_error(monkeypatch):
    """A fresh clone has no `data/` directory, so `preds` is not an empty table -- there is
    no view, and DuckDB raises CatalogException. This is the guard that flows that through
    to the `not enough calibration` message.

    The tests above must declare `tables` too, or they exercise *this* path by accident and
    pass or fail on whether the developer's machine happens to have a preds directory. That
    is how they came to pass here and fail on a fresh clone.
    """
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: set())
    got = conformal.load_scored("m", schedules=_sched([("g1", 7.0)]))
    assert got.is_empty() and got.columns == ["season", "week", "margin_mean", "margin_actual"]


# --- against the store's own output, not a hand-built frame (issue #25) ------
#
# Every test above hands `rolling_coverage` a frame carrying a numeric `week`, which is what
# `PREDICTION_SCHEMA` declares. The store returned the Hive partition key instead -- the
# zero-padded string `"01"` -- so `pl.col("week").is_in([1, 2])` compared a string column
# against integers and this path had never once been run on real data. It exits early today
# for want of calibration points, which is why the divergence was latent rather than loud.

def _store_with_predictions(base, n_weeks=12, per_week=16, seed=0):
    """A real store, written the way `ratings.fit` writes one: one partition per week."""
    import datetime as dt

    from hub import store
    rng = np.random.default_rng(seed)
    for w in range(1, n_weeks + 1):
        mean = rng.normal(0.0, 6.0, per_week)
        store.write(
            pl.DataFrame({
                "game_id": [f"2026_{w:02d}_g{i}" for i in range(per_week)],
                "league": ["nfl"] * per_week,
                "season": pl.Series([2026] * per_week, dtype=pl.Int32),
                "week": pl.Series([w] * per_week, dtype=pl.Int32),
                "home_win_prob": [0.5] * per_week, "margin_mean": mean,
                "margin_lo": mean - 17.0, "margin_hi": mean + 17.0,
                "model": ["market_baseline"] * per_week, "version": ["v1"] * per_week,
                "fit_through_week": pl.Series([w - 1] * per_week, dtype=pl.Int32),
                "predicted_at": [dt.datetime(2026, 9, 1)] * per_week}),
            "preds", "nfl", 2026, w, base=base)
    return base


def test_an_adjusted_game_is_in_what_conformal_scores(tmp_path):
    """#284 named an adjusted row `market_baseline-qb`, and this reader filtered on the
    exact string -- so once the quarterback layer moved a few games, the calibration window
    quietly scored the unadjusted subset, which is biased: the backup-quarterback games are
    the ones that left. `market_baseline` here means the family, one row per game."""
    import datetime as dt

    from hub import store
    base = _store_with_predictions(tmp_path / "processed", n_weeks=1, per_week=4)
    store.write(
        pl.DataFrame({
            "game_id": ["2026_01_adjusted"], "league": ["nfl"],
            "season": pl.Series([2026], dtype=pl.Int32), "week": pl.Series([1], dtype=pl.Int32),
            "home_win_prob": [0.6], "margin_mean": [4.8], "margin_lo": [-12.2],
            "margin_hi": [21.8], "model": ["market_baseline-qb"], "version": ["v1-qb"],
            "fit_through_week": pl.Series([0], dtype=pl.Int32),
            "predicted_at": [dt.datetime(2026, 9, 1)]}),
        "preds", "nfl", 2026, 1, base=base, name="v1-qb")
    sched = pl.DataFrame(
        {"game_id": [f"2026_01_g{i}" for i in range(4)] + ["2026_01_adjusted"],
         "result": [3.0, -7.0, 10.0, 1.0, 6.0]},
        schema={"game_id": pl.Utf8, "result": pl.Float64})
    scored = conformal.load_scored("market_baseline", base=base, schedules=sched)
    assert scored.height == 5, "four plain games and the adjusted one"
    assert 4.8 in scored["margin_mean"].to_list(), "the adjusted game's own number"


def test_the_rolling_window_runs_on_what_the_store_actually_returns(tmp_path):
    """The end-to-end shape: store -> load_scored -> rolling_coverage. Nothing hand-built,
    so a week that arrives as a padded string fails here rather than in October."""
    base = _store_with_predictions(tmp_path / "processed")
    rng = np.random.default_rng(1)
    sched = pl.DataFrame(
        {"game_id": [f"2026_{w:02d}_g{i}" for w in range(1, 13) for i in range(16)],
         "result": rng.normal(0.0, 13.0, 12 * 16)},
        schema={"game_id": pl.Utf8, "result": pl.Float64})
    scored = conformal.load_scored("market_baseline", base=base, schedules=sched)
    assert scored.schema["week"].is_numeric(), "the store must hand back a numeric week"
    assert scored.schema["season"].is_numeric(), "and a numeric season (#262)"

    got = conformal.rolling_coverage(scored, alpha=0.2, min_calibration=32)
    assert got["n_weeks_scored"] > 0
    assert got["first_scored_week"] > 1, "the first week has no earlier week to calibrate on"
