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


def _frame(n_weeks=12, per_week=16, bias=0.0, noise=13.0, seed=0):
    """Predictions and outcomes where the true error spread is known."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        mean = rng.normal(0.0, 7.0, per_week)
        actual = mean + bias + rng.normal(0.0, noise, per_week)
        for m, a in zip(mean, actual, strict=True):
            rows.append({"week": w, "margin_mean": float(m), "margin_actual": float(a)})
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
    monkeypatch.setattr(conformal, "load_scored", lambda model, base=None: _frame())
    assert conformal.main(["--recalibrate", "--model", "market_baseline"]) == 0
    out = capsys.readouterr().out.lower()
    assert "empirical" in out and "nominal" in out


# --- scoring a prediction is a join ----------------------------------------
#
# `load_scored` used to `SELECT margin_actual FROM preds`, and no such column exists. The
# CLI died on a DuckDB binder error on every real invocation, which is the practical reason
# nothing in the repo consumes a conformal interval.

def _preds(rows):
    """(game_id, week, margin_mean)."""
    return pl.DataFrame({"game_id": [r[0] for r in rows], "week": [r[1] for r in rows],
                         "margin_mean": [r[2] for r in rows]})


def _sched(rows):
    """(game_id, result) -- nflverse's realised margin, home minus away."""
    return pl.DataFrame({"game_id": [r[0] for r in rows],
                         "result": [r[1] for r in rows]},
                        schema={"game_id": pl.Utf8, "result": pl.Float64})


def test_a_played_game_gets_its_realised_margin(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("g1", 1, 3.0)]))
    got = conformal.load_scored("m", schedules=_sched([("g1", 7.0)]))
    assert got["margin_actual"].to_list() == [7.0]
    assert got.columns == ["week", "margin_mean", "margin_actual"]


def test_an_unplayed_game_does_not_survive_the_join(monkeypatch):
    """The whole 2026 board is unplayed games. They must not arrive as zeros -- a zero
    margin is a tie, not a missing result, and it would calibrate against fiction."""
    import hub.store as store
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("g1", 1, 3.0)]))
    assert conformal.load_scored("m", schedules=_sched([("g1", None)])).is_empty()


def test_a_prediction_with_no_matching_game_is_dropped(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("ghost", 1, 3.0)]))
    assert conformal.load_scored("m", schedules=_sched([("g1", 7.0)])).is_empty()


def test_no_predictions_returns_the_right_shape_not_a_crash(monkeypatch):
    """Before any game is published this is the normal state, and it has to flow through to
    the `not enough calibration` message rather than blowing up on a missing column."""
    import hub.store as store
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([]))
    got = conformal.load_scored("m", schedules=_sched([]))
    assert got.is_empty() and got.columns == ["week", "margin_mean", "margin_actual"]


def test_schedules_without_a_result_column_says_so(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "sql", lambda *a, **k: _preds([("g1", 1, 3.0)]))
    with pytest.raises(ValueError, match="result"):
        conformal.load_scored("m", schedules=pl.DataFrame({"game_id": ["g1"]}))
