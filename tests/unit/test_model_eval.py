"""Comparing two models honestly.

`docs/foundation-plan.md` 3.4. The point of this harness is not to produce a number that
favours whatever was just built -- it is to make "is this better than the market?" answerable
with an interval, so the answer can be no.

The correctness test the plan names is deliberately the boring one: score a model against
itself and the delta must be exactly zero with an interval containing zero. A comparison
harness that cannot return "no difference" when there is none will happily report an edge
that is not there, and every subsequent result from it is worthless.

Splitting is temporal by default because a random split leaks: predictions for week 10 made
after week 12 are not predictions. `docs/track-record.md` rule 1 is the same rule one layer
up.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import eval as me


def _preds(probs, outcomes, week=1, model="a"):
    n = len(probs)
    return pl.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "season": pl.Series([2026] * n, dtype=pl.Int32),
        "week": pl.Series([week] * n, dtype=pl.Int32),
        "model": [model] * n,
        "home_win_prob": list(probs),
        "home_won": list(outcomes)})


# --- the correctness test the plan names ----------------------------------

def test_a_model_scored_against_itself_reports_no_edge():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.15, 0.85, 300)
    y = (rng.uniform(size=300) < p).astype(int)
    got = me.compare(_preds(p, y, model="a"), _preds(p, y, model="b"))
    assert got["delta"] == pytest.approx(0.0, abs=1e-12)
    assert got["ci95"][0] <= 0.0 <= got["ci95"][1]


def test_the_interval_around_no_edge_is_not_degenerate():
    """A zero-width interval would pass the test above while telling you nothing. Two
    genuinely different models must produce a real interval."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 300)
    y = (rng.uniform(size=300) < p).astype(int)
    got = me.compare(_preds(p, y), _preds(np.clip(p + 0.05, 0.01, 0.99), y, model="b"))
    assert got["ci95"][1] > got["ci95"][0]


# --- it can find a real difference ----------------------------------------

def test_a_better_model_wins():
    """The sharper model is the one that knows the outcome-generating probability; the
    other is a coin flip. If this cannot be detected the harness is inert."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 600)
    y = (rng.uniform(size=600) < p).astype(int)
    got = me.compare(_preds(p, y), _preds(np.full(600, 0.5), y, model="b"))
    assert got["delta"] < 0, "log loss is lower for the better model"
    assert got["ci95"][1] < 0, "and the interval excludes zero"


def test_a_confidently_wrong_model_loses_badly():
    got = me.compare(_preds([0.99] * 50, [0] * 50), _preds([0.5] * 50, [0] * 50, model="b"))
    assert got["delta"] > 0


# --- splitting ------------------------------------------------------------

def test_a_temporal_split_holds_out_the_later_weeks():
    """A random split leaks: a prediction for week 10 made with week 12 in the fit is not a
    prediction. Same rule as docs/track-record.md, one layer up."""
    a = pl.concat([_preds([0.6] * 20, [1] * 20, week=w) for w in range(1, 11)])
    b = pl.concat([_preds([0.6] * 20, [1] * 20, week=w, model="b") for w in range(1, 11)])
    got = me.compare(a, b, split="temporal", holdout=0.3)
    assert got["n_scored"] == 60, "the last 3 of 10 weeks"
    assert got["holdout_weeks"] == [8, 9, 10]


def test_scoring_everything_is_available_but_not_the_default():
    a = pl.concat([_preds([0.6] * 20, [1] * 20, week=w) for w in range(1, 11)])
    b = pl.concat([_preds([0.6] * 20, [1] * 20, week=w, model="b") for w in range(1, 11)])
    assert me.compare(a, b, split="all")["n_scored"] == 200


def test_models_are_compared_only_on_games_they_both_predicted():
    """Otherwise the comparison is partly a comparison of which games each chose to touch,
    and the easy games are not evenly distributed."""
    a = _preds([0.6, 0.7, 0.8], [1, 1, 1])
    b = _preds([0.6, 0.7], [1, 1], model="b")
    assert me.compare(a, b)["n_scored"] == 2


def test_no_overlap_is_an_error_rather_than_a_delta_of_zero():
    a = _preds([0.6], [1]).with_columns(pl.lit("x").alias("game_id"))
    b = _preds([0.6], [1], model="b").with_columns(pl.lit("y").alias("game_id"))
    with pytest.raises(me.NoOverlap):
        me.compare(a, b)


# --- reporting ------------------------------------------------------------

def test_it_reports_both_models_absolute_scores():
    """A delta alone hides the case where both models are bad."""
    got = me.compare(_preds([0.6] * 40, [1] * 40), _preds([0.5] * 40, [1] * 40, model="b"))
    assert got["log_loss_a"] > 0 and got["log_loss_b"] > 0
    assert got["brier_a"] > 0 and got["brier_b"] > 0


def test_a_reliability_diagram_comes_with_it():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.uniform(size=400) < p).astype(int)
    got = me.compare(_preds(p, y), _preds(np.full(400, 0.5), y, model="b"))
    assert sum(b["n"] for b in got["reliability_a"]) == 400


def test_the_cli_runs_a_comparison(capsys, tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    p = rng.uniform(0.1, 0.9, 200)
    y = (rng.uniform(size=200) < p).astype(int)
    monkeypatch.setattr(me, "load_predictions",
                        lambda model, base=None: _preds(p, y, model=model))
    assert me.main(["--compare", "market_baseline,market_baseline"]) == 0
    out = capsys.readouterr().out
    assert "log loss" in out.lower() and "0.000" in out


def test_an_unknown_model_fails_with_a_message_not_a_traceback(capsys, monkeypatch):
    def _boom(model, base=None):
        raise me.NoOverlap(f"{model} has no scored outcomes yet")
    monkeypatch.setattr(me, "load_predictions", _boom)
    assert me.main(["--compare", "a,b"]) == 1
    assert "no scored outcomes" in capsys.readouterr().err


def test_the_cli_refuses_anything_but_two_models(capsys):
    assert me.main(["--compare", "only_one"]) == 2
    assert "exactly two" in capsys.readouterr().err


def test_an_empty_holdout_window_is_an_error():
    a = _preds([0.6] * 5, [1] * 5, week=1)
    b = _preds([0.6] * 5, [1] * 5, week=1, model="b")
    got = me.compare(a, b, split="temporal", holdout=0.5)
    assert got["n_scored"] == 5, "a single week cannot be split; score it rather than fail"
