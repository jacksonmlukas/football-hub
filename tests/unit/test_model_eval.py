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
    """An unknown model reads back empty, and two empty frames share no games. The message
    names both models and names *overlap* -- it used to say `has no scored outcomes yet`,
    which was the #59 defect borrowing this exception rather than a real condition, and an
    empty frame carries no `model` value to read a name off, so the CLI supplies them."""
    monkeypatch.setattr(me, "load_predictions",
                        lambda model, base=None: _preds([], [], model=model).head(0))
    assert me.main(["--compare", "alpha,beta"]) == 1
    err = capsys.readouterr().err
    assert "alpha and beta share no games in common" in err


def test_the_cli_refuses_anything_but_two_models(capsys):
    assert me.main(["--compare", "only_one"]) == 2
    assert "exactly two" in capsys.readouterr().err


def test_an_empty_holdout_window_is_an_error():
    a = _preds([0.6] * 5, [1] * 5, week=1)
    b = _preds([0.6] * 5, [1] * 5, week=1, model="b")
    got = me.compare(a, b, split="temporal", holdout=0.5)
    assert got["n_scored"] == 5, "a single week cannot be split; score it rather than fail"


# --- scoring a prediction is a join (issue #59) ------------------------------
#
# `load_predictions` used to raise `NoOverlap("<model> has no scored outcomes yet")` for any
# frame without `home_won`, and the store never carries one -- it records what was predicted
# and never what happened. So the store path was unreachable and `--compare` could not score
# against a real store at all. Every test here passed a hand-built frame that already carried
# `home_won`, which is why the dead path read as working. `conformal.load_scored` had already
# settled the shape one file over: the outcome is a join from schedules, injectable so no test
# needs the network.

def _sched(rows):
    """(game_id, result) -- nflverse's realised margin, home minus away."""
    return pl.DataFrame({"game_id": [r[0] for r in rows], "result": [r[1] for r in rows]},
                        schema={"game_id": pl.Utf8, "result": pl.Float64})


def _stored(rows):
    """What `store.predictions` hands back: no outcome column, `week` as the declared Int32."""
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([r[1] for r in rows], dtype=pl.Int32),
         "model": ["m"] * len(rows),
         "home_win_prob": [r[2] for r in rows]})


def test_a_played_game_gets_its_outcome_joined_on(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("g1", 1, 0.6)]))
    got = me.load_predictions("m", schedules=_sched([("g1", 7.0)]))
    assert got["home_won"].to_list() == [1]


def test_a_home_loss_is_scored_as_one(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("g1", 1, 0.6)]))
    got = me.load_predictions("m", schedules=_sched([("g1", -3.0)]))
    assert got["home_won"].to_list() == [0]


def test_an_unplayed_game_does_not_survive_the_join(monkeypatch):
    """The whole 2026 board is unplayed. An unplayed game arriving as a home loss would be
    scored against a fabricated outcome, and log loss cannot tell that from a real one."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("g1", 1, 0.6)]))
    assert me.load_predictions("m", schedules=_sched([("g1", None)])).is_empty()


def test_a_tie_is_dropped_rather_than_scored_as_a_home_loss(monkeypatch):
    """`hub.models.margin.DROP_TIES` states the rule this follows: a tie is neither a home
    win nor an away win, and there is no sensible probability to score it against."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("g1", 1, 0.6),
                                                              ("g2", 1, 0.6)]))
    got = me.load_predictions("m", schedules=_sched([("g1", 0.0), ("g2", 4.0)]))
    assert got["game_id"].to_list() == ["g2"]


def test_a_prediction_with_no_matching_game_is_dropped(monkeypatch):
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("ghost", 1, 0.6)]))
    assert me.load_predictions("m", schedules=_sched([("g1", 7.0)])).is_empty()


def test_no_predictions_returns_the_right_shape_not_a_crash(monkeypatch):
    """Before any game is published this is the normal state, and it has to flow through to
    the no-overlap message rather than blowing up on a missing column."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([]))
    got = me.load_predictions("m", schedules=_sched([]))
    assert got.is_empty() and "home_won" in got.columns


def test_a_store_with_no_preds_at_all_is_empty_not_a_catalog_error(monkeypatch):
    """A fresh clone has no `data/` directory, so `preds` is not an empty table -- there is
    no view, and DuckDB raises CatalogException."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: set())
    got = me.load_predictions("m", schedules=_sched([("g1", 7.0)]))
    assert got.is_empty() and "home_won" in got.columns


def test_schedules_without_a_result_column_says_so(monkeypatch):
    """`OutcomesUnavailable` rather than a bare `ValueError`, so the CLI can catch it
    without also catching every unrelated value error raised beneath it (issue #63)."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("g1", 1, 0.6)]))
    with pytest.raises(me.OutcomesUnavailable, match="result"):
        me.load_predictions("m", schedules=pl.DataFrame({"game_id": ["g1"]}))


# --- against the store's own output, not a hand-built frame (issues #25, #59) --
#
# `_preds` above builds `week` as the `Int32` that `PREDICTION_SCHEMA` declares, so every
# temporal-split test ran on a frame the store never produces. The Hive partition key
# shadowed the written column and `week` arrived as `"01"`, which `_holdout_weeks` turns
# into integers and `pl.col("week").is_in(...)` then compares against a string column.

def _write_predictions(base, model, probs, week, season=2026):
    import datetime as dt

    from hub import store
    n = len(probs)
    store.write(
        pl.DataFrame({
            "game_id": [f"{season}_{week:02d}_g{i}" for i in range(n)],
            "league": ["nfl"] * n,
            "season": pl.Series([season] * n, dtype=pl.Int32),
            "week": pl.Series([week] * n, dtype=pl.Int32),
            "home_win_prob": list(probs), "margin_mean": [0.0] * n,
            "margin_lo": [-17.0] * n, "margin_hi": [17.0] * n,
            "model": [model] * n, "version": ["v1"] * n,
            "fit_through_week": pl.Series([week - 1] * n, dtype=pl.Int32),
            "predicted_at": [dt.datetime(2026, 9, 1)] * n}),
        "preds", "nfl", season, week, base=base, name=model)


def test_a_comparison_runs_end_to_end_from_a_real_store(tmp_path):
    """The path `--compare` actually takes: store -> load_predictions -> compare, with the
    outcome joined on rather than handed in. This is the test that would have caught #59 --
    every other test in this file pre-joins `home_won` and so never enters the store path.

    It is also the test that would have caught #25: a padded string week makes the holdout
    filter compare a string column against integers.
    """
    base = tmp_path / "processed"
    for w in range(1, 11):
        _write_predictions(base, "a", [0.6] * 8, w)
        _write_predictions(base, "b", [0.5] * 8, w)
    sched = _sched([(f"2026_{w:02d}_g{i}", 7.0) for w in range(1, 11) for i in range(8)])

    a = me.load_predictions("a", base=base, schedules=sched)
    assert a.schema["week"].is_numeric(), "the store must hand back a numeric week"
    b = me.load_predictions("b", base=base, schedules=sched)

    got = me.compare(a, b, split="temporal", holdout=0.3)
    assert got["holdout_weeks"] == [8, 9, 10]
    assert got["n_scored"] == 24, "the last 3 of 10 weeks, 8 games each"
    assert got["delta"] < 0, "every home team won; 0.6 beats 0.5"


def test_the_store_path_reaches_the_cli(capsys, tmp_path, monkeypatch):
    """`--compare` against a store, with only the schedules stubbed. The defect was that no
    real store could get this far, so the CLI's own store path needs driving too."""
    base = tmp_path / "processed"
    for w in range(1, 11):
        _write_predictions(base, "a", [0.6] * 8, w)
        _write_predictions(base, "b", [0.5] * 8, w)
    sched = _sched([(f"2026_{w:02d}_g{i}", 7.0) for w in range(1, 11) for i in range(8)])
    monkeypatch.setattr(me, "_schedules", lambda: sched)

    assert me.main(["--compare", "a,b", "--store", str(base)]) == 0
    out = capsys.readouterr().out
    assert "on 24 games" in out and "weeks 8-10 held out" in out


# --- the outcome source can be down (issue #63) ------------------------------
#
# Closing #59 gave `load_predictions` a network call it never had -- the outcome is joined
# from nflverse's schedules. `main` was not widened to match: it wrapped only `NoOverlap`, so
# an unreachable nflverse left a traceback on a command that previously could not fail that
# way. `hub.publish._scored` had already settled the shape one file over -- catch broadly
# around the fetch, name the failure type rather than presenting every cause as one, and
# degrade. These drive the unreachable case by monkeypatching `_schedules`, so no test here
# touches the network.

def _two_model_store(tmp_path):
    """A store `--compare a,b` can actually read, so the CLI reaches the fetch."""
    base = tmp_path / "processed"
    for w in (1, 2):
        _write_predictions(base, "a", [0.6] * 4, w)
        _write_predictions(base, "b", [0.5] * 4, w)
    return base


def test_an_unreachable_schedules_source_is_reported_not_raised(capsys, tmp_path,
                                                                monkeypatch):
    def _down():
        raise ConnectionError("nflverse unreachable")

    monkeypatch.setattr(me, "_schedules", _down)
    assert me.main(["--compare", "a,b", "--store", str(_two_model_store(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "ConnectionError" in err, "the failure type, as `hub.publish._scored` prints it"
    assert "schedules" in err


def test_schedules_missing_the_realised_margin_is_reported_not_raised(capsys, tmp_path,
                                                                      monkeypatch):
    """A frame that arrived but has no `result` is a different fix from a frame that never
    arrived, and it used to leave a `ValueError` traceback in exactly the same place."""
    monkeypatch.setattr(me, "_schedules", lambda: pl.DataFrame({"game_id": ["g1"]}))
    assert me.main(["--compare", "a,b", "--store", str(_two_model_store(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "result" in err


def test_the_three_failure_causes_do_not_share_one_message(capsys, tmp_path, monkeypatch):
    """An unreachable source, a frame with no realised margin, and two models sharing no
    games are three different fixes. Collapsing them into one sentence is what sends the
    reader to check the wrong thing first."""
    base = _two_model_store(tmp_path)

    def _down():
        raise TimeoutError("nflverse timed out")

    said = []
    for schedules in (_down,
                      lambda: pl.DataFrame({"game_id": ["g1"]}),
                      lambda: _sched([("no_such_game", 7.0)])):
        monkeypatch.setattr(me, "_schedules", schedules)
        assert me.main(["--compare", "a,b", "--store", str(base)]) == 1
        said.append(capsys.readouterr().err.strip())

    assert len(set(said)) == 3, f"three causes, {len(set(said))} distinct messages: {said}"
    assert "TimeoutError" in said[0]
    assert "result" in said[1]
    assert "share no games" in said[2]


def test_the_underlying_fetch_failure_is_kept_as_the_cause(monkeypatch):
    """Broad like `_scored`'s, but not swallowing: the original is chained, so a stack trace
    is still there for whoever goes looking."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: {"preds"})
    monkeypatch.setattr(store, "sql", lambda *a, **k: _stored([("g1", 1, 0.6)]))

    boom = ConnectionError("nflverse unreachable")

    def _down():
        raise boom

    monkeypatch.setattr(me, "_schedules", _down)
    with pytest.raises(me.OutcomesUnavailable) as e:
        me.load_predictions("m")
    assert e.value.__cause__ is boom


def test_a_model_with_nothing_stored_never_reaches_the_fetch(monkeypatch):
    """The normal state before the first prediction is written, and the frame it returns is
    named for meaning that: nothing predicted. It must not cost a fetch to say so -- and a
    fresh clone with nflverse down still has to reach the no-overlap message."""
    import hub.store as store
    monkeypatch.setattr(store, "tables", lambda *a, **k: set())

    def _down():
        raise ConnectionError("nflverse unreachable")

    monkeypatch.setattr(me, "_schedules", _down)
    got = me.load_predictions("m")
    assert got.is_empty() and "home_won" in got.columns
