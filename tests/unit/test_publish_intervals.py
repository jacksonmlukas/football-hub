"""#481: an interval on every calibration bin, and the baselines each score is read against."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from hub import publish, store
from hub.fetch.replay import serve


def test_wilson_interval_matches_the_textbook_value():
    """7 home wins in 10 at 90%, worked by hand from z = 1.6449: centre 0.6575, half-width
    0.2157. Wald would give 0.4466-0.9534, so a change of method moves it."""
    got = publish.wilson_interval(7, 10)
    assert got is not None
    lo, hi = got
    assert lo == pytest.approx(0.4418, abs=5e-4)
    assert hi == pytest.approx(0.8732, abs=5e-4)


def test_an_empty_bin_has_a_null_interval_never_nan():
    assert publish.wilson_interval(0, 0) is None
    got = publish.with_interval({"bin": "0.0-0.1", "n": 0, "predicted": None, "actual": None,
                                 "gap": None})
    assert got["lo"] is None and got["hi"] is None


def test_all_wins_reach_one_and_no_wins_reach_zero_exactly():
    all_won, none_won = publish.wilson_interval(5, 5), publish.wilson_interval(0, 5)
    assert all_won is not None and none_won is not None
    assert all_won[1] == 1.0 and 0.0 < all_won[0] < 1.0
    assert none_won[0] == 0.0 and 0.0 < none_won[1] < 1.0


def _preds(ids, prob):
    n = len(ids)
    return pl.DataFrame(
        {"game_id": ids, "league": ["nfl"] * n,
         "season": pl.Series([2026] * n, dtype=pl.Int32),
         "week": pl.Series([1] * n, dtype=pl.Int32), "home_win_prob": [prob] * n,
         "margin_mean": [3.0] * n, "margin_lo": [-14.0] * n, "margin_hi": [20.0] * n,
         "model": ["market_baseline"] * n, "version": ["v1"] * n,
         "fit_through_week": pl.Series([0] * n, dtype=pl.Int32),
         "predicted_at": [dt.datetime(2026, 9, 1)] * n})


def _schedule(history_seasons, wins_per_season, now_ids, now_wins):
    ids, seasons, results = [], [], []
    for s in history_seasons:
        for i in range(4):
            ids.append(f"{s}_{i}"), seasons.append(s)
            results.append(7.0 if i < wins_per_season else -7.0)
    for i, g in enumerate(now_ids):
        ids.append(g), seasons.append(2026)
        results.append(7.0 if i < now_wins else -7.0)
    n = len(ids)
    return pl.DataFrame(
        {"game_id": ids, "season": seasons, "week": [1] * n, "home_team": ["H"] * n,
         "away_team": ["A"] * n, "result": results},
        schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "result": pl.Float64})


@pytest.fixture
def record(tmp_path):
    """Four 2026 games priced 0.65, three won; six earlier seasons at 3 home wins in 4."""
    site, base = tmp_path / "site" / "data", tmp_path / "processed"
    ids = [f"g{i}" for i in range(4)]
    store.write(_preds(ids, 0.65), "preds", "nfl", 2026, 1, base=base)
    publish.predictions(2026, 1, base=base, out=site)
    serve(schedules=_schedule(range(2020, 2026), 3, ids, 3))
    got = publish.track_record(base=base, out=site)
    assert isinstance(got, dict)
    return got


def test_every_bin_carries_an_interval_and_empty_bins_keep_their_place(record):
    bins = record["seasons"][0]["bins"]
    full = [b for b in bins if b["n"]]
    assert len(bins) == 10 and len(full) == 1
    assert full[0]["lo"] == pytest.approx(0.3562, abs=1e-3)
    assert full[0]["hi"] == pytest.approx(0.9421, abs=1e-3)
    empty = [b for b in bins if not b["n"]]
    assert len(empty) == 9 and all(b["lo"] is None and b["hi"] is None for b in empty)
    assert record["bins"] == bins


def test_the_coin_flip_is_computed_to_ln2_and_a_quarter(record):
    cf = record["seasons"][0]["baselines"]["coin_flip"]
    assert cf["log_loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert cf["brier"] == pytest.approx(0.25, abs=1e-4)
    assert cf["n"] == 4 and cf["p"] == 0.5


def test_the_home_constant_names_seasons_that_end_before_the_one_it_scores(record):
    s = record["seasons"][0]
    hc = s["baselines"]["home_constant"]
    assert hc["p"] == pytest.approx(0.75) and hc["n"] == 4
    assert hc["source_seasons"] == [2020, 2021, 2022, 2023, 2024, 2025]
    assert hc["source_n_games"] == 24
    assert max(hc["source_seasons"]) < s["season"]
    assert record["baselines"] == s["baselines"]


def test_the_home_constant_is_null_when_no_earlier_season_has_a_result():
    """A schedule holding only the scored season has no history to take a rate from; the
    season itself must not stand in for it (no lookahead)."""
    sched = _schedule([2025], 0, ["g0"], 1)
    serve(schedules=sched.with_columns(
        pl.when(pl.col("season") == 2025).then(None).otherwise(pl.col("result")).alias("result")))
    assert publish._home_rate_before(2026) is None


def test_the_home_constant_is_null_when_history_cannot_be_read():
    def down(keys):
        raise RuntimeError("down")
    serve(schedules=down)
    assert publish._home_rate_before(2026) is None
    assert publish._home_rate_before(None) is None
    out = publish._baselines(2026, [0.6, 0.4], [1, 0])
    assert out["home_constant"] is None and out["coin_flip"]["n"] == 2
