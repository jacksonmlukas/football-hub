"""Kickoff and the betting market's own price on every prediction row (#482)."""
import datetime as dt
import json

import polars as pl
import pytest

from hub import market_row, publish, store
from hub.fetch.replay import serve
from hub.models.market import MARGIN_SD, normal_cdf

KICK = dt.datetime(2026, 10, 18, 17, 0)          # Sunday 13:00 Eastern


@pytest.fixture(autouse=True)
def offline():
    serve(schedules=_sched(["g1", "g2"]))


@pytest.fixture
def site(tmp_path):
    return tmp_path / "site" / "data"


@pytest.fixture
def base(tmp_path):
    return tmp_path / "processed"


def _sched(ids, *, times=True):
    frame = pl.DataFrame(
        {"game_id": [*ids, "pad"], "season": [2026] * (len(ids) + 1),
         "week": [6] * (len(ids) + 1), "home_team": ["H"] * (len(ids) + 1),
         "away_team": ["A"] * (len(ids) + 1), "result": [None] * (len(ids) + 1)},
        schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "result": pl.Float64})
    return (frame.with_columns(gameday=pl.lit("2026-10-18"), gametime=pl.lit("13:00"))
            if times else frame)


def _preds(rows):
    """rows: (game_id, spread, priced_at | None)."""
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows], "league": ["nfl"] * len(rows),
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([6] * len(rows), dtype=pl.Int32),
         "home_win_prob": [normal_cdf(r[1] / MARGIN_SD) for r in rows],
         "margin_mean": [r[1] for r in rows],
         "margin_lo": [r[1] - 17.0 for r in rows], "margin_hi": [r[1] + 17.0 for r in rows],
         "model": ["market_baseline"] * len(rows), "version": ["v1"] * len(rows),
         "price_source": ["live" if r[2] else "schedule" for r in rows],
         "priced_at": pl.Series([r[2] for r in rows], dtype=pl.Datetime),
         "fit_through_week": pl.Series([5] * len(rows), dtype=pl.Int32),
         "predicted_at": [dt.datetime(2026, 10, 17)] * len(rows)})


def _capture(base, game, when, spread, total=44.5, ml=None):
    store.write_snapshot(
        pl.DataFrame({"game_id": [game], "close_spread": [spread], "spread_price": [-110.0],
                      "close_total": [total], "total_price": [-110.0], "ml_home_prob": [ml],
                      "captured_at": [when], "polls_unmoved": [1], "unmoved_since": [when]},
                     schema=store.SNAPSHOT_COLUMNS), 2026, 6, when, base=base)


def _rows(site, base, preds):
    store.write(preds, "preds", "nfl", 2026, 6, base=base)
    publish.predictions(2026, 6, base=base, out=site)
    return {r["game_id"]: r for r in
            json.loads((site / "preds_2026_wk06.json").read_text())["rows"]}


def test_a_priced_row_carries_kickoff_with_offset_and_the_captures_own_price(site, base):
    when = dt.datetime(2026, 10, 17, 9, 0)
    _capture(base, "g1", when, -3.5, total=47.5, ml=0.61)
    row = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    assert dt.datetime.fromisoformat(row["kickoff"]).utcoffset() == dt.timedelta(0)
    assert dt.datetime.fromisoformat(row["kickoff"]).replace(tzinfo=None) == KICK
    assert row["market_spread"] == -3.5 and row["market_total"] == 47.5
    assert row["market_prob_ml"] == pytest.approx(0.61)
    assert 0 < row["market_prob"] < 1
    assert row["market_missing"] is None and row["kickoff_missing"] is None
    assert row["priced_at"] is not None and row["price_source"] == "live"


def test_on_a_market_baseline_row_the_two_probabilities_cannot_drift(site, base):
    """The page draws no model-versus-market gap while these are the same number."""
    when = dt.datetime(2026, 10, 17, 9, 0)
    _capture(base, "g1", when, 6.5)
    row = _rows(site, base, _preds([("g1", 6.5, when)]))["g1"]
    assert row["model"] == "market_baseline"
    assert abs(row["home_win_prob"] - row["market_prob"]) < 1e-9


def test_a_game_with_no_capture_has_null_market_fields_and_a_reason_never_zero(site, base):
    row = _rows(site, base, _preds([("g1", 3.0, None)]))["g1"]
    assert row["market_prob"] is None and row["market_spread"] is None
    assert row["market_total"] is None and row["market_prob_ml"] is None
    assert "no capture" in row["market_missing"]
    assert row["kickoff"] is not None                      # the schedule still answers


def test_a_capture_after_kickoff_is_never_the_published_market_price(site, base):
    late = KICK + dt.timedelta(minutes=40)
    _capture(base, "g1", late, 9.0)
    row = _rows(site, base, _preds([("g1", 9.0, late)]))["g1"]
    assert row["market_prob"] is None and row["market_spread"] is None
    assert "after kickoff" in row["market_missing"]


def test_without_a_moneyline_the_ml_probability_is_null_not_the_spread_one(site, base):
    when = dt.datetime(2026, 10, 17, 9, 0)
    _capture(base, "g1", when, -3.5, ml=None)
    row = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    assert row["market_prob"] is not None and row["market_prob_ml"] is None


def test_the_kickoff_join_is_what_puts_a_kickoff_on_the_row(site, base, monkeypatch):
    """Planted: with the kickoff join removed the row has no kickoff and withholds the price."""
    when = dt.datetime(2026, 10, 17, 9, 0)
    _capture(base, "g1", when, -3.5)
    monkeypatch.setattr(market_row, "kickoffs", lambda season: ({}, "planted"))
    row = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    assert row["kickoff"] is None and row["kickoff_missing"] == "planted"
    assert row["market_prob"] is None                     # cannot show the capture precedes it


def test_a_schedule_without_times_degrades_to_a_reason(site, base):
    serve(schedules=_sched(["g1"], times=False))
    when = dt.datetime(2026, 10, 17, 9, 0)
    _capture(base, "g1", when, -3.5)
    row = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    assert row["kickoff"] is None and "no kickoff times" in row["kickoff_missing"]


def test_a_row_already_published_keeps_the_price_it_was_published_with(site, base):
    when = dt.datetime(2026, 10, 17, 9, 0)
    _capture(base, "g1", when, -3.5)
    first = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    _capture(base, "g1", dt.datetime(2026, 10, 17, 10, 0), 10.0)       # a later poll
    again = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    assert {k: again[k] for k in market_row.MARKET_FIELDS} == \
        {k: first[k] for k in market_row.MARKET_FIELDS}


def test_attach_leaves_rows_that_carry_the_fields_untouched():
    rows = [{"game_id": "g1", "market_prob": 0.5}]
    assert market_row.attach(rows, 2026) == [{"game_id": "g1", "market_prob": 0.5}]


def test_moment_reads_what_a_row_holds_and_refuses_what_it_cannot():
    assert market_row._moment("2026-10-17 09:00:00") == dt.datetime(2026, 10, 17, 9)
    assert market_row._moment("not a time") is None and market_row._moment("") is None
    aware = dt.datetime(2026, 10, 17, 5, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    assert market_row._moment(aware) == dt.datetime(2026, 10, 17, 9)


def test_an_unreadable_store_prices_nothing_and_the_row_says_the_capture_is_absent(
        site, base, monkeypatch):
    when = dt.datetime(2026, 10, 17, 9, 0)

    def boom(season, base=None):
        raise RuntimeError("store locked")
    monkeypatch.setattr(market_row.store, "lines", boom)
    row = _rows(site, base, _preds([("g1", -3.5, when)]))["g1"]
    assert row["market_prob"] is None and "not in the store" in row["market_missing"]
