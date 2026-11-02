"""`hub.forecasters.market`: the study's benchmark, a de-vigged moneyline from the hub's feed.

The prereg's rule, tested by hand-computed values: convert each side to decimal odds, normalise
each *book's* implied probabilities to sum to 1 (proportional de-vig), then take the median
home probability across books. With two sides proportional de-vig reduces to
p_home = o_away / (o_home + o_away), which is how every expected value below was worked.
Nothing here reaches the network.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from hub.fetch import odds
from hub.forecasters import market

HOME, AWAY = "Kansas City Chiefs", "Buffalo Bills"
NOW = datetime(2026, 10, 11, 13, 35, tzinfo=UTC)


def _book(key: str, home: float | None, away: float | None, *, spread: float | None = None,
          extra: list[dict] | None = None) -> dict:
    h2h = [o for o in ({"name": HOME, "price": home} if home is not None else None,
                       {"name": AWAY, "price": away} if away is not None else None) if o]
    markets = [{"key": "h2h", "last_update": "2026-10-11T13:20:00Z", "outcomes": h2h + (extra or [])}]
    if spread is not None:
        markets.append({"key": "spreads", "last_update": "2026-10-11T13:20:00Z",
                        "outcomes": [{"name": HOME, "price": 1.91, "point": spread},
                                     {"name": AWAY, "price": 1.91, "point": -spread}]})
    return {"key": key, "title": key, "markets": markets}


def _event(*books: dict, eid: str = "ev1") -> dict:
    return {"id": eid, "home_team": HOME, "away_team": AWAY,
            "commence_time": "2026-10-11T17:00:00Z", "bookmakers": list(books)}


# --- the de-vig ---------------------------------------------------------------------------

def test_one_book_de_vigs_to_known_values_and_sums_to_one():
    p_home, p_away = market.devig(1.80, 2.10)
    assert p_home == pytest.approx(2.1 / 3.9)            # 0.538461...
    assert p_away == pytest.approx(1.8 / 3.9)
    assert p_home + p_away == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(("home", "away"), [(1.01, 30.0), (1.91, 1.91), (3.4, 1.33), (2.0, 2.0)])
def test_every_book_sums_to_one(home, away):
    assert sum(market.devig(home, away)) == pytest.approx(1.0, abs=1e-12)


def test_even_odds_with_vig_de_vig_to_a_coin_flip():
    assert market.devig(1.91, 1.91)[0] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -110.0, float("nan"), float("inf")])
def test_a_price_that_is_not_decimal_odds_is_refused(bad):
    with pytest.raises(ValueError):
        market.devig(bad, 2.0)


def test_the_median_is_taken_after_de_vigging_each_book_not_of_each_side():
    """Per book: 3.0/4.5 = .6667, 1.9/3.8 = .5, 2.0/4.4 = .4545 -> median .5. Median sides
    first would pair home 1.9 with away 2.0, a quote no single book offered: 2.0/3.9 = .5128."""
    quote = market.quote(_event(_book("a", 1.5, 3.0), _book("b", 1.9, 1.9), _book("c", 2.4, 2.0)))
    assert quote is not None
    assert quote.home_win_prob == pytest.approx(0.5)
    assert quote.home_win_prob != pytest.approx(2.0 / 3.9)
    assert quote.books == 3


def test_an_even_number_of_books_takes_the_mean_of_the_middle_two():
    # .6667 and .5 -> .58333
    quote = market.quote(_event(_book("a", 1.5, 3.0), _book("b", 1.9, 1.9)))
    assert quote is not None and quote.home_win_prob == pytest.approx((2 / 3 + 0.5) / 2)


def test_a_book_missing_a_side_or_quoting_a_third_outcome_is_skipped_not_guessed():
    draw = [{"name": "Draw", "price": 25.0}]
    quote = market.quote(_event(_book("a", 1.80, 2.10), _book("b", 1.5, None),
                                _book("c", 1.5, 2.7, extra=draw)))
    assert quote is not None and quote.books == 1
    assert quote.home_win_prob == pytest.approx(2.1 / 3.9)


def test_no_moneyline_is_a_missing_line_not_a_coin_flip():
    assert market.quote(_event()) is None
    assert market.quote(_event(_book("a", None, None, spread=-3.0))) is None


def test_the_spread_is_the_median_home_point_across_books():
    quote = market.quote(_event(_book("a", 1.8, 2.1, spread=-3.0), _book("b", 1.8, 2.1, spread=-2.5),
                                _book("c", 1.8, 2.1, spread=-3.5)))
    assert quote is not None and quote.home_spread == pytest.approx(-3.0)


def test_a_moneyline_without_any_spread_carries_a_null_spread():
    quote = market.quote(_event(_book("a", 1.8, 2.1)))
    assert quote is not None and quote.home_spread is None


# --- the pull: budget, floor, one call, snapshot id ---------------------------------------

class _Get:
    def __init__(self, payload: list[dict], remaining: str = "400"):
        self.payload, self.remaining, self.calls = payload, remaining, []

    def __call__(self, params, key):
        self.calls.append(dict(params))
        return self.payload, {"x-requests-remaining": self.remaining, "x-requests-last": "2"}


def _pull(tmp_path, get, *, job="forecast", balance: int | None = 400, now=NOW):
    state = tmp_path / "odds.json"
    if balance is not None:
        state.write_text(json.dumps({"remaining": balance, "checked_at": now.isoformat()}))
    return market.snapshot(job, get=get, key="k" * 32, state_path=state, now=now,
                           match=lambda ev: "2026_06_BUF_KC" if ev["id"] == "ev1" else None), state


def test_the_forecast_pull_asks_for_moneyline_and_spreads_in_one_region_once(tmp_path):
    get = _Get([_event(_book("a", 1.8, 2.1, spread=-3.0))])
    _pull(tmp_path, get)
    assert get.calls == [{"markets": "h2h,spreads", "regions": "us", "oddsFormat": "decimal"}]


def test_the_closing_pull_asks_for_the_moneyline_only(tmp_path):
    get = _Get([_event(_book("a", 1.8, 2.1))])
    _pull(tmp_path, get, job="closing")
    assert get.calls[0]["markets"] == "h2h"


def test_an_undeclared_job_is_refused_before_a_request(tmp_path):
    get = _Get([])
    with pytest.raises(ValueError):
        _pull(tmp_path, get, job="props")
    assert get.calls == []


def test_the_study_floor_sits_below_the_hubs_so_the_hub_stops_first():
    assert market.STUDY_FLOOR < odds.CREDIT_FLOOR
    # the hub's floor must leave the study a worst-case month above the study's own floor
    assert odds.CREDIT_FLOOR - market.STUDY_FLOOR >= market.WORST_MONTH_CREDITS


def test_a_balance_below_the_study_floor_spends_nothing(tmp_path):
    get = _Get([])
    with pytest.raises(odds.QuotaFloor):
        _pull(tmp_path, get, balance=market.STUDY_FLOOR - 1)
    assert get.calls == []


def test_a_balance_the_hub_refuses_on_still_lets_the_study_pull(tmp_path):
    get = _Get([_event(_book("a", 1.8, 2.1, spread=-3.0))])
    _pull(tmp_path, get, balance=odds.CREDIT_FLOOR - 1)
    assert len(get.calls) == 1


def test_the_pull_writes_the_shared_balance_the_hubs_floor_reads(tmp_path):
    _, state = _pull(tmp_path, _Get([_event(_book("a", 1.8, 2.1, spread=-3.0))], remaining="398"))
    assert odds.credits_remaining(state) == 398


def test_the_snapshot_keys_quotes_by_game_and_drops_unmatched_events(tmp_path):
    snap, _ = _pull(tmp_path, _Get([_event(_book("a", 1.8, 2.1, spread=-3.0)),
                                    _event(_book("a", 1.5, 2.7), eid="unknown")]))
    assert set(snap.quotes) == {"2026_06_BUF_KC"}
    assert snap.unmatched == ("unknown",)
    assert snap.captured_at == NOW


def test_the_snapshot_id_names_the_moment_and_the_payload(tmp_path):
    a, _ = _pull(tmp_path, _Get([_event(_book("a", 1.8, 2.1, spread=-3.0))]))
    b, _ = _pull(tmp_path, _Get([_event(_book("a", 1.8, 2.2, spread=-3.0))]))
    assert a.snapshot_id.startswith("20261011T133500Z-")
    assert a.snapshot_id != b.snapshot_id                  # same moment, different bytes
    assert a.raw_sha256 != b.raw_sha256


def test_the_packet_input_carries_this_snapshots_id_and_time(tmp_path):
    snap, _ = _pull(tmp_path, _Get([_event(_book("a", 1.8, 2.1, spread=-3.0))]))
    m = snap.for_packet("2026_06_BUF_KC")
    assert m is not None
    assert (m.snapshot_id, m.snapshot_time) == (snap.snapshot_id, snap.captured_at)
    assert m.home_win_prob == pytest.approx(2.1 / 3.9) and m.home_spread == -3.0


def test_a_game_with_no_spread_cannot_feed_a_line_aware_packet(tmp_path):
    snap, _ = _pull(tmp_path, _Get([_event(_book("a", 1.8, 2.1))]))
    assert snap.for_packet("2026_06_BUF_KC") is None
    assert snap.quotes["2026_06_BUF_KC"].home_win_prob == pytest.approx(2.1 / 3.9)


def test_a_transport_failure_never_carries_the_key(tmp_path):
    key = "s3cr3t-key-value-0123456789"

    def boom(params, k):
        raise RuntimeError(f"500 for https://api.the-odds-api.com/v4/x?apiKey={k}")
    with pytest.raises(odds.MarketUnreachable) as exc:
        market.snapshot("forecast", get=boom, key=key, state_path=tmp_path / "s.json",
                        now=NOW, match=lambda ev: None)
    assert key not in str(exc.value)
