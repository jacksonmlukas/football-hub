"""The prop card and its scorecard (#217).

Every test here runs against fixtures. There is no props pull in this repo -- a props market
runs about four credits an event -- so the pricing and the logging are proved on hand-built
quotes, and the day a payload is paid for the scorecard exists to receive it.
"""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from hub.models import components as C
from hub.models import props
from hub.models.predict import components as component_line

T0 = dt.datetime(2026, 9, 10, 12)


def _at(hours: float) -> dt.datetime:
    return T0 + dt.timedelta(hours=hours)


def _players(**extra) -> pl.DataFrame:
    return pl.DataFrame({"player": ["Justin Jefferson", "Kicker Guy", "Bijan Robinson"],
                         "pos": ["WR", "K", "RB"], "adp": [3.0, 150.0, 2.0],
                         "proj_blend": [17.0, 8.0, 18.0], **extra})


def _poll(player, market, point, over, under=-110.0, at=T0, game="2026_02_MIN_ATL",
          unmoved=1, since=None):
    return {"game_id": game, "player_key": props.player_key(player), "player": player,
            "market": market, "point": point, "over_price": float(over),
            "under_price": None if under is None else float(under), "captured_at": at,
            "polls_unmoved": unmoved, "unmoved_since": since or at}


_POLL_SCHEMA = {"game_id": pl.Utf8, "player_key": pl.Utf8, "player": pl.Utf8,
                "market": pl.Utf8, "point": pl.Float64, "over_price": pl.Float64,
                "under_price": pl.Float64, "captured_at": pl.Datetime,
                "polls_unmoved": pl.Int64, "unmoved_since": pl.Datetime}


def _polls(*rows) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=_POLL_SCHEMA)


WR = component_line(3.0, "WR", 17.0)


# --- the distribution behind a component -----------------------------------------

@pytest.mark.parametrize(("stat", "phase"), [("receptions", "rec"), ("carries", "rush"),
                                             ("attempts", "pass")])
def test_a_count_carries_the_measured_dispersion(stat, phase):
    """Var/mean is the number `hub.models.components` measured for the phase, not one."""
    line = {stat: 6.0}
    d = props.stat_draws(line, stat, n=200_000)
    assert d.mean() == pytest.approx(6.0, rel=0.02)
    assert d.var() / d.mean() == pytest.approx(C.COUNT_DISPERSION[phase], rel=0.05)


def test_a_touchdown_count_is_underdispersed():
    d = props.stat_draws({"passing_tds": 1.8}, "passing_tds", n=200_000)
    assert d.var() / d.mean() == pytest.approx(C.TD_DISPERSION["pass"], rel=0.05)
    assert d.var() / d.mean() < 1.0


def test_yardage_spread_grows_with_the_root_of_the_mean_not_the_mean():
    """The square-root law is an output of compounding on the count. A Gamma with a fixed CV
    on the weekly total would put the ratio at four."""
    lo = props.stat_draws({"receptions": 2.0, "receiving_yards": 24.0}, "receiving_yards",
                          n=100_000)
    hi = props.stat_draws({"receptions": 8.0, "receiving_yards": 96.0}, "receiving_yards",
                          n=100_000)
    assert hi.mean() / lo.mean() == pytest.approx(4.0, rel=0.03)
    assert hi.std() / lo.std() == pytest.approx(2.0, abs=0.25)


def test_the_mean_is_the_lines_own():
    for stat in ("receiving_yards", "receptions", "receiving_tds", "rushing_yards"):
        assert props.stat_draws(WR, stat, n=100_000).mean() == pytest.approx(WR[stat], rel=0.03)


def test_yards_without_a_count_infer_one_rather_than_losing_the_compound_shape():
    d = props.stat_draws({"rushing_yards": 43.0}, "rushing_yards", n=100_000)
    assert d.mean() == pytest.approx(43.0, rel=0.03)
    assert d.std() > 0


def test_anytime_touchdown_is_at_least_one_from_either_phase():
    both = props.price({"rushing_tds": 0.3, "receiving_tds": 0.2}, "player_anytime_td", None)
    rush = props.price({"rushing_tds": 0.3}, "player_anytime_td", None)
    assert both.point is None and both.p_push == 0.0
    assert both.p_over is not None and rush.p_over is not None
    assert both.p_over > rush.p_over
    assert both.p_over == pytest.approx(1 - (1 - rush.p_over) * (1 - 0.2 * 0.97), abs=0.03)


def test_over_under_and_push_partition_the_mass():
    """A count quoted at a whole number has mass on the point; a yardage quote does not."""
    p = props.price(WR, "player_receptions", 5.0)
    assert p.p_push is not None and p.p_push > 0.05
    assert p.p_over is not None and p.p_under is not None
    assert p.p_over + p.p_under + p.p_push == pytest.approx(1.0)
    y = props.price(WR, "player_reception_yds", 72.5)
    assert y.p_push == 0.0
    assert y.p50 < y.mean, "yardage is right-skewed, so the median week is under the mean"


def test_a_market_with_no_point_is_priced_for_its_distribution_alone():
    p = props.price(WR, "player_reception_yds", None)
    assert p.p_over is None and p.fair_over is None
    assert p.mean == pytest.approx(WR["receiving_yards"], rel=0.03)
    assert p.p10 < p.p50 < p.p90


def test_the_same_line_prices_the_same_on_every_call():
    a, b = props.price(WR, "player_reception_yds", 72.5), props.price(WR, "player_reception_yds", 72.5)
    assert a == b


def test_the_component_draws_recompose_to_the_points_draw():
    """The prop distribution is the one `sample_weeks` aggregates, drawn one component at a
    time. If the two ever disagree on the points they imply, one of them has drifted.

    The mean is what recomposes. The spread does not, and should not: inside `sample_weeks`
    one drawn reception count carries both the catches and the yards on them, so the points
    total is a *correlated* sum, while a prop is one component on its own and each is drawn
    apart here. The independent sum is narrower by construction, which is asserted rather
    than tolerated."""
    pts = C.sample_weeks(WR, "WR", 40_000, np.random.default_rng(1))
    total = np.zeros(40_000)
    for i, s in enumerate(("receiving_yards", "rushing_yards", "receptions",
                           "receiving_tds", "rushing_tds")):
        total += C.SCORING[s] * props.stat_draws(WR, s, n=40_000, seed=i)
    assert total.mean() == pytest.approx(pts.mean(), rel=0.03)
    assert total.std() < pts.std()


# --- prices --------------------------------------------------------------------------

def test_implied_probability_carries_the_vig_and_novig_removes_it():
    assert props.implied(-110) == pytest.approx(0.5238, abs=1e-3)
    assert props.implied(+150) == pytest.approx(0.4)
    assert props.implied(50) is None, "inside the hole American odds have is not a price"
    assert props.novig_over(-110, -110) == pytest.approx(0.5)
    assert props.novig_over(-130, +110) == pytest.approx(0.5427, abs=1e-3)
    assert props.novig_over(-110, None) == pytest.approx(0.5238, abs=1e-3), \
        "a one-sided quote cannot be de-vigged and is returned as charged"
    assert props.novig_over(None, -110) is None


def test_a_fair_price_round_trips_through_implied():
    for p in (0.3, 0.5, 0.62, 0.9):
        assert props.implied(props.fair_price(p)) == pytest.approx(p, abs=1e-6)


# --- the card ------------------------------------------------------------------------

def test_every_posted_prop_is_priced_and_an_unposted_one_is_kept_not_dropped():
    quotes = _polls(_poll("Justin Jefferson", "player_reception_yds", 72.5, -115, -105),
                    _poll("Kicker Guy", "player_pass_yds", 10.5, -110, -110))
    got = props.card(_players(), quotes, decided_at=T0, n=4000)
    by = {(r["player"], r["market"]): r for r in got.iter_rows(named=True)}
    priced = by[("Justin Jefferson", "player_reception_yds")]
    assert priced["status"] == props.PRICED
    assert priced["decision_point"] == 72.5 and priced["side"] in (props.OVER, props.UNDER)
    assert priced["decided_at"] == T0 and priced["game_id"] == "2026_02_MIN_ATL"
    unposted = by[("Justin Jefferson", "player_receptions")]
    assert unposted["status"] == props.NO_LINE
    assert unposted["our_mean"] == pytest.approx(WR["receptions"], rel=0.05)
    assert unposted["decision_point"] is None and unposted["our_p_over"] is None
    assert unposted["decided_at"] == T0, "the decision point is stamped on the kept row too"
    kicker = by[("Kicker Guy", "player_pass_yds")]
    assert kicker["status"] == props.NO_NUMBER
    assert kicker["our_mean"] is None and kicker["decision_point"] == 10.5
    assert ("Justin Jefferson", "player_rush_yds") not in by, \
        "a receiver's rushing yards clear zero one week in eight; no book hangs the prop"
    assert ("Justin Jefferson", "player_pass_tds") not in by
    assert ("Justin Jefferson", "player_anytime_td") in by
    assert ("Bijan Robinson", "player_rush_attempts") in by
    assert ("Bijan Robinson", "player_reception_yds") in by


def _rec_yds(frame: pl.DataFrame) -> dict:
    return frame.filter((pl.col("market") == "player_reception_yds")
                        & (pl.col("player") == "Justin Jefferson")).row(0, named=True)


def test_the_side_is_where_our_probability_beats_the_vig_free_implied():
    low = _polls(_poll("Justin Jefferson", "player_reception_yds", 40.5, -110, -110))
    high = _polls(_poll("Justin Jefferson", "player_reception_yds", 120.5, -110, -110))
    over = _rec_yds(props.card(_players(), low, decided_at=T0, n=4000))
    under = _rec_yds(props.card(_players(), high, decided_at=T0, n=4000))
    assert over["side"] == props.OVER and over["edge"] > 0.2
    assert under["side"] == props.UNDER and under["edge"] > 0.2


def test_the_edge_is_against_the_vig_free_price_not_the_charged_one():
    """At a point where we sit exactly on 50%, a -110/-110 quote is no edge either way. Read
    against the charged 52.4% instead, every coin flip would look like an Under."""
    p = props.price(WR, "player_reception_yds", None)
    median = round(p.p50 * 2) / 2 + 0.5
    rows = [_rec_yds(props.card(_players(), _polls(
                _poll("Justin Jefferson", "player_reception_yds", median, over, under)),
                decided_at=T0))
            for over, under in ((-110, -110), (-105, -105), (100, 100), (-120, -120))]
    assert rows[0]["edge"] < 0.03
    assert len({r["side"] for r in rows}) == 1
    assert len({round(r["edge"], 9) for r in rows}) == 1, \
        "four quotes at the same vig-free 50% are one quote; only the charge differs"
    shaded = _rec_yds(props.card(_players(), _polls(
        _poll("Justin Jefferson", "player_reception_yds", median, -130, 110)), decided_at=T0))
    assert shaded["edge"] != rows[0]["edge"]


def test_a_push_is_refunded_before_the_sides_are_compared():
    """A count quoted at a whole number pushes on the point. The sides are compared on the
    mass that is not refunded, so a prop that is 51% over, 33% under and 16% push is a
    60% Over -- read on the raw 51% against a quote shaded to 53% it would be an Under."""
    p = props.price(WR, "player_receptions", 5.0)
    assert p.p_over is not None and p.p_push is not None and p.p_push > 0.05
    q_over = props.novig_over(-125, +105)
    assert q_over is not None
    quotes = _polls(_poll("Justin Jefferson", "player_receptions", 5.0, -125, +105))
    row = (props.card(_players(), quotes, decided_at=T0)
                .filter(pl.col("market") == "player_receptions").row(0, named=True))
    conditional = p.p_over / (1 - p.p_push)
    assert row["side"] == props.OVER and conditional > q_over
    assert row["edge"] == pytest.approx(conditional - q_over, abs=1e-6)
    assert p.p_over < q_over, \
        "the fixture must be one where the refund flips the side, or it proves nothing"


def test_the_decision_quote_is_the_one_live_at_the_moment():
    """`docs/method.md` rule 2: the predictor is read strictly before. The poll after the
    decision is not the decision's quote, and a prop first posted after it was not priced."""
    polls = _polls(
        _poll("Justin Jefferson", "player_reception_yds", 70.5, -110, at=_at(0), unmoved=1),
        _poll("Justin Jefferson", "player_reception_yds", 70.5, -110, at=_at(2), unmoved=2,
              since=_at(0)),
        _poll("Justin Jefferson", "player_reception_yds", 74.5, -110, at=_at(5), unmoved=1,
              since=_at(5)),
        _poll("Justin Jefferson", "player_receptions", 5.5, -110, at=_at(5)),
    )
    live = props.quotes_as_of(polls, _at(3))
    assert live.height == 1
    row = live.row(0, named=True)
    assert row["point"] == 70.5 and row["captured_at"] == _at(2)
    assert row["polls_unmoved"] == 2 and row["unmoved_since"] == _at(0)
    card = props.card(_players(), live, decided_at=_at(3), n=2000)
    by = {r["market"]: r for r in card.filter(pl.col("player") == "Justin Jefferson")
                                    .iter_rows(named=True)}
    assert by["player_reception_yds"]["decision_captured_at"] == _at(2)
    assert by["player_reception_yds"]["decision_polls_unmoved"] == 2
    assert by["player_reception_yds"]["decision_unmoved_since"] == _at(0)
    assert by["player_receptions"]["status"] == props.NO_LINE, \
        "posted two hours after the decision is not a line we could have decided on"


# --- the close and the scorecard -----------------------------------------------------

def _priced(side_point: float, close_point: float, *, over=-110, under=-110,
            close_over=-110, close_under=-110):
    polls = _polls(_poll("Justin Jefferson", "player_reception_yds", side_point, over, under,
                         at=_at(0)),
                   _poll("Justin Jefferson", "player_reception_yds", close_point, close_over,
                         close_under, at=_at(30)))
    return props.log_decisions(_players(), polls, decided_at=_at(1), n=4000).filter(
        pl.col("market") == "player_reception_yds").row(0, named=True)


def test_clv_is_signed_toward_our_side():
    """Our number is about 75. Over at 40.5 and the close at 44.5 is +4 for us; Under at
    120.5 and the close at 124.5 moved away, so it is -4."""
    over = _priced(40.5, 44.5)
    assert over["side"] == props.OVER and over["clv_points"] == pytest.approx(4.0)
    under = _priced(120.5, 124.5)
    assert under["side"] == props.UNDER and under["clv_points"] == pytest.approx(-4.0)
    assert over["close_point"] == 44.5 and over["close_captured_at"] == _at(30)


def test_clv_in_probability_is_the_vig_free_move_on_our_side():
    q_dec, q_close = props.novig_over(-120, 100), props.novig_over(-130, 110)
    assert q_dec is not None and q_close is not None and q_dec != 0.5
    over = _priced(40.5, 40.5, over=-120, under=100, close_over=-130, close_under=+110)
    assert over["clv_points"] == 0.0
    assert over["clv_prob"] == pytest.approx(q_close - q_dec, abs=1e-6)
    under = _priced(120.5, 120.5, over=-120, under=100, close_over=-130, close_under=+110)
    assert under["clv_prob"] == pytest.approx(q_dec - q_close, abs=1e-6)


def test_the_close_can_be_read_as_of_kickoff():
    polls = _polls(_poll("Justin Jefferson", "player_receptions", 5.5, -110, at=_at(0)),
                   _poll("Justin Jefferson", "player_receptions", 6.5, -110, at=_at(10)),
                   _poll("Justin Jefferson", "player_receptions", 7.5, -110, at=_at(20)))
    assert props.closing_quotes(polls).row(0, named=True)["close_point"] == 7.5
    assert props.closing_quotes(polls, _at(12)).row(0, named=True)["close_point"] == 6.5


def test_clv_is_null_where_there_was_no_decision():
    polls = _polls(_poll("Kicker Guy", "player_pass_yds", 10.5, -110, at=_at(0)),
                   _poll("Kicker Guy", "player_pass_yds", 12.5, -110, at=_at(30)))
    log = props.log_decisions(_players(), polls, decided_at=_at(1), n=2000)
    for r in log.iter_rows(named=True):
        assert r["clv_points"] is None and r["clv_prob"] is None
    assert set(log["status"]) == {props.NO_LINE, props.NO_NUMBER}
    assert log["model"].unique().to_list() == [props.MODEL]


def test_the_version_moves_with_the_dispersions_this_module_reads(monkeypatch):
    before = props.version()
    monkeypatch.setattr(C, "COUNT_DISPERSION", {**C.COUNT_DISPERSION, "rec": 1.5})
    assert props.version() != before
    assert props.version().startswith(f"{props.MODEL}-")


# --- the report ----------------------------------------------------------------------

def _log_row(player, market, clv_pts, clv_prob, *, our=None, close=None, status=props.PRICED,
             decided=T0):
    return {"game_id": "g", "player_key": props.player_key(player), "player": player,
            "position": "WR", "market": market, "stat": props.MARKET_STATS[market],
            "status": status, "decided_at": decided, "our_mean": our, "our_sd": 10.0,
            "our_p50": our, "our_p_over": 0.5, "side": props.OVER, "edge": 0.05,
            "decision_point": None if close is None else close - clv_pts,
            "decision_over_price": -110.0, "decision_under_price": -110.0,
            "decision_captured_at": decided, "decision_polls_unmoved": 1,
            "decision_unmoved_since": decided, "close_point": close,
            "close_over_price": -110.0, "close_under_price": -110.0,
            "close_captured_at": decided, "clv_points": clv_pts, "clv_prob": clv_prob,
            "model": props.MODEL, "version": "v"}


def _log(*rows) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=props._CARD_SCHEMA | {
        "close_point": pl.Float64, "close_over_price": pl.Float64,
        "close_under_price": pl.Float64, "close_captured_at": pl.Datetime,
        "clv_points": pl.Float64, "clv_prob": pl.Float64, "model": pl.Utf8,
        "version": pl.Utf8})


def test_the_error_is_taken_over_players_not_props():
    """`docs/method.md` rule 3. Six props on one player at +1 and one on another at -1 are
    two observations of a process, not seven: the mean over players is zero and the mean
    over props would be +0.71."""
    rows = [_log_row("A", "player_reception_yds", 1.0, 0.01, our=70, close=71 + i)
            for i in range(6)]
    rows.append(_log_row("B", "player_reception_yds", -1.0, -0.01, our=70, close=69))
    got = props.clv_by_market(_log(*rows)).filter(pl.col("market") == "player_reception_yds")
    r = got.row(0, named=True)
    assert r["props"] == 7 and r["players"] == 2
    assert r["clv_points"] == pytest.approx(0.0)
    assert r["clv_prob"] == pytest.approx(0.0)
    assert r["se_points"] == pytest.approx(1.0)      # sd of (+1, -1) over sqrt(2)


def test_the_ceiling_is_the_absolute_move_and_the_share_is_against_it():
    """Rule 8. A side-picker that was always right logs the whole move; we log what we
    captured of it, and the report says both."""
    rows = [_log_row("A", "player_receptions", 1.0, 0.02, our=5, close=6),
            _log_row("B", "player_receptions", -1.0, -0.02, our=5, close=4),
            _log_row("C", "player_receptions", 2.0, 0.04, our=5, close=7),
            _log_row("D", "player_receptions", 0.0, 0.0, our=5, close=5)]
    r = (props.clv_by_market(_log(*rows)).filter(pl.col("market") == "player_receptions")
              .row(0, named=True))
    assert r["ceiling_points"] == pytest.approx(1.0)
    assert r["clv_points"] == pytest.approx(0.5)
    assert r["ceiling_prob"] == pytest.approx(0.02)
    assert r["share"] == pytest.approx(0.5)
    assert r["unmoved"] == 1 and r["hit"] == pytest.approx(2 / 3)


def test_the_anytime_market_has_no_points_column():
    r = (props.clv_by_market(_log(_log_row("A", "player_anytime_td", None, 0.03)))
              .filter(pl.col("market") == "player_anytime_td").row(0, named=True))
    assert r["clv_points"] is None and r["ceiling_points"] is None
    assert r["clv_prob"] == pytest.approx(0.03)


def test_the_baseline_comparison_is_our_number_against_the_close_on_yardage_only():
    rows = [_log_row("A", "player_reception_yds", 0.0, 0.0, our=80.0, close=70.0),
            _log_row("B", "player_rush_yds", 0.0, 0.0, our=45.0, close=50.0),
            _log_row("C", "player_receptions", 0.0, 0.0, our=9.0, close=5.0),
            _log_row("D", "player_pass_yds", 0.0, 0.0, our=None, close=250.0,
                     status=props.NO_NUMBER)]
    b = props.baseline_comparison(_log(*rows))
    assert b["n"] == 2 and b["players"] == 2
    assert b["bias_yards"] == pytest.approx(2.5)
    assert b["bias_share"] == pytest.approx(5.0 / 120.0)
    assert b["mae_yards"] == pytest.approx(7.5)
    assert props.baseline_comparison(_log(rows[2]))["n"] == 0


def test_the_report_names_the_recorded_baseline_beside_the_new_figure(capsys):
    props.report(_log(_log_row("A", "player_reception_yds", 1.0, 0.01, our=80.0, close=70.0),
                      _log_row("B", "player_receptions", None, None, status=props.NO_LINE)))
    out = capsys.readouterr().out
    assert "1 priced, 1 with no posted line" in out
    assert "bias  10.00 yds ( 14.3%), MAE 10.00 yds" in out
    assert f"n={props.BASELINE_N}" in out and f"+{props.BASELINE_BIAS_YARDS} yds" in out
    assert f"MAE {props.BASELINE_MAE_YARDS} yds" in out


def test_coverage_counts_every_state():
    log = _log(_log_row("A", "player_receptions", 0.0, 0.0, our=5, close=5),
               _log_row("B", "player_receptions", None, None, status=props.NO_LINE),
               _log_row("C", "player_receptions", None, None, status=props.NO_NUMBER),
               _log_row("D", "player_receptions", None, None, status=props.NO_NUMBER))
    assert props.coverage(log) == {props.PRICED: 1, props.NO_LINE: 1, props.NO_NUMBER: 2}


# --- the store round trip and the CLI --------------------------------------------------

def test_the_log_is_written_under_its_contract_and_read_back_by_the_report(tmp_path, capsys):
    polls = _polls(_poll("Justin Jefferson", "player_reception_yds", 40.5, -110, at=_at(0)),
                   _poll("Justin Jefferson", "player_reception_yds", 44.5, -110, at=_at(30)))
    log = props.log_decisions(_players(), polls, decided_at=_at(1), n=2000)
    path = props.write_log(log, 2026, 2, base=tmp_path)
    assert path.exists() and "decided-" in path.name
    assert props.main(["--report", "--season", "2026", "--base", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "player_reception_yds" in out and "1 priced" in out


def test_the_log_command_prices_a_players_file_against_the_archive(tmp_path, capsys):
    from hub import store
    polls = _polls(_poll("Justin Jefferson", "player_reception_yds", 40.5, -110, at=_at(0)),
                   _poll("Justin Jefferson", "player_reception_yds", 44.5, -110, at=_at(30)))
    store.write(polls, "prop_lines", "nfl", 2026, 2, base=tmp_path, name="snap-a")
    players = tmp_path / "players.parquet"
    _players().write_parquet(players)
    code = props.main(["--log", "--players", str(players), "--decided-at", _at(1).isoformat(),
                       "--season", "2026", "--week", "2", "--base", str(tmp_path)])
    assert code == 0
    assert "prop_log" in store.tables(tmp_path)
    assert "wrote" in capsys.readouterr().out


def test_a_fresh_clone_is_told_there_is_no_log(tmp_path, capsys):
    assert props.main(["--report", "--base", str(tmp_path)]) == 1
    assert "no prop_log" in capsys.readouterr().err


def test_log_without_its_inputs_is_a_sentence(tmp_path, capsys):
    assert props.main(["--log", "--base", str(tmp_path)]) == 2
    assert "--players" in capsys.readouterr().err
    code = props.main(["--log", "--players", str(tmp_path / "nope.parquet"),
                       "--decided-at", T0.isoformat(), "--week", "2", "--base", str(tmp_path)])
    assert code == 1
    assert "unavailable" in capsys.readouterr().err


def test_a_poisson_count_is_the_family_at_a_dispersion_of_one():
    """No measured phase sits at exactly one, so the branch is reached only by asking."""
    d = props._counts(np.random.default_rng(0), 4.0, 1.0, 100_000)
    assert d.var() / d.mean() == pytest.approx(1.0, rel=0.05)


def test_a_player_with_no_pick_or_no_projection_has_no_line_and_is_not_an_error():
    players = pl.DataFrame({"player": ["Has Both", "No Pick", "No Proj"],
                            "pos": ["WR", "WR", "WR"], "adp": [3.0, None, 4.0],
                            "proj_blend": [17.0, 15.0, None]})
    assert set(props.component_lines(players)) == {"has both"}


def test_a_quote_with_no_readable_over_price_is_decided_against_even_money():
    quotes = _polls(_poll("Justin Jefferson", "player_reception_yds", 40.5, -110, None)
                    ).with_columns(pl.lit(None, dtype=pl.Float64).alias("over_price"))
    row = _rec_yds(props.card(_players(), quotes, decided_at=T0, n=2000))
    assert row["side"] == props.OVER and row["edge"] > 0.2


def test_the_log_command_names_the_missing_archive(tmp_path, capsys):
    players = tmp_path / "players.parquet"
    _players().write_parquet(players)
    code = props.main(["--log", "--players", str(players), "--decided-at", T0.isoformat(),
                       "--week", "2", "--base", str(tmp_path)])
    assert code == 1
    assert "prop_lines" in capsys.readouterr().err


def test_the_report_names_a_season_the_log_does_not_hold(tmp_path, capsys):
    polls = _polls(_poll("Justin Jefferson", "player_reception_yds", 40.5, -110, at=_at(0)))
    log = props.log_decisions(_players(), polls, decided_at=_at(1), n=2000)
    props.write_log(log, 2026, 2, base=tmp_path)
    assert props.main(["--report", "--season", "2027", "--base", str(tmp_path)]) == 1
    assert "no 2027 rows" in capsys.readouterr().err


def test_players_without_a_pick_or_a_projection_are_refused_by_name():
    with pytest.raises(ValueError, match="proj_blend"):
        props.component_lines(pl.DataFrame({"player": ["a"], "pos": ["WR"], "adp": [1.0]}))
    with pytest.raises(ValueError, match="position"):
        props.component_lines(pl.DataFrame({"player": ["a"], "adp": [1.0], "proj_ppg": [1.0]}))


def test_no_props_market_is_in_the_odds_budget():
    """The one line in this file that is about money. `hub.fetch.odds.MARKETS` is the whole
    of what a poll may ask for, and every market this module prices is outside it."""
    from hub.fetch import odds
    assert not set(props.MARKET_STATS) & set(odds.MARKETS)
    assert set(props.MARKET_STATS) == set(odds.PROP_MARKETS)
