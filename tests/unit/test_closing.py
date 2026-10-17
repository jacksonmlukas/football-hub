"""The close of each game, its lead time, the capture gate, and closing-line value (#495).

Rule-18 controls, each planted so it could fail:
  * a capture after kickoff is never the close;
  * a capture six hours before kickoff is recorded with its 360-minute lead, not as the close;
  * a planted budget overrun (weekly cap, credit floor, an undeclared market) is refused before
    any request is formed.
"""
import datetime as dt

import polars as pl
import pytest

from hub import closing
from hub.fetch import odds

KICK = dt.datetime(2026, 10, 18, 17, 0)           # naive UTC, as every stamp in this repo is


def _kicks(*rows):
    return pl.DataFrame({"game_id": [g for g, _ in rows], "kickoff": [k for _, k in rows]},
                        schema={"game_id": pl.Utf8, "kickoff": pl.Datetime})


def _arc(*rows):
    """(game_id, captured_at, spread[, ml_home_prob])."""
    cols = {"game_id": [r[0] for r in rows], "captured_at": [r[1] for r in rows],
            "close_spread": [r[2] for r in rows]}
    schema = {"game_id": pl.Utf8, "captured_at": pl.Datetime, "close_spread": pl.Float64}
    if rows and len(rows[0]) > 3:
        cols["ml_home_prob"] = [r[3] for r in rows]
        schema["ml_home_prob"] = pl.Float64
    return pl.DataFrame(cols, schema=schema)


def _min(m):
    return KICK - dt.timedelta(minutes=m)


# --- closes ------------------------------------------------------------------------------

def test_the_last_capture_at_or_before_kickoff_is_the_close_with_its_lead():
    got = closing.closes(_arc(("g", _min(600), 3.0), ("g", _min(12), 3.5)), _kicks(("g", KICK)))
    row = got.row(0, named=True)
    assert row["close_spread"] == 3.5 and row["lead_minutes"] == 12.0
    assert row["is_true_close"] is True


def test_a_capture_after_kickoff_is_never_the_close():
    # Planted: the only late capture is the one that disagrees. If the filter were dropped the
    # close would be 9.0 at a lead of -45 minutes.
    arc = _arc(("g", _min(90), 3.0), ("g", KICK + dt.timedelta(minutes=45), 9.0))
    got = closing.closes(arc, _kicks(("g", KICK)))
    assert got["close_spread"].to_list() == [3.0]
    assert got["lead_minutes"].to_list() == [90.0]


def test_a_game_with_only_post_kickoff_captures_has_no_close():
    arc = _arc(("g", KICK + dt.timedelta(minutes=5), 9.0))
    assert closing.closes(arc, _kicks(("g", KICK))).is_empty()


def test_a_capture_six_hours_early_is_recorded_with_its_lead_and_is_not_the_close():
    got = closing.closes(_arc(("g", _min(360), 3.0)), _kicks(("g", KICK)))
    row = got.row(0, named=True)
    assert row["lead_minutes"] == 360.0
    assert row["is_true_close"] is False


def test_a_capture_exactly_at_the_lead_limit_is_still_a_true_close():
    got = closing.closes(_arc(("g", _min(closing.TRUE_CLOSE_LEAD_MIN), 3.0)),
                         _kicks(("g", KICK)))
    assert got["is_true_close"].to_list() == [True]


def test_a_game_without_a_kickoff_has_no_close():
    arc = _arc(("g", _min(10), 3.0))
    assert closing.closes(arc, _kicks(("g", None))).is_empty()


# --- due ---------------------------------------------------------------------------------

def test_due_inside_the_window_with_no_capture_yet():
    now = _min(10)
    assert closing.due(_arc(("g", _min(600), 3.0)), _kicks(("g", KICK)), now)["game_id"] \
        .to_list() == ["g"]


def test_not_due_outside_the_window():
    assert closing.due(_arc(), _kicks(("g", KICK)), _min(closing.CLOSE_WINDOW_MIN + 5)).is_empty()


def test_nothing_is_due_at_or_after_kickoff():
    assert closing.due(_arc(), _kicks(("g", KICK)), KICK).is_empty()
    assert closing.due(_arc(), _kicks(("g", KICK)), KICK + dt.timedelta(minutes=1)).is_empty()


def test_a_second_cycle_in_the_same_window_is_a_no_op():
    arc = _arc(("g", _min(11), 3.0))
    assert closing.due(arc, _kicks(("g", KICK)), _min(6)).is_empty()


def test_a_capture_from_the_morning_does_not_cover_the_window():
    assert not closing.due(_arc(("g", _min(600), 3.0)), _kicks(("g", KICK)), _min(10)).is_empty()


# --- capture: one call per kickoff time, and the guards ----------------------------------

class _Fetch:
    def __init__(self):
        self.calls = 0

    def __call__(self, season, **kw):
        self.calls += 1


def test_one_call_serves_every_game_at_that_kickoff():
    kicks = _kicks(("a", KICK), ("b", KICK), ("c", KICK))
    f = _Fetch()
    assert closing.capture(2026, now=_min(10), kickoffs=kicks, archive=_arc(), fetch=f) == 1
    assert f.calls == 1


def test_no_call_when_nothing_is_due():
    f = _Fetch()
    assert closing.capture(2026, now=_min(300), kickoffs=_kicks(("a", KICK)),
                           archive=_arc(), fetch=f) == 0
    assert f.calls == 0


def test_a_planted_weekly_overrun_is_refused_before_a_request_is_formed():
    # Ten captures this week inside some game's window: the cap is reached.
    k = [(f"p{i}", KICK - dt.timedelta(days=1, hours=i)) for i in range(closing.CLOSE_CALLS_PER_WEEK)]
    arc = _arc(*[(g, kk - dt.timedelta(minutes=5), 1.0) for g, kk in k])
    kicks = _kicks(*k, ("g", KICK))
    f = _Fetch()
    with pytest.raises(closing.CloseBudgetExceeded):
        closing.capture(2026, now=_min(10), kickoffs=kicks, archive=arc, fetch=f)
    assert f.calls == 0


def test_a_balance_under_the_floor_is_refused_by_the_poller(tmp_path, monkeypatch):
    state = tmp_path / "odds.json"
    state.write_text('{"remaining": 10, "checked_at": "2026-10-18T10:00:00"}')
    monkeypatch.setattr(odds, "_http_get", lambda *a, **k: pytest.fail("a request was formed"))
    monkeypatch.setattr(odds, "_api_key", lambda: "k" * 12)
    with pytest.raises(odds.QuotaFloor):
        closing.capture(2026, now=_min(10), kickoffs=_kicks(("g", KICK)), archive=_arc(),
                        state_path=state)


def test_an_undeclared_market_is_refused_by_the_poller(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "_http_get", lambda *a, **k: pytest.fail("a request was formed"))
    with pytest.raises(odds.MultiplierRefused):
        odds.snapshot(2026, markets="spreads,totals,h2h,player_pass_yds",
                      state_path=tmp_path / "odds.json")


# --- closing-line value ------------------------------------------------------------------

def _preds(*rows):
    """(game_id, home_win_prob, predicted_at[, home_won])."""
    cols = {"game_id": [r[0] for r in rows], "home_win_prob": [r[1] for r in rows],
            "predicted_at": [r[2] for r in rows]}
    if rows and len(rows[0]) > 3:
        cols["home_won"] = [r[3] for r in rows]
    return pl.DataFrame(cols)


def _close(game, spread, lead):
    return closing.closes(_arc((game, _min(lead), spread)), _kicks((game, KICK)))


def test_clv_is_close_minus_published_on_the_picked_side():
    p = closing.normal_cdf(3.0 / closing.MARGIN_SD)
    q = closing.normal_cdf(6.0 / closing.MARGIN_SD)
    got = closing.clv(_preds(("g", p, _min(3000))), _close("g", 6.0, 12))
    assert got["n_true_close"] == 1
    assert got["true_close"]["mean"] == pytest.approx(q - p)


def test_clv_flips_sign_for_an_away_pick():
    p = closing.normal_cdf(-3.0 / closing.MARGIN_SD)         # away favoured at publication
    q = closing.normal_cdf(-6.0 / closing.MARGIN_SD)         # and more so at the close
    got = closing.clv(_preds(("g", p, _min(3000))), _close("g", -6.0, 12))
    assert got["true_close"]["mean"] == pytest.approx((1 - q) - (1 - p))
    assert got["true_close"]["mean"] > 0


def test_a_pick_em_prediction_has_no_side_and_is_counted_apart():
    got = closing.clv(_preds(("g", 0.5, _min(3000))), _close("g", 6.0, 12))
    assert got["n"] == 0 and got["n_no_pick"] == 1


def test_a_prediction_published_after_the_capture_is_dropped_not_compared():
    got = closing.clv(_preds(("g", 0.6, _min(5))), _close("g", 6.0, 12))
    assert got["n"] == 0 and got["n_published_after_close"] == 1


def test_an_early_close_is_reported_apart_with_its_lead_and_not_as_a_true_close():
    got = closing.clv(_preds(("g", 0.6, _min(3000))), _close("g", 6.0, 360))
    assert got["n_true_close"] == 0 and got["n_early_close"] == 1
    assert got["early_close"]["median_lead_minutes"] == 360.0
    assert got["true_close"]["mean"] is None


def test_the_figure_carries_n_an_interval_and_no_verdict_word():
    rows = [(f"g{i}", 0.6, _min(3000)) for i in range(6)]
    close = pl.concat([_close(f"g{i}", 3.0 + i, 12) for i in range(6)])
    got = closing.clv(_preds(*rows), close)
    tc = got["true_close"]
    assert tc["n"] == 6 and tc["lo"] < tc["mean"] < tc["hi"]
    assert got["descriptive"] is True
    text = repr(got).lower()
    for word in ("beat", "win", "edge", "significant", "proves"):
        assert word not in text.replace("home_win", "").replace("window", "")


def test_log_loss_against_the_close_uses_scored_games_only():
    got = closing.clv(_preds(("g", 0.6, _min(3000), 1)), _close("g", 6.0, 12))
    assert got["log_loss"]["n"] == 1
    assert got["log_loss"]["close"] < got["log_loss"]["published"]


def test_the_earliest_published_row_per_game_is_the_one_compared():
    got = closing.clv(_preds(("g", 0.6, _min(3000)), ("g", 0.9, _min(2000))),
                      _close("g", 6.0, 12))
    assert got["n"] == 1
    assert got["true_close"]["mean"] == pytest.approx(
        closing.normal_cdf(6.0 / closing.MARGIN_SD) - 0.6)


def test_nothing_published_or_no_close_is_an_empty_figure_not_an_error():
    assert closing.clv(pl.DataFrame(), _close("g", 6.0, 12))["n"] == 0
    assert closing.clv(_preds(("g", 0.6, _min(3000))), pl.DataFrame())["n"] == 0


# --- the CLI ------------------------------------------------------------------------------

def test_due_exits_zero_when_due_one_when_not_and_two_when_it_cannot_tell(monkeypatch):
    monkeypatch.setattr(closing, "kickoff_table", lambda season: _kicks(("g", KICK)))
    monkeypatch.setattr(closing.store, "lines", lambda season, base=None: _arc())
    monkeypatch.setattr(closing, "datetime", _Clock(_min(10)))
    assert closing.main(["--due"]) == 0
    monkeypatch.setattr(closing, "datetime", _Clock(_min(300)))
    assert closing.main(["--due"]) == 1

    def boom(season):
        raise OSError("no schedule")
    monkeypatch.setattr(closing, "kickoff_table", boom)
    assert closing.main(["--due"]) == 2


class _Clock:
    """`datetime` with `now` pinned; everything else is the real class."""

    def __init__(self, at):
        self._at = at

    def now(self, tz=None):
        return self._at.replace(tzinfo=tz)

    def __getattr__(self, name):
        return getattr(dt.datetime, name)


def test_capture_cli_refuses_with_a_sentence_and_exit_one(monkeypatch, capsys):
    def refuse(season, **kw):
        raise closing.CloseBudgetExceeded("cap reached")
    monkeypatch.setattr(closing, "capture", refuse)
    assert closing.main(["--capture"]) == 1
    assert "cap reached" in capsys.readouterr().err


def test_capture_cli_reports_an_unreachable_market_as_unavailable(monkeypatch):
    def down(season, **kw):
        raise OSError("down")
    monkeypatch.setattr(closing, "capture", down)
    assert closing.main(["--capture"]) != 0


def test_report_summarises_the_closes_and_spends_nothing(monkeypatch, capsys):
    monkeypatch.setattr(closing, "kickoff_table", lambda season: _kicks(("g", KICK)))
    monkeypatch.setattr(closing.store, "lines",
                        lambda season, base=None: _arc(("g", _min(12), 3.0)))
    assert closing.main(["--report"]) == 0
    assert "1 games with a last pre-kickoff capture" in capsys.readouterr().out
    monkeypatch.setattr(closing.store, "lines", lambda season, base=None: _arc())
    assert closing.main(["--report"]) == 0
    assert "none archived" in capsys.readouterr().out


def test_predictions_for_games_without_a_close_give_an_empty_figure():
    got = closing.clv(_preds(("other", 0.6, _min(3000))), _close("g", 6.0, 12))
    assert got["n"] == 0


def test_kickoff_table_reads_kickoffs_from_the_schedule_and_degrades_without_times(monkeypatch):
    from hub.fetch import nflverse
    sched = pl.DataFrame({"game_id": ["g"], "season": [2026], "gameday": ["2026-10-18"],
                          "gametime": ["13:00"]})
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched)
    got = closing.kickoff_table(2026)
    assert got["kickoff"].to_list() == [KICK]
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched.drop("gametime"))
    assert closing.kickoff_table(2026).is_empty()


def test_capture_cli_returns_zero_when_a_gated_capture_completes(monkeypatch):
    monkeypatch.setattr(closing, "capture", lambda season, **kw: 1)
    assert closing.main(["--capture"]) == 0


# --- credits that cannot be wasted (review of #495) ----------------------------------------

@pytest.fixture(autouse=True)
def _attempts_in_tmp(tmp_path, monkeypatch):
    """Attempts are committed state; a test must never write the repo's."""
    monkeypatch.setattr(closing, "ATTEMPTS_FILE", tmp_path / "attempts.json")


def test_a_game_missing_from_the_response_does_not_keep_the_window_due():
    # `a` was captured 11 minutes out; `b` kicks off with it but the response had no row for it.
    arc = _arc(("a", _min(11), 3.0))
    kicks = _kicks(("a", KICK), ("b", KICK))
    assert closing.due(arc, kicks, _min(6)).is_empty()


def test_a_window_whose_response_matched_nothing_is_spent_exactly_once():
    kicks = _kicks(("a", KICK), ("b", KICK))
    f = _Fetch()
    assert closing.capture(2026, now=_min(10), kickoffs=kicks, archive=_arc(), fetch=f) == 1
    # The response matched no game, so the archive is still empty -- the attempt is what counts.
    assert closing.capture(2026, now=_min(6), kickoffs=kicks, archive=_arc(), fetch=f) == 0
    assert f.calls == 1


def test_a_lost_capture_push_leaves_the_window_marked_attempted():
    kicks = _kicks(("a", KICK))
    f = _Fetch()
    closing.capture(2026, now=_min(10), kickoffs=kicks, archive=_arc(), fetch=f,
                    publish_attempt=lambda: None)
    assert closing.read_attempts() == [_min(10)]
    # The capture never reached the archive (push lost); a second run in the window pays nothing.
    assert closing.capture(2026, now=_min(5), kickoffs=kicks, archive=_arc(), fetch=f) == 0
    assert f.calls == 1


def test_an_attempt_that_cannot_be_published_makes_no_request():
    f = _Fetch()

    def push_fails():
        raise OSError("push rejected")
    with pytest.raises(closing.AttemptNotRecorded):
        closing.capture(2026, now=_min(10), kickoffs=_kicks(("a", KICK)), archive=_arc(),
                        fetch=f, publish_attempt=push_fails)
    assert f.calls == 0


def test_attempts_without_captures_count_toward_the_weekly_cap():
    for i in range(closing.CLOSE_CALLS_PER_WEEK):
        closing.record_attempt(KICK - dt.timedelta(days=1, hours=i))
    f = _Fetch()
    with pytest.raises(closing.CloseBudgetExceeded):
        closing.capture(2026, now=_min(10), kickoffs=_kicks(("a", KICK)), archive=_arc(), fetch=f)
    assert f.calls == 0


def test_an_attempt_and_its_own_capture_are_one_call_not_two():
    closing.record_attempt(_min(12))
    arc = _arc(("a", _min(11), 3.0))
    assert closing.calls_in_week(arc, _kicks(("a", KICK)), _min(5), closing.read_attempts()) == 1


def test_the_floor_refuses_before_an_attempt_is_recorded(tmp_path):
    state = tmp_path / "odds.json"
    state.write_text('{"remaining": 10, "checked_at": "2026-10-18T10:00:00"}')
    with pytest.raises(odds.QuotaFloor):
        closing.capture(2026, now=_min(10), kickoffs=_kicks(("a", KICK)), archive=_arc(),
                        state_path=state, fetch=_Fetch())
    assert closing.read_attempts() == []


def test_a_missing_or_corrupt_attempt_file_reads_as_no_attempts(tmp_path):
    assert closing.read_attempts(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert closing.read_attempts(bad) == []


def test_old_attempts_are_pruned_when_a_new_one_is_recorded(tmp_path):
    p = tmp_path / "a.json"
    closing.record_attempt(KICK - dt.timedelta(days=40), p)
    closing.record_attempt(KICK, p)
    assert closing.read_attempts(p) == [KICK]


def test_the_commit_attempt_flag_publishes_through_the_shared_script(monkeypatch):
    seen = {}

    def fake_capture(season, publish_attempt=None, **kw):
        seen["publish"] = publish_attempt
        return 1
    ran = []
    monkeypatch.setattr(closing, "capture", fake_capture)
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: ran.append(cmd))
    assert closing.main(["--capture", "--commit-attempt"]) == 0
    publish = seen["publish"]
    assert publish is not None
    publish()
    assert ran and ran[0][0].endswith("commit-state.sh")
    assert closing.main(["--capture"]) == 0 and seen["publish"] is None


def test_an_unrecorded_attempt_is_a_refusal_with_exit_one(monkeypatch, capsys):
    def refuse(season, **kw):
        raise closing.AttemptNotRecorded("push rejected")
    monkeypatch.setattr(closing, "capture", refuse)
    assert closing.main(["--capture", "--commit-attempt"]) == 1
    assert "push rejected" in capsys.readouterr().err
