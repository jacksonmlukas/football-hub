"""The quarterback adjustment (#218): relative, decaying, and applied only where the
staleness field marks no live price.

Three things the ticket asks to be held, and they are held one test each: a game with a
live price is byte-identical before and after; a long-tenured backup's adjustment has
decayed to near zero; a fresh backup gets the full relative gap. The rest is the seam --
which rows the rule reaches, what it writes on them, and the one line it reports.
"""
import datetime as dt

import polars as pl
import pytest

from hub.models import quarterback as qb

AT = dt.datetime(2026, 9, 12, 12)


def games(rows):
    """A slate the shape `hub.schedule.priced_games` returns: one row per game, the two
    staleness columns on the snapshot-priced rows and null elsewhere.

    rows: (game_id, home, away, close_spread, price_source, unmoved_since | None)
    """
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "league": ["nfl"] * len(rows),
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([3] * len(rows), dtype=pl.Int32),
         "home_team": [r[1] for r in rows],
         "away_team": [r[2] for r in rows],
         "close_spread": pl.Series([r[3] for r in rows], dtype=pl.Float64),
         "price_source": [r[4] for r in rows],
         "priced_at": pl.Series([r[5] for r in rows], dtype=pl.Datetime),
         "polls_unmoved": pl.Series([None if r[5] is None else 1 for r in rows],
                                    dtype=pl.Int64),
         "unmoved_since": pl.Series([r[5] for r in rows], dtype=pl.Datetime)})


def state(rows):
    """rows: (team, qb, qb_value, arrival_value, arrival_adj, tenure)"""
    return pl.DataFrame(
        {"team": [r[0] for r in rows], "qb": [r[1] for r in rows],
         "qb_value": pl.Series([r[2] for r in rows], dtype=pl.Float64),
         "arrival_value": pl.Series([r[3] for r in rows], dtype=pl.Float64),
         "arrival_adj": pl.Series([r[4] for r in rows], dtype=pl.Float64),
         "tenure": pl.Series([r[5] for r in rows], dtype=pl.Int64),
         "as_of": ["2026-09-10"] * len(rows)})


# Two established starters whose gap to what their teams embed is exactly zero, so a game
# between them is adjusted by exactly nothing -- which is what lets a test read the *other*
# side's adjustment off the row directly.
FLAT = [("KC", "Patrick Mahomes", 140.0, 140.0, 0.0, 40),
        ("LAC", "Justin Herbert", 120.0, 120.0, 0.0, 40)]

FRESH = dt.datetime(2026, 9, 12, 11)               # an hour old: a live quote
FROZEN = dt.datetime(2026, 8, 20)                  # three weeks unmoved: not one


# --- the three the ticket names -----------------------------------------------------------

def test_a_game_with_a_live_price_is_byte_identical_before_and_after():
    """The first of the ticket's three. LV's starter is a backup with a gap of -100 value
    units and no tenure, the largest adjustment this file constructs; the game is priced by
    a snapshot polled an hour ago, so none of it reaches the row."""
    before = games([("live", "KC", "LV", 7.0, "snapshot", FRESH)])
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    assert after.select(before.columns).row(0) == before.row(0)
    assert after.select(before.columns).write_csv() == before.write_csv(), "byte for byte"
    assert after["qb_adjustment"].to_list() == [None]
    assert after["adjusted_by"].to_list() == [None]


def test_a_long_tenured_backup_has_decayed_to_near_zero():
    """The second. Same backup, same gap, thirty starts in: the team rating has absorbed
    him and the adjustment says so. 0.9^30 is 0.042, so a 13.2-point full gap is 0.56."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 30)])
    after = qb.apply(before, st, at=AT)
    full = qb.POINTS_PER_VALUE * 100.0
    assert abs(after["qb_adjustment"][0]) < 0.05 * full
    assert after["qb_adjustment"][0] == pytest.approx(full * 0.9 ** 30, abs=1e-9)


def test_a_fresh_backup_gets_the_full_relative_gap():
    """The third. Named for the coming game and yet to start: nothing has been absorbed.
    arrival_adj of -330 Elo is a gap of -100 value units (3.3 per unit); at 0.132 points a
    unit the away side is 13.2 points worse, so the home side's spread rises by 13.2."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    assert after["qb_adjustment"][0] == pytest.approx(13.2)
    assert after["close_spread"][0] == pytest.approx(20.2)
    assert after["adjusted_by"][0] == "nfeloqb"


# --- which rows the rule reaches -----------------------------------------------------------

def test_a_snapshot_whose_quote_has_stood_past_the_threshold_is_not_a_live_price():
    before = games([("g", "KC", "LV", 7.0, "snapshot", FROZEN)])
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    assert after["adjusted_by"][0] == "nfeloqb"


def test_the_threshold_is_the_declared_number_of_days():
    """A quote unmoved for exactly the threshold is still live; a second past it is not."""
    edge = AT - dt.timedelta(days=qb.STALE_AFTER_DAYS)
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    live = qb.apply(games([("g", "KC", "LV", 7.0, "snapshot", edge)]), st, at=AT)
    stale = qb.apply(games([("g", "KC", "LV", 7.0, "snapshot",
                             edge - dt.timedelta(seconds=1))]), st, at=AT)
    assert live["adjusted_by"][0] is None
    assert stale["adjusted_by"][0] == "nfeloqb"


def test_an_unpriced_game_has_no_rating_to_adjust():
    """`priced_games` never emits a source without a number, so the row here is the
    caller-built shape the guard exists for: a source named and no spread to move. An
    adjustment added to nothing is a null the row would then call an adjustment."""
    before = games([("g", "KC", "LV", None, "schedule", None),
                    ("h", "KC", "LV", None, None, None)])
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    assert after["close_spread"].to_list() == [None, None]
    assert after["qb_adjustment"].to_list() == [None, None]
    assert after["adjusted_by"].to_list() == [None, None]


def test_a_team_the_source_does_not_list_leaves_the_game_untouched():
    """Half an adjustment is worse than none: a game is moved only when both sides are
    known, and the row says it was not moved rather than carrying one side's number."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    after = qb.apply(before, state(FLAT), at=AT)
    assert after["close_spread"].to_list() == [7.0]
    assert after["qb_adjustment"].to_list() == [None]


def test_the_adjustment_is_home_minus_away():
    """Both sides move the home spread: the home side's points add, the away side's
    subtract. KC's backup at a gap of -50 and LV's at -100: the home side is 6.6 worse and
    the away side 13.2 worse, so the home spread rises by 6.6."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    st = state([("KC", "Carson Wentz", 90.0, 90.0, -165.0, 0),
                ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    assert after["qb_adjustment"][0] == pytest.approx(6.6)


def test_a_slate_without_the_staleness_columns_adjusts_only_the_moving_field():
    """A caller-built frame with no staleness evidence: a snapshot row is not *marked* as
    having no live price, so it is left alone; the moving field still is."""
    before = games([("s", "KC", "LV", 7.0, "snapshot", FROZEN),
                    ("m", "KC", "LV", 7.0, "schedule", None)]).drop(
        "polls_unmoved", "unmoved_since")
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    assert after["adjusted_by"].to_list() == [None, "nfeloqb"]


def test_no_state_at_all_is_the_passthrough_with_the_columns_present():
    """Zero attention: no cached file means every row is what it was, and the two
    provenance columns are still on the frame so a reader downstream finds one schema."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    after = qb.apply(before, None, at=AT)
    assert after["close_spread"].to_list() == [7.0]
    assert after["qb_adjustment"].to_list() == [None]
    assert after["adjusted_by"].to_list() == [None]


# --- the one line it reports ---------------------------------------------------------------

def test_the_report_counts_the_games_touched_and_the_mean_absolute_change():
    before = games([("live", "KC", "LV", 7.0, "snapshot", FRESH),
                    ("a", "KC", "LV", 7.0, "schedule", None),
                    ("b", "LAC", "LV", -3.0, "snapshot", FROZEN),
                    ("none", "KC", "LAC", None, None, None)])
    st = state([*FLAT, ("LV", "Aidan O'Connell", 40.0, 40.0, -330.0, 0)])
    after = qb.apply(before, st, at=AT)
    line = qb.report_line(after)
    assert "2 of 3 priced games" in line
    assert "13.20 points" in line
    # win probability moves too, and is reported beside the spread
    assert "win probability" in line


def test_the_report_says_when_nothing_was_touched():
    before = games([("live", "KC", "LV", 7.0, "snapshot", FRESH)])
    after = qb.apply(before, state(FLAT), at=AT)
    assert "0 of 1 priced games" in qb.report_line(after)


# --- the stated constants --------------------------------------------------------------------

def test_the_scale_is_the_recipes_two_conversions_multiplied_out():
    """0.132 points per value unit is 3.3 Elo per value unit over 25 Elo per point, both
    538's published figures. Stated as one number so the digest hashes the choice."""
    assert qb.POINTS_PER_VALUE == pytest.approx(qb.ELO_PER_VALUE / qb.ELO_PER_POINT)


def test_the_constants_identify_the_model_version():
    from hub.config import fitted_constants
    got = fitted_constants()
    for name in ("quarterback.POINTS_PER_VALUE", "quarterback.DECAY_PER_GAME",
                 "quarterback.STALE_AFTER_DAYS"):
        assert name in got, name
