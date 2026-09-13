"""The quarterback adjustment (#218, restated under #268): the source's own adjustment on
the latest row, in spread points, applied only where the staleness field marks no live price.

Three things are held one test each: a game with a live price is byte-identical before and
after; the adjustment is `qb_adj / ELO_PER_POINT` and nothing else enters it; tenure does
not enter it, because the source's rolling value has already decayed the gap. The rest is
the seam -- which rows the rule reaches, what it writes on them, and the one line it reports.

**Every state fixture here has a tenure above zero and a starter whose value has drifted
since he arrived.** The estimator #268 replaced subtracted an arrival-time baseline from
the current value and decayed the difference by tenure; it was exactly right when the
arrival row *was* the latest row, and every fixture set that condition, so the suite was
green on an input the estimator never met. `docs/method.md` rule 15: a fixture that sets
the condition under which the estimator is trivially correct is not a test of the estimator.
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
    """rows: (team, qb, qb_value, qb_adj, tenure) -- the shape `hub.fetch.nfeloqb.state`
    returns. `qb_adj` is the source's adjustment on the latest row, in Elo points."""
    for r in rows:
        assert r[4] > 0, "a fixture with no tenure is the one the old estimator could not fail"
    return pl.DataFrame(
        {"team": [r[0] for r in rows], "qb": [r[1] for r in rows],
         "qb_value": pl.Series([r[2] for r in rows], dtype=pl.Float64),
         "qb_adj": pl.Series([r[3] for r in rows], dtype=pl.Float64),
         "tenure": pl.Series([r[4] for r in rows], dtype=pl.Int64),
         "as_of": ["2026-09-10"] * len(rows)})


# Two established starters whose adjustment on the latest row is exactly zero, so a game
# between them is adjusted by exactly nothing -- which is what lets a test read the *other*
# side's adjustment off the row directly. Both are forty starts in.
FLAT = [("KC", "Patrick Mahomes", 140.0, 0.0, 40),
        ("LAC", "Justin Herbert", 120.0, 0.0, 40)]

# A backup twelve starts in, whose value has drifted since he arrived, on a row where the
# source's adjustment is -330 Elo: -13.2 spread points, and tenure has nothing to say about
# it. Under the estimator #268 replaced this row would have decayed to 0.9 ** 12 of the gap.
BACKUP = ("LV", "Aidan O'Connell", 40.0, -330.0, 12)

FRESH = dt.datetime(2026, 9, 12, 11)               # an hour old: a live quote
FROZEN = dt.datetime(2026, 8, 20)                  # three weeks unmoved: not one


# --- the three the ticket names -----------------------------------------------------------

def test_a_game_with_a_live_price_is_byte_identical_before_and_after():
    """The first of the ticket's three. LV's starter is a backup at -330 Elo, the largest
    adjustment this file constructs; the game is priced by a snapshot polled an hour ago,
    so none of it reaches the row."""
    before = games([("live", "KC", "LV", 7.0, "snapshot", FRESH)])
    st = state([*FLAT, BACKUP])
    after = qb.apply(before, st, at=AT)
    assert after.select(before.columns).row(0) == before.row(0)
    assert after.select(before.columns).write_csv() == before.write_csv(), "byte for byte"
    assert after["qb_adjustment"].to_list() == [None]
    assert after["adjusted_by"].to_list() == [None]


def test_the_points_are_the_latest_rows_adjustment_over_elo_per_point():
    """The second. The source publishes the gap between this starter and what the team's
    rolling value embeds on every row, in Elo, already decayed by its own update rule; the
    team layer divides by 25 and adds nothing. -330 Elo on the away side is 13.2 points, so
    the home side's spread rises by 13.2 -- twelve starts in, with the value drifted."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    after = qb.apply(before, state([*FLAT, BACKUP]), at=AT)
    assert after["qb_adjustment"][0] == pytest.approx(330.0 / qb.ELO_PER_POINT)
    assert after["qb_adjustment"][0] == pytest.approx(13.2)
    assert after["close_spread"][0] == pytest.approx(20.2)
    assert after["adjusted_by"][0] == "nfeloqb"


def test_tenure_does_not_enter_the_adjustment():
    """The third. The same row at one start and at thirty gives the same number: the decay
    is the source's, applied when it computed `qb_adj`, and applying one here again would
    count it twice. This is the test the old fixtures could not have run -- every one of
    them set tenure to zero, where 0.9 ** tenure is 1 and the decay is invisible."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    one = qb.apply(before, state([*FLAT, ("LV", "Aidan O'Connell", 40.0, -330.0, 1)]), at=AT)
    thirty = qb.apply(before, state([*FLAT, ("LV", "Aidan O'Connell", 40.0, -330.0, 30)]),
                      at=AT)
    assert one["qb_adjustment"][0] == thirty["qb_adjustment"][0] == pytest.approx(13.2)


def test_the_value_does_not_enter_the_adjustment():
    """Nor does the starter's own value. Two rows with the same adjustment and values a
    hundred apart adjust the same, because the value's contribution is already inside the
    source's `qb_adj` -- the quantity the old estimator re-derived, wrongly, from a value
    drift that has nothing to do with the gap being priced."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    low = qb.apply(before, state([*FLAT, ("LV", "Aidan O'Connell", 40.0, -330.0, 12)]), at=AT)
    high = qb.apply(before, state([*FLAT, ("LV", "Aidan O'Connell", 140.0, -330.0, 12)]),
                    at=AT)
    assert low["qb_adjustment"][0] == high["qb_adjustment"][0] == pytest.approx(13.2)


def test_points_per_team_off_the_captured_file_are_the_sources_adjustment_over_25():
    """The captured live rows (2026-09-12): for every one of the 32 teams, `points` is the
    source's own `qb_adj` on that team's latest row over 25, read off the file here rather
    than off the state, so a state carrying anything but the latest row's number would show.
    #268 measured the replaced estimator against this on the live cache at a mean absolute
    error of 0.4 points, worst 3.6 (Baltimore, an established starter with no change)."""
    import json
    from pathlib import Path

    from hub.fetch import nfeloqb
    fixtures = Path(__file__).resolve().parents[1] / "golden" / "fixtures"
    rows = json.loads((fixtures / "nfeloqb_qb_elos.json").read_text())
    latest: dict[str, tuple[str, float]] = {}
    for r in rows:
        if r["qb1"] is None:
            continue
        for n in ("1", "2"):
            team = nfeloqb.ABBREVIATIONS.get(r[f"team{n}"], r[f"team{n}"])
            if team not in latest or r["date"] >= latest[team][0]:
                latest[team] = (r["date"], r[f"qb{n}_adj"])
    cols = list(rows[0])
    text = "\n".join([",".join(cols)] + [",".join("" if r[c] is None else str(r[c]) for c in cols)
                                         for r in rows]) + "\n"
    got = qb.points(nfeloqb.state(nfeloqb.parse(text.encode())))
    pts = dict(zip(got["team"].to_list(), got["points"].to_list(), strict=True))
    assert len(pts) == 32
    for team, (_, adj) in latest.items():
        assert pts[team] == pytest.approx(adj / 25, abs=1e-9), team
    assert max(abs(v) for v in pts.values()) > 0.5, "the file carries a non-trivial adjustment"


# --- which rows the rule reaches -----------------------------------------------------------

def test_a_snapshot_whose_quote_has_stood_past_the_threshold_is_not_a_live_price():
    before = games([("g", "KC", "LV", 7.0, "snapshot", FROZEN)])
    after = qb.apply(before, state([*FLAT, BACKUP]), at=AT)
    assert after["adjusted_by"][0] == "nfeloqb"


def test_the_threshold_is_the_declared_number_of_days():
    """A quote unmoved for exactly the threshold is still live; a second past it is not."""
    edge = AT - dt.timedelta(days=qb.STALE_AFTER_DAYS)
    st = state([*FLAT, BACKUP])
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
    after = qb.apply(before, state([*FLAT, BACKUP]), at=AT)
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
    subtract. KC's backup at -165 Elo and LV's at -330: the home side is 6.6 worse and
    the away side 13.2 worse, so the home spread rises by 6.6."""
    before = games([("g", "KC", "LV", 7.0, "schedule", None)])
    st = state([("KC", "Carson Wentz", 90.0, -165.0, 3), BACKUP])
    after = qb.apply(before, st, at=AT)
    assert after["qb_adjustment"][0] == pytest.approx(6.6)


def test_a_slate_without_the_staleness_columns_adjusts_only_the_moving_field():
    """A caller-built frame with no staleness evidence: a snapshot row is not *marked* as
    having no live price, so it is left alone; the moving field still is."""
    before = games([("s", "KC", "LV", 7.0, "snapshot", FROZEN),
                    ("m", "KC", "LV", 7.0, "schedule", None)]).drop(
        "polls_unmoved", "unmoved_since")
    after = qb.apply(before, state([*FLAT, BACKUP]), at=AT)
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
    after = qb.apply(before, state([*FLAT, BACKUP]), at=AT)
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

def test_the_constants_identify_the_model_version():
    """Two numbers, and no third: the conversion and the staleness threshold. The decay
    and the value-unit scale left with the estimator under #268, and a digest that still
    hashed them would be claiming a difference between two runs that compute the same
    thing."""
    from hub.config import fitted_constants
    got = {k for k in fitted_constants() if k.startswith("quarterback.")}
    assert got == {"quarterback.ELO_PER_POINT", "quarterback.STALE_AFTER_DAYS"}
