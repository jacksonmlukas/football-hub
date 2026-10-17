"""The roster artifact's quantiles, swaps with odds and unprojected players (#479)."""
import json
import math
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

from hub import contracts, publish
from hub.contracts import ContractViolation, check_roster_artifact
from hub.models import predict
from hub.season import swaps


def _frame(rows):
    """rows: (player, pos, team, mu, sd, projected, extra-dict)."""
    n = len(rows)
    return pl.DataFrame(
        {"player": [r[0] for r in rows], "pos": [r[1] for r in rows],
         "nfl_team": [r[2] for r in rows], "mu": [r[3] for r in rows],
         "sd": [r[4] for r in rows], "projected": [r[5] for r in rows],
         "starting": [True] * n, "injury_status": ["ACTIVE"] * n,
         "available": [True] * n, "can_start": [True] * n, "missing_games": [0] * n},
        schema={"player": pl.Utf8, "pos": pl.Utf8, "nfl_team": pl.Utf8, "mu": pl.Float64,
                "sd": pl.Float64, "projected": pl.Boolean, "starting": pl.Boolean,
                "injury_status": pl.Utf8, "available": pl.Boolean, "can_start": pl.Boolean,
                "missing_games": pl.Int64})


# --- quantiles --------------------------------------------------------------

def test_a_planted_player_gets_the_textbook_quantiles():
    """mu 100, sd 10, skew 0.05 (the floor): far from the zero clip, so the 20 quantiles are
    those of a near-normal at the 20 levels (i + .5) / 20, to the skew term's small correction."""
    df = _frame([("Planted", "WR", "CIN", 100.0, 10.0, True)]).with_columns(
        pl.lit(0.05).alias("skew"))
    q = swaps.quantiles(df)["Planted"]
    nd = NormalDist(100.0, 10.0)
    want = [nd.inv_cdf((i + 0.5) / 20) for i in range(20)]
    assert len(q) == 20
    assert q == sorted(q)
    assert q == pytest.approx(want, abs=0.25)
    assert q[9] < 100.0 < q[10] + 0.2     # median straddles the mean


def test_the_quantiles_are_the_skewed_transform_not_a_normal():
    """Right-skewed and clipped: the median sits under the mean, and a low-mean player's
    lowest quantiles are exactly zero -- what a normal on mu and sd gets wrong."""
    df = _frame([("Low", "WR", "CIN", 3.0, 3.7, True)])
    q = swaps.quantiles(df)["Low"]
    assert q[0] == 0.0
    z = NormalDist().inv_cdf(0.525)
    assert q[10] == pytest.approx(float(predict.skewed(3.0, 3.7, 0.66, z)), abs=0.01)
    assert q[10] < 3.0 + 3.7 * z


def test_unprojected_and_bye_rows_have_no_quantiles_and_say_why():
    df = _frame([("Kicker", "K", "CIN", None, None, False),
                 ("Bye Guy", "WR", "GB", 10.0, 5.0, True),
                 ("Fine", "WR", "CIN", 10.0, 5.0, True)]).with_columns(
        pl.Series("on_bye", [False, True, False]))
    q = swaps.quantiles(df)
    assert set(q) == {"Fine"}
    rows = {r["player"]: r for r in df.iter_rows(named=True)}
    assert "outside the board's scope" in (swaps.quantile_reason(rows["Kicker"]) or "")
    assert (swaps.quantile_reason(rows["Bye Guy"]) or "").startswith("bye")
    assert swaps.quantile_reason(rows["Fine"]) is None


def test_unprojected_lists_names_with_reasons():
    df = _frame([("K One", "K", "CIN", None, None, False),
                 ("D One", "D/ST", "GB", None, None, False),
                 ("Rookie", "WR", "GB", None, None, False),
                 ("Chase", "WR", "CIN", 19.0, 6.0, True)])
    names, why = swaps.unprojected(df)
    assert names == ["K One", "D One", "Rookie"]
    assert "no projection" in why["K One"] and "no projection" in why["D One"]
    assert why["Rookie"] == "the board carries no projection for this player"


# --- the pairing ------------------------------------------------------------

WEEK6 = _frame([("DJ Moore", "WR", "CHI", 12.0, 6.0, True),
                ("Denzel Boston", "WR", "CLE", 9.0, 5.5, True),
                ("Bhayshul Tuten", "RB", "JAX", 7.0, 5.0, True),
                ("Ja'Marr Chase", "WR", "CIN", 19.0, 7.0, True)])


def test_the_week_six_shape_pairs_by_position_then_flex():
    got, reason = swaps.swaps(WEEK6, sit=["Bhayshul Tuten", "DJ Moore"],
                              start=["Denzel Boston", "Ja'Marr Chase"],
                              withheld=["DJ Moore"])
    assert reason is None
    moore, tuten = got[0], got[1]
    assert (moore["out"], moore["in"]) == ("DJ Moore", "Denzel Boston")
    assert moore["forced"] is True and moore["p_in_beats_out"] is None
    assert moore["gain"] == pytest.approx(9.0 - 12.0)
    assert (tuten["out"], tuten["in"]) == ("Bhayshul Tuten", "Ja'Marr Chase")
    assert tuten["forced"] is False
    assert tuten["gain"] == pytest.approx(12.0)
    assert 0.5 < tuten["p_in_beats_out"] < 1.0


def test_a_side_with_nothing_to_pair_against_is_null_with_no_odds():
    got, _ = swaps.swaps(WEEK6, sit=["DJ Moore"], start=[], withheld=[])
    assert got == [{"out": "DJ Moore", "in": None, "gain": None, "forced": False,
                    "p_in_beats_out": None}]


# --- the odds ---------------------------------------------------------------

def test_a_swap_between_identical_distributions_is_a_coin_flip():
    df = _frame([("A", "WR", "CIN", 12.0, 6.0, True), ("B", "WR", "GB", 12.0, 6.0, True)])
    got, _ = swaps.swaps(df, sit=["A"], start=["B"], withheld=[])
    assert got[0]["p_in_beats_out"] == pytest.approx(0.5, abs=0.02)
    assert got[0]["gain"] == 0.0


def test_identical_low_mean_players_are_still_a_coin_flip_where_the_clip_ties_them():
    """At a mean of 2 a fifth of draws are 0 for both. Ties count half; counting them as a
    loss would put this well under 0.5."""
    df = _frame([("A", "WR", "CIN", 2.0, 3.0, True), ("B", "WR", "GB", 2.0, 3.0, True)])
    got, _ = swaps.swaps(df, sit=["A"], start=["B"], withheld=[])
    assert got[0]["p_in_beats_out"] == pytest.approx(0.5, abs=0.02)


def test_a_better_incoming_player_wins_more_often_and_a_worse_one_less():
    df = _frame([("Lo", "WR", "CIN", 8.0, 5.0, True), ("Hi", "WR", "GB", 16.0, 5.0, True)])
    up, _ = swaps.swaps(df, sit=["Lo"], start=["Hi"], withheld=[])
    down, _ = swaps.swaps(df, sit=["Hi"], start=["Lo"], withheld=[])
    assert up[0]["p_in_beats_out"] > 0.8
    assert up[0]["p_in_beats_out"] + down[0]["p_in_beats_out"] == pytest.approx(1.0, abs=2e-4)


def test_teammates_use_the_correlation_not_the_independent_answer(monkeypatch):
    """Two same-team players with rho ~ 1 move together, so the one with the higher mean
    outscores the other nearly every time; independent normals say Phi(0.28) = 0.61."""
    df = _frame([("Out", "WR", "CIN", 10.0, 5.0, True), ("In", "WR", "CIN", 12.0, 5.0, True)])
    independent, _ = swaps.swaps(df, sit=["Out"], start=["In"], withheld=[])
    monkeypatch.setitem(predict.TEAMMATE_RHO, ("WR", "WR"), 0.999)
    together, _ = swaps.swaps(df, sit=["Out"], start=["In"], withheld=[])
    reverse, _ = swaps.swaps(df, sit=["In"], start=["Out"], withheld=[])
    assert independent[0]["p_in_beats_out"] == pytest.approx(0.61, abs=0.03)
    assert together[0]["p_in_beats_out"] > 0.99
    assert reverse[0]["p_in_beats_out"] < 0.01


def test_the_same_sync_publishes_the_same_odds():
    a, _ = swaps.swaps(WEEK6, ["Bhayshul Tuten"], ["Ja'Marr Chase"], [])
    b, _ = swaps.swaps(WEEK6, ["Bhayshul Tuten"], ["Ja'Marr Chase"], [])
    assert a == b


def test_a_void_correlation_nulls_the_odds_and_says_why(monkeypatch):
    def refuse(*a, **k):
        raise predict.CorrelationVoid("no block factored")
    monkeypatch.setattr(predict, "correlated_normal", refuse)
    got, reason = swaps.swaps(WEEK6, ["Bhayshul Tuten"], ["Ja'Marr Chase"], [])
    assert got[0]["p_in_beats_out"] is None and got[0]["gain"] == pytest.approx(12.0)
    assert "no block factored" in (reason or "")


# --- through the publisher --------------------------------------------------

def _published(tmp_path, df):
    src = tmp_path / "roster.parquet"
    df.write_parquet(src)
    publish.roster(out=tmp_path / "site", path=src)
    return json.loads((tmp_path / "site" / "roster.json").read_text())


def test_the_publisher_carries_quantiles_swaps_and_unprojected(tmp_path, monkeypatch):
    from hub.season.roster import Lock
    # A thin fixture cannot field a legal lineup, so the lock's answer is planted: the swaps
    # are the week-6 shape and the publisher's job is to carry them with their odds.
    monkeypatch.setattr(publish, "lock", lambda df: Lock(
        50.0, 60.0, 10.0, ["Denzel Boston", "Ja'Marr Chase"],
        ["Denzel Boston", "Ja'Marr Chase"], ["Bhayshul Tuten", "DJ Moore"], ["DJ Moore"]))
    df = pl.concat([WEEK6, _frame([("Kicker", "K", "CIN", None, None, False)])]).with_columns(
        pl.Series("starting", [True, False, True, False, True]),
        pl.Series("available", [False, True, True, True, True]),
        pl.Series("slot", ["WR", "BE", "RB", "BE", "K"]))
    art = _published(tmp_path, df)
    by = {r["player"]: r for r in art["rows"]}
    assert len(by["Ja'Marr Chase"]["quantiles"]) == 20 and by["Ja'Marr Chase"]["quantiles_reason"] is None
    assert by["Kicker"]["quantiles"] is None and by["Kicker"]["quantiles_reason"]
    assert art["unprojected"] == ["Kicker"] and "Kicker" in art["unprojected_reasons"]
    outs = {(s["out"], s["in"]): s for s in art["swaps"]}
    assert outs[("DJ Moore", "Denzel Boston")]["forced"] is True
    assert outs[("DJ Moore", "Denzel Boston")]["p_in_beats_out"] is None
    flex = outs[("Bhayshul Tuten", "Ja'Marr Chase")]
    assert 0.0 <= flex["p_in_beats_out"] <= 1.0
    assert art["odds_reason"] is None
    check_roster_artifact(art)


# --- the contract -----------------------------------------------------------

def _art(**over):
    q = [float(i) for i in range(20)]
    art = {"rows": [{"player": "A", "quantiles": q}, {"player": "B", "quantiles": None}],
           "swaps": [{"out": "A", "in": "B", "gain": 1.0, "forced": False,
                      "p_in_beats_out": 0.4}]}
    art.update(over)
    return art


def test_a_well_formed_artifact_passes_and_an_old_one_without_the_fields_passes():
    check_roster_artifact(_art())
    check_roster_artifact({"rows": [{"player": "A", "mu": 1.0}]})
    check_roster_artifact({})


@pytest.mark.parametrize("q", [
    [1.0] * 19, [1.0] * 21, [float(i) for i in range(20)][::-1],
    [-1.0] + [1.0] * 19, [math.nan] * 20, "not a list"])
def test_bad_quantiles_are_refused(q):
    with pytest.raises(ContractViolation, match="quantiles"):
        check_roster_artifact(_art(rows=[{"player": "A", "quantiles": q}]))


@pytest.mark.parametrize("p", [1.0001, -0.01, "0.5"])
def test_odds_outside_the_unit_interval_are_refused(p):
    swap = {"out": "A", "in": "B", "gain": 0.0, "forced": False, "p_in_beats_out": p}
    with pytest.raises(ContractViolation, match="p_in_beats_out"):
        check_roster_artifact(_art(swaps=[swap]))


@pytest.mark.parametrize("p", [0.0, 1.0, None])
def test_the_ends_of_the_interval_and_null_are_fine(p):
    check_roster_artifact(_art(swaps=[{"out": "A", "in": "B", "p_in_beats_out": p}]))


def test_a_swap_naming_nobody_on_the_roster_is_refused():
    with pytest.raises(ContractViolation, match="not a rostered row"):
        check_roster_artifact(_art(swaps=[{"out": "A", "in": "Ghost", "p_in_beats_out": None}]))


def test_the_publisher_serves_last_good_when_its_new_fields_break_the_contract(
        tmp_path, monkeypatch):
    good = _published(tmp_path, WEEK6)
    monkeypatch.setattr(swaps, "quantiles", lambda df: {"DJ Moore": [-1.0] * 20})
    src = tmp_path / "roster.parquet"
    kept = publish.roster(out=tmp_path / "site", path=src)
    assert isinstance(kept, publish.Kept)
    assert json.loads((tmp_path / "site" / "roster.json").read_text()) == good


def test_contracts_module_still_exports_the_checker():
    assert contracts.check_roster_artifact is check_roster_artifact
    assert np  # numpy is imported for the transform controls above
