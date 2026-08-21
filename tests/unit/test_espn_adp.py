"""ESPN ADP parsing.

espn_api reads `ownership.percentOwned` and throws the rest of the block away --
including `averageDraftPosition`, which is the only real ADP ESPN exposes and the
whole basis of the `edge` column. We read the raw kona_player_info view instead.

ESPN writes 0.0 for a player nobody drafts. That is a sentinel, not a first
overall pick, and letting it through would invert the top of the board.
"""
import polars as pl
from hub.fetch.espn import _parse_adp


def _entry(name, adp, pos_id=0):
    p = {"fullName": name, "defaultPositionId": pos_id}
    if adp is not None:
        p["ownership"] = {"averageDraftPosition": adp, "percentOwned": 50.0}
    return {"player": p}


def test_parses_name_and_adp():
    df = _parse_adp({"players": [_entry("Jahmyr Gibbs", 1.49), _entry("Ja'Marr Chase", 2.1)]})
    assert df.height == 2
    assert df["adp"].dtype.is_numeric()
    assert df.filter(pl.col("player") == "Jahmyr Gibbs")["adp"][0] == 1.49


def test_zero_adp_is_a_sentinel_for_undrafted_not_pick_one():
    df = _parse_adp({"players": [_entry("Real Player", 4.2), _entry("Deep Bench Guy", 0.0)]})
    assert df.height == 1
    assert df["player"].to_list() == ["Real Player"]


def test_missing_ownership_block_is_dropped():
    df = _parse_adp({"players": [_entry("Has ADP", 5.0), _entry("No Ownership", None)]})
    assert df["player"].to_list() == ["Has ADP"]


def test_null_adp_is_dropped():
    df = _parse_adp({"players": [_entry("Has ADP", 5.0), _entry("Null ADP", None)]})
    assert df.height == 1


def test_empty_payload_yields_typed_empty_frame():
    """Must stay numeric so the board's substance guard reads dtype, not emptiness alone."""
    df = _parse_adp({"players": []})
    assert df.is_empty()
    assert df["adp"].dtype.is_numeric()
    assert df.columns == ["player", "adp"]


def test_absent_players_key_does_not_raise():
    df = _parse_adp({})
    assert df.is_empty()
    assert df["adp"].dtype.is_numeric()


def test_adp_is_an_overall_pick_number_so_it_scales_with_consensus_rank():
    """The old bug: posRank (WR5 -> 5) was subtracted from an overall rank. ADP is overall."""
    df = _parse_adp({"players": [_entry(f"P{i}", float(i)) for i in range(1, 51)]})
    assert df["adp"].max() == 50.0
    assert df["adp"].min() == 1.0
