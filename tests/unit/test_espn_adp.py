"""ESPN ADP parsing.

espn_api reads `ownership.percentOwned` and throws the rest of the block away --
including `averageDraftPosition`, which is the only real ADP ESPN exposes and the
whole basis of the `edge` column. We read the raw kona_player_info view instead.

ESPN writes 0.0 for a player nobody drafts. That is a sentinel, not a first
overall pick, and letting it through would invert the top of the board.

These tests used to point at `_parse_adp`, which had no production callers -- the live
path, `player_market`, reimplemented the same parse inline and untested. So the sentinel
rule was verified on a function nobody ran. They point at the live parser now, and the
difference in behaviour is real and deliberate: an undrafted player keeps his row with a
*null* adp rather than being dropped, because his projection and injury status are still
wanted. Unpriced is not the same as unknown.
"""
import polars as pl
from hub.fetch.espn import _parse_market

SEASON = 2026


def _entry(name, adp, pos_id=0, proj=None):
    p = {"fullName": name, "defaultPositionId": pos_id}
    if adp is not None:
        p["ownership"] = {"averageDraftPosition": adp, "percentOwned": 50.0}
    if proj is not None:
        p["stats"] = [{"statSourceId": 1, "statSplitTypeId": 0, "seasonId": SEASON,
                       "appliedAverage": proj}]
    return {"player": p}


def _parse(entries):
    return _parse_market({"players": entries}, SEASON)


def test_parses_name_and_adp():
    df = _parse([_entry("Jahmyr Gibbs", 1.49), _entry("Ja'Marr Chase", 2.1)])
    assert df.height == 2
    assert df["adp"].dtype.is_numeric()
    assert df.filter(pl.col("player") == "Jahmyr Gibbs")["adp"][0] == 1.49


def test_zero_adp_is_a_sentinel_for_undrafted_not_pick_one():
    """The failure that would invert the top of the board."""
    df = _parse([_entry("Real Player", 4.2), _entry("Deep Bench Guy", 0.0)])
    assert df.filter(pl.col("player") == "Deep Bench Guy")["adp"][0] is None
    assert df.filter(pl.col("player") == "Real Player")["adp"][0] == 4.2


def test_an_undrafted_player_keeps_his_projection():
    """Why the live parser nulls the adp instead of dropping the row, as the dead one did:
    the board still wants what ESPN thinks of a player nobody drafts."""
    df = _parse([_entry("Deep Bench Guy", 0.0, proj=6.5)])
    assert df.height == 1
    assert df["adp"][0] is None
    assert df["proj_ppg"][0] == 6.5


def test_missing_ownership_block_leaves_adp_null():
    df = _parse([_entry("Has ADP", 5.0), _entry("No Ownership", None)])
    assert df.filter(pl.col("player") == "No Ownership")["adp"][0] is None


def test_a_boolean_is_not_an_adp():
    """`isinstance(True, int)` is True in Python, so a stray boolean would otherwise become
    ADP 1.0 -- the first overall pick."""
    df = _parse([_entry("Bool ADP", True)])
    assert df["adp"][0] is None


def test_a_player_with_no_name_is_dropped_entirely():
    """No name means nothing can join to him; a null-keyed row would fan out on the join."""
    df = _parse([{"player": {"ownership": {"averageDraftPosition": 3.0}}},
                 _entry("Real", 4.0)])
    assert df["player"].to_list() == ["Real"]


def test_empty_payload_yields_typed_empty_frame():
    """Must stay numeric so the board's substance guard reads dtype, not emptiness alone."""
    df = _parse([])
    assert df.is_empty()
    assert df["adp"].dtype.is_numeric()
    assert df.columns == ["player", "adp", "proj_ppg", "injury_status"]


def test_absent_players_key_does_not_raise():
    df = _parse_market({}, SEASON)
    assert df.is_empty()
    assert df["adp"].dtype.is_numeric()


def test_adp_is_an_overall_pick_number_so_it_scales_with_consensus_rank():
    """The old bug: posRank (WR5 -> 5) was subtracted from an overall rank. ADP is overall."""
    df = _parse([_entry(f"P{i}", float(i)) for i in range(1, 51)])
    assert df["adp"].max() == 50.0
    assert df["adp"].min() == 1.0


# --- the projection half of the same payload -------------------------------

def test_the_season_projection_is_picked_out_of_the_stats_block():
    """statSourceId 1 is the projection, 0 is actuals; statSplitTypeId 0 is the season.
    The same payload carries a *current week* projection and last season's actuals, and
    picking the wrong one substitutes a number off by an order of magnitude."""
    df = _parse([_entry("Projected", 3.0, proj=18.4)])
    assert df["proj_ppg"][0] == 18.4


def test_a_weekly_projection_is_not_mistaken_for_a_season_one():
    entry = _entry("Weekly Only", 3.0)
    entry["player"]["stats"] = [
        {"statSourceId": 1, "statSplitTypeId": 1, "seasonId": SEASON, "appliedAverage": 18.4},
        {"statSourceId": 0, "statSplitTypeId": 0, "seasonId": SEASON, "appliedAverage": 12.0},
    ]
    assert _parse([entry])["proj_ppg"][0] is None


def test_last_seasons_projection_is_not_used_for_this_one():
    entry = _entry("Stale", 3.0)
    entry["player"]["stats"] = [
        {"statSourceId": 1, "statSplitTypeId": 0, "seasonId": SEASON - 1,
         "appliedAverage": 18.4}]
    assert _parse([entry])["proj_ppg"][0] is None


def test_injury_status_is_carried_through():
    entry = _entry("Hurt", 3.0)
    entry["player"]["injuryStatus"] = "QUESTIONABLE"
    assert _parse([entry])["injury_status"][0] == "QUESTIONABLE"
