"""The roster producer: the file `hub.season.lineup` has always read and nothing ever wrote.

Everything here is offline. ESPN's league and team objects are duck-typed, because the
producer only ever reads seven attributes off a player and pinning those seven is the point --
if `espn_api` renames one, this fails rather than silently writing a column of empty strings.
"""
import polars as pl
import pytest

from hub.season import roster as R


class _Player:
    def __init__(self, name, position, slot="BE", injury="ACTIVE", pid=1, pro="GB"):
        self.name, self.position, self.lineupSlot = name, position, slot
        self.injuryStatus, self.playerId, self.proTeam = injury, pid, pro


class _Team:
    def __init__(self, owners, players, team_id=1):
        self.owners, self.roster, self.team_id = owners, players, team_id


class _League:
    def __init__(self, teams):
        self.teams = teams


def _board(**over):
    base = {"player": ["Ja'Marr Chase", "Jordan Love"], "proj_blend": [19.6, 14.9],
            "pos": ["WR", "QB"]}
    base.update(over)
    return pl.DataFrame(base)


# --- identity, which is inferred rather than configured ---

def test_the_team_is_the_one_whose_owner_matches_the_swid():
    a = _Team([{"id": "{AAA}"}], [], team_id=3)
    b = _Team([{"id": "{BBB}"}], [], team_id=14)
    assert R.mine(_League([a, b]), swid="BBB").team_id == 14


def test_swid_matching_ignores_braces_and_case():
    t = _Team([{"id": "{abc-123}"}], [], team_id=7)
    assert R.mine(_League([t]), swid="ABC-123").team_id == 7


def test_a_swid_owning_no_team_raises_rather_than_picking_one():
    """A wrong team here computes cleanly against sixteen players who are not yours."""
    with pytest.raises(LookupError, match="no team in this league"):
        R.mine(_League([_Team([{"id": "{AAA}"}], [])]), swid="ZZZ")


def test_an_absent_swid_is_its_own_error():
    with pytest.raises(LookupError, match="ESPN_SWID is not set"):
        R.mine(_League([]), swid="")


def test_a_team_owned_by_several_people_still_matches():
    t = _Team([{"id": "{AAA}"}, {"id": "{BBB}"}], [], team_id=9)
    assert R.mine(_League([t]), swid="BBB").team_id == 9


# --- reading the roster off ESPN ---

def test_every_attribute_the_producer_depends_on_is_read():
    got = R.from_team(_Team([], [_Player("Ja'Marr Chase", "WR", "WR", "QUESTIONABLE", 42, "CIN")]))
    r = got.row(0, named=True)
    assert r["player"] == "Ja'Marr Chase"
    assert (r["pos"], r["slot"], r["injury_status"]) == ("WR", "WR", "QUESTIONABLE")
    assert (r["espn_id"], r["nfl_team"]) == (42, "CIN")


def test_an_empty_roster_yields_the_right_empty_shape():
    got = R.from_team(_Team([], []))
    assert got.height == 0
    assert set(R.REQUIRED) - set(got.columns) == {"mu", "sd"}


# --- the join, and the zero that must not be mistaken for a forecast ---

def test_a_rostered_player_gets_the_boards_projection():
    got = R.build(R.from_team(_Team([], [_Player("Ja'Marr Chase", "WR")])), _board())
    assert got.row(0, named=True)["mu"] == pytest.approx(19.6)
    assert got.row(0, named=True)["sd"] > 0


def test_a_kicker_carries_no_projection_rather_than_a_zero_one():
    """`moments` fills a missing projection with 0.0, and that zero is indistinguishable
    from a real one once written. K and D/ST are out of the board's scope by decision."""
    got = R.build(R.from_team(_Team([], [_Player("Eddy Pineiro", "K")])), _board())
    r = got.row(0, named=True)
    assert r["projected"] is False
    assert r["mu"] is None and r["sd"] is None


def test_the_name_key_survives_a_suffix_and_punctuation():
    got = R.build(R.from_team(_Team([], [_Player("Ja'Marr Chase Jr.", "WR")])),
                  _board(player=["JaMarr Chase Jr", "Jordan Love"]))
    assert got.row(0, named=True)["mu"] == pytest.approx(19.6)


def test_a_duplicated_board_row_does_not_fan_the_roster_out():
    """A join that multiplies rows would silently double a player's week."""
    b = _board(player=["Ja'Marr Chase", "Ja'Marr Chase"], proj_blend=[19.6, 12.0],
               pos=["WR", "WR"])
    got = R.build(R.from_team(_Team([], [_Player("Ja'Marr Chase", "WR")])), b)
    assert got.height == 1


def test_the_bench_flag_is_the_slot_and_not_the_position():
    players = [_Player("Josh Jacobs", "RB", slot="BE"),
               _Player("MarShawn Lloyd", "RB", slot="RB")]
    got = R.build(R.from_team(_Team([], players)), _board())
    starting = dict(zip(got["player"].to_list(), got["starting"].to_list(), strict=True))
    assert starting == {"Josh Jacobs": False, "MarShawn Lloyd": True}


def test_projected_players_sort_above_unprojected_ones():
    players = [_Player("Eddy Pineiro", "K"), _Player("Jordan Love", "QB")]
    got = R.build(R.from_team(_Team([], players)), _board())
    assert got["player"].to_list() == ["Jordan Love", "Eddy Pineiro"]


def test_a_board_with_no_projection_column_at_all_is_refused_by_moments():
    with pytest.raises(ValueError, match="proj_blend"):
        R.build(R.from_team(_Team([], [_Player("Jordan Love", "QB")])),
                pl.DataFrame({"player": ["Jordan Love"], "pos": ["QB"]}))


# --- writing, which must never produce a file lineup.py cannot read ---

def test_write_refuses_a_frame_missing_what_lineup_requires(tmp_path):
    with pytest.raises(ValueError, match=r"lineup\.py requires"):
        R.write(pl.DataFrame({"player": ["x"], "pos": ["WR"]}), tmp_path / "r.parquet")


def test_a_written_roster_round_trips_with_every_required_column(tmp_path):
    got = R.build(R.from_team(_Team([], [_Player("Ja'Marr Chase", "WR")])), _board())
    p = R.write(got, tmp_path / "roster.parquet")
    back = pl.read_parquet(p)
    assert not set(R.REQUIRED) - set(back.columns)
    assert back.height == 1
