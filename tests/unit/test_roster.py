"""The roster producer: the file `hub.season.lineup` has always read and nothing ever wrote.

Everything here is offline. ESPN's league and team objects are duck-typed, because the
producer only ever reads seven attributes off a player and pinning those seven is the point --
if `espn_api` renames one, this fails rather than silently writing a column of empty strings.
"""
import polars as pl
import pytest

from hub.season import roster as R


class _Player:
    def __init__(self, name, position, slot="BE", injury="ACTIVE", pid=1, pro="GB",
                 avg=10.0, total=170.0):
        self.name, self.position, self.lineupSlot = name, position, slot
        self.injuryStatus, self.playerId, self.proTeam = injury, pid, pro
        # ESPN's two projections; their ratio is the availability signal. The defaults are a
        # full 17-game slate, so a test that says nothing about availability gets a fit player.
        self.projected_avg_points, self.projected_total_points = avg, total


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


# --- availability, which `injury_status` does not carry ---

def _p(name, pos, avg, total, slot="BE", injury="ACTIVE"):
    p = _Player(name, pos, slot=slot, injury=injury)
    p.projected_avg_points, p.projected_total_points = avg, total
    return p


def test_a_suspended_player_is_found_by_his_games_not_his_designation():
    """The defect this exists for: ESPN reported Jacobs ACTIVE/DAY_TO_DAY through a six-game
    ban, and only his projected total gave it away."""
    got = R.availability(R.from_team(_Team([], [
        _p("Josh Jacobs", "RB", 15.15, 166.67, injury="DAY_TO_DAY"),
        _p("Bhayshul Tuten", "RB", 12.29, 208.86),
    ])))
    by = dict(zip(got["player"].to_list(), got["missing_games"].to_list(), strict=True))
    assert by == {"Josh Jacobs": 6, "Bhayshul Tuten": 0}
    assert got.filter(pl.col("player") == "Josh Jacobs")["available"][0] is False


def test_the_full_slate_is_the_rosters_own_maximum_not_a_hardcoded_17():
    """Season length is ESPN's to change. A roster where nobody plays 17 must not read as a
    roster where everybody is suspended."""
    got = R.availability(R.from_team(_Team([], [
        _p("A", "RB", 10.0, 140.0), _p("B", "WR", 10.0, 140.0)])))
    assert got["missing_games"].to_list() == [0, 0]
    assert got["available"].all()


def test_a_player_with_no_projection_is_not_declared_unavailable():
    """K and D/ST have no ESPN projection here; absence of evidence is not a suspension."""
    got = R.availability(R.from_team(_Team([], [
        _p("Kicker", "K", 0.0, 0.0), _p("Real", "WR", 10.0, 170.0)])))
    assert got.filter(pl.col("player") == "Kicker")["available"][0] is True


def test_an_empty_roster_yields_the_availability_columns_anyway():
    got = R.availability(R.from_team(_Team([], [])))
    assert {"espn_games", "missing_games", "available"} <= set(got.columns)


# --- the lock decision ---

# A roster that can actually fill QB/RB2/WR3/TE1/FLEX1, shaped like the real one. `swap`
# overrides one player, which is how each test states the single thing it is about.
_SQUAD = [
    ("Jordan Love", "QB", 14.9, "QB"), ("Josh Jacobs", "RB", 15.5, "BE"),
    ("Bhayshul Tuten", "RB", 8.8, "RB"), ("MarShawn Lloyd", "RB", 5.6, "RB"),
    ("Ja'Marr Chase", "WR", 19.6, "WR"), ("Rashee Rice", "WR", 14.9, "WR"),
    ("Nico Collins", "WR", 14.4, "WR"), ("Terry McLaurin", "WR", 10.6, "BE"),
    ("Mark Andrews", "TE", 9.2, "TE"),
]


def _squad(unavailable=()):
    """The nine, with named players optionally projected to miss six games."""
    players = [_p(n, pos, 10.0, 110.0 if n in unavailable else 170.0, slot=slot)
               for n, pos, _, slot in _SQUAD]
    board = _board(player=[n for n, _, _, _ in _SQUAD],
                   proj_blend=[mu for _, _, mu, _ in _SQUAD],
                   pos=[p for _, p, _, _ in _SQUAD])
    return R.build(R.from_team(_Team([], players)), board)


def test_the_lock_withholds_an_unavailable_player_however_high_he_projects():
    """The bug in one test: ranking on projection and caveating the result is what
    recommended starting a suspended player. Jacobs is the roster's second-best projection
    and must still not be started."""
    lk = R.lock(_squad(unavailable={"Josh Jacobs"}))
    assert lk.withheld == ["Josh Jacobs"]
    assert "Josh Jacobs" not in lk.start


def test_withholding_changes_the_recommendation_it_does_not_just_annotate_it():
    """The available-only answer must differ from the availability-blind one, or the
    filtering is decorative."""
    df = _squad(unavailable={"Josh Jacobs"})
    blind, aware = R.lock(df, include_unavailable=True), R.lock(df)
    assert "Josh Jacobs" in blind.start
    assert aware.gain is not None and blind.gain is not None
    assert aware.gain < blind.gain


def test_include_unavailable_puts_him_back():
    """The override exists so the operator can see the number they are declining."""
    assert R.lock(_squad(unavailable={"Josh Jacobs"}), include_unavailable=True).withheld == []


def test_the_lock_names_both_sides_of_every_swap():
    lk = R.lock(_squad())
    assert "Josh Jacobs" in lk.start and "MarShawn Lloyd" in lk.bench
    assert lk.gain is not None and lk.gain > 0


def test_a_lineup_that_cannot_be_filled_reports_no_gain_rather_than_raising():
    """Withholding can thin a roster past the slots -- here every receiver, leaving nobody for
    the three WR slots. A lock that raised would take the whole slate down with it."""
    lk = R.lock(_squad(unavailable={n for n, pos, _, _ in _SQUAD if pos == "WR"}))
    assert lk.gain is None
    assert len(lk.withheld) == 4


def test_a_roster_where_nobody_plays_a_full_slate_is_not_a_roster_of_suspensions():
    """`full` is the roster's own maximum, so a uniformly-short roster reads as available.
    Marking everyone unavailable would be the same bug pointed the other way."""
    lk = R.lock(_squad(unavailable={n for n, _, _, _ in _SQUAD}))
    assert lk.withheld == []
    assert lk.gain is not None


# --- the market half, refreshed against live ESPN ---

def _with_board(players, board):
    return R.build(R.from_team(_Team([], players)), board)


def test_the_live_projection_replaces_the_boards_frozen_copy():
    """The board is a draft-day artifact. ESPN repriced MarShawn Lloyd 5.10 -> 8.01 when the
    back ahead of him was suspended, and the board could not know."""
    got = _with_board(
        [_p("MarShawn Lloyd", "RB", 8.01, 136.0, slot="RB")],
        pl.DataFrame({"player": ["MarShawn Lloyd"], "pos": ["RB"],
                      "proj_ppg": [5.10], "xfp_per_game": [6.08]}))
    # blend of the *live* 8.01 with the board's xfp, not of the frozen 5.10
    assert got.row(0, named=True)["mu"] == pytest.approx((8.01 + 6.08) / 2)


def test_a_player_espn_no_longer_projects_keeps_the_draft_day_number():
    """The refresh is a coalesce, not a replacement: losing a live projection must not cost a
    player the one the board already had."""
    got = _with_board(
        [_p("Someone", "RB", 0.0, 0.0, slot="RB")],
        pl.DataFrame({"player": ["Someone"], "pos": ["RB"],
                      "proj_ppg": [9.0], "xfp_per_game": [7.0]}))
    assert got.row(0, named=True)["mu"] == pytest.approx(8.0)


def test_the_refresh_does_not_project_a_kicker_espn_happens_to_price():
    """The regression this ordering exists for. ESPN projects kickers and defences perfectly
    happily; refreshing before scoping let a D/ST into the lineup at 4.6 points."""
    got = _with_board(
        [_p("Eddy Pineiro", "K", 9.13, 155.2, slot="K"),
         _p("Jaguars D/ST", "D/ST", 4.61, 78.4, slot="D/ST"),
         _p("Real Back", "RB", 10.0, 170.0, slot="RB")],
        pl.DataFrame({"player": ["Real Back"], "pos": ["RB"],
                      "proj_ppg": [9.0], "xfp_per_game": [9.0]}))
    by = {r["player"]: r for r in got.iter_rows(named=True)}
    assert by["Eddy Pineiro"]["projected"] is False and by["Eddy Pineiro"]["mu"] is None
    assert by["Jaguars D/ST"]["projected"] is False and by["Jaguars D/ST"]["mu"] is None
    assert by["Real Back"]["mu"] == pytest.approx(9.5)


def test_market_is_a_no_op_when_the_frame_carries_no_live_column():
    """`build` is not the only caller shape -- a frame without `espn_avg` must pass through
    rather than raise."""
    df = pl.DataFrame({"proj_ppg": [9.0], "xfp_per_game": [7.0]})
    assert R.market(df).equals(df)


def test_the_board_and_the_roster_blend_the_same_way():
    """One owner for `proj_blend`. The board builds it at draft time and the roster rebuilds
    it in-season, and the two must not be able to disagree about what it means."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    bad = []
    for path in (root / "draft" / "board.py", root / "season" / "roster.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            # the literal formula, written out anywhere other than predict.blend
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.right, ast.Constant) and node.right.value == 2.0
                    and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Add)):
                bad.append(f"{path.name}:{node.lineno}")
    assert not bad, f"the blend formula is restated in: {bad}"
