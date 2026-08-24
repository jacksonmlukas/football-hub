"""Tracking who is off the board.

Name matching is the whole problem. ESPN says "Marvin Harrison Jr.", FantasyPros says
"Marvin Harrison", and a draft board that silently fails to match leaves a drafted
player sitting at the top of your recommendations all night.
"""
import polars as pl
import pytest
from hub.draft import state as st


@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "draft_state.json"


def _board(names):
    return pl.DataFrame({"player": names, "vor": [1.0] * len(names)})


def test_round_trips_through_disk(tmp_state):
    s = st.take(st.DraftState(), "Ja'Marr Chase", "Bijan Robinson")
    st.save(s, tmp_state)
    assert st.load(tmp_state).taken == ["Ja'Marr Chase", "Bijan Robinson"]


def test_missing_file_loads_empty_rather_than_raising(tmp_state):
    """Draft night is the worst possible time to hit a missing-file traceback."""
    assert st.load(tmp_state).taken == []


def test_take_preserves_pick_order():
    s = st.take(st.DraftState(), "A", "B", "C")
    assert s.taken == ["A", "B", "C"] and s.n_taken == 3


def test_undo_removes_the_last_pick():
    s = st.undo(st.take(st.DraftState(), "A", "B", "C"))
    assert s.taken == ["A", "B"]


def test_undo_on_empty_is_a_noop():
    assert st.undo(st.DraftState()).taken == []


# --- name matching --------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Marvin Harrison Jr.", "Marvin Harrison"),
    ("Chris Godwin Jr.", "Chris Godwin"),
    ("Ja'Marr Chase", "JaMarr Chase"),
    ("Michael Pittman Jr.", "Michael Pittman"),
    ("Kenneth Walker III", "Kenneth Walker"),
    ("D.J. Moore", "DJ Moore"),
    ("  Amon-Ra  St. Brown ", "Amon-Ra St. Brown"),
])
def test_names_that_must_match(a, b):
    assert st._norm(a) == st._norm(b)


def test_distinct_players_do_not_collide():
    assert st._norm("Justin Jefferson") != st._norm("Justin Herbert")
    assert st._norm("Josh Allen") != st._norm("Keenan Allen")


# --- filtering the board --------------------------------------------------

def test_remaining_drops_taken_players():
    b = _board(["Ja'Marr Chase", "Bijan Robinson", "Puka Nacua"])
    out = st.remaining(b, st.take(st.DraftState(), "Bijan Robinson"))
    assert out["player"].to_list() == ["Ja'Marr Chase", "Puka Nacua"]


def test_remaining_matches_across_suffix_differences():
    """The failure that would actually bite: ESPN's name, FantasyPros' board."""
    b = _board(["Marvin Harrison Jr.", "Puka Nacua"])
    out = st.remaining(b, st.take(st.DraftState(), "Marvin Harrison"))
    assert out["player"].to_list() == ["Puka Nacua"]


def test_unmatched_pick_is_reported_not_swallowed():
    b = _board(["Puka Nacua"])
    assert st.unmatched(b, st.take(st.DraftState(), "Nonexistent Guy")) == ["Nonexistent Guy"]


def test_empty_state_returns_the_whole_board():
    b = _board(["A", "B"])
    assert st.remaining(b, st.DraftState()).height == 2


# --- my roster ------------------------------------------------------------

def test_my_roster_is_derived_from_slot_and_pick_order():
    """Picks 3 and 22 are mine from slot 3 of 12; nothing else in the first round is."""
    s = st.DraftState(taken=[f"P{i}" for i in range(1, 23)])
    assert st.my_roster(s, slot=3, teams=12) == ["P3", "P22"]


def test_my_roster_is_empty_before_my_first_pick():
    s = st.DraftState(taken=["P1", "P2"])
    assert st.my_roster(s, slot=3, teams=12) == []


# --- espn sync ------------------------------------------------------------

def test_sync_falls_back_to_local_state_when_espn_is_unreachable(monkeypatch, tmp_path, capsys):
    """A traceback while you are on the clock is the failure that matters."""
    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "league_settings", lambda: (_ for _ in ()).throw(RuntimeError("403")))
    monkeypatch.setattr(st, "STATE", tmp_path / "s.json")
    st.save(st.take(st.DraftState(), "Local Pick"), tmp_path / "s.json")
    monkeypatch.setattr(st, "load", lambda path=tmp_path / "s.json": st.DraftState(taken=["Local Pick"]))
    assert st.sync_from_espn().taken == ["Local Pick"]
    assert "unavailable" in capsys.readouterr().out


def test_sync_reads_picks_in_order(monkeypatch):
    class _P:
        def __init__(self, n): self.playerName = n
    class _L:
        draft = [_P("First"), _P("Second")]
    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "league_settings", lambda: (_L(), {}))
    assert st.sync_from_espn().taken == ["First", "Second"]


def test_empty_espn_draft_keeps_local_state(monkeypatch, capsys):
    class _L:
        draft = []
    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "league_settings", lambda: (_L(), {}))
    monkeypatch.setattr(st, "load", lambda path=None: st.DraftState(taken=["Kept"]))
    assert st.sync_from_espn().taken == ["Kept"]
    assert "not started" in capsys.readouterr().out


# --- pointing the sync at a different room --------------------------------

def test_sync_can_target_another_league():
    """ESPN mock-draft rooms are their own leagues with their own ids, so a sync hardcoded
    to ESPN_LEAGUE_ID cannot see one. Being able to point it elsewhere is what makes a
    practice draft a real test of the live path rather than a test of everything except it.
    """
    import hub.draft.state as st

    seen = {}

    class _Pick:
        def __init__(self, name): self.playerName = name

    class _League:
        draft = [_Pick("Ja'Marr Chase"), _Pick("Bijan Robinson")]

    def fake(year, league_id):
        seen["year"], seen["league_id"] = year, league_id
        return _League()

    got = st.sync_from_espn(year=2026, league_id=999888, _league_factory=fake)
    assert seen["league_id"] == 999888
    assert got.taken == ["Ja'Marr Chase", "Bijan Robinson"]


def test_omitting_the_league_id_still_uses_the_configured_league():
    """Back-compat: draft night must not require an extra argument."""
    import inspect

    import hub.draft.state as st
    sig = inspect.signature(st.sync_from_espn)
    assert sig.parameters["league_id"].default is None


# --- catching a mistyped pick ---------------------------------------------

def test_a_typo_is_reported_with_the_name_it_probably_meant():
    """The draft-night failure that costs a pick. `--taken` records whatever it is given
    and saves it; a name that does not match the board leaves that player sitting on the
    board as available, and the next recommendation can hand back someone already gone.

    ESPN publishes nothing mid-draft for a mock (docs/decisions.md), so typing picks is the
    primary path and this is its sharp edge."""
    import polars as pl

    import hub.draft.state as st
    board = pl.DataFrame({"player": ["Bijan Robinson", "Ja'Marr Chase"],
                          "pos": ["RB", "WR"]})
    got = st.suggest_unmatched(board, ["Bijan Robinsen"])
    assert got == {"Bijan Robinsen": "Bijan Robinson"}


def test_a_name_matching_nothing_is_reported_without_a_guess():
    """Kickers and defences are drafted and the board excludes them by design, so they are
    always unmatched. Offering a wild guess for them would train the reader to ignore the
    warning, which is worse than no warning."""
    import polars as pl

    import hub.draft.state as st
    board = pl.DataFrame({"player": ["Bijan Robinson"], "pos": ["RB"]})
    got = st.suggest_unmatched(board, ["Steelers D/ST"])
    assert got == {"Steelers D/ST": None}


def test_a_name_that_matches_is_not_reported_at_all():
    import polars as pl

    import hub.draft.state as st
    board = pl.DataFrame({"player": ["Bijan Robinson"], "pos": ["RB"]})
    assert st.suggest_unmatched(board, ["bijan robinson"]) == {}


def test_punctuation_and_suffixes_still_match():
    """The normaliser already handles these; this pins that the check uses it rather than
    doing its own comparison and crying wolf on `A.J. Brown`."""
    import polars as pl

    import hub.draft.state as st
    board = pl.DataFrame({"player": ["A.J. Brown", "Marvin Harrison Jr."],
                          "pos": ["WR", "WR"]})
    assert st.suggest_unmatched(board, ["AJ Brown", "Marvin Harrison"]) == {}
