"""`hub.fetch.nfeloqb`: the published quarterback ratings, read into a per-team state.

Nothing here reaches the network. The rows are the hand-built fixture in 538's schema, so
what these prove is that the reader walks the shape the module *assumes* -- the first live
pull is what says whether the assumption held, and the module docstring lists what it must
confirm.
"""
import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from hub.contracts import ContractViolation
from hub.fetch import nfeloqb

FIXTURES = Path(__file__).resolve().parents[1] / "golden" / "fixtures"


def rows() -> list[dict]:
    return json.loads((FIXTURES / "nfeloqb_qb_elos.synthetic.json").read_text())


def csv_text(got: list[dict] | None = None) -> str:
    """The fixture as the CSV the source publishes: the same rows, the same columns."""
    got = rows() if got is None else got
    cols = list(got[0])
    def cell(v):
        return "" if v is None else str(v)
    return "\n".join([",".join(cols)] + [",".join(cell(r[c]) for c in cols) for r in got]) + "\n"


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "nfeloqb"


@pytest.fixture
def transport(monkeypatch):
    """The one function that touches the network, answering with the fixture."""
    calls = []

    def _get(url):
        calls.append(url)
        return csv_text().encode()
    monkeypatch.setattr(nfeloqb, "_http_get", _get)
    return calls


# --- parsing the published shape -------------------------------------------------

def test_the_csv_parses_to_the_contracted_frame():
    """Cast to the contract's dtypes rather than inferred: a file whose adjustments all
    happen to be whole numbers infers `qb1_adj` as an integer column, and the contract would
    refuse a dtype the source never chose."""
    whole = [{**r, "qb1_adj": int(r["qb1_adj"])} for r in rows()]
    df = nfeloqb.parse(csv_text(whole).encode())
    assert df.height == 7
    assert df.schema["qb1_value_pre"] == pl.Float64
    assert df.schema["qb1_adj"] == pl.Float64
    assert df.schema["score1"] == pl.Int64
    assert df["score1"].null_count() == 2, "the coming week's rows have no score"


def test_a_row_with_no_quarterback_on_either_side_is_dropped_and_counted_not_refused(capsys):
    """The live file's first pull (2026-09-12) carried 2,162 pre-1950 rows and, for each of
    the two played 2026 games, an Elo-only twin of the quarterback row: same teams, same
    date, same score, no quarterback on either side. The contract refused the whole file.
    Both kinds are dropped and counted, and the line says what the this-season ones are."""
    got = rows()
    twin = dict(got[0])
    twin.update({"qb1": None, "qb2": None, "qb1_value_pre": None, "qb2_value_pre": None,
                 "qb1_adj": None, "qb2_adj": None})
    frame = nfeloqb.parse(csv_text([twin, *got]).encode())
    assert frame.height == len(got)
    said = capsys.readouterr().out
    assert "dropped 1 rows with no quarterback on either side" in said
    assert "1 of them this season, the Elo-only twin" in said


def test_one_null_pre_game_value_drops_that_side_and_serves_the_other_31_teams(capsys):
    """#283: one unknown quarterback with a null prior -- a rookie the source has no value
    for -- refused the whole eighteen-thousand-row file and every team fell back to
    last-good. The refusal is scoped to the side: LV's value on its coming game is null, so
    LV's state is its previous row and its opponent's game is served as it was."""
    got = rows()
    got[-2]["qb1_value_pre"] = None                       # LV @ WSH on 09-27, LV's side
    by = {r["team"]: r for r in nfeloqb.state(nfeloqb.parse(csv_text(got).encode())).to_dicts()}
    assert len(by) == 6, "every team is served"
    assert by["LV"]["as_of"] == "2026-09-21" and by["LV"]["qb"] == "Gardner Minshew"
    assert by["WAS"]["as_of"] == "2026-09-27" and by["WAS"]["qb_adj"] == pytest.approx(7.0)
    said = capsys.readouterr().out
    assert "1 rows blank on one side" in said and "LV 2026-09-27" in said


def test_a_row_blank_on_one_side_keeps_the_other_sides_game(capsys):
    """The twin-row structure the source emits is blank on *both* sides, and the filter was
    two-sided too: a row blank on one side dropped the opponent's game with it. KC's side of
    the 09-27 row is blanked; LA's side of the same row is the row LA's state is built from,
    and KC's tenure counts the two played rows before it."""
    got = rows()
    got[-1].update({"qb1": None, "qb1_value_pre": None, "qb1_adj": None})   # KC @ LAR, KC's side
    by = {r["team"]: r for r in nfeloqb.state(nfeloqb.parse(csv_text(got).encode())).to_dicts()}
    assert by["LA"]["as_of"] == "2026-09-27" and by["LA"]["qb"] == "Matthew Stafford"
    assert by["KC"]["as_of"] == "2026-09-20" and by["KC"]["qb_adj"] == pytest.approx(11.0)
    assert by["KC"]["tenure"] == 2
    assert "KC 2026-09-27" in capsys.readouterr().out


def test_the_captured_file_parses_through_the_contract():
    """The first live pull, 2026-09-12: the last 300 rows of `qb_elos.csv` frozen as a
    capture, two of them the Elo-only twins of the two played 2026 games -- same teams,
    same date, same score as their quarterback rows, and no quarterback. The shape the
    synthetic rows guessed is the shape the source publishes -- which is what moves
    `NFELOQB.verified_against_live` to True."""
    captured = json.loads((FIXTURES / "nfeloqb_qb_elos.json").read_text())
    twins = [r for r in captured if r["qb1"] is None]
    assert len(twins) == 2 and all(r["score1"] is not None for r in twins)
    for t in twins:
        named = [r for r in captured if r["game_id"] == t["game_id"] and r["qb1"] is not None]
        assert len(named) == 1 and named[0]["score1"] == t["score1"], "the twin names the starter"
    frame = nfeloqb.parse(csv_text(captured).encode())
    assert frame.height == 298 and set(frame["season"].to_list()) == {2025, 2026}
    state = nfeloqb.state(frame)
    assert state.height == 32, "every team has a starter on the captured file"


def test_a_renamed_column_is_refused_by_name():
    text = csv_text().replace("qb1_value_pre", "qb1_val_pre")
    with pytest.raises(ContractViolation, match="qb1_value_pre"):
        nfeloqb.parse(text.encode())


# --- the per-team state ----------------------------------------------------------

def test_each_team_appears_once_with_its_latest_starter():
    state = nfeloqb.state(nfeloqb.parse(csv_text().encode()))
    assert state["team"].n_unique() == state.height == 6
    by = {r["team"]: r for r in state.to_dicts()}
    assert by["KC"]["qb"] == "Patrick Mahomes"
    assert by["LV"]["qb"] == "Aidan O'Connell", "the unplayed row names the coming starter"


def test_tenure_counts_the_games_the_starter_has_played_in_his_current_run():
    by = {r["team"]: r for r in nfeloqb.state(nfeloqb.parse(csv_text().encode())).to_dicts()}
    assert by["LV"]["tenure"] == 0, "named for the coming game, has started none"
    assert by["KC"]["tenure"] == 2, "two played rows before the unplayed one"
    assert by["DEN"]["tenure"] == 2, "two played rows and no unplayed one"


def test_the_state_carries_the_latest_rows_adjustment_and_not_the_arrival_rows():
    """#268: the source's `qb_adj` is already the gap between the starter and what the
    team's rolling value embeds, already decayed by its own update rule, so the state hands
    over the latest row's number and nothing about the row he arrived on. KC's run in the
    fixture is three rows with the value drifting 145.2 -> 147.0 and the adjustment
    12.0 -> 10.0; the state says 10.0, which is the row the old state never read."""
    by = {r["team"]: r for r in nfeloqb.state(nfeloqb.parse(csv_text().encode())).to_dicts()}
    assert by["KC"]["qb_adj"] == pytest.approx(10.0), "the latest row, not the arrival row's 12"
    assert by["KC"]["qb_value"] == pytest.approx(147.0), "drifted since he arrived at 145.2"
    assert by["KC"]["tenure"] == 2, "an input the replaced estimator would have decayed on"
    assert by["LV"]["qb_adj"] == pytest.approx(-110.0)
    assert set(by["KC"]) == {"team", "qb", "qb_value", "qb_adj", "tenure", "as_of"}


# --- the as-of (#272) ------------------------------------------------------------------
#
# The quarterback state for a fixture is built from rows strictly earlier than that
# fixture's kickoff, so a backtest of the layer can be leak-free. `docs/method.md` rule 2:
# measure the predictor strictly before the outcome window. Asserted on the split -- which
# rows reach the state -- rather than on what the state says about them.

def test_the_split_is_strictly_before_the_as_of_day():
    """`before` is the split. Given a day, every row it keeps is dated before that day and
    every row it drops is dated on or after it -- so a fixture's own row, dated its game
    day, never reaches a state built as of its kickoff, and neither does any later one."""
    rows = nfeloqb.parse(csv_text().encode())
    kept = nfeloqb.before(rows, dt.date(2026, 9, 20))
    assert kept["date"].to_list() == ["2026-09-10", "2026-09-13", "2026-09-14"]
    assert kept.height + rows.filter(pl.col("date") >= "2026-09-20").height == rows.height
    assert nfeloqb.before(rows, dt.date(2026, 9, 10)).height == 0, "the first day has nothing before it"


def test_a_kickoff_is_read_as_the_eastern_game_day_before_the_split():
    """The source dates a row by its Eastern game day; the repo's kickoffs are naive UTC. A
    Thursday 20:15 ET kickoff is 00:15 UTC Friday, and a split on the UTC day would keep
    the Thursday row -- the game's own -- as if it were earlier. The datetime form of the
    as-of is converted to the Eastern day first."""
    rows = nfeloqb.parse(csv_text().encode())
    friday_utc = dt.datetime(2026, 9, 11, 0, 15)         # Thursday 09-10, 20:15 ET
    assert nfeloqb.before(rows, friday_utc).height == 0, "the 09-10 row is the game's own"
    assert nfeloqb.before(rows, dt.datetime(2026, 9, 14, 17, 0))["date"].to_list() == [
        "2026-09-10", "2026-09-13"]


def test_state_as_of_a_fixture_carries_nothing_from_that_game_or_later():
    """Every game in the captured file: the state built as of its kickoff is built from a
    split that holds no row of that game and no row dated on or after its day. On the split,
    by `game_id`, over all 298 rows the file names a quarterback on."""
    captured = json.loads((FIXTURES / "nfeloqb_qb_elos.json").read_text())
    rows = nfeloqb.parse(csv_text(captured).encode())
    checked = 0
    for r in rows.select("game_id", "date").unique().iter_rows(named=True):
        day = dt.date.fromisoformat(r["date"])
        split = nfeloqb.before(rows, day)
        assert r["game_id"] not in split["game_id"].to_list()
        assert all(d < r["date"] for d in split["date"].to_list())
        checked += 1
    assert checked == 298, "every game in the capture, once"
    st = nfeloqb.state(rows, as_of=dt.date(2026, 9, 13))
    assert all(d < "2026-09-13" for d in st["as_of"].to_list())


def test_state_as_of_is_the_latest_row_before_the_day_per_team():
    """KC's rows in the fixture are 09-10, 09-20 and 09-27. As of the 09-20 game the state
    is the 09-10 row: the adjustment the source published there (12.0, not the 10.0 of the
    latest row), one game played in the run. A team with no row before the day is absent
    rather than served from its future."""
    rows = nfeloqb.parse(csv_text().encode())
    by = {r["team"]: r for r in nfeloqb.state(rows, as_of=dt.date(2026, 9, 20)).to_dicts()}
    assert by["KC"]["qb_adj"] == pytest.approx(12.0) and by["KC"]["as_of"] == "2026-09-10"
    assert by["KC"]["tenure"] == 1
    assert by["LV"]["qb"] == "Gardner Minshew", "the 09-13 row; O'Connell is named on 09-27"
    assert set(by) == {"KC", "LAC", "LV", "DEN", "WAS", "LA"}
    assert set(nfeloqb.state(rows, as_of=dt.date(2026, 9, 13)).to_dicts()[0].keys()) == set(
        nfeloqb.STATE_SCHEMA)
    assert nfeloqb.state(rows, as_of=dt.date(2026, 9, 13))["team"].to_list() == ["KC", "LAC"]


def test_without_an_as_of_the_state_is_unchanged():
    """No current consumer moves: the default is every row, which is the latest state."""
    rows = nfeloqb.parse(csv_text().encode())
    assert nfeloqb.state(rows).equals(nfeloqb.state(rows, as_of=None))
    assert max(nfeloqb.state(rows)["as_of"].to_list()) == "2026-09-27"


def test_538_abbreviations_are_spelled_as_nflverse_spells_them():
    teams = set(nfeloqb.state(nfeloqb.parse(csv_text().encode()))["team"].to_list())
    assert "WAS" in teams and "LA" in teams
    assert "WSH" not in teams and "LAR" not in teams


# nflverse's 32 spellings, which every join in this repo is on.
NFLVERSE = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
            "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
            "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]


def test_the_raiders_are_spelled_as_nflverse_spells_them_and_no_team_is_unknown():
    """#283: nfeloqb spells the Raiders `OAK` and nflverse `LV`, and the map carried the two
    other differences and not this one, so every Raiders game was unadjustable on both
    sides. On the captured live file every one of the 32 teams now matches a schedule in
    nflverse's spellings, and `unknown_teams` -- the sentence a missing mapping would
    otherwise be the only sign of -- is empty."""
    captured = json.loads((FIXTURES / "nfeloqb_qb_elos.json").read_text())
    assert any(r["team1"] == "OAK" or r["team2"] == "OAK" for r in captured), "the source's spelling"
    st = nfeloqb.state(nfeloqb.parse(csv_text(captured).encode()))
    assert "LV" in st["team"].to_list() and "OAK" not in st["team"].to_list()
    games = pl.DataFrame({"home_team": NFLVERSE[:16], "away_team": NFLVERSE[16:]})
    assert nfeloqb.unknown_teams(st, games) == []


def test_the_spellings_the_docstring_reports_are_the_captured_files():
    """The re-read #283 asks for: 538 spelled Washington `WSH`, and the docstring said the
    map carried that difference; the live file spells it `WAS` already, spells the Rams
    `LAR` and the Raiders `OAK`. The docstring now says what each was found to be, and this
    holds the capture to it."""
    captured = json.loads((FIXTURES / "nfeloqb_qb_elos.json").read_text())
    spelled = {r["team1"] for r in captured} | {r["team2"] for r in captured}
    assert {"WAS", "LAR", "OAK"} <= spelled
    assert not {"WSH", "LA", "LV"} & spelled
    assert set(nfeloqb.ABBREVIATIONS) == {"WSH", "LAR", "OAK"}
    assert nfeloqb.ABBREVIATIONS["OAK"] == "LV"


# --- cache, last-good, refusal -----------------------------------------------------

def test_a_refresh_writes_the_file_and_a_stamp_and_reads_back(cache, transport):
    got = nfeloqb.refresh(cache=cache, now=dt.datetime(2026, 9, 12, 12))
    assert got.height == 7
    assert (cache / nfeloqb.FILE).exists()
    assert nfeloqb.captured_at(cache) == "2026-09-12T12:00:00"
    again = nfeloqb.read_rows(cache)
    assert again is not None and again.height == 7


def test_a_failed_fetch_serves_the_last_good_file_and_says_why(cache, transport, monkeypatch,
                                                                capsys):
    nfeloqb.refresh(cache=cache)

    def _down(url):
        raise OSError("no route to github")
    monkeypatch.setattr(nfeloqb, "_http_get", _down)
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 0
    out = capsys.readouterr()
    assert "no route to github" in out.err and "last-good" in out.err
    assert "6 teams" in out.out


def test_a_fresh_clone_with_nothing_cached_is_a_sentence_and_a_non_zero_exit(cache, monkeypatch,
                                                                             capsys):
    def _down(url):
        raise OSError("no route to github")
    monkeypatch.setattr(nfeloqb, "_http_get", _down)
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "Traceback" not in err


def test_a_drifted_cache_is_refused_rather_than_served(cache, transport, monkeypatch, capsys):
    nfeloqb.refresh(cache=cache)
    (cache / nfeloqb.FILE).write_text(csv_text().replace("qb1_adj", "qb1_adjustment"))

    def _down(url):
        raise OSError("no route to github")
    monkeypatch.setattr(nfeloqb, "_http_get", _down)
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 1
    assert "qb1_adj" in capsys.readouterr().err


def test_status_reads_the_cache_and_nothing_else(cache, transport, capsys):
    nfeloqb.refresh(cache=cache)
    transport.clear()
    assert nfeloqb.main(["--status", "--cache", str(cache)]) == 0
    assert transport == []
    assert "Aidan O'Connell" in capsys.readouterr().out


def test_a_refresh_from_the_cli_prints_the_state(cache, transport, capsys):
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 0
    out = capsys.readouterr().out
    assert "6 teams" in out and "Patrick Mahomes" in out


def test_a_pull_the_contract_refuses_serves_the_last_good_file(cache, transport, monkeypatch,
                                                                capsys):
    nfeloqb.refresh(cache=cache)

    def _renamed(url):
        return csv_text().replace("qb1_adj", "qb1_adjustment").encode()
    monkeypatch.setattr(nfeloqb, "_http_get", _renamed)
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 0
    out = capsys.readouterr()
    assert "refused" in out.err and "qb1_adj" in out.err
    assert "6 teams" in out.out, "the last-good file, not the refused one"
    kept = nfeloqb.read_rows(cache)
    assert kept is not None and kept.height == 7, "the refused pull did not overwrite it"


def test_status_on_a_fresh_clone_is_a_sentence_and_a_non_zero_exit(cache, capsys):
    assert nfeloqb.main(["--status", "--cache", str(cache)]) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "--refresh" in err


def test_status_on_a_drifted_cache_is_refused_by_name(cache, transport, capsys):
    nfeloqb.refresh(cache=cache)
    (cache / nfeloqb.FILE).write_text(csv_text().replace("team1", "home"))
    assert nfeloqb.main(["--status", "--cache", str(cache)]) == 1
    assert "team1" in capsys.readouterr().err


def test_an_unreadable_stamp_is_no_capture_time_rather_than_a_traceback(cache, transport):
    nfeloqb.refresh(cache=cache)
    (cache / nfeloqb.STAMP).write_text("{not json")
    assert nfeloqb.captured_at(cache) is None


def test_a_team_the_schedule_does_not_spell_is_named():
    st = nfeloqb.state(nfeloqb.parse(csv_text().encode()))
    games = pl.DataFrame({"home_team": ["KC", "WAS"], "away_team": ["LV", "LA"]})
    assert nfeloqb.unknown_teams(st, games) == ["DEN", "LAC"]


def test_the_transport_refuses_a_live_call_from_the_default_suite(monkeypatch):
    """`hub.fetch.pool`'s guard, at the one function here that touches the network."""
    import urllib.request

    def _never(*a, **k):                                     # pragma: no cover - must not run
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(urllib.request, "urlopen", _never)
    with pytest.raises(nfeloqb.LiveCallRefused) as e:
        nfeloqb._http_get(nfeloqb.URL)
    assert "test_the_transport_refuses_a_live_call" in str(e.value)


# --- the pin (#271) -------------------------------------------------------------------
#
# Two runs a week apart from the same pinned input produce the same number. The URL names a
# commit of the source repository rather than its default branch, the stamp records that
# commit and whether the bytes matched the pinned hash, and the commit is in the model
# digest so advancing it is a deliberate edit that moves the version.

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_url_names_a_commit_not_a_branch(cache, transport, monkeypatch):
    monkeypatch.setattr(nfeloqb, "COMMIT", COMMIT)
    assert f"/greerreNFL/nfeloqb/{COMMIT}/qb_elos.csv" in nfeloqb.url(COMMIT)
    assert "/main/" not in nfeloqb.url(COMMIT)
    nfeloqb.refresh(cache=cache)
    assert transport == [nfeloqb.url(COMMIT)], "the pull went to the pinned commit"
    assert nfeloqb.stamp(cache)["commit"] == COMMIT, "the stamp records the commit"


def test_a_pull_whose_hash_differs_from_the_pin_is_a_source_change_said_out_loud(
        cache, transport, monkeypatch, capsys):
    """Served -- the file validated -- but never silently: the stamp says the bytes did not
    match, the CLI says so on stderr, and the fit's sentence repeats it (test_ratings.py)."""
    monkeypatch.setattr(nfeloqb, "COMMIT", COMMIT)
    monkeypatch.setattr(nfeloqb, "PINNED_SHA256", "0" * 64)
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 0
    out = capsys.readouterr()
    assert "source change" in out.err and COMMIT[:12] in out.err
    assert "6 teams" in out.out, "served, not refused"
    st = nfeloqb.stamp(cache)
    assert st["matches_pin"] is False and st["pinned_sha256"] == "0" * 64
    assert "source change" in (nfeloqb.source_change(cache) or "")


def test_a_pull_that_matches_the_pin_is_not_a_source_change(cache, transport, monkeypatch,
                                                            capsys):
    monkeypatch.setattr(nfeloqb, "COMMIT", COMMIT)
    monkeypatch.setattr(nfeloqb, "PINNED_SHA256", _sha(csv_text()))
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 0
    assert "source change" not in capsys.readouterr().err
    assert nfeloqb.stamp(cache)["matches_pin"] is True
    assert nfeloqb.source_change(cache) is None


def test_an_unpinned_pull_is_said_rather_than_silent(cache, transport, monkeypatch, capsys):
    """No commit pinned is the default branch at pull time, which is what the URL always
    was. It is allowed -- a fetch must serve with zero attention -- and it is said: the
    stamp records no commit and the run names the two constants to set from it."""
    monkeypatch.setattr(nfeloqb, "COMMIT", None)
    monkeypatch.setattr(nfeloqb, "PINNED_SHA256", None)
    assert nfeloqb.main(["--refresh", "--cache", str(cache)]) == 0
    err = capsys.readouterr().err
    assert "unpinned" in err and "COMMIT" in err and "PINNED_SHA256" in err
    st = nfeloqb.stamp(cache)
    assert st["commit"] is None and st["matches_pin"] is None
    assert transport == [nfeloqb.url(None)] and "/main/" in transport[0]


def test_advancing_the_pin_is_an_edit_that_moves_the_model_digest(monkeypatch):
    """The predictions under a new input are distinguishable from the old: `COMMIT` is in
    `config_digest` through `FITTED_EXTRA`, so the version string moves with the pin and
    nothing else about the run has to."""
    from hub.config import HubConfig, config_digest, fitted_constants
    assert "nfeloqb.COMMIT" in fitted_constants()
    before = config_digest(HubConfig())
    monkeypatch.setattr(nfeloqb, "COMMIT", COMMIT)
    assert config_digest(HubConfig()) != before
