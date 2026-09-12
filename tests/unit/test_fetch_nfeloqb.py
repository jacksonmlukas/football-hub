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


def test_a_row_with_no_quarterback_is_dropped_and_counted_not_refused(capsys):
    """The live file's first pull (2026-09-12) carried 2,162 pre-1950 rows and two played
    2026 games with no quarterback filled in yet; the contract refused the whole file. Both
    kinds are dropped and counted, and the this-season count is the one printed to watch."""
    got = rows()
    blank = dict(got[-1])
    blank.update({"qb1": None, "qb2": None, "qb1_value_pre": None, "qb2_value_pre": None,
                  "qb1_adj": None, "qb2_adj": None, "date": "2026-09-09"})
    frame = nfeloqb.parse(csv_text(got + [blank]).encode())
    assert frame.height == len(got)
    said = capsys.readouterr().out
    assert "dropped 1 rows with no quarterback" in said and "1 of them this season" in said


def test_the_captured_file_parses_through_the_contract():
    """The first live pull, 2026-09-12: the last 300 rows of `qb_elos.csv` frozen as a
    capture, two of them 2026 games with no quarterback filled in yet. The shape the
    synthetic rows guessed is the shape the source publishes -- which is what moves
    `NFELOQB.verified_against_live` to True."""
    captured = json.loads((FIXTURES / "nfeloqb_qb_elos.json").read_text())
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


def test_the_arrival_row_is_the_first_of_the_current_run():
    """What the team rating embeds is read off the row the starter arrived on, not the
    latest one -- the latest one's gap is already decayed by the source's rolling value."""
    by = {r["team"]: r for r in nfeloqb.state(nfeloqb.parse(csv_text().encode())).to_dicts()}
    assert by["KC"]["arrival_value"] == pytest.approx(145.2)
    assert by["KC"]["arrival_adj"] == pytest.approx(12.0)
    assert by["KC"]["qb_value"] == pytest.approx(147.0)
    assert by["LV"]["arrival_value"] == pytest.approx(52.0)
    assert by["LV"]["arrival_adj"] == pytest.approx(-110.0)


def test_538_abbreviations_are_spelled_as_nflverse_spells_them():
    teams = set(nfeloqb.state(nfeloqb.parse(csv_text().encode()))["team"].to_list())
    assert "WAS" in teams and "LA" in teams
    assert "WSH" not in teams and "LAR" not in teams


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
