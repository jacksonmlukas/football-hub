"""roster.json carries weekly ECR and ESPN's weekly projection beside `mu` (#492)."""
import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from hub import atomic, publish
from hub.fetch import consensus

WEEK = 7
FIRST = date(2026, 10, 25)


def _frame(week: int | None = WEEK, espn: bool = True) -> pl.DataFrame:
    n = 2
    cols: dict[str, list] = {
        "player": ["Ja'Marr Chase", "Nobody Ranked"], "pos": ["WR", "RB"],
        "nfl_team": ["CIN", "GB"], "mu": [19.7, 11.0], "sd": [6.0, 5.0],
        "projected": [True] * n, "starting": [True] * n, "injury_status": ["ACTIVE"] * n,
        "slot": ["WR", "RB"], "available": [True] * n, "can_start": [True] * n,
        "missing_games": [0] * n}
    if espn:
        cols["espn_week"] = [week] * n
        cols["espn_week_proj"] = [17.5, None]
    return pl.DataFrame(cols, schema_overrides={"espn_week": pl.Int64} if espn else None)


def _published(site, tmp_path, df):
    src = tmp_path / "roster.parquet"
    df.write_parquet(src)
    publish.roster(out=site, path=src)
    return json.loads((site / "roster.json").read_text())


@pytest.fixture(autouse=True)
def _no_stream_wire(monkeypatch):
    """The K / D/ST heuristic (#494) reads the schedule; not what is under test here."""
    from hub.season import stream
    monkeypatch.setattr(publish, "_roster_stream", lambda df: stream.unavailable("stubbed"))


@pytest.fixture
def site(tmp_path):
    return tmp_path / "site"


def _write_capture(root):
    when = datetime(2026, 10, 20, 12, tzinfo=UTC)
    rows = pl.DataFrame({"player": ["Ja'Marr Chase"], "pos": ["WR"], "team": ["CIN"],
                         "ecr": [4.0]})
    doc = consensus.build(2026, WEEK, rows, date(2026, 10, 24), FIRST, when, "d")
    atomic.write_text(consensus.capture_path(2026, WEEK, when, root), json.dumps(doc))


def test_ecr_and_the_espn_weekly_are_published_per_player(site, tmp_path, monkeypatch):
    monkeypatch.setattr(consensus, "ROOT", tmp_path / "cons")
    _write_capture(tmp_path / "cons")
    art = _published(site, tmp_path, _frame())
    by = {r["player"]: r for r in art["rows"]}
    assert by["Ja'Marr Chase"]["ecr"] == 4.0
    assert by["Ja'Marr Chase"]["espn_week_proj"] == 17.5
    assert art["week"] == WEEK and art["ecr_reason"] is None


def test_an_unranked_player_is_null_not_zero(site, tmp_path, monkeypatch):
    monkeypatch.setattr(consensus, "ROOT", tmp_path / "cons")
    _write_capture(tmp_path / "cons")
    nobody = {r["player"]: r for r in _published(site, tmp_path, _frame())["rows"]}[
        "Nobody Ranked"]
    assert nobody["ecr"] is None and nobody["espn_week_proj"] is None


def test_no_capture_publishes_null_ecr_with_the_reason(site, tmp_path, monkeypatch):
    monkeypatch.setattr(consensus, "ROOT", tmp_path / "none")
    art = _published(site, tmp_path, _frame())
    assert all(r["ecr"] is None for r in art["rows"])
    assert "no pre-kickoff consensus capture for week 7" in art["ecr_reason"]


def test_a_roster_with_no_weekly_column_publishes_null_with_the_reason(site, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(consensus, "ROOT", tmp_path / "none")
    art = _published(site, tmp_path, _frame(espn=False))
    assert all(r["espn_week_proj"] is None for r in art["rows"])
    assert "predates" in art["espn_week_reason"] and art["week"] is None
