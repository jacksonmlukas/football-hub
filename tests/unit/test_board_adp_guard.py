"""The ADP guard.

The Sep 2026 failure was not a crash: espn_api sets `posRank` from a
`positionalRanking` key that free-agent payloads omit, so it comes back as `[]`.
An empty list is not null, so the null filter passed it, `espn_adp` returned a
non-None frame, the ECR-only branch never fired, and `edge` shipped as 366 empty
lists. Shape checks passed; substance was absent.

These tests pin the substance check: anything that is not a numeric ADP must
return None so the caller degrades loudly instead of quietly.
"""
import polars as pl
import pytest
from hub.draft import board


class _Player:
    def __init__(self, name, posRank):
        self.name, self.posRank = name, posRank


class _League:
    def __init__(self, players):
        self._players = players

    def free_agents(self, size=400):
        return self._players[:size]


@pytest.fixture
def fake_league(monkeypatch):
    def _install(players):
        import hub.fetch.espn as espn
        monkeypatch.setattr(espn, "league_settings", lambda: (_League(players), {}))
    return _install


def test_empty_positional_ranking_degrades_to_ecr_only(fake_league, capsys):
    """The actual bug: `positionalRanking` absent -> posRank is []."""
    fake_league([_Player(f"P{i}", []) for i in range(50)])
    assert board.espn_adp() is None
    assert "ECR-only" in capsys.readouterr().out


def test_all_null_positional_ranking_degrades_to_ecr_only(fake_league):
    fake_league([_Player(f"P{i}", None) for i in range(50)])
    assert board.espn_adp() is None


def test_no_free_agents_degrades_to_ecr_only(fake_league):
    fake_league([])
    assert board.espn_adp() is None


def test_real_numeric_adp_is_returned(fake_league):
    fake_league([_Player(f"P{i}", i + 1) for i in range(50)])
    adp = board.espn_adp()
    assert adp is not None
    assert adp.height == 50
    assert adp["adp"].dtype.is_numeric()
    assert adp["adp"].is_not_null().all()


def test_partial_nulls_keep_the_numeric_rows(fake_league):
    fake_league([_Player("a", 1), _Player("b", None), _Player("c", 3)])
    adp = board.espn_adp()
    assert adp is not None
    assert adp.height == 2
