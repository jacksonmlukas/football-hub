"""The board's ADP substance guard.

The fetch layer is typed now, so the guard is defence in depth rather than the
primary fix. It stays because the original failure -- ESPN returning something
structurally valid and semantically empty -- is the documented failure mode for
this whole data source, not a one-off.

Seam under test: whatever `player_adp` hands back, does the board degrade loudly
instead of shipping an edge column with nothing in it?
"""
import polars as pl
import pytest
from hub.draft import board


@pytest.fixture
def fake_adp(monkeypatch):
    def _install(frame_or_exc):
        import hub.fetch.espn as espn

        def _fake(*a, **kw):
            if isinstance(frame_or_exc, Exception):
                raise frame_or_exc
            return frame_or_exc
        monkeypatch.setattr(espn, "player_adp", _fake)
    return _install


def _frame(rows, dtype=pl.Float64):
    return pl.DataFrame(rows, schema={"player": pl.Utf8, "adp": dtype})


def test_usable_adp_is_returned(fake_adp):
    fake_adp(_frame([{"player": "Jahmyr Gibbs", "adp": 1.49}]))
    adp = board.espn_adp()
    assert adp is not None and adp.height == 1


def test_empty_frame_degrades_to_ecr_only(fake_adp, capsys):
    fake_adp(_frame([]))
    assert board.espn_adp() is None
    assert "ECR-only" in capsys.readouterr().out


def test_non_numeric_adp_degrades_to_ecr_only(fake_adp, capsys):
    """The original bug shape: structurally present, semantically empty."""
    fake_adp(pl.DataFrame({"player": ["a", "b"], "adp": [[], []]}))
    assert board.espn_adp() is None
    assert "ECR-only" in capsys.readouterr().out


def test_fetch_failure_degrades_to_ecr_only(fake_adp, capsys):
    fake_adp(RuntimeError("ESPN 403"))
    assert board.espn_adp() is None
    out = capsys.readouterr().out
    assert "ECR-only" in out and "RuntimeError" in out
