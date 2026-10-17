"""The kickoff and betting-market fields on a published prediction row (#482).

An artifact written before the fields existed carries none of them and is read as it always
was; a row that carries them must keep them true. Checked on whatever is committed, so the
first publish after this lands is held to it by the next push.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[2] / "site" / "data"


def _built():
    """Rows built by the writer's own function, so the contract bites before any is committed."""
    from hub import market_row
    from hub.models.market import MARGIN_SD, normal_cdf
    kick = datetime(2026, 10, 18, 17, 0)
    cap = {("g", datetime(2026, 10, 17, 9)): {"close_spread": -3.5, "close_total": 44.5,
                                              "ml_home_prob": None}}
    for priced in (datetime(2026, 10, 17, 9), None):
        row = {"game_id": "g", "model": "market_baseline", "priced_at": priced,
               "home_win_prob": normal_cdf(-3.5 / MARGIN_SD),
               "predicted_at": "2026-10-17 10:00:00"}
        row |= market_row.market_fields(row, kick, None, cap)
        if priced:
            row["priced_at"] = str(priced)
        yield pytest.param(row, id=f"built-priced-{priced is not None}")


def _rows():
    yield from _built()
    for path in sorted(SITE.glob("preds_*.json")):
        for r in json.loads(path.read_text()).get("rows", []):
            if "market_prob" in r:
                yield pytest.param(r, id=f"{path.name}:{r.get('game_id')}")


@pytest.mark.parametrize("row", _rows())
def test_a_row_with_market_fields_keeps_them_true(row):
    kickoff = None
    if row["kickoff"] is not None:
        kickoff = datetime.fromisoformat(row["kickoff"])
        assert kickoff.utcoffset() is not None
    else:
        assert row["kickoff_missing"], "a null kickoff says why"
    if row["market_prob"] is None:
        assert row["market_missing"], "a null price says why, and is never 0"
        return
    assert 0 < row["market_prob"] < 1
    assert datetime.fromisoformat(row["priced_at"]) < datetime.fromisoformat(row["predicted_at"])
    assert kickoff is not None, "a price is published only against a known kickoff"
    assert datetime.fromisoformat(row["priced_at"]) <= kickoff.replace(tzinfo=None)
    if row["model"] == "market_baseline":
        assert abs(row["home_win_prob"] - row["market_prob"]) < 1e-9
