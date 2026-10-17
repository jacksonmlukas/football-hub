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


def _rows():
    for path in sorted(SITE.glob("preds_*.json")):
        for r in json.loads(path.read_text()).get("rows", []):
            if "market_prob" in r:
                yield pytest.param(r, id=f"{path.name}:{r.get('game_id')}")


@pytest.mark.parametrize("row", _rows())
def test_a_row_with_market_fields_keeps_them_true(row):
    if row["kickoff"] is not None:
        assert datetime.fromisoformat(row["kickoff"]).utcoffset() is not None
    else:
        assert row["kickoff_missing"], "a null kickoff says why"
    if row["market_prob"] is None:
        assert row["market_missing"], "a null price says why, and is never 0"
        return
    assert 0 < row["market_prob"] < 1
    assert datetime.fromisoformat(row["priced_at"]) < datetime.fromisoformat(row["predicted_at"])
    assert datetime.fromisoformat(row["priced_at"]) <= datetime.fromisoformat(row["kickoff"])
    if row["model"] == "market_baseline":
        assert abs(row["home_win_prob"] - row["market_prob"]) < 1e-9
