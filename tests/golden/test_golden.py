"""Nightly: hit the live APIs and diff against the frozen fixtures.

This is the job that actually catches an upstream rename. `tests/contracts/` proves we
parse the fixture; only this proves the fixture still resembles reality.

Marked `golden` and deselected by default -- `data-contracts/SKILL.md` is explicit that a
third-party outage must never block a docs commit. Run deliberately:

    uv run pytest -m golden

Sources without a key on this machine (CFBD, The Odds API) are skipped rather than failed.
A skip is honest; a pass would claim verification that did not happen.
"""
import json
from pathlib import Path

import pytest

from hub.contracts import FF_OPPORTUNITY, PBP, SCHEDULES
from hub.fetch import nflverse
from hub.fetch.nflverse import PBP_COLS

pytestmark = pytest.mark.golden

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_columns(name: str) -> set[str]:
    rows = json.loads((FIXTURES / name).read_text())
    return set(rows[0]) if rows else set()


def test_live_pbp_still_has_every_column_we_freeze(tmp_path):
    live = nflverse.load("pbp", seasons=[2025], cols=list(PBP_COLS), cache=tmp_path)
    missing = fixture_columns("nflverse_pbp.json") - set(live.columns)
    assert not missing, f"nflverse dropped or renamed: {sorted(missing)}"


def test_live_pbp_still_satisfies_its_contract(tmp_path):
    live = nflverse.load("pbp", seasons=[2025], cols=list(PBP_COLS), cache=tmp_path)
    assert PBP.validate(live).height == live.height


def test_live_ff_opportunity_still_satisfies_its_contract(tmp_path):
    live = nflverse.load("ff_opportunity", seasons=[2025], cache=tmp_path)
    assert FF_OPPORTUNITY.validate(live).height == live.height


def test_live_schedules_still_satisfies_its_contract(tmp_path):
    live = nflverse.load("schedules", seasons=[2025], cache=tmp_path)
    assert SCHEDULES.validate(live).height == live.height


def test_live_espn_scoreboard_still_has_the_shape_the_poller_reads():
    from hub.fetch.espn import scoreboard
    events = (scoreboard("nfl") or {}).get("events") or []
    if not events:
        pytest.skip("no games on the board right now")
    e = events[0]
    assert e.get("id") and (e.get("status") or {}).get("type", {}).get("state")
    sides = (e.get("competitions") or [{}])[0].get("competitors") or []
    assert {c.get("homeAway") for c in sides} == {"home", "away"}


def test_live_cfbd_matches_its_fixture_shape():
    from hub.fetch import cfbd
    if not cfbd._api_key():
        pytest.skip("no CFBD_API_KEY: this source is unverified against reality")
    live = cfbd.bulk("games", year=2025, week=1)
    missing = fixture_columns("cfbd_games.synthetic.json") - set(live.columns)
    assert not missing, f"CFBD shape differs from the hand-built fixture: {sorted(missing)}"


def test_live_odds_matches_its_fixture_shape():
    from hub.fetch import odds
    key = odds._api_key()
    if key is None:
        pytest.skip("no ODDS_API_KEY: this source is unverified against reality")
    payload, _ = odds._http_get(
        {"markets": odds.MARKET, "regions": odds.REGION}, key)
    if not payload:
        pytest.skip("no events on the board right now")
    missing = fixture_columns("odds_spreads.synthetic.json") - set(payload[0])
    assert not missing, f"Odds API shape differs from the fixture: {sorted(missing)}"
