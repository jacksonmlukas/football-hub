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

from hub.contracts import ESPN_SCOREBOARD, FF_OPPORTUNITY, PBP, SCHEDULES
from hub.fetch import espn, nflverse
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


@pytest.mark.parametrize("league", ["nfl", "cfb"])
def test_the_live_board_still_resolves_through_the_reader(league, monkeypatch):
    """The live board, put through `live_state` -- the function the poller actually calls.

    Until 2026-09-05 this asserted `events[0]["status"]["type"]["state"]`: the *event*-level
    status, which the poller does not read. `_overlay_row` reads
    `competitions[0].status.type.state`, so ESPN could have dropped the path production
    depends on while this stayed green -- a nightly check named for the shape the poller
    reads, watching a shape beside it. That is the defect #73 removed from the frozen
    capture; it survived here because the check and the reader were written apart.

    So the board goes through the reader rather than through a second copy of its lookups.
    Fetched once and replayed, because `live_state` fetches for itself and one nightly GET
    per league is the budget.

    Both leagues, deliberately: the poller defaults to `nfl` and the frozen capture is the
    college board, and the endpoint is shared, so a rename shows up in whichever board has
    games that night.

    Every event has to resolve, not merely one. A dropped event is `_overlay_row` refusing a
    shape, which is exactly what this is here to notice -- the alternative tolerates a
    partial rename until it reaches all of them, and `live_state` only raises when *nothing*
    resolves.
    """
    # `_get` degrades to last-good when every host fails -- right for the dashboard, wrong
    # here: it would let this pass on a replayed file and claim a verification that did not
    # happen, which is the one thing this file's header says a golden test must not do. The
    # cache is written only on a successful fetch, so an unadvanced mtime *is* the outage.
    cached = espn.CACHE / f"sb_{league}_now.json"
    before = cached.stat().st_mtime if cached.exists() else 0.0
    try:
        payload = espn.scoreboard(league)
    except RuntimeError as e:                       # unreachable, and nothing to serve
        pytest.skip(f"ESPN unreachable for {league}: {e}")
    if cached.exists() and cached.stat().st_mtime <= before:
        pytest.skip(f"ESPN unreachable for {league}; `_get` served last-good, which proves "
                    f"nothing about the live shape")

    events = (payload or {}).get("events") or []
    if not events:
        pytest.skip(f"no games on the {league} board right now")
    monkeypatch.setattr(espn, "scoreboard", lambda *a, **k: payload)

    got = ESPN_SCOREBOARD.validate(espn.scoreboard_frame(espn.live_state(league)))
    assert got.height == len(events), (
        f"the {league} board carried {len(events)} event(s) and the reader resolved "
        f"{got.height}; the dropped one(s) are named on stdout above. An event ESPN still "
        f"lists but the overlay cannot read is a shape change in the making.")


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
