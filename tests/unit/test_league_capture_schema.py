"""The league capture's schema contract (#545): each refusal is planted against the frozen,
hand-built fixture. Self-contained on purpose -- the guard harness runs this file against a
mutated source tree with `tests/` off the path, so it imports nothing from there."""
import copy
import json
from pathlib import Path

import pytest

from hub.contracts import ContractViolation, check_league_capture

FIXTURE = (Path(__file__).resolve().parents[1] / "golden" / "fixtures"
           / "league_capture.synthetic.json")


def good_capture():
    return json.loads(FIXTURE.read_text())


def test_league_capture_contract_admits_the_frozen_fixture_and_a_null_deadline():
    c = good_capture()
    check_league_capture(c)
    c["settings"]["trade_deadline"] = None
    c["free_agents"] = None
    check_league_capture(c)


def _drop_team(c):
    c["teams"] = []


def _dup_id(c):
    c["teams"][0]["team_id"] = c["teams"][1]["team_id"]


def _two_ours(c):
    c["teams"][0]["is_mine"] = True


def _no_projection_column(c):
    del c["teams"][0]["roster"][0]["week_proj"]


def _epoch_zero(c):
    c["settings"]["trade_deadline"] = "1970-01-01T00:00:00+00:00"


def _naive_deadline(c):
    c["settings"]["trade_deadline"] = "2026-11-30T12:00:00"


def _stranger(c):
    c["schedule"][0]["opponent_id"] = 99


def _played_week(c):
    c["schedule"][0]["week"] = c["week"] - 1


def _bad_pool(c):
    c["free_agents"]["rows"][0]["espn_avg"] = 900.0


@pytest.mark.parametrize("plant, says", [
    (_drop_team, "no teams"), (_dup_id, "team ids are not unique"),
    (_two_ours, "teams marked ours"), (_no_projection_column, "roster row lacks"),
    (_epoch_zero, "trade_deadline"), (_naive_deadline, "trade_deadline"),
    (_stranger, "schedule pair"), (_played_week, "schedule pair"), (_bad_pool, "free_agents:"),
])
def test_league_capture_contract_refuses_each_planted_break(plant, says):
    """Each plant is matched on its own sentence: several breaks also trip a neighbouring
    check, and a bare "it raised" would stay green with the check under test deleted."""
    c = copy.deepcopy(good_capture())
    plant(c)
    with pytest.raises(ContractViolation, match=says):
        check_league_capture(c)
