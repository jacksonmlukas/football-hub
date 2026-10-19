"""The league's free agents, parsed from espn_api's Player into the contract's frame (#493)."""
from types import SimpleNamespace

import polars as pl
import pytest

import hub.fetch.espn as E
from hub.contracts import ESPN_FREE_AGENTS, ContractViolation


def _p(name, pos, pid, avg=0.0, total=0.0, week=None, owned=10.0, team="GB", status="ACTIVE"):
    stats = {} if week is None else {7: {"projected_points": week}}
    return SimpleNamespace(name=name, position=pos, playerId=pid, proTeam=team,
                           injuryStatus=status, projected_avg_points=avg,
                           projected_total_points=total, percent_owned=owned, stats=stats)


def test_a_free_agent_row_carries_the_roster_vocabulary_and_the_weekly_projection():
    got = E.free_agent_rows([_p("A Runner", "RB", 1, 9.5, 133.0, week=10.1, owned=41.2)], week=7)
    row = got.row(0, named=True)
    assert dict(got.schema) == E.FREE_AGENT_SCHEMA
    assert (row["player"], row["key"], row["pos"], row["espn_id"]) == (
        "A Runner", "a runner", "RB", 1)
    assert (row["espn_avg"], row["espn_total"], row["espn_week"], row["espn_week_proj"]) == (
        9.5, 133.0, 7, 10.1)
    assert row["percent_owned"] == 41.2
    assert "slot" not in got.columns


def test_an_unprojected_player_is_zero_season_and_null_week_never_a_fake_number():
    row = E.free_agent_rows([_p("B C", "TE", 2)], week=7).row(0, named=True)
    assert row["espn_avg"] == 0.0 and row["espn_total"] == 0.0 and row["espn_week_proj"] is None


def test_no_ownership_figure_is_espns_minus_one_not_zero():
    row = E.free_agent_rows([_p("D E", "K", 3, owned=None)], week=None).row(0, named=True)
    assert row["percent_owned"] == -1.0 and row["espn_week"] is None


def test_the_parsed_pool_meets_its_contract_and_a_duplicate_does_not():
    pool = E.free_agent_rows([_p("A B", "RB", 1, 5.0, 70.0), _p("C D", "WR", 2, 6.0, 84.0)], 7)
    assert ESPN_FREE_AGENTS.validate(pool).height == 2
    with pytest.raises(ContractViolation):
        ESPN_FREE_AGENTS.validate(pl.concat([pool, pool.head(1)]))


def test_an_empty_pool_is_refused_by_the_contract():
    with pytest.raises(ContractViolation):
        ESPN_FREE_AGENTS.validate(E.free_agent_rows([], 7))
