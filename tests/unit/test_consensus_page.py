"""Which FantasyPros page the consensus rankings come from.

`load_ff_rankings("draft")` is 31 ranking pages stacked in one frame -- redraft,
dynasty, best-ball, superflex, and IDP -- each carrying its own independent `ecr`
scale. The original consensus() selected `page_type` away and then took
`.unique(keep="first")`, so each player's ECR came from whichever page happened to
sort first: 27 distinct pages, the largest contributors being best-ball overall
and IDP linebackers.

The visible symptom was kickers topping the edge list on positional ECRs of 11-30.
This league is 12-team redraft full PPR, no superflex, no IDP, and does not draft
K or DST off this board.
"""
import polars as pl
import pytest
from hub.contracts import ContractViolation
from hub.draft.board import _select_consensus


def _rows(*specs):
    return pl.DataFrame(
        [{"player": p, "pos": pos, "team": "X", "ecr": e, "sd": 1.0,
          "best": 1.0, "worst": 9.0, "page_type": page} for p, pos, e, page in specs])


def test_only_the_redraft_ppr_page_survives():
    df = _select_consensus(_rows(
        ("Redraft Guy", "RB", 5.0, "redraft-overall"),
        ("Dynasty Guy", "RB", 3.0, "dynasty-overall"),
        ("Bestball Guy", "WR", 2.0, "best-overall"),
        ("Superflex Guy", "QB", 1.0, "redraft-op"),
        ("IDP Guy", "LB", 4.0, "redraft-lb"),
    ))
    assert df["player"].to_list() == ["Redraft Guy"]


def test_kickers_and_dst_are_dropped():
    df = _select_consensus(_rows(
        ("Skill Guy", "WR", 10.0, "redraft-overall"),
        ("Will Reichard", "K", 186.0, "redraft-overall"),
        ("Some Defense", "DST", 190.0, "redraft-overall"),
    ))
    assert df["player"].to_list() == ["Skill Guy"]


def test_all_four_skill_positions_are_kept():
    df = _select_consensus(_rows(
        ("A", "QB", 4.0, "redraft-overall"), ("B", "RB", 1.0, "redraft-overall"),
        ("C", "WR", 2.0, "redraft-overall"), ("D", "TE", 3.0, "redraft-overall"),
    ))
    assert sorted(df["pos"].to_list()) == ["QB", "RB", "TE", "WR"]


def test_result_is_sorted_by_ecr():
    df = _select_consensus(_rows(
        ("Third", "WR", 30.0, "redraft-overall"), ("First", "RB", 1.0, "redraft-overall"),
        ("Second", "TE", 12.0, "redraft-overall"),
    ))
    assert df["player"].to_list() == ["First", "Second", "Third"]


def test_null_ecr_is_dropped():
    df = _select_consensus(_rows(
        ("Ranked", "WR", 5.0, "redraft-overall"), ("Unranked", "WR", None, "redraft-overall"),
    ))
    assert df["player"].to_list() == ["Ranked"]


def test_one_row_per_player():
    df = _select_consensus(_rows(
        ("Dup", "WR", 5.0, "redraft-overall"), ("Dup", "WR", 6.0, "redraft-overall"),
    ))
    assert df.height == 1


def test_missing_page_raises_rather_than_silently_returning_a_mongrel():
    """Schema drift here is exactly the failure this whole change exists to stop."""
    with pytest.raises(ContractViolation):
        _select_consensus(_rows(("Only Dynasty", "RB", 1.0, "dynasty-overall")))


def test_absent_page_type_column_raises():
    df = pl.DataFrame({"player": ["A"], "pos": ["RB"], "ecr": [1.0]})
    with pytest.raises(ContractViolation):
        _select_consensus(df)
