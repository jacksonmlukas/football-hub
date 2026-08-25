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


def test_the_consensus_spread_is_named_for_what_it_measures():
    """FantasyPros calls it `sd`, which is also what a weekly *points* spread is called.
    Renamed at the producer so the two never share a column name on the same frame."""
    df = _select_consensus(_rows(("Guy", "RB", 5.0, "redraft-overall")))
    assert "ecr_sd" in df.columns
    assert "sd" not in df.columns


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


# --- the board as it stood on a past date ---------------------------------
#
# `consensus(as_of=...)` exists so a past draft can be replayed against rankings the room
# could actually have seen. Scoring a 2022 draft on 2026 rankings would be hindsight, and
# hindsight is the failure the whole realised-outcome backtest exists to escape.

def _dated(*specs):
    """(player, ecr, scrape_date) rows on the consensus page."""
    return pl.DataFrame(
        [{"player": p, "pos": "RB", "team": "X", "ecr": e, "sd": 1.0, "best": 1.0,
          "worst": 9.0, "page_type": "redraft-overall", "scrape_date": d}
         for p, e, d in specs])


def _patch(monkeypatch, frame):
    import hub.draft.board as board_mod
    monkeypatch.setattr(board_mod.nfl, "load_ff_rankings", lambda which: frame)


def test_as_of_takes_the_latest_scrape_before_the_date(monkeypatch):
    from hub.draft.board import consensus
    _patch(monkeypatch, _dated(("Guy", 30.0, "2022-07-01"),
                               ("Guy", 12.0, "2022-08-28"),
                               ("Guy", 99.0, "2026-08-20")))
    got = consensus(as_of="2022-09-01")
    assert got.height == 1
    assert got["ecr"][0] == 12.0, "must be the last scrape before the draft, not the first"


def test_as_of_excludes_anything_scraped_after(monkeypatch):
    """The hindsight guard, stated directly."""
    from hub.draft.board import consensus
    _patch(monkeypatch, _dated(("Old", 5.0, "2022-08-01"),
                               ("Future", 1.0, "2026-08-20")))
    assert consensus(as_of="2022-09-01")["player"].to_list() == ["Old"]


def test_the_live_path_is_untouched(monkeypatch):
    """No `as_of` must still read the small `draft` table, not the 1.8M-row archive."""
    import hub.draft.board as board_mod
    from hub.draft.board import consensus
    seen = []

    def _spy(which):
        seen.append(which)
        return _dated(("Guy", 3.0, "2022-08-20"))

    monkeypatch.setattr(board_mod.nfl, "load_ff_rankings", _spy)
    consensus()
    assert seen == ["draft"]
    consensus(as_of="2022-09-01")
    assert seen == ["draft", "all"]


def test_a_date_before_the_archive_starts_is_a_contract_violation(monkeypatch):
    """The archive begins 2020-10-16. Returning an empty board would be worse than raising:
    every downstream stage would degrade quietly and produce a plausible empty result."""
    from hub.contracts import ContractViolation
    from hub.draft.board import consensus
    _patch(monkeypatch, _dated(("Guy", 3.0, "2022-08-01")))
    with pytest.raises(ContractViolation, match="scraped before"):
        consensus(as_of="2019-09-01")
