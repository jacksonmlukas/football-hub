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


def _patch(monkeypatch, tmp_path, frame):
    """Fake the archive, and point the loader's cache at a tmp tree.

    `consensus(as_of=...)` reads through `hub.fetch.nflverse.load_rankings` now, which writes
    a dated parquet and a pin beside it under `nflverse.RAW`. Redirecting RAW is not tidiness:
    without it these tests write into the developer's own `data/raw`, and the *next* test
    naming the same as-of is served the previous test's frame from cache instead of its own.
    That is not a hypothetical -- it is what the first run of this file did.
    """
    import hub.draft.board as board_mod
    import hub.fetch.nflverse as nv
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: frame)
    monkeypatch.setattr(board_mod.nfl, "load_ff_rankings", lambda which: frame)


def test_as_of_takes_the_latest_scrape_before_the_date(monkeypatch, tmp_path):
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Guy", 30.0, "2022-07-01"),
                                         ("Guy", 12.0, "2022-08-28"),
                                         ("Guy", 99.0, "2026-08-20")))
    got = consensus(as_of="2022-09-01")
    assert got.height == 1
    assert got["ecr"][0] == 12.0, "must be the last scrape before the draft, not the first"


def test_as_of_excludes_anything_scraped_after(monkeypatch, tmp_path):
    """The hindsight guard, stated directly."""
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Old", 5.0, "2022-08-01"),
                                         ("Future", 1.0, "2026-08-20")))
    assert consensus(as_of="2022-09-01")["player"].to_list() == ["Old"]


def test_the_live_path_is_untouched(monkeypatch, tmp_path):
    """No `as_of` must still read the small `draft` table, not the 1.8M-row archive.

    And it must read it *directly*: `load` serves whatever is already in the cache, so a live
    board routed through it would print yesterday's ECR on draft night without saying so.
    """
    import hub.draft.board as board_mod
    import hub.fetch.nflverse as nv
    from hub.draft.board import consensus
    frame = _dated(("Guy", 3.0, "2022-08-20"))
    seen, routed = [], []
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: routed.append(pages) or frame)
    monkeypatch.setattr(board_mod.nfl, "load_ff_rankings",
                        lambda which: seen.append(which) or frame)
    consensus()
    assert (seen, routed) == (["draft"], []), "the live path does not go through the loader"
    consensus(as_of="2022-09-01")
    assert routed == [["all"]], "the dated path is the loader's, keyed by page type"


def test_a_date_before_the_archive_starts_is_a_contract_violation(monkeypatch, tmp_path):
    """The archive begins 2020-10-16. Returning an empty board would be worse than raising:
    every downstream stage would degrade quietly and produce a plausible empty result.

    Which refusal fires moved with the routing and both are asserted here. An as-of before the
    archive leaves the *loader* holding nothing, and `FF_RANKINGS.min_rows` refuses there --
    earlier than before, on the same evidence, and naming the source rather than the page.
    """
    from hub.contracts import ContractViolation
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Guy", 3.0, "2022-08-01")))
    with pytest.raises(ContractViolation, match="nflverse_ff_rankings: 0 rows"):
        consensus(as_of="2019-09-01")


def test_an_archive_with_no_consensus_page_before_the_date_still_names_the_page(monkeypatch,
                                                                               tmp_path):
    """The board's own refusal, on the case only the board can see.

    Rows survive the loader and none of them is this league's page, so `_select_consensus`
    would be handed an empty frame. The message has to say the page and the date, because
    "FantasyPros renamed the page" and "the as-of is too early" are different problems.
    """
    from hub.contracts import ContractViolation
    from hub.draft.board import consensus
    early = _dated(("Guy", 3.0, "2022-08-01"))
    _patch(monkeypatch, tmp_path,
           early.with_columns(pl.lit("dynasty-overall").alias("page_type")))
    with pytest.raises(ContractViolation, match="scraped on or before 2022-09-01"):
        consensus(as_of="2022-09-01")


# --- the as-of boundary, which is one convention and used to be two -------------
#
# `consensus` filtered `scrape_date < as_of` -- strictly before -- while `hub.fetch.nflverse`
# filtered `<= as_of` and documented itself as inclusive. Nothing was wrong while the call site
# had not moved; the day it moved, a board built "as of" some date would silently gain the rows
# scraped that day and every number downstream of it would move with it. The convention that
# survived is the loader's, because `hub.store.lines_as_of` already means "at or before" and
# because the loader is what pins and caches on the date.
#
# These are the tests that go red if the two ever drift apart again.

def test_the_boundary_day_is_inside_the_as_of_on_both_sides(monkeypatch, tmp_path):
    """One frame, one date, both readers: the scrape *on* the as-of counts, in both places."""
    import hub.fetch.nflverse as nv
    from hub.draft.board import consensus
    frame = _dated(("Guy", 30.0, "2022-08-30"),
                   ("Guy", 12.0, "2022-09-01"),      # the boundary day itself
                   ("Guy", 99.0, "2022-09-02"))
    _patch(monkeypatch, tmp_path, frame)

    served = nv.load_rankings("all", as_of="2022-09-01")
    assert sorted(served["scrape_date"].to_list()) == ["2022-08-30", "2022-09-01"], \
        "the loader is inclusive of its as-of day and exclusive of the one after"

    got = consensus(as_of="2022-09-01")
    assert got["ecr"][0] == 12.0, \
        ("the board's consensus read must take the boundary day too. If this fails and the "
         "loader's assertion above passes, the two conventions have drifted apart again -- "
         "which is a silent re-pricing of every historical board, not a test detail.")


def test_the_board_keeps_no_date_comparison_of_its_own(monkeypatch):
    """The structural half of the same guarantee.

    Two implementations of one convention agree right up until somebody edits one of them, so
    `consensus` keeps none: it hands the as-of to the loader and filters on no date itself.
    Handed a frame the loader should have bounded and did not, it returns those rows -- which
    looks like the hindsight bug and is the opposite of it. This test fails the moment a
    second comparison is reintroduced here, whichever direction it points.
    """
    from hub.draft import board as board_mod
    frame = _dated(("Guy", 30.0, "2022-08-01"), ("Guy", 1.0, "2030-01-01"))
    monkeypatch.setattr(board_mod, "load_rankings",
                        lambda page, as_of=None, cols=None: frame)
    assert board_mod.consensus(as_of="2022-09-01")["ecr"][0] == 1.0, \
        ("consensus() bounded the archive itself. The as-of boundary belongs to "
         "hub.fetch.nflverse.load_rankings and to nothing else -- two copies of one "
         "convention is exactly what this migration removed.")


def test_board_as_of_asks_for_the_cutoff_that_holds_the_replay_still():
    """August 31 inclusive is the September 1 exclusive the strict comparison meant.

    Measured on the live archive before the switch: `< {yr}-09-01` and `<= {yr}-08-31` return
    the same rows for every season 2021-26, and only 2023 has any `redraft-overall` scrape on
    September 1 at all -- reading that day in would have moved 463 ECRs on the 2023 board.
    """
    import inspect

    from hub.draft import board as board_mod
    src = inspect.getsource(board_mod.board_as_of)
    assert "-08-31" in src and "-09-01" not in src.split('"""')[2], \
        "the replay cutoff moved back to the day the strict comparison actually selected"
