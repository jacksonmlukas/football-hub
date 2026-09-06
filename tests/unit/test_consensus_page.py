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
    import hub.fetch.nflverse as nv
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    # One stub, not two. The live path used to read `nflreadpy` directly and needed its own
    # double; since #36 both paths reach the archive through `load_rankings`, so stubbing the
    # fetch layer's one entry point covers them -- and a test that has to double two seams to
    # cover one source is a test asserting that the two agree, which nothing guaranteed.
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: frame)


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


def test_the_live_path_reads_the_small_table_and_never_a_cached_one(monkeypatch, tmp_path):
    """No `as_of` must still read the small `draft` table, not the 1.8M-row archive -- and it
    must reach the wire every time.

    This used to assert the opposite of the first half: that the live path did *not* go
    through the loader. The reason given was right and is why the change had to keep it --
    `load` serves whatever is already in the cache, and a live board printing yesterday's ECR
    on draft night without saying so is the failure that matters. Going direct bought that at
    the price of the pin, so the live board could not say what it had read.

    `refresh=True` buys both: the loader fetches every time and still writes the pin. So the
    property asserted here is now the one that was actually wanted -- never cached -- rather
    than the mechanism that happened to deliver it (#36).
    """
    import hub.fetch.nflverse as nv
    from hub.draft.board import consensus
    frame = _dated(("Guy", 3.0, "2022-08-20"))
    routed = []
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: routed.append(pages) or frame)

    consensus()
    assert routed == [["draft"]], "the live path reads the small table through the loader"
    consensus()
    assert routed == [["draft"], ["draft"]], (
        "the second live board was served from cache, so draft night can print yesterday's "
        "ECR without saying so")

    consensus(as_of="2022-09-01")
    assert routed[-1] == ["all"], "the dated path is the loader's, keyed by page type"


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
    with pytest.raises(ContractViolation, match="scraped between 2022-07-01 and 2022-09-01"):
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


def test_the_replay_as_of_selects_what_the_strict_comparison_selected(monkeypatch, tmp_path):
    """The replay date held by the rows it selects, not by the string in the source.

    `board_as_of` asks for August 31 because the as-of filter is inclusive, and inclusive of
    August 31 is what *strictly before* September 1 used to mean. That equivalence is why no
    published number moved when the convention changed, so it is the thing worth asserting --
    and the previous version of this test asserted it by reading the function's own source for
    the literal `-08-31`. That passes for a function rewritten to do something else entirely
    so long as the string survives, and fails for a correct function that spells the date
    differently.

    Here the archive carries a scrape on the boundary day and another the day after, with
    different values, so the two rules are distinguishable: asking for the wrong day picks up
    the September scrape and this fails.

    **The equivalence is a property of plain-date scrape dates.** Inclusive-of-August-31 and
    strictly-before-September-1 coincide only because no scrape carries a time. A move to
    timestamps has to revisit the convention itself, not merely this test.
    """
    import hub.draft.board as board_mod
    from hub.draft.board import consensus

    yr = 2023
    frame = _dated(("Guy", 50.0, f"{yr}-08-30"),
                   ("Guy", 12.0, f"{yr}-08-31"),      # the boundary day itself
                   ("Guy", 99.0, f"{yr}-09-01"))      # the day the strict rule excluded
    _patch(monkeypatch, tmp_path, frame)

    asked: dict[str, object] = {}
    monkeypatch.setattr(board_mod, "build",
                        lambda **kw: asked.update(kw) or (pl.DataFrame(), None))
    board_mod.board_as_of(yr)

    got = consensus(as_of=str(asked["as_of"]))
    assert got["ecr"][0] == 12.0, (
        f"the replay asked for {asked['as_of']} and selected ecr {got['ecr'][0]}. The strict "
        f"comparison it replaced took the last scrape *before* {yr}-09-01, which is the "
        f"{yr}-08-31 one at 12.0; reading 99.0 means the day moved forward and the replay now "
        f"sees a scrape the original never did.")

    strict = frame.filter(pl.col("scrape_date") < f"{yr}-09-01").sort("scrape_date")
    assert got["ecr"][0] == strict["ecr"][-1], (
        "the inclusive as-of and the strict comparison no longer select the same scrape")


def test_the_boundary_day_is_inside_the_replay_as_of(monkeypatch, tmp_path):
    """The case that makes the equivalence fragile, pinned on its own.

    Were the boundary day excluded rather than included, the two rules would still agree on
    any archive with no scrape on August 31 -- which is most of them. Only 2023 has one in the
    real archive, which is why this needs a fixture rather than whichever season the suite
    happens to exercise.
    """
    from hub.draft.board import consensus

    yr = 2023
    _patch(monkeypatch, tmp_path, _dated(("Guy", 50.0, f"{yr}-08-29"),
                                         ("Guy", 12.0, f"{yr}-08-31")))
    assert consensus(as_of=f"{yr}-08-31")["ecr"][0] == 12.0, (
        "a scrape landing exactly on the as-of was dropped; the filter is documented as "
        "inclusive and the replay date depends on it being so")


# --- the window is bounded below, not only above (issue #38) ---------------

def test_a_player_ranked_only_in_an_earlier_preseason_is_off_the_board(monkeypatch, tmp_path):
    """The defect. `consensus` took the latest scrape per player at or before its as-of and
    had no lower bound, so a player ranked once in 2022 and never again sat mid-pool on the
    2024 board -- draftable, carrying an ECR from a season he did not play."""
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Gone", 40.0, "2022-08-10"),
                                         ("Current", 12.0, "2024-08-20")))
    got = consensus(as_of="2024-08-31")
    assert got["player"].to_list() == ["Current"]


def test_a_player_ranked_in_two_preseasons_is_ranked_on_the_current_one(monkeypatch, tmp_path):
    """The bound must not reach back for an older rank when a newer one exists -- that is the
    same defect wearing a window."""
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Both", 40.0, "2022-08-10"),
                                         ("Both", 11.0, "2024-08-20")))
    got = consensus(as_of="2024-08-31")
    assert got.height == 1 and got["ecr"][0] == 11.0


def test_the_bound_follows_a_mid_season_as_of(monkeypatch, tmp_path):
    """Derived from the as-of it is given. A November as-of belongs to the season that opened
    that July, and a January one to the season that opened the *previous* July -- twelve
    months back instead would put two preseasons in the window."""
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Old", 40.0, "2023-08-10"),
                                         ("New", 12.0, "2024-08-20")))
    assert consensus(as_of="2024-11-15")["player"].to_list() == ["New"]
    assert consensus(as_of="2025-01-20")["player"].to_list() == ["New"]


def test_the_live_path_is_exempt_and_that_is_deliberate(monkeypatch, tmp_path):
    """No `as_of` reads the current `draft` page, which is one scrape of today's board -- it
    has no historical window to bound and nothing older in it to exclude.

    Asserted rather than left implicit: the exemption is the reason the bound cannot simply be
    pushed into the loader for every caller, and a reader meeting `preseason_start` only on
    the dated path should find out here why.
    """
    import hub.fetch.nflverse as nv
    from hub.draft.board import consensus
    frame = _dated(("Only", 3.0, "2019-01-01"))       # far outside any preseason window
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: frame)
    assert consensus()["player"].to_list() == ["Only"], (
        "the live board applied a historical window to today's scrape")


def test_the_dropped_count_is_reported(monkeypatch, tmp_path, capsys):
    """Criterion six. This drops players from every historical board, which re-prices the
    backtest and everything downstream; a height that moves with no line saying so is the
    change nobody connects to its cause."""
    from hub.draft.board import consensus
    _patch(monkeypatch, tmp_path, _dated(("Gone", 40.0, "2022-08-10"),
                                         ("Current", 12.0, "2024-08-20")))
    consensus(as_of="2024-08-31")
    out = capsys.readouterr().out
    assert "1 dropped" in out and "2024-07-01" in out
