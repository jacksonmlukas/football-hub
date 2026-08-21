"""Written before board.py's replacement_levels existed. This is the TDD anchor:
the full-PPR flex effect is a claim about the world, so it gets a test.
"""
import polars as pl
from hub.draft.board import replacement_levels


def _pool(n_per_pos=60):
    rows = []
    for pos, base in (("QB", 22.0), ("RB", 18.0), ("WR", 18.0), ("TE", 12.0)):
        for i in range(n_per_pos):
            rows.append({"position": pos, "xfp_per_game": base - i * 0.2})
    return pl.DataFrame(rows)


def test_replacement_deepens_with_league_size():
    small = replacement_levels(_pool(), teams=8)
    large = replacement_levels(_pool(), teams=14)
    for pos in ("RB", "WR", "TE"):
        assert large[pos] < small[pos], f"{pos} replacement must fall as leagues grow"


def test_flex_pushes_wr_replacement_deeper_than_rb_in_ppr():
    """Given identical talent curves, full-PPR flex allocation is WR-heavy, so WR
    replacement should sit deeper into its pool than RB does."""
    lv = replacement_levels(_pool(), teams=12)
    assert lv["WR"] < lv["RB"]


def test_qb_unaffected_by_flex():
    lv12 = replacement_levels(_pool(), teams=12)
    assert abs(lv12["QB"] - (22.0 - (12 - 1) * 0.2)) < 1e-6


def test_three_wr_league_pushes_wr_replacement_deep():
    """3 WR starters + flex in a 12-team league means ~43 startable WRs, so replacement
    sits far deeper into the pool than the 2WR default would put it."""
    from hub.draft.board import SLOTS
    assert SLOTS["WR"] == 3, "this league starts three WRs"
    lv = replacement_levels(_pool(n_per_pos=60), teams=12)
    # WR replacement should be at least 10 slots deeper than RB replacement given
    # identical talent curves (36+flex WR vs 24+flex RB).
    assert lv["RB"] - lv["WR"] > 1.5
