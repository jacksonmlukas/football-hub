"""Written before board.py's replacement_levels existed. This is the TDD anchor:
the full-PPR flex effect is a claim about the world, so it gets a test.
"""
import polars as pl
import pytest

from hub.draft.board import (
    FLEX_ELIGIBLE,
    FLEX_SHARES,
    REPLACEMENT_POPULATIONS,
    SLOTS,
    replacement_levels,
)

# These fixtures are a synthetic full pool, so they are asking the prior-season question.
# Naming it is the whole of #184's mechanism: a call that does not say which population it is
# over is making a modelling claim nobody wrote down, and a test is not exempt from that.
POP = "prior_season_pool"


def _pool(n_per_pos=60):
    """The three Series `replacement_levels` takes, ready to splat.

    It used to take a frame, and the frame's *column names* were the real interface -- so
    every caller, tests included, had to know that `position`/`xfp_per_game`/`games` were
    the magic words. Now the names are the caller's business.
    """
    rows = []
    for pos, base in (("QB", 22.0), ("RB", 18.0), ("WR", 18.0), ("TE", 12.0)):
        for i in range(n_per_pos):
            # `games` gained meaning when replacement_levels started requiring a real
            # denominator: a one-game sample is not a per-game rate. Full seasons here,
            # so every assertion below is unchanged by that filter.
            rows.append({"position": pos, "xfp_per_game": base - i * 0.2, "games": 16})
    df = pl.DataFrame(rows)
    return df["position"], df["xfp_per_game"], df["games"]


def test_replacement_deepens_with_league_size():
    small = replacement_levels(*_pool(), teams=8, population=POP)
    large = replacement_levels(*_pool(), teams=14, population=POP)
    for pos in ("RB", "WR", "TE"):
        assert large[pos] < small[pos], f"{pos} replacement must fall as leagues grow"


def test_flex_pushes_wr_replacement_deeper_than_rb_in_ppr():
    """Given identical talent curves, full-PPR flex allocation is WR-heavy, so WR
    replacement should sit deeper into its pool than RB does."""
    lv = replacement_levels(*_pool(), teams=12, population=POP)
    assert lv["WR"] < lv["RB"]


def test_qb_unaffected_by_flex():
    lv12 = replacement_levels(*_pool(), teams=12, population=POP)
    assert abs(lv12["QB"] - (22.0 - (12 - 1) * 0.2)) < 1e-6


def test_three_wr_league_pushes_wr_replacement_deep():
    """3 WR starters + flex in a 12-team league means ~43 startable WRs, so replacement
    sits far deeper into the pool than the 2WR default would put it."""
    from hub.draft.board import SLOTS
    assert SLOTS["WR"] == 3, "this league starts three WRs"
    lv = replacement_levels(*_pool(n_per_pos=60), teams=12, population=POP)
    # WR replacement should be at least 10 slots deeper than RB replacement given
    # identical talent curves (36+flex WR vs 24+flex RB).
    assert lv["RB"] - lv["WR"] > 1.5


# --- the flex shares: a declared assumption, with its sensitivity (#184) ----
#
# `test_flex_shares_sum_to_one` came here from `test_config.py` when the constant did. The
# tests below it are new, and they are the "sensitivity published" half of #184's second
# acceptance criterion -- the half that can be established without a lineup archive. The
# shares reach the board through one expression, `round(teams * FLEX * share)`, so they are
# a step function, and the useful statement about them is where the steps are.

def test_flex_shares_sum_to_one():
    assert abs(sum(FLEX_SHARES.values()) - 1.0) < 1e-9
    assert set(FLEX_SHARES) == set(FLEX_ELIGIBLE)


def _flex_slots(share: float, teams: int = 12) -> int:
    """The only thing a share does: how many extra players it makes startable."""
    return round(teams * SLOTS["FLEX"] * share)


def test_the_published_flex_slot_counts_are_what_the_shares_produce():
    """The head of the table beside `FLEX_SHARES`: 5 RB, 6 WR, 1 TE in this league."""
    assert (_flex_slots(FLEX_SHARES["RB"]), _flex_slots(FLEX_SHARES["WR"]),
            _flex_slots(FLEX_SHARES["TE"])) == (5, 6, 1)


def test_each_flex_share_is_inert_over_the_published_window():
    """The first half of the sensitivity, and the reassuring half.

    A share is not a dial. Restating one anywhere inside its window changes no number
    anywhere in this repo, so most of what a usage study could plausibly return would leave
    the board byte-identical. The windows are the ones written beside `FLEX_SHARES`.
    """
    windows = {"RB": (-0.074, +0.008), "WR": (-0.041, +0.041), "TE": (-0.008, +0.074)}
    for pos, (lo, hi) in windows.items():
        base = _flex_slots(FLEX_SHARES[pos])
        for delta in (lo, 0.0, hi):
            assert _flex_slots(FLEX_SHARES[pos] + delta) == base, (
                f"{pos} at {FLEX_SHARES[pos] + delta} adds "
                f"{_flex_slots(FLEX_SHARES[pos] + delta)} flex slots, not {base}. The window "
                f"published beside FLEX_SHARES is wrong, so the sensitivity is wrong.")
        # And the other side of it, or "inert" would be satisfied by a share that moved
        # nothing anywhere -- the window is a claim about where the step is, so the step has
        # to be shown. A thousandth further out in each direction flips the count.
        for delta in (lo - 0.002, hi + 0.001):
            assert _flex_slots(FLEX_SHARES[pos] + delta) != base, (
                f"{pos} at {FLEX_SHARES[pos] + delta} still adds {base} flex slots, so the "
                f"published window is narrower than the real one and understates the "
                f"latitude these shares have.")


def test_rb_is_the_share_that_sits_next_to_a_boundary():
    """The second half, and the one worth knowing.

    RB is 0.008 below a step and WR is 0.042 from one in either direction, so the hand-setting
    error that matters is not the size of the RB share but which side of 0.4583 it falls. A
    usage study returning RB 0.46 adds a sixth RB flex slot; one returning WR 0.52 changes
    nothing. Asserted rather than described, because the whole finding is a distance, and a
    described distance is what nobody rechecks.
    """
    import hub.draft.board as board_mod

    assert _flex_slots(0.46) == _flex_slots(FLEX_SHARES["RB"]) + 1
    assert _flex_slots(0.52) == _flex_slots(FLEX_SHARES["WR"])
    # And the step is exactly one player deeper into the pool, which is what makes the
    # published window the whole of the sensitivity rather than a summary of it.
    pool = _pool()
    base = replacement_levels(*pool, teams=12, population=POP)["RB"]
    orig = board_mod.FLEX_SHARES
    try:
        board_mod.FLEX_SHARES = dict(FLEX_SHARES, RB=0.46)
        moved = replacement_levels(*pool, teams=12, population=POP)["RB"]
    finally:
        board_mod.FLEX_SHARES = orig
    # The fixture's RB curve falls 0.2 a player, so one extra startable back is one step.
    assert abs((base - moved) - 0.2) < 1e-9


# --- which population, declared (#184) -------------------------------------

def test_a_replacement_level_must_name_its_population():
    """The mechanism, and the reason it is an argument rather than a comment.

    Three call sites took levels over three populations and two value columns, and none had
    to say which -- so nothing distinguished a deliberate second definition from a third
    nobody had noticed. There is no default, because a default is exactly what a fourth site
    would have taken.
    """
    with pytest.raises(ValueError, match="undeclared population"):
        replacement_levels(*_pool(), teams=12, population="whatever_is_lying_around")


def test_every_declared_population_says_what_question_it_answers():
    """A registry of bare names would be the same defect with a longer spelling."""
    assert set(REPLACEMENT_POPULATIONS) == {"in_draft", "published_board",
                                            "prior_season_pool"}
    for name, says in REPLACEMENT_POPULATIONS.items():
        assert len(says) > 40, f"{name} names itself and answers nothing"


def test_the_declaration_does_not_change_the_answer():
    """It is a declaration, and a declaration that steered the arithmetic would be a fourth
    definition hiding inside the repair for three."""
    pool = _pool()
    got = {p: replacement_levels(*pool, teams=12, population=p)
           for p in REPLACEMENT_POPULATIONS}
    first = got["in_draft"]
    assert all(lv == first for lv in got.values())
