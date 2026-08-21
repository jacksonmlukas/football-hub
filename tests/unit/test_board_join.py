"""Joining the consensus board to expected points.

An exact string join drops every player whose two sources disagree about a suffix.
Observed Aug 2026: 20 of the top 168 by ADP had no xFP, including James Cook III
(ADP 12.8) and Michael Pittman Jr. -- not rookies, just names.

They then carry null VOR, which is worse than it sounds: anything ranking on VOR skips
them, and the season simulation scores them at zero, so a first-round RB reads as an
empty roster spot.
"""
import polars as pl
from hub.draft.board import _join_expected_points


def _ecr(names):
    return pl.DataFrame({"player": names, "ecr": [float(i + 1) for i in range(len(names))]})


def _xp(names):
    return pl.DataFrame({"full_name": names, "position": ["RB"] * len(names),
                         "xfp_per_game": [10.0] * len(names), "games": [16] * len(names),
                         "vor": [1.0] * len(names)})


def test_suffix_mismatch_still_joins():
    out = _join_expected_points(_ecr(["James Cook III"]), _xp(["James Cook"]))
    assert out["xfp_per_game"][0] == 10.0


def test_punctuation_mismatch_still_joins():
    out = _join_expected_points(_ecr(["Ja'Marr Chase"]), _xp(["JaMarr Chase"]))
    assert out["xfp_per_game"][0] == 10.0


def test_genuinely_absent_player_stays_null():
    out = _join_expected_points(_ecr(["Nobody At All"]), _xp(["Someone Else"]))
    assert out["xfp_per_game"][0] is None


def test_row_count_is_preserved():
    """A left join that duplicates rows would corrupt every rank on the board."""
    out = _join_expected_points(_ecr([f"P{i}" for i in range(20)]), _xp([f"P{i}" for i in range(20)]))
    assert out.height == 20


def test_distinct_players_do_not_cross_match():
    out = _join_expected_points(_ecr(["Josh Allen"]), _xp(["Keenan Allen"]))
    assert out["xfp_per_game"][0] is None


def test_helper_column_does_not_leak():
    out = _join_expected_points(_ecr(["A"]), _xp(["A"]))
    assert not [c for c in out.columns if c.startswith("_")]
