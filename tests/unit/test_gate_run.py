"""One gate run: the sequence around the rule, written once (issue #135).

ADR-0019 unified the three verdict branches into `experiment.gate`. The sequence *around* it --
summarise, break out by season, take the verdict, render, stamp -- stayed written out in three
entry points, which is why "void the draft backtest" (#46), "stamp the lineup gate" (#133) and
"reach the ceiling from the season gates" (#134) each existed as a ticket to do in one gate what
another already did. `experiment.run_gate` is that sequence, once.

All offline. The frames here are synthetic; nothing reaches a board or the network.
"""
import inspect
import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hub.models import experiment
from hub.models.experiment import SEASON_CLUSTER, Actions, Ceiling, run_gate

_ACTIONS = Actions(adopt="ADOPT: ship it.", remove="REMOVE: delete it.",
                   show="SHOW: beside the incumbent.")


def _paired(seed=0, seasons=4, per=6, shift=0.0):
    """One row per (season, draft) with a real season effect, like the draft gate's frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2022, 2022 + seasons):
        level = rng.normal(0.0, 2.0)
        for k in range(per):
            rows.append({"season": s, "draft": k,
                         "diff": float(level + rng.normal(0.0, 1.0) + shift)})
    return pl.DataFrame(rows)


def _run(paired, **kw: Any):
    """The shared call with the state file pointed nowhere, so a test writes no history."""
    base: dict[str, Any] = {"cluster": SEASON_CLUSTER, "actions": _ACTIONS, "name": "test",
                            "arm_a": "a", "arm_b": "b", "bootstrap": 200,
                            "record_width": False}
    return run_gate(paired, **(base | kw))


# --- the shape: no default for the one argument that must not have one --------------------

def test_the_cluster_unit_has_no_default():
    """`summarise`'s docstring says there is no safe default, and a shared run that guessed
    one would be the most expensive mistake in this repo's record by its own account. So the
    parameter exists, is keyword-only, and has no default -- a caller that omits it does not
    get the row silently, it gets a TypeError."""
    p = inspect.signature(run_gate).parameters["cluster"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        run_gate(_paired(), actions=_ACTIONS, name="x", arm_a="a", arm_b="b")  # type: ignore[call-arg]


def test_the_run_returns_the_five_things_the_entry_points_used_to_assemble():
    got = _run(_paired())
    assert set(got._fields) == {"summary", "seasons", "verdict", "lines", "stamped"}
    assert isinstance(got.summary, dict) and isinstance(got.seasons, pl.DataFrame)
    assert isinstance(got.verdict, tuple) and len(got.verdict) == 2
    assert isinstance(got.lines, list) and all(isinstance(ln, str) for ln in got.lines)
    assert isinstance(got.stamped, pl.DataFrame)


# --- nothing moves: the old path and the new, on the same frame -------------------------
#
# The acceptance criterion. Each gate's `main` used to spell the sequence by hand, and the
# three spellings are reproduced here exactly as they stood at `e14ab56` -- including the
# weekly gate's, which handed its ceiling to `ceiling_report` and never to `summarise`. The
# summary, the season table and the verdict must be the same objects the old sequence made.

def _old_draft(paired, bound, seed):
    s = experiment.summarise(paired, cluster=SEASON_CLUSTER, seed=seed, bootstrap=200)
    if bound is not None:
        top = float(np.asarray(bound["diff"].to_numpy()).mean())
        s = dict(s, ceiling=top)
    seasons = experiment.per_season(paired)
    return s, seasons, experiment.gate(s, seasons, _ACTIONS)


def _old_lineup(paired, seed):
    top = (float(np.asarray(paired["ceiling_diff"].to_numpy()).mean())
           if "ceiling_diff" in paired.columns else None)
    s = experiment.summarise(paired, cluster=SEASON_CLUSTER, seed=seed, bootstrap=200,
                             ceiling=top)
    seasons = experiment.per_season(paired)
    return s, seasons, experiment.gate(s, seasons, _ACTIONS)


def _old_weekly(paired, seed, void):
    s = experiment.summarise(paired, cluster=SEASON_CLUSTER, seed=seed, bootstrap=200)
    seasons = experiment.per_season(paired)
    return s, seasons, experiment.gate(s, seasons, _ACTIONS, void=void)


def _same(old, new):
    s_old, seasons_old, verdict_old = old
    assert new.summary == s_old, "the summary moved"
    assert new.seasons.equals(seasons_old), "the season table moved"
    assert new.verdict == verdict_old, "the verdict moved"


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("shift", [0.0, -6.0, 6.0])
def test_the_draft_gate_s_numbers_are_the_old_path_s(seed, shift):
    paired = _paired(seed=seed, shift=shift)
    bound = paired.with_columns((pl.col("diff") + 9.0).alias("diff"))
    _same(_old_draft(paired, None, seed), _run(paired, seed=seed))
    _same(_old_draft(paired, bound, seed),
          _run(paired, seed=seed, ceiling=Ceiling("foresight", bound["diff"])))


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("shift", [0.0, -6.0, 6.0])
def test_the_lineup_gate_s_numbers_are_the_old_path_s(seed, shift):
    paired = _paired(seed=seed, shift=shift).rename({"draft": "roster"})
    with_c = paired.with_columns((pl.col("diff") + 4.0).alias("ceiling_diff"))
    _same(_old_lineup(paired, seed), _run(paired, seed=seed, unit="points per game"))
    _same(_old_lineup(with_c, seed),
          _run(with_c, seed=seed, unit="points per game",
               ceiling=Ceiling("a perfect spread", with_c["ceiling_diff"])))


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("void", [None, "VOID: 6.2% of roster-weeks are a join failure"])
def test_the_weekly_gate_s_numbers_are_the_old_path_s(seed, void):
    paired = _paired(seed=seed).rename({"draft": "roster"}).with_columns(pl.lit(5).alias("week"))
    _same(_old_weekly(paired, seed, void),
          _run(paired, seed=seed, unit="points per team-week", places=3, show_n=False,
               void=void))


def test_the_weekly_gate_s_ceiling_now_reaches_the_rule_and_that_is_the_one_departure():
    """Recorded rather than hidden. The weekly gate's `main` handed its ceiling to a report
    and never to `summarise`, so `docs/weekly-blend-gate.md` says its NOT-RUNNABLE branch
    "does not fire and cannot". Under one run the ceiling reaches the summary the verdict
    reads, as the lineup gate's already did (#134). No published weekly verdict was reached
    with a ceiling, so none moves; a `--ceiling` run can now be declared unable to run."""
    paired = _paired(seed=1).rename({"draft": "roster"})
    # A ceiling far below this frame's MDE: the rule must now see it.
    c = pl.Series("ceiling_diff", [0.001] * paired.height)
    got = _run(paired, seed=1, unit="points per team-week", places=3, show_n=False,
               ceiling=Ceiling("perfect foresight", c))
    assert experiment.reading(got.summary, "ceiling") is experiment.Field.VALUE
    assert got.summary["mde"] > got.summary["ceiling"]
    assert got.verdict[0] == "NOT-RUNNABLE"


# --- the pieces the run owns ---------------------------------------------------------------

def test_a_void_condition_preempts_every_branch_and_is_the_caller_s_sentence():
    got = _run(_paired(shift=50.0), void="VOID: the join failed at 9.0% against a floor of 2%")
    assert got.verdict == ("VOID", "VOID: the join failed at 9.0% against a floor of 2%")
    assert got.summary["lo"] > 0, "the frame alone would have adopted"


def test_the_ceiling_line_carries_the_arm_s_name_and_its_number():
    """The reason two of three gates bypassed the shared block: it hard-coded 'perfect
    foresight' and the lineup gate's ceiling is a variance oracle. The run carries the name
    beside the number, so one renderer serves all three without lying about two of them."""
    paired = _paired(seed=2)
    c = pl.Series("c", [30.0] * paired.height)
    got = _run(paired, seed=2, ceiling=Ceiling("a perfect spread, not foresight", c))
    joined = "\n".join(got.lines)
    assert "ceiling (a perfect spread, not foresight) +30.00 points per team game" in joined
    assert "perfect foresight)" not in joined
    assert "not comparable with another gate's" in joined
    assert got.summary["ceiling"] == 30.0


def test_no_ceiling_means_no_ceiling_line_and_no_slot():
    got = _run(_paired())
    assert "ceiling" not in "\n".join(got.lines).lower()
    assert experiment.reading(got.summary, "ceiling") is experiment.Field.NO_SLOT


def test_a_ceiling_below_the_effect_says_so_loudly_in_the_gate_s_own_places():
    """`backtest.with_ceiling`, `lineup_gate.ceiling_report` and `weekly_gate.ceiling_report`
    each said this, at two, two and three places. One sentence now, at the caller's."""
    paired = _paired(seed=4, shift=8.0)
    c = pl.Series("c", [1.0] * paired.height)
    got = _run(paired, seed=4, places=3, ceiling=Ceiling("x", c))
    warn = [ln for ln in got.lines if "CEILING BELOW THE EFFECT" in ln]
    assert len(warn) == 1
    assert f"+1.000 < {got.summary['mean']:+.3f}" in warn[0]
    assert "do not read the interval above as bounded" in warn[0]
    bounded = _run(paired, seed=4, ceiling=Ceiling("x", pl.Series("c", [99.0] * paired.height)))
    assert not any("CEILING BELOW" in ln for ln in bounded.lines)


def test_the_report_is_the_block_then_the_small_sample_lines_then_the_width_review(tmp_path):
    paired = _paired(seed=5)
    path = tmp_path / "gate-width.json"
    first = _run(paired, seed=5, record_width=True, width_path=path)
    assert first.lines[0].startswith("\n  n=24  a - b = ")
    assert "95% CI" in first.lines[1] and "MDE at 80% power" in first.lines[2]
    assert any("4 clusters" in ln for ln in first.lines)
    assert not any("interval width" in ln for ln in first.lines), "a first run has no history"
    second = _run(paired, seed=5, record_width=True, width_path=path)
    assert any("interval width" in ln for ln in second.lines), "the second run compares"
    assert path.exists()


def test_the_width_history_is_keyed_by_the_gate_s_name(tmp_path):
    import json
    path = tmp_path / "gate-width.json"
    _run(_paired(), name="draft", record_width=True, width_path=path)
    _run(_paired(), name="lineup", record_width=True, width_path=path)
    assert set(json.loads(path.read_text())) == {"draft", "lineup"}


def test_an_empty_frame_runs_and_says_nothing_was_measured():
    """The weekly gate's `compare` returns a frame with no rows and no columns when nothing
    is covered. The run must still answer rather than fall over on a season column that is
    not there."""
    got = _run(pl.DataFrame())
    assert got.verdict[0] == "SHOW" and "Nothing measured" in got.verdict[1]
    assert got.seasons.is_empty() and got.seasons.columns == ["season", "gain", "n"]
    assert got.stamped.is_empty(), "a literal must not broadcast one row onto no rows"
    assert math.isnan(got.summary["mean"])


# --- every gate stamps the data it scored against: a property of the one function --------
#
# Until now this was assertable only by reading three entry points a network has to reach.
# The stamps are the run's, so whatever frame a gate hands in comes back naming its config,
# its data, its boards and its commit, with every row it had and nothing else touched.

STAMPS = ("cfg_digest", "data_digest", "board_digest", "commit")

_SITES = st.sampled_from([
    {"cluster": SEASON_CLUSTER, "unit": "points per team game", "places": 2, "show_n": True},
    {"cluster": SEASON_CLUSTER, "unit": "points per game", "places": 2, "show_n": True},
    {"cluster": SEASON_CLUSTER, "unit": "points per team-week", "places": 3, "show_n": False},
    {"cluster": None, "unit": "points per team game", "places": 2, "show_n": True},
])


@st.composite
def _frames(draw):
    seasons = draw(st.integers(min_value=1, max_value=4))
    per = draw(st.integers(min_value=1, max_value=5))
    rows = []
    for s in range(2022, 2022 + seasons):
        for k in range(per):
            rows.append({"season": s, "roster": k,
                         "diff": draw(st.floats(min_value=-30, max_value=30,
                                                allow_nan=False, allow_infinity=False))})
    return pl.DataFrame(rows)


@settings(max_examples=25, deadline=None)
@given(paired=_frames(), site=_SITES, boards=st.booleans())
def test_every_gate_stamps_the_data_it_scored_against(paired, site, boards):
    from hub.config import NO_FRAMES

    played = {2024: paired} if boards else None
    got = run_gate(paired, actions=_ACTIONS, name="p", arm_a="a", arm_b="b",
                   bootstrap=50, record_width=False, boards=played, **site)
    assert got.stamped.height == paired.height
    assert set(got.stamped.columns) == set(paired.columns) | set(STAMPS)
    assert got.stamped.select(paired.columns).equals(paired), "a stamp changed a row"
    for stamp in STAMPS:
        assert got.stamped[stamp].n_unique() == 1, f"{stamp} differs between rows"
    assert (got.stamped["board_digest"][0] == NO_FRAMES) is not boards
    # And said, not only stored: the reader deciding whether two runs compare is at the
    # terminal.
    joined = "\n".join(got.lines)
    assert "  data: " in joined and "  board: " in joined and "  commit: " in joined
