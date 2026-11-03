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
from typing import Any, NamedTuple

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hub.ledger import Ledger
from hub.models import experiment
from hub.models.experiment import SEASON_CLUSTER, Actions, Ceiling, run_gate


class _Played(NamedTuple):
    frame: pl.DataFrame
    report: Any

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
    """The shared call with a fresh in-memory ledger, so a test writes no history unless it
    hands in its own `ledger=` to share across two calls.

    `within` defaults to `"draft"`, this module's own fixture column, and is overridden by
    callers whose frame renamed it to `"roster"` -- the same rename `_paired` itself takes.
    """
    base: dict[str, Any] = {"cluster": SEASON_CLUSTER, "within": ("draft",),
                            "actions": _ACTIONS, "name": "test",
                            "arm_a": "a", "arm_b": "b", "bootstrap": 200,
                            "ledger": Ledger(path=None)}
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


def test_the_resolved_and_abstained_counts_sum_to_the_seasons_run():
    """Since #381 the verdict sentence and the ledger carry how many seasons resolved and
    how many abstained; together they are the seasons run."""
    got = _run(_paired())
    assert got.resolved + got.abstained == got.seasons.height


# --- nothing moves: the numbers each gate produced when the sequence was unified ---------
#
# The acceptance criterion of #135. Each gate's `main` used to spell the sequence by hand;
# `run_gate` must give the numbers those spellings gave. The expected values below are
# literals captured from that unified run for these fixtures and seeds, not re-derived from
# the experiment internals `run_gate` itself calls -- so a dropped or reordered seed,
# bootstrap count, cluster or ceiling argument moves a number and fails here.
#
# Each row: (mean, lo, hi, p_better, se, mde, ceiling or None, per-season gains, verdict kind).

_TOL = 1e-6


def _pin(got, want):
    mean, lo, hi, p_better, se, mde, ceiling, gains, kind = want
    s = got.summary
    for key, val in (("mean", mean), ("lo", lo), ("hi", hi), ("p_better", p_better),
                     ("se", se), ("mde", mde)):
        assert s[key] == pytest.approx(val, abs=_TOL), f"summary[{key!r}] moved"
    if ceiling is None:
        assert "ceiling" not in s, "a ceiling appeared that the run was not given"
    else:
        assert s["ceiling"] == pytest.approx(ceiling, abs=_TOL), "the ceiling moved"
    assert got.seasons["gain"].to_list() == pytest.approx(list(gains), abs=_TOL), \
        "the season table moved"
    assert got.verdict[0] == kind, "the verdict moved"


def _draft_runs(seed, shift):
    paired = _paired(seed=seed, shift=shift)
    bound = paired.with_columns((pl.col("diff") + 9.0).alias("diff"))
    return {"plain": _run(paired, seed=seed),
            "ceiling": _run(paired, seed=seed,
                            ceiling=Ceiling("foresight", bound["diff"]))}


def _lineup_runs(seed, shift):
    paired = _paired(seed=seed, shift=shift).rename({"draft": "roster"})
    with_c = paired.with_columns((pl.col("diff") + 4.0).alias("ceiling_diff"))
    return {"plain": _run(paired, seed=seed, within=("roster",), unit="points per game"),
            "ceiling": _run(with_c, seed=seed, within=("roster",), unit="points per game",
                            ceiling=Ceiling("a perfect spread", with_c["ceiling_diff"]))}


def _weekly_run(seed, void):
    paired = _paired(seed=seed).rename({"draft": "roster"}).with_columns(pl.lit(5).alias("week"))
    return _run(paired, seed=seed, within=("roster",), unit="points per team-week", places=3,
                show_n=False, void=void)


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("shift", [0.0, -6.0, 6.0])
def test_the_draft_gate_s_numbers_are_pinned(seed, shift):
    for arm, got in _draft_runs(seed, shift).items():
        _pin(got, _DRAFT[(seed, shift, arm)])


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("shift", [0.0, -6.0, 6.0])
def test_the_lineup_gate_s_numbers_are_pinned(seed, shift):
    for arm, got in _lineup_runs(seed, shift).items():
        _pin(got, _LINEUP[(seed, shift, arm)])


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("void", [None, "VOID: 6.2% of roster-weeks are a join failure"])
def test_the_weekly_gate_s_numbers_are_pinned(seed, void):
    _pin(_weekly_run(seed, void), _WEEKLY[(seed, void)])


_DRAFT = {
    (0, 0.0, 'plain'): (0.404997, -1.269457, 2.062513, 0.67, 1.063757, 4.28063, None, (0.541984, 1.045007, -2.536358, 2.569356), 'NOT-RUNNABLE'),
    (0, 0.0, 'ceiling'): (0.404997, -1.269457, 2.062513, 0.67, 1.063757, 4.28063, 9.404997, (0.541984, 1.045007, -2.536358, 2.569356), 'SHOW'),
    (0, -6.0, 'plain'): (-5.595003, -7.269457, -3.937487, 0.0, 1.063757, 4.28063, None, (-5.458016, -4.954993, -8.536358, -3.430644), 'NOT-RUNNABLE'),
    (0, -6.0, 'ceiling'): (-5.595003, -7.269457, -3.937487, 0.0, 1.063757, 4.28063, 3.404997, (-5.458016, -4.954993, -8.536358, -3.430644), 'NOT-RUNNABLE'),
    (0, 6.0, 'plain'): (6.404997, 4.730543, 8.062513, 1.0, 1.063757, 4.28063, None, (6.541984, 7.045007, 3.463642, 8.569356), 'NOT-RUNNABLE'),
    (0, 6.0, 'ceiling'): (6.404997, 4.730543, 8.062513, 1.0, 1.063757, 4.28063, 15.404997, (6.541984, 7.045007, 3.463642, 8.569356), 'ADOPT'),
    (3, 0.0, 'plain'): (1.095073, -1.130116, 3.309195, 0.765, 1.423716, 5.729129, None, (3.18291, -0.233597, -2.0045, 3.43548), 'NOT-RUNNABLE'),
    (3, 0.0, 'ceiling'): (1.095073, -1.130116, 3.309195, 0.765, 1.423716, 5.729129, 10.095073, (3.18291, -0.233597, -2.0045, 3.43548), 'SHOW'),
    (3, -6.0, 'plain'): (-4.904927, -7.130116, -2.690805, 0.0, 1.423716, 5.729129, None, (-2.81709, -6.233597, -8.0045, -2.56452), 'NOT-RUNNABLE'),
    (3, -6.0, 'ceiling'): (-4.904927, -7.130116, -2.690805, 0.0, 1.423716, 5.729129, 4.095073, (-2.81709, -6.233597, -8.0045, -2.56452), 'NOT-RUNNABLE'),
    (3, 6.0, 'plain'): (7.095073, 4.869884, 9.309195, 1.0, 1.423716, 5.729129, None, (9.18291, 5.766403, 3.9955, 9.43548), 'NOT-RUNNABLE'),
    (3, 6.0, 'ceiling'): (7.095073, 4.869884, 9.309195, 1.0, 1.423716, 5.729129, 16.095073, (9.18291, 5.766403, 3.9955, 9.43548), 'ADOPT'),
}

_LINEUP = {
    (0, 0.0, 'plain'): (0.404997, -1.269457, 2.062513, 0.67, 1.063757, 4.28063, None, (0.541984, 1.045007, -2.536358, 2.569356), 'NOT-RUNNABLE'),
    (0, 0.0, 'ceiling'): (0.404997, -1.269457, 2.062513, 0.67, 1.063757, 4.28063, 4.404997, (0.541984, 1.045007, -2.536358, 2.569356), 'SHOW'),
    (0, -6.0, 'plain'): (-5.595003, -7.269457, -3.937487, 0.0, 1.063757, 4.28063, None, (-5.458016, -4.954993, -8.536358, -3.430644), 'NOT-RUNNABLE'),
    (0, -6.0, 'ceiling'): (-5.595003, -7.269457, -3.937487, 0.0, 1.063757, 4.28063, -1.595003, (-5.458016, -4.954993, -8.536358, -3.430644), 'NOT-RUNNABLE'),
    (0, 6.0, 'plain'): (6.404997, 4.730543, 8.062513, 1.0, 1.063757, 4.28063, None, (6.541984, 7.045007, 3.463642, 8.569356), 'NOT-RUNNABLE'),
    (0, 6.0, 'ceiling'): (6.404997, 4.730543, 8.062513, 1.0, 1.063757, 4.28063, 10.404997, (6.541984, 7.045007, 3.463642, 8.569356), 'ADOPT'),
    (3, 0.0, 'plain'): (1.095073, -1.130116, 3.309195, 0.765, 1.423716, 5.729129, None, (3.18291, -0.233597, -2.0045, 3.43548), 'NOT-RUNNABLE'),
    (3, 0.0, 'ceiling'): (1.095073, -1.130116, 3.309195, 0.765, 1.423716, 5.729129, 5.095073, (3.18291, -0.233597, -2.0045, 3.43548), 'NOT-RUNNABLE'),
    (3, -6.0, 'plain'): (-4.904927, -7.130116, -2.690805, 0.0, 1.423716, 5.729129, None, (-2.81709, -6.233597, -8.0045, -2.56452), 'NOT-RUNNABLE'),
    (3, -6.0, 'ceiling'): (-4.904927, -7.130116, -2.690805, 0.0, 1.423716, 5.729129, -0.904927, (-2.81709, -6.233597, -8.0045, -2.56452), 'NOT-RUNNABLE'),
    (3, 6.0, 'plain'): (7.095073, 4.869884, 9.309195, 1.0, 1.423716, 5.729129, None, (9.18291, 5.766403, 3.9955, 9.43548), 'NOT-RUNNABLE'),
    (3, 6.0, 'ceiling'): (7.095073, 4.869884, 9.309195, 1.0, 1.423716, 5.729129, 11.095073, (9.18291, 5.766403, 3.9955, 9.43548), 'ADOPT'),
}

_WEEKLY = {
    (0, None): (0.404997, -1.269457, 2.062513, 0.67, 1.063757, 4.28063, None, (0.541984, 1.045007, -2.536358, 2.569356), 'NOT-RUNNABLE'),
    (0, 'VOID: 6.2% of roster-weeks are a join failure'): (0.404997, -1.269457, 2.062513, 0.67, 1.063757, 4.28063, None, (0.541984, 1.045007, -2.536358, 2.569356), 'VOID'),
    (3, None): (1.095073, -1.130116, 3.309195, 0.765, 1.423716, 5.729129, None, (3.18291, -0.233597, -2.0045, 3.43548), 'NOT-RUNNABLE'),
    (3, 'VOID: 6.2% of roster-weeks are a join failure'): (1.095073, -1.130116, 3.309195, 0.765, 1.423716, 5.729129, None, (3.18291, -0.233597, -2.0045, 3.43548), 'VOID'),
}

def test_the_weekly_gate_s_ceiling_now_reaches_the_rule_and_that_is_the_one_departure():
    """Recorded rather than hidden. The weekly gate's `main` handed its ceiling to a report
    and never to `summarise`, so `docs/weekly-blend-gate.md` says its NOT-RUNNABLE branch
    "does not fire and cannot". Under one run the ceiling reaches the summary the verdict
    reads, as the lineup gate's already did (#134). No published weekly verdict was reached
    with a ceiling, so none moves; a `--ceiling` run can now be declared unable to run."""
    paired = _paired(seed=1).rename({"draft": "roster"})
    # A ceiling far below this frame's MDE: the rule must now see it.
    c = pl.Series("ceiling_diff", [0.001] * paired.height)
    got = _run(paired, seed=1, within=("roster",), unit="points per team-week", places=3,
               show_n=False, ceiling=Ceiling("perfect foresight", c))
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


def test_the_report_is_the_block_then_the_small_sample_lines_then_the_width_review():
    """One `Ledger` shared across both calls -- the in-memory adapter (#385) in place of a tmp
    file -- so the second run's comparison is asserted on `run.lines`, not on a file read."""
    paired = _paired(seed=5)
    ledger = Ledger(path=None)
    first = _run(paired, seed=5, ledger=ledger)
    assert first.lines[0].startswith("\n  n=24  a - b = ")
    assert "95% CI" in first.lines[1] and "MDE at 80% power" in first.lines[2]
    assert any("4 clusters" in ln for ln in first.lines)
    assert not any("interval width" in ln for ln in first.lines), "a first run has no history"
    second = _run(paired, seed=5, ledger=ledger)
    assert any("interval width" in ln for ln in second.lines), "the second run compares"


def test_the_width_history_is_keyed_by_the_gate_s_name():
    """Keyed by name: a `draft` run and a `lineup` run share one `Ledger` and neither sees the
    other's history -- checked on the `Comparison` each run got, not a file's contents."""
    ledger = Ledger(path=None)
    draft = _run(_paired(), name="draft", ledger=ledger)
    lineup = _run(_paired(), name="lineup", ledger=ledger)
    assert not any("interval width" in ln for ln in draft.lines)
    assert not any("interval width" in ln for ln in lineup.lines)


def test_the_ledger_entry_and_the_run_carry_the_abstention_count():
    """#381 (C): four seasons that win and one whose diffs are all exactly zero (an
    Abstention: the sign fallback reads exact zero as a tie). The verdict, the run's counts and
    the recorded entry must all say 4 resolved of 5, 1 abstained -- the same numbers in three
    places, so a verdict on fewer seasons than were run cannot lose that fact on its way to
    the record. Rule 18: with the entry's two fields dropped this fails on the ledger line.

    #520 (2026-10-20): this was three wins and the zero over four seasons. Three equal gains
    beside an exact zero cannot ADOPT at k=4 under the unbiased t-reference SE (mean `3a/4`
    against a half-width of `3.18 * a/4`); the biased SE was what let it, so a fifth season
    was added rather than the interval weakened."""
    rows = [{"season": 2022 + s, "draft": k, "diff": 0.0 if s == 4 else 3.0 + 0.01 * (k % 3)}
            for s in range(5) for k in range(6)]
    ledger = Ledger(path=None)
    got = _run(pl.DataFrame(rows), ledger=ledger, ceiling=Ceiling("x", np.array([1e6])))
    assert got.verdict[0] == "ADOPT", got.verdict
    assert "4 resolved of 5, 1 abstained" in got.verdict[1]
    assert (got.resolved, got.abstained) == (4, 1)
    recorded = ledger._read()
    assert recorded is not None
    assert (recorded[-1].resolved, recorded[-1].abstained) == (4, 1)
    assert recorded[-1].verdict == "ADOPT"


def test_the_ledger_entry_names_each_source_the_run_read_even_when_the_digest_is_unpinned():
    """#429: the draft gate's re-run of #376's recipe did not reproduce it, and nothing the run
    recorded could say why -- one read with no pin turns the whole `data_digest` into `unpinned`,
    which discriminates nothing. The entry now carries the run's reads one by one (source, as-of,
    digest; `unpinned` where there is no pin), so a later replay can be diffed source by source
    and the unpinned read is named. Rule 18: with the field dropped this fails on the entry."""
    from dataclasses import dataclass

    from hub.config import UnpinnedRead
    from hub.fetch import nflverse as nv

    @dataclass(frozen=True)
    class _Pin:
        source: str
        as_of: str | None
        digest: str

    ledger = Ledger(path=None)
    with nv.reads_of_one_run() as reads:
        reads.remember("a", _Pin("ff_rankings", "2025-08-31", "abc123"))
        reads.remember("b", UnpinnedRead("player_stats", None))
        got = _run(_paired(), ledger=ledger)
    assert got.verdict[0] in {"SHOW", "ADOPT", "REMOVE", "NOT-RUNNABLE"}
    recorded = ledger._read()
    assert recorded is not None and recorded[-1].inputs == [
        {"source": "ff_rankings", "as_of": "2025-08-31", "digest": "abc123"},
        {"source": "player_stats", "as_of": None, "digest": "unpinned"}]


def test_a_run_records_the_digest_of_the_modules_it_was_given_and_a_changed_one_does_not_compare(
        tmp_path, monkeypatch):
    """#435 end to end: `run_gate(code_modules=...)` puts the arm modules' digest in the entry,
    and editing one of them between two otherwise identical runs leaves the second with nothing
    to compare against (planted: a real module on disk, rewritten between the runs)."""
    import importlib
    import sys
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "planted435_gate_arm.py").write_text("BASE = 'within-season'\n")
    importlib.invalidate_caches()
    sys.modules.pop("planted435_gate_arm", None)
    ledger = Ledger(path=None)
    first = _run(_paired(), ledger=ledger, code_modules=("planted435_gate_arm",))
    _run(_paired(), ledger=ledger, code_modules=("planted435_gate_arm",))
    entries = ledger._read()
    assert entries is not None and entries[0].code_digest == entries[1].code_digest
    assert entries[0].code_digest is not None and len(entries[0].code_digest) == 8
    assert entries[1].comparable(entries[0]), "unchanged source still compares"
    (tmp_path / "planted435_gate_arm.py").write_text("BASE = 'strictly prior'\n")
    _run(_paired(), ledger=ledger, code_modules=("planted435_gate_arm",))
    entries = ledger._read()
    assert entries is not None
    assert not entries[2].comparable(entries[1]) and not entries[2].comparable(entries[0])
    assert first.verdict[0]


def test_a_direct_call_that_declares_no_code_is_of_unknown_code_and_never_compared(tmp_path):
    """#439: `run_gate(code_modules=())` used to record `code_digest=None`, a *known* "no modules
    declared" that compares equal to another such run -- the hole #435 closed, reopened by saying
    nothing. It is now an undeclared run: unknown code, written with no `code_digest` key, named
    and compared with nothing, not even with itself. Declaring modules makes the same two runs
    compare (the flip)."""
    import json
    path = tmp_path / "gate-width.json"
    led = Ledger(path)
    _run(_paired(), ledger=led)
    second = _run(_paired(), ledger=led)
    rows = json.loads(path.read_text())["entries"]
    assert all("code_digest" not in r for r in rows), "undeclared must not be written as known"
    assert any("unknown code" in ln for ln in second.lines)
    entries = led._read()
    assert entries is not None and not entries[1].comparable(entries[0])

    declared = Ledger(tmp_path / "declared.json")
    _run(_paired(), ledger=declared, code_modules=("hub.ledger",))
    _run(_paired(), ledger=declared, code_modules=("hub.ledger",))
    got = declared._read()
    assert got is not None and got[1].comparable(got[0])


def test_an_empty_frame_runs_and_says_nothing_was_measured():
    """The weekly gate's `compare` returns a frame with no rows and no columns when nothing
    is covered. The run must still answer rather than fall over on a season column that is
    not there."""
    got = _run(pl.DataFrame())
    assert got.verdict[0] == "SHOW" and "Nothing measured" in got.verdict[1]
    assert got.seasons.is_empty()
    assert got.seasons.columns == ["season", "gain", "n", "se", "m"]
    assert got.stamped.is_empty(), "a literal must not broadcast one row onto no rows"
    assert math.isnan(got.summary["mean"])


# --- every gate stamps the data it scored against: a property of the one function --------
#
# Until now this was assertable only by reading three entry points a network has to reach.
# The stamps are the run's, so whatever frame a gate hands in comes back naming its config,
# its data, its boards and its commit, with every row it had and nothing else touched.

STAMPS = ("cfg_digest", "data_digest", "board_digest", "commit")

_SITES = st.sampled_from([
    {"cluster": SEASON_CLUSTER, "within": ("roster",), "unit": "points per team game",
     "places": 2, "show_n": True},
    {"cluster": SEASON_CLUSTER, "within": ("roster",), "unit": "points per game",
     "places": 2, "show_n": True},
    {"cluster": SEASON_CLUSTER, "within": ("roster",), "unit": "points per team-week",
     "places": 3, "show_n": False},
    {"cluster": None, "within": ("roster",), "unit": "points per team game",
     "places": 2, "show_n": True},
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

    # The Boards a run played, as `walk_forward_inputs` returns them: the frame with a report
    # attached (#295). Structural, like `ReportedFrame`; the digest is over the frame.
    played = {2024: _Played(paired, None)} if boards else None
    got = run_gate(paired, actions=_ACTIONS, name="p", arm_a="a", arm_b="b",
                   bootstrap=50, ledger=Ledger(path=None), boards=played, **site)
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


def test_the_run_prints_the_posterior_beside_the_verdict_for_a_gate_with_a_prior():
    """#375: a gate named in `posterior.S_REF` prints the block after its per-season table,
    reading its own frozen prior and naming the verdict word that still decides."""
    got = _run(_paired(shift=3.0), name="weekly")
    text = "\n".join(got.lines)
    assert got.posterior is not None and got.posterior.primary is not None
    assert "posterior on the pooled effect" in text
    assert "prior N(0, 0.648), season scatter tau ~ HalfNormal(0.648); costly" in text
    assert f"(verdict word: {got.verdict[0]})" in text
    seasons_at = next(i for i, ln in enumerate(got.lines) if "posterior on the pooled" in ln)
    assert any("2022" in ln for ln in got.lines[:seasons_at]), "after the per-season table"


def test_the_run_says_why_a_gate_without_a_prior_has_no_posterior():
    """A gate absent from `S_REF` prints one line naming the absence, never nothing."""
    got = _run(_paired())
    assert got.posterior is not None and got.posterior.primary is None
    assert any("not computed -- this gate has no pre-registered S_REF" in ln for ln in got.lines)
