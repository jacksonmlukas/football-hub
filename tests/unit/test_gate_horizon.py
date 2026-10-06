"""`scripts/gate_horizon.py` is #388's harness: the shipped `experiment.gate` driven over a grid of
season counts and within-season rows. Its table is only worth reading if the harness is (a) the
rule that ships, (b) able to tell the shipped tie handling from the pre-#335 one, and (c) able to
vary with the effect it is about -- the three things the ticket's controls ask of the full run.
These are the same three at a size a unit test can afford, written the way rule 18 asks: plant the
condition, watch the harness respond to it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.models import experiment

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gate_horizon.py"


@pytest.fixture(scope="module")
def gh():
    spec = importlib.util.spec_from_file_location("gate_horizon", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_horizon"] = mod     # a dataclass resolves its annotations through here
    spec.loader.exec_module(mod)
    return mod


def _run(gh, cell, n, seed=1):
    return gh.run_chunk(cell, gh.estimates()[cell.path], n, np.random.SeedSequence(seed))


def test_the_estimates_are_the_published_history_disattenuated(gh):
    est = gh.estimates()
    # weekly: a cluster mean over 13 weeks must reproduce the recorded season SEs (0.41-0.55)
    se13 = np.sqrt(est["weekly"].row_var / 13 / gh.WEEKLY_M)
    assert 0.40 < se13 < 0.60
    # tau is the observed between-season SD less its own sampling noise, so it sits below it
    assert 0 < est["weekly"].tau < float(np.std(gh.WEEKLY_GAINS, ddof=1))
    assert 0 < est["draft"].tau < float(np.std(gh.DRAFT_GAINS, ddof=1))


def test_the_harness_adopts_on_an_effect_it_cannot_miss(gh):
    """Control 1 at unit-test size: a rate that cannot reach 1 is a harness whose ADOPT branch
    something upstream of the rule has made unreachable (the `rule16_combined_power` shape)."""
    res = _run(gh, gh.weekly_cell(6, 13, gh.PLANTED_DELTA["weekly"]), 30)
    assert res["ship"].get("ADOPT", 0) / res["n"] > 0.95, res["ship"]
    assert res["ship"].get("NOT-RUNNABLE", 0) == 0


def test_the_harness_does_not_adopt_a_null_and_can_tell_the_two_apart(gh):
    null = _run(gh, gh.weekly_cell(4, 13, 0.0), 120)
    big = _run(gh, gh.weekly_cell(4, 13, gh.PLANTED_DELTA["weekly"]), 120)
    assert null["ship"].get("ADOPT", 0) / null["n"] < 0.1
    assert big["ship"]["ADOPT"] > null["ship"].get("ADOPT", 0)


def test_the_pre_335_reading_never_sees_a_tie_and_the_shipped_one_does(gh):
    """Control 3 at unit-test size. At the weekly noise a gain of 0.3 sits inside two season
    SEs, so the shipped rule ties; the sign-alone reading has no ties by construction and so can
    only adopt where the shipped rule does and more. #335's prediction, as an inclusion per frame."""
    res = _run(gh, gh.weekly_cell(4, 13, 0.3), 200)
    assert res["tot"].get("ties", 0) > 0
    assert res["violations"] == 0
    assert res["sign"].get("ADOPT", 0) >= res["ship"].get("ADOPT", 0)


def test_the_sign_only_reading_is_what_the_floor_override_gives(gh):
    """`_read` takes the pre-#335 verdict by dropping `se`/`m` from the seasons frame. The same
    verdict must come from raising `TIE_MIN_CLUSTERS` past any cluster count: two routes to one
    rule, or the control is measuring something other than what #335 replaced."""
    rng = np.random.default_rng(7)
    proc = gh.estimates()["weekly"]
    for i in range(25):
        frame = gh._frame(rng, proc, ((20, 13),) * 4, 0.3)
        via_drop = gh._read(frame, i)["sign"]
        original = experiment.TIE_MIN_CLUSTERS
        experiment.TIE_MIN_CLUSTERS = 10**9
        try:
            via_floor = experiment.gate(
                frame, cluster=experiment.SEASON_CLUSTER, within=("unit",), ceiling=gh._TOP,
                actions=gh._ACTIONS, bootstrap=experiment.BOOTSTRAP, seed=i).verdict[0]
        finally:
            experiment.TIE_MIN_CLUSTERS = original
        assert via_drop == via_floor


def test_a_collapsed_cluster_frame_gives_the_row_level_verdict(gh):
    """The harness hands `gate` one row per cluster carrying the cluster's mean. For a balanced
    design that must be the verdict the row-level frame gives -- the claim the module docstring
    makes in place of simulating every roster-week."""
    rng = np.random.default_rng(3)
    rows = []
    for yr in range(4):
        mu = rng.normal(0.4, 0.4)
        for roster in range(20):
            for week in range(13):
                rows.append((2022 + yr, roster, week, rng.normal(mu, 8.0)))
    long = pl.DataFrame(rows, schema=["season", "unit", "week", "diff"], orient="row")
    wide = long.group_by("season", "unit").agg(pl.col("diff").mean()).sort("season", "unit")
    kw = {"cluster": experiment.SEASON_CLUSTER, "within": ("unit",), "ceiling": gh._TOP,
          "actions": gh._ACTIONS, "bootstrap": 400, "seed": 5}
    a, b = experiment.gate(long, **kw), experiment.gate(wide, **kw)
    assert a.verdict[0] == b.verdict[0]
    assert a.summary["mean"] == pytest.approx(b.summary["mean"])
    assert a.summary["se"] == pytest.approx(b.summary["se"])
    assert a.seasons["se"].to_list() == pytest.approx(b.seasons["se"].to_list())


def test_a_cell_does_not_depend_on_how_its_trials_are_split_across_workers(gh):
    cell = gh.weekly_cell(4, 13, 0.3)
    procs = gh.estimates()
    once = gh.run_cells([cell], procs, trials=gh.CHUNK, workers=1)
    again = gh.run_cells([cell], procs, trials=gh.CHUNK, workers=1)
    assert once == again


def test_the_render_reads_every_cell_the_grids_ask_for(gh, monkeypatch):
    monkeypatch.setattr(gh, "CHUNK", 2)
    procs = gh.estimates()
    table = gh.run_cells(gh.table_cells(), procs, trials=2, workers=1)
    sens = gh.run_cells(gh.sensitivity_cells(), procs, trials=2, workers=1)
    text = gh.render(table, sens)
    assert "Where the gap to 80% power sits" in text and "Sensitivity to the between-season" in text


def test_the_rule16_ladder_starts_at_rule16s_process_and_ends_at_388s_cell(gh):
    """#418: step 1 is the precedent script's process (cluster SD = between-season s, m as the
    script set it, bootstrap 200); the last full-bootstrap step is `weekly_cell`/`draft_cell`
    itself, so the ladder's two ends are the two published numbers' own generating processes."""
    procs, cells, steps = gh.rule16_ladder()
    for path, (s, m, delta), end in (("weekly", gh.R16_WEEKLY, gh.weekly_cell(4, 13, 0.3)),
                                     ("draft", gh.R16_DRAFT, gh.draft_cell(4, 20, gh.DRAFT_DELTA))):
        first, = [lab for p, st, lab in steps if p == path and st.startswith("1 ")]
        last, = [lab for p, st, lab in steps if p == path and st.startswith("5 ")]
        assert procs[first].tau == s and procs[first].row_var == pytest.approx(s * s)
        at_first = [c for c in cells if c.path == first]
        assert {c.seasons for c in at_first} == {((m, 1),) * 4} and {c.bootstrap for c in at_first} == {200}
        at_last = [c for c in cells if c.path == last and c.delta == delta]
        assert at_last[0].seasons == end.seasons and at_last[0].bootstrap == end.bootstrap
        assert procs[last] == gh.estimates()[path]


def test_the_rule16_ladder_renders_and_its_null_stays_under_alpha(gh, monkeypatch):
    monkeypatch.setattr(gh, "CHUNK", 5)
    procs, cells, steps = gh.rule16_ladder()
    res = gh.run_cells(cells, procs, trials=5, workers=1)
    assert "ties/season at the observed gain" in gh.render_rule16(res, steps)
    # a planted effect through the same harness is adopted where a null is not (rule 18)
    big = gh.run_chunk(gh.Cell("weekly:big", ((40, 1),) * 6, 20.0, 1.0, 200),
                       procs["weekly:1 rule-16 process, bootstrap 200"], 20, np.random.SeedSequence(3))
    assert big["ship"].get("ADOPT", 0) / big["n"] > 0.9


# --- #381: (C) beside (A), read off the same `gate` run ---------------------------------------

def _recs(*disps):
    return [{"season": 2022 + i, "gain": 0.0, "se": 1.0, "m": 20, "disposition": d}
            for i, d in enumerate(disps)]


def test_the_two_rules_read_a_resolved_unanimous_frame_with_one_abstention_differently(gh):
    """Rule 18 for the harness's two columns: plant the one case they exist to tell apart. Three
    wins and a tie with an interval above zero is SHOW to (A) and ADOPT to (C); mirrored for
    REMOVE; and the interval half vetoes both."""
    up, down = {"t_lo": 0.5, "t_hi": 2.0}, {"t_lo": -2.0, "t_hi": -0.5}
    straddle = {"t_lo": -0.5, "t_hi": 2.0}
    wwwt, llltt = _recs("win", "win", "win", "tie"), _recs("loss", "loss", "loss", "tie")
    assert gh.verdict_abstain(up, wwwt) == "SHOW" and gh.verdict_resolved(up, wwwt) == "ADOPT"
    assert gh.verdict_abstain(down, llltt) == "SHOW" and gh.verdict_resolved(down, llltt) == "REMOVE"
    assert gh.verdict_resolved(straddle, wwwt) == "SHOW"          # the interval half still vetoes
    assert gh.verdict_resolved(up, _recs("win", "win", "loss", "tie")) == "SHOW"
    # no resolved season: "every resolved season wins" is vacuous, and must not adopt
    assert gh.verdict_resolved(up, _recs("tie", "tie", "tie", "tie")) == "SHOW"
    assert gh.verdict_resolved(down, _recs("tie", "tie")) == "SHOW"
    assert gh.verdict_resolved(up, _recs("win", "tie", "tie", "tie")) == "ADOPT"   # one is enough


def test_gate_reads_the_same_verdict_as_the_rule_the_harness_says_is_landed(gh):
    """The harness's "ship" column is `gate`'s own verdict. Per frame it must be one of the two
    functions above, and over a planted-effect run where (A) and (C) differ it must be the same
    one every time -- the landed rule -- or the harness is reading one rule and reporting another."""
    rng = np.random.default_rng(11)
    proc = gh.estimates()["weekly"]
    outs = [gh._read(gh._frame(rng, proc, ((20, 13),) * 4, 0.9), i) for i in range(60)]
    assert all(o["ship"] in (o["abstain"], o["resolved"]) for o in outs)
    assert any(o["abstain"] != o["resolved"] for o in outs)      # the plant: the rules do differ
    assert all(o["ship"] == o["abstain"] for o in outs)          # LANDED: (A)


def test_c_adopts_wherever_a_does_and_never_beyond_the_interval_alone(gh):
    res = _run(gh, gh.weekly_cell(4, 13, 0.9), 120)
    assert res["tot"].get("c_not_superset_of_a", 0) == 0
    assert res["tot"].get("c_beyond_interval", 0) == 0
    assert res["resolved"].get("ADOPT", 0) > res["abstain"].get("ADOPT", 0)


def test_the_rule_c_render_reads_every_cell_its_grid_asks_for(gh, monkeypatch):
    monkeypatch.setattr(gh, "CHUNK", 2)
    res = gh.run_cells(gh.rule_c_cells(), gh.estimates(), trials=2, workers=1)
    text = gh.render_rule_c(res)
    assert "null (C)" in text and "Per-frame inclusions" in text
