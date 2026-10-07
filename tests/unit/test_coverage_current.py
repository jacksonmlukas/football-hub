"""#423 (V5): the current season is a separate labelled row, the closed backtest is read back,
a run that changes nothing writes nothing, and looks 2 and 3 read the row only when it is time.

Written to the pre-registration of 2026-10-07 (`docs/weekly-coverage.md`): look 1 reads the
2021-2025 backtest; look 2 reads the current-season row after week 14 is complete and look 3
after week 18, on the same marginal bar, and nothing else reads that row. Every control plants
the condition its check exists to detect, and every write goes to `tmp_path`: **no test here
takes a look against the committed artifact**.

The world is three synthetic seasons (2023-2025) of 18 weeks drawn through the shipped law, with
`CURRENT_SEASON` moved to 2025, so "the current season" has the full 18 weeks the real one will
and a calendar a test can put a run date on.
"""
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub import jsonio
from hub.models import coverage, predict

ROOT = Path(__file__).resolve().parents[2]
CURRENT = 2025
FIRST_THURSDAY = date(2025, 9, 4)


def _last_game(week: int) -> date:
    """Week w's Monday game: the schedule below has a Thursday and a Monday game a week."""
    return FIRST_THURSDAY + timedelta(days=7 * (week - 1) + 4)


def _schedule(season=CURRENT, weeks=18):
    rows = []
    for w in range(1, weeks + 1):
        thu = FIRST_THURSDAY + timedelta(days=7 * (w - 1))
        for k, d in enumerate((thu, thu + timedelta(days=4))):
            rows.append({"game_id": f"{season}_{w:02d}_{k}", "season": season, "week": w,
                         "game_type": "REG", "gameday": d.isoformat()})
    return pl.DataFrame(rows)


def _drawn(seasons=(2023, 2024, 2025), weeks=18, n_players=120, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        mu = 18.0 if pos == "QB" else 12.0
        sd = predict.WEEKLY_K[pos] * math.sqrt(mu)
        sk = predict.WEEKLY_SKEW[pos]
        for season in seasons:
            for p in range(n_players):
                pts = predict.skewed(mu, sd, sk, rng.standard_normal(weeks))
                rows += [{"player_id": f"{pos}{p}", "position": pos, "season": season,
                          "week": w + 1, "season_type": "REG",
                          "fantasy_points_ppr": float(x)} for w, x in enumerate(pts)]
    return pl.DataFrame(rows, schema={
        "player_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
        "season_type": pl.Utf8, "fantasy_points_ppr": pl.Float64})


@pytest.fixture(scope="module")
def stats():
    return _drawn()


@pytest.fixture
def world(tmp_path, monkeypatch, stats):
    """The current season moved to 2025; loaders that log what they were asked for; an artifact
    in tmp_path. `w.calls` is `[("stats", [seasons]) | ("schedules", [seasons])]`."""
    monkeypatch.setattr(coverage, "CURRENT_SEASON", CURRENT)
    monkeypatch.setattr(coverage, "ARTIFACT", tmp_path / "interval_coverage.json")

    class World:
        calls: list
        frame = stats
        survivor: pl.DataFrame | None = None
        path: Path = tmp_path / "interval_coverage.json"

    w = World()
    w.calls = []

    def _stats(seasons, cache):
        w.calls.append(("stats", list(seasons)))
        return w.frame.filter(pl.col("season").is_in(list(seasons)))

    def _schedules(seasons, cache):
        w.calls.append(("schedules", list(seasons)))
        if list(seasons) == [CURRENT]:
            return _schedule()
        if w.survivor is not None:
            return w.survivor
        raise OSError("no schedule for this fixture")

    monkeypatch.setattr(coverage, "_stats", _stats)
    monkeypatch.setattr(coverage, "_schedules", _schedules)
    return w


def _after(week: int, days: int = 2) -> str:
    return (_last_game(week) + timedelta(days=days)).isoformat()


def _file(w):
    return json.loads(w.path.read_text())


# --- the current-season row: labelled, separate, complete weeks only ---------------------


def test_the_current_row_is_labelled_out_of_sample_and_scores_only_complete_weeks(stats):
    """Week 10's Thursday game is played and its Monday game is not: the week is not complete,
    so week 10 is out of the row even though its rows are in the data (a Saturday run holds
    Thursday's game of a week with three days left). Mutation: count a week at its first game
    (`#263`'s `weeks_played`) and `weeks_complete` is 10."""
    sched = _schedule()
    mid10 = _last_game(10) - timedelta(days=1)
    cur = coverage.measure_current(stats, sched, mid10, season=CURRENT)
    assert cur["label"] == "out-of-sample" and cur["season"] == CURRENT
    assert cur["weeks_complete"] == 9
    done10 = coverage.measure_current(stats, sched, _last_game(10) + timedelta(days=2),
                                      season=CURRENT)
    assert done10["weeks_complete"] == 10 and done10["n"] > cur["n"]


def test_a_week_the_calendar_closed_but_the_data_lacks_is_not_complete(stats):
    """Dates then data (#437): week 14 is over by the calendar and its rows are not in nflverse.
    Planted by cutting the frame at week 13. Mutation: drop the `w not in have` clause."""
    cut = stats.filter(~((pl.col("season") == CURRENT) & (pl.col("week") >= 14)))
    got = coverage.measure_current(cut, _schedule(), date(2026, 1, 20), season=CURRENT)
    assert got["weeks_complete"] == 13


def test_the_row_carries_n_cov80_cov68_clipped_share_and_sigma_at_the_binomial_se(stats):
    cur = coverage.measure_current(stats, _schedule(), date(2026, 1, 20), season=CURRENT)
    for k in ("n", "cov80", "cov68", "clipped_share", "sigma", "se", "mde"):
        assert cur[k] is not None, k
    nominal = 0.80
    assert cur["se"] == pytest.approx(
        math.sqrt(2.0) * math.sqrt(nominal * (1 - nominal) / cur["n"]))
    assert cur["sigma"] == pytest.approx((cur["cov80"] - nominal) / cur["se"])
    assert cur["weeks_complete"] == 18


def test_the_row_has_no_verdict_and_every_position_is_not_runnable_below_the_minimum(stats):
    """The row carries no verdict of its own: the pool has none, the positions say NOT-RUNNABLE
    with the MDE printed. Mutation: lower `N_MIN_GROUP` to 100 and a position gets a verdict."""
    cur = coverage.measure_current(stats, _schedule(), date(2026, 1, 20), season=CURRENT)
    pool = next(r for r in cur["by_position"] if r["group"] == "all")
    assert "verdict" not in pool and "verdict" not in cur
    positions = [r for r in cur["by_position"] if r.get("position")]
    assert len(positions) == 4
    for r in positions:
        assert r["n"] < coverage.N_MIN_GROUP
        assert r["verdict"] == "NOT-RUNNABLE" and r["mde"] > 0.02


def test_a_season_with_no_complete_week_is_an_empty_row_not_an_error(stats):
    """Before week 5 nothing is scored in a prior-centre season (four weeks of centre are
    needed): the row exists, says so, and the positions are NOT-RUNNABLE."""
    cur = coverage.measure_current(stats, _schedule(), date(2025, 9, 1), season=CURRENT)
    assert cur["weeks_complete"] == 0 and cur["n"] == 0 and cur["cov80"] is None
    assert all(r["verdict"] == "NOT-RUNNABLE" for r in cur["by_position"] if r.get("position"))


def test_two_seasons_of_history_give_the_calibration_a_full_window_as_full_history_does(stats):
    """The row reads `CURRENT_HISTORY` seasons behind the current one rather than all five. The
    window is 14 cells, so that must give the same calibration the full history does: the same
    row from three seasons and from five. Planted: with no history at all the early weeks lose
    their calibration and the row shrinks, which is what a window that reached past the data
    would look like."""
    wide = _drawn(seasons=(2021, 2022, 2023, 2024, 2025), seed=3)
    # Same draws for the shared seasons would need the same rng order, so compare on one frame.
    three = wide.filter(pl.col("season") >= 2023)
    a = coverage.measure_current(three, _schedule(), date(2026, 1, 20), season=CURRENT)
    b = coverage.measure_current(wide, _schedule(), date(2026, 1, 20), season=CURRENT)
    drop = ("seasons_read",)
    assert {k: v for k, v in a.items() if k not in drop} == \
        {k: v for k, v in b.items() if k not in drop}
    alone = coverage.measure_current(wide.filter(pl.col("season") == CURRENT), _schedule(),
                                     date(2026, 1, 20), season=CURRENT)
    assert alone["n"] < a["n"], "no history: the early weeks have no calibration -- the control"


def test_the_current_row_is_never_pooled_into_the_backtest(world):
    """A weekly write leaves the backtest's figure exactly what the backtest seasons give, and
    the 2025 weeks are only under `current_season`."""
    assert coverage.main(["--measure", "--write", "--seasons", "2023,2024",
                          "--as-of", _after(18)]) == 0
    got = _file(world)
    alone = coverage.measure(world.frame.filter(pl.col("season") < CURRENT), "prior")
    assert got["n"] == alone["n"] and got["gate_n"] == alone["gate_n"]
    assert got["current_season"]["season"] == CURRENT and got["current_season"]["n"] > 0
    assert got["gate_n"] != got["gate_n"] + got["current_season"]["n"]
    assert got["current_season"]["weeks_complete"] == 18


# --- the closed backtest is measured once and read back ------------------------------------


def test_the_closed_backtest_is_read_back_not_pulled_again(world, capsys):
    """Run twice: the second run asks for the three-season current window and never for the
    backtest's seasons. Mutation: drop the `read_back_backtest` call and the second run pulls
    [2023, 2024] again."""
    args = ["--measure", "--write", "--seasons", "2023,2024", "--as-of", _after(14)]
    assert coverage.main(args) == 0
    first = [c for c in world.calls if c == ("stats", [2023, 2024])]
    assert len(first) == 1
    first_text = world.path.read_text()
    world.calls.clear()
    capsys.readouterr()
    assert coverage.main(args) == 0
    assert ("stats", [2023, 2024]) not in world.calls, "the closed span was pulled again"
    assert ("stats", [2023, 2024, 2025]) in world.calls, "the current row is still measured"
    assert "read back" in capsys.readouterr().out
    assert world.path.read_text() == first_text


def test_a_changed_setting_or_remeasure_pulls_the_backtest_again(world):
    base = ["--measure", "--write", "--seasons", "2023,2024", "--as-of", _after(14)]
    assert coverage.main(base) == 0
    for extra in (["--band", "0.03"], ["--min-prior", "5"], ["--remeasure"]):
        world.calls.clear()
        assert coverage.main([*base, *extra]) == 0, extra
        assert ("stats", [2023, 2024]) in world.calls, f"{extra} read back a stale backtest"


def test_a_changed_module_is_a_changed_key(world, monkeypatch):
    """The code half of the key: edit the source a figure depends on and the stored one is not
    read. Planted by changing what `code_digest` returns."""
    base = ["--measure", "--write", "--seasons", "2023,2024", "--as-of", _after(14)]
    assert coverage.main(base) == 0
    monkeypatch.setattr(coverage, "code_digest", lambda modules: "deadbeef")
    world.calls.clear()
    assert coverage.main(base) == 0
    assert ("stats", [2023, 2024]) in world.calls


def test_a_span_that_includes_the_current_season_is_never_read_back(world):
    """It can still move. `backtest_key` is None for it."""
    assert coverage.backtest_key([2023, 2024, 2025]) is None
    assert coverage.backtest_key([2023, 2024]) is not None
    args = ["--measure", "--write", "--seasons", "2024,2025", "--as-of", _after(14)]
    assert coverage.main(args) == 0
    world.calls.clear()
    assert coverage.main(args) == 0
    assert ("stats", [2024, 2025]) in world.calls


def test_the_survivor_block_is_read_back_under_its_key(world, capsys):
    games = [(10.0, 7.0)] * 40 + [(10.0, -7.0)] * 10
    world.survivor = pl.DataFrame({"spread_line": [g[0] for g in games],
                                   "result": [g[1] for g in games]})
    args = ["--survivor", "--write", "--seasons", "2023,2024"]
    assert coverage.main(args) == 0
    text = world.path.read_text()
    world.calls.clear()
    assert coverage.main(args) == 0
    assert ("schedules", [2023, 2024]) not in world.calls
    assert "read back" in capsys.readouterr().out
    assert world.path.read_text() == text
    assert coverage.main([*args, "--remeasure"]) == 0
    assert ("schedules", [2023, 2024]) in world.calls


# --- a run that changes nothing writes nothing --------------------------------------------


def test_two_identical_runs_write_byte_identical_files(world, monkeypatch):
    """The clock moves between the runs and the file does not: the stamp is when the numbers
    last moved. Mutation: stamp unconditionally in `write_summary` and the second run differs."""
    ticks = iter(f"2026-10-0{i}T00:00:00+00:00" for i in range(1, 9))
    monkeypatch.setattr(jsonio, "stamp", lambda: next(ticks))
    args = ["--measure", "--survivor", "--write", "--seasons", "2023,2024", "--as-of", _after(14)]
    world.survivor = pl.DataFrame({"spread_line": [10.0] * 40, "result": [7.0] * 40})
    assert coverage.main(args) == 0
    first = world.path.read_bytes()
    assert coverage.main(args) == 0
    assert world.path.read_bytes() == first
    assert coverage.main(args) == 0
    assert world.path.read_bytes() == first


def test_a_run_that_changes_a_number_does_stamp_it(world, monkeypatch):
    """The other side, or the test above would pass for a writer that never writes: the same
    run a week later (week 14 complete instead of 13) moves the row and the stamp."""
    ticks = iter(f"2026-10-0{i}T00:00:00+00:00" for i in range(1, 9))
    monkeypatch.setattr(jsonio, "stamp", lambda: next(ticks))
    assert coverage.main(["--measure", "--write", "--seasons", "2023,2024",
                          "--as-of", _after(13)]) == 0
    first = _file(world)
    assert coverage.main(["--measure", "--write", "--seasons", "2023,2024",
                          "--as-of", _after(14)]) == 0
    second = _file(world)
    assert second["current_season"]["weeks_complete"] == 14 > first["current_season"]["weeks_complete"]
    assert second["generated_at"] != first["generated_at"]


def test_float_noise_below_the_stated_precision_writes_nothing(tmp_path, monkeypatch):
    """The Audit V finding: four `sd_ratio` values changing in the 16th digit made every slate
    commit carry a change. Planted at 1e-13; a change at the 4th digit does move the file."""
    p = tmp_path / "interval_coverage.json"
    ticks = iter(f"2026-10-0{i}T00:00:00+00:00" for i in range(1, 9))
    monkeypatch.setattr(jsonio, "stamp", lambda: next(ticks))
    base = {"verdict": "COVERS", "sd_ratio": 0.9876543210987, "nested": [{"x": 0.1 + 0.2}]}
    coverage.write_summary(base, p)
    first = p.read_bytes()
    noisy = {"verdict": "COVERS", "sd_ratio": 0.9876543210987 + 3e-13,
             "nested": [{"x": 0.30000000000000004}]}
    coverage.write_summary(noisy, p)
    assert p.read_bytes() == first
    coverage.write_summary({**base, "sd_ratio": 0.9877}, p)
    assert p.read_bytes() != first


def test_every_float_written_is_at_the_stated_precision(world):
    assert coverage.main(["--measure", "--write", "--seasons", "2023,2024",
                          "--as-of", _after(18)]) == 0

    def walk(x):
        if isinstance(x, float):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)

    floats = list(walk(_file(world)))
    assert floats, "nothing to check"
    assert all(round(f, coverage.ROUND_DIGITS) == f for f in floats)


# --- looks 2 and 3 read the current row, and only when it is time ------------------------


def test_look_1_reads_the_backtest_and_never_the_schedule(world, capsys):
    assert coverage.main(["--audit", "--look", "1", "--write", "--seasons", "2023,2024"]) in (0, 1)
    assert not [c for c in world.calls if c[0] == "schedules"]
    got = _file(world)
    assert got["audit"]["reads"] == "backtest" and got["audit_looks"][0]["reads"] == "backtest"
    assert got["n"] == got["audit"]["n"] or got["audit"]["n"] == got["gate_n"]


def test_look_2_is_refused_before_week_14_is_complete_and_nothing_is_loaded_or_spent(
        world, capsys):
    """The planted control: the run date is the last game of week 14 itself (and one day after,
    inside `LAG_DAYS`). Refused, exit 2, no outcome loaded, nothing written. Mutation: drop the
    date check in `_audit_current` and the stats are loaded and a look is written."""
    for as_of in (_last_game(13).isoformat(), _last_game(14).isoformat(),
                  (_last_game(14) + timedelta(days=1)).isoformat()):
        world.calls.clear()
        assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", as_of]) == 2
        err = capsys.readouterr().err
        assert "REFUSED" in err and "week 14" in err and "No outcome was loaded" in err
        assert not [c for c in world.calls if c[0] == "stats"], "an outcome was loaded"
        assert not world.path.exists(), "a refused look must not touch the artifact"


def test_look_2_runs_once_week_14_is_complete_and_records_that_it_read_the_current_row(
        world, capsys):
    """The positive control for the refusal above: two days after week 14's last game."""
    assert coverage.main(["--audit", "--look", "1", "--write", "--seasons", "2023,2024"]) in (0, 1)
    before = _file(world)
    capsys.readouterr()
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) in (0, 1)
    out = capsys.readouterr().out
    assert "AUDIT look 2 of 3" in out and "out-of-sample row" in out and "NOT-RUNNABLE" in out
    after = _file(world)
    assert [r["look"] for r in after["audit_looks"]] == [1, 2]
    assert after["audit"]["look"] == 2 and after["audit"]["reads"] == "current_season"
    assert after["audit"]["season_coverage"]["season"] == CURRENT
    assert after["audit"]["season_coverage"]["weeks_complete"] == 14
    # The backtest at the file's top level is the backtest still: look 2 did not overwrite it.
    keep = [k for k in before if k not in ("audit", "audit_looks")]
    assert {k: after[k] for k in keep} == {k: before[k] for k in keep}


def test_look_2_is_refused_when_the_date_is_past_but_week_14_is_not_in_the_data(world, capsys):
    cut = world.frame.filter(~((pl.col("season") == CURRENT) & (pl.col("week") >= 14)))
    world.frame = cut
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "not in nflverse" in err
    assert not world.path.exists()


def test_look_3_is_refused_before_week_18_and_runs_after_it(world, capsys):
    """Week 14 complete is not week 18 complete. Mutation: read `LOOK_WEEKS[2]` for look 3."""
    assert coverage.main(["--audit", "--look", "3", "--write", "--as-of", _after(17)]) == 2
    assert "week 18" in capsys.readouterr().err and not world.path.exists()
    assert coverage.main(["--audit", "--look", "3", "--write", "--as-of", _after(18)]) in (0, 1)
    assert coverage.looks_taken() == [3]
    assert _file(world)["audit"]["season_coverage"]["weeks_complete"] == 18


def test_a_refusal_is_not_a_failed_audit_and_the_third_look_is_still_open(world):
    """Exit 2 is distinct from the audit failing (1); after a refused look 2 the record is
    empty, so look 2 can still be taken when it is time."""
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(13)]) == 2
    assert coverage.looks_taken() == []
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) in (0, 1)
    assert coverage.looks_taken() == [2]


def test_a_look_already_taken_is_refused_for_the_current_row_too(world, capsys):
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) in (0, 1)
    first = world.path.read_text()
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(15)]) == 1
    assert "already recorded" in capsys.readouterr().err
    assert world.path.read_text() == first


def test_the_look_weeks_are_the_pre_registered_ones():
    assert coverage.LOOK_WEEKS == {2: 14, 3: 18}


def test_the_lag_is_the_one_the_forward_gate_uses():
    from hub.season import weekly_forward
    assert coverage.LAG_DAYS == weekly_forward.LAG_DAYS


# --- the publisher reads what the same run wrote ------------------------------------------


def test_the_published_summary_carries_the_current_row_and_the_files_own_stamp(world):
    assert coverage.main(["--measure", "--write", "--seasons", "2023,2024",
                          "--as-of", _after(14)]) == 0
    pub = coverage.published_summary(world.path)
    assert pub is not None
    assert pub["current_season"]["label"] == "out-of-sample"
    assert pub["generated_at"] == _file(world)["generated_at"]


def test_the_slate_measures_before_it_publishes_and_checks_the_stamp_it_published():
    """#423 box 4. `make slate` runs `hub.publish`, which reads the state file into the track
    record, so the measure step must precede it; and a step after the publish compares the stamp
    embedded in the record to the file's. Mutation: move the measure step after `make slate` and
    the first assertion is red (that was the bug)."""
    text = (ROOT / ".github" / "workflows" / "slate.yml").read_text()
    measure = text.index("hub.models.coverage --measure --survivor --write")
    publish = text.index("run: make slate")
    commit = text.index("git add site/data state")
    check = text.index("interval_coverage.generated_at")
    assert measure < publish < check < commit
    step = text[text.rfind("- name:", 0, measure): measure]
    assert "continue-on-error: true" in step, "a failed fetch still must not take the slate down"


# --- a recorded look is carried through every regeneration exactly -----------------------

RECORDED = {
    "audit": {"look": 1, "looks": 3, "alpha_per_look": 0.016666666666666666,
              "z_per_look": 2.39397979981851, "taken_at": "2026-10-07T20:18:01+00:00",
              "claim": 0.8, "band": 0.02, "marginal": "COVERS", "passes": True, "n": 15678,
              "cov80": 0.80035718841689, "mde": 0.014617887814570909,
              "season_coverage": {"season": 2025, "n": 3192, "cov80": 0.7985588972431078},
              "groups": [{"group": "QB", "cov80": 0.788365650969529, "sigma": -0.8737872482044272,
                          "verdict": "NOT-RUNNABLE"}]},
    "audit_looks": [{"look": 1, "taken_at": "2026-10-07T20:18:01+00:00", "marginal": "COVERS",
                     "passes": True, "cov80": 0.80035718841689, "n": 15678, "claim": 0.8}],
}


def _recorded_text(p: Path) -> str:
    got = json.loads(p.read_text())
    return json.dumps({k: got[k] for k in ("audit", "audit_looks")}, sort_keys=True)


def test_a_regeneration_without_audit_keeps_the_recorded_look_exactly(world):
    """Look 1 is on `main` with sixteen-place values. Every non-audit writer -- the weekly
    measure, the survivor, `write_summary` called directly -- must hand them back untouched, or
    the rounding that keeps noise out of the weekly file would edit a recorded look. Planted: the
    values carry places past `ROUND_DIGITS`. Mutation: `round_values` over the whole body
    instead of `rounded_body` and `cov80` reads 0.800357 afterwards."""
    world.path.write_text(json.dumps({"name": "interval_coverage", **RECORDED}))
    want = _recorded_text(world.path)
    assert "0.80035718841689" in want and "0.7985588972431078" in want
    world.survivor = pl.DataFrame({"spread_line": [10.0] * 40, "result": [7.0] * 40})
    assert coverage.main(["--measure", "--survivor", "--write", "--seasons", "2023,2024",
                          "--as-of", _after(14)]) == 0
    assert _recorded_text(world.path) == want, "the weekly regeneration changed a recorded look"
    assert coverage.looks_taken(world.path) == [1]
    coverage.write_summary({"verdict": "COVERS", "gate_cov80": 0.7999999999999999}, world.path)
    assert _recorded_text(world.path) == want
    coverage.write_survivor({"verdict": "HOLDS", "favourite_gap": 0.0123456789123}, world.path)
    assert _recorded_text(world.path) == want
    got = _file(world)
    assert got["survivor"]["favourite_gap"] == 0.012346, "the report is rounded, the look is not"


def test_writing_a_later_look_leaves_the_earlier_one_byte_for_byte(world):
    world.path.write_text(json.dumps({"name": "interval_coverage", **RECORDED}))
    before = json.loads(world.path.read_text())["audit_looks"][0]
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) in (0, 1)
    after = _file(world)["audit_looks"]
    assert [r["look"] for r in after] == [1, 2] and after[0] == before


# --- the seams that say no ---------------------------------------------------------------


def test_an_open_span_has_no_key_and_a_half_written_file_is_not_read_back(world):
    """The survivor key is None for a span holding the current season; a file whose key matches
    but which carries no backtest rows is a miss, never a read of nothing."""
    assert coverage.survivor_key([2024, 2025]) is None
    assert coverage.survivor_key([2023, 2024]) is not None
    key = coverage.backtest_key([2023, 2024])
    world.path.write_text(json.dumps({"name": "interval_coverage", "backtest_key": key}))
    assert coverage.read_back_backtest(key, world.path) is None


def test_a_schedule_with_no_week_14_refuses_look_2_by_name(world, monkeypatch, capsys):
    short = _schedule(weeks=10)
    monkeypatch.setattr(coverage, "_schedules", lambda seasons, cache: short)
    assert coverage.main(["--audit", "--look", "2", "--as-of", _after(18)]) == 2
    assert "has no week 14" in capsys.readouterr().err


def test_look_2_with_unreachable_stats_is_a_sentence_and_spends_nothing(world, monkeypatch,
                                                                       capsys):
    def _down(seasons, cache):
        raise OSError("nflverse is down")

    monkeypatch.setattr(coverage, "_stats", _down)
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) != 0
    assert not world.path.exists()


def test_look_2_with_no_player_weeks_at_all_is_an_error_not_a_look(world, monkeypatch, capsys):
    world.frame = world.frame.clear()
    assert coverage.main(["--audit", "--look", "2", "--write", "--as-of", _after(14)]) == 1
    assert "no player-week survived" in capsys.readouterr().err
    assert not world.path.exists()


# --- an unmeasurable row is a failed run that loses nothing --------------------------------


def test_a_current_row_that_cannot_be_measured_fails_the_run_and_keeps_the_last_one(
        world, monkeypatch, capsys):
    """Planted: the schedule fetch dies on the second run. Exit 1 (an unreachable source has
    always failed this command), the stderr line says why, and the file still holds the last
    committed row and the backtest -- the page degrades to last-good, not to a hole."""
    args = ["--measure", "--write", "--seasons", "2023,2024", "--as-of", _after(14)]
    assert coverage.main(args) == 0
    good = _file(world)
    assert good["current_season"]["weeks_complete"] == 14

    def _dies(seasons, cache):
        raise OSError("nflverse is down")

    monkeypatch.setattr(coverage, "_schedules", _dies)
    capsys.readouterr()
    assert coverage.main(args) == 1
    assert "could not be measured" in capsys.readouterr().err
    assert _file(world) == good, "a failed fetch must not erase the row or the backtest"
