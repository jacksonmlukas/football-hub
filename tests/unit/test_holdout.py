"""`hub.holdout`: a per-season constant set, fitted without that season, that a gate can
replay under (#294).

The eight fitted constants the draft gate's LIMITATIONS name were fitted on the seasons
the gate replays. A hold-out set is what a fitting script writes with `--exclude-season N`;
the gate reads it for season N instead of the shipped constants, says so on its run line,
and the shipped constants -- and every digest on the default path -- are untouched.
"""
from __future__ import annotations

import json

import pytest

from hub import holdout
from hub.config import fitted_digest
from hub.models import predict


def test_a_script_records_one_constant_at_a_time_and_the_file_keeps_the_rest(tmp_path):
    """Five scripts write one file per season, each its own keys; a later record must not
    erase an earlier script's. The command that produced each value travels with it."""
    holdout.record(2024, "predict.WEEKLY_K", {"QB": 1.9, "RB": 2.1}, command="fit_weekly_spread.py --exclude-season 2024",
                   root=tmp_path)
    holdout.record(2024, "predict.TEAMMATE_RHO", {("QB", "WR"): 0.21},
                   command="fit_teammate_rho.py --exclude-season 2024", root=tmp_path)
    raw = json.loads((tmp_path / "2024.json").read_text())
    assert raw["excluded_season"] == 2024 and set(raw["constants"]) == {"predict.WEEKLY_K", "predict.TEAMMATE_RHO"}
    # The set is complete only when every script has run; the rest are recorded as not refitted.
    for key in holdout.HELD_OUT:
        if key not in raw["constants"]:
            holdout.record(2024, key, None, command="x", root=tmp_path, why_not="not fitted here")
    got = holdout.load(2024, root=tmp_path)
    assert got.season == 2024
    assert got.values["predict.WEEKLY_K"] == {"QB": 1.9, "RB": 2.1}
    assert got.values["predict.TEAMMATE_RHO"] == {("QB", "WR"): 0.21}   # tuple keys survive
    assert got.commands["predict.WEEKLY_K"].startswith("fit_weekly_spread.py")


def test_a_key_that_is_not_a_held_out_constant_is_refused(tmp_path):
    with pytest.raises(KeyError, match="not one of the held-out constants"):
        holdout.record(2024, "predict.MIN_SKEW", 0.1, command="x", root=tmp_path)


def test_a_set_is_applied_for_the_duration_and_the_shipped_value_comes_back(
        tmp_path, record_holdout_set):
    """The gate rebinds the module attribute inside `applied` and nothing outside it: the
    shipped constant, and with it the shipped digest, is what every other reader sees."""
    shipped, before = predict.TALENT_CV, fitted_digest()
    record_holdout_set(2024, {"predict.TALENT_CV": 0.99}, root=tmp_path)
    with holdout.applied(2024, root=tmp_path) as said:
        assert predict.TALENT_CV == 0.99
        assert fitted_digest() != before
        assert "season 2024" in said and "predict.TALENT_CV" in said and "refitted" in said
    assert predict.TALENT_CV == shipped
    assert fitted_digest() == before


def test_the_run_line_names_what_was_refitted_and_what_stayed_shipped(tmp_path, record_holdout_set):
    """A set that refits some of the eight and not the rest says which is which, with the
    reason a key was not refitted carried from the file."""
    record_holdout_set(2024, {"predict.WEEKLY_K": {"QB": 1.9}}, root=tmp_path)
    holdout.record(2024, "availability.PICK_NOISE_SLOPE", None, command="scripts/fit_pick_noise.py",
                   root=tmp_path, why_not="needs an ESPN session; the 2026-09-13 lane had none")
    got = holdout.load(2024, root=tmp_path)
    said = holdout.describe(got)
    assert "refitted: predict.WEEKLY_K" in said
    assert "shipped (not refitted): " in said and "availability.PICK_NOISE_SLOPE" in said
    assert "needs an ESPN session" in said
    # Every other held-out constant is recorded as not refitted, and is listed.
    assert "predict.TALENT_CV" in said


def test_a_season_with_no_set_is_refused_rather_than_silently_shipped(tmp_path, record_holdout_set):
    """Replaying "under hold-out" on a season nobody fitted a set for would be the shipped
    run wearing the hold-out's name -- and a gate asks for every season's line before it
    plays a draft, so the refusal comes first."""
    with pytest.raises(FileNotFoundError, match="no hold-out constant set for 2019"):
        holdout.load(2019, root=tmp_path)
    record_holdout_set(2024, {"predict.TALENT_CV": 0.91}, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="no hold-out constant set for 2025"):
        holdout.run_lines([2024, 2025], root=tmp_path)
    lines = holdout.run_lines([2024], root=tmp_path)
    assert len(lines) == 1 and "season 2024" in lines[0] and "predict.TALENT_CV" in lines[0]


def test_applied_restores_after_an_exception(tmp_path, record_holdout_set):
    shipped = predict.TALENT_CV
    record_holdout_set(2024, {"predict.TALENT_CV": 0.5}, root=tmp_path)
    with pytest.raises(RuntimeError), holdout.applied(2024, root=tmp_path):
        assert predict.TALENT_CV == 0.5
        raise RuntimeError("a season blew up")
    assert predict.TALENT_CV == shipped


def test_a_scripts_recorder_writes_only_under_record_and_needs_the_season(tmp_path, monkeypatch, capsys):
    """The seam every fitting script uses: `--record` without `--exclude-season` is refused
    up front, without `--record` the recorder is a no-op, and with both it writes the key
    under the command line that names the script and the season."""
    import argparse

    monkeypatch.setattr(holdout, "SETS", tmp_path)
    ap = argparse.ArgumentParser()
    holdout.add_arguments(ap)
    with pytest.raises(SystemExit, match="--record needs --exclude-season"):
        holdout.recording(ap.parse_args(["--record"]), "x")
    quiet = holdout.recording(ap.parse_args([]), "x")
    quiet("predict.WEEKLY_K_POOLED", 2.0)
    assert not list(tmp_path.glob("*.json"))
    a = ap.parse_args(["--exclude-season", "2024", "--record"])
    note = holdout.recording(a, holdout.command_line("scripts/fit_x.py", a))
    note("predict.WEEKLY_K_POOLED", 2.0)
    note("predict.TALENT_CV", None, why_not="no session")
    raw = json.loads((tmp_path / "2024.json").read_text())["constants"]
    assert raw["predict.WEEKLY_K_POOLED"]["value"] == 2.0
    assert raw["predict.TALENT_CV"]["why_not"] == "no session"
    assert raw["predict.TALENT_CV"]["command"] == "uv run python scripts/fit_x.py --exclude-season 2024 --record"
    assert "recorded predict.WEEKLY_K_POOLED into" in capsys.readouterr().out


def test_a_set_that_lies_about_its_season_or_carries_a_stray_key_is_refused(tmp_path, record_holdout_set):
    record_holdout_set(2024, {"predict.TALENT_CV": 0.5}, root=tmp_path)
    raw = json.loads((tmp_path / "2024.json").read_text())
    raw["excluded_season"] = 2023
    (tmp_path / "2024.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="says excluded_season=2023"):
        holdout.load(2024, root=tmp_path)
    raw["excluded_season"] = 2024
    raw["constants"]["predict.MIN_SKEW"] = {"value": 0.1, "command": "x"}
    (tmp_path / "2024.json").write_text(json.dumps(raw))
    with pytest.raises(KeyError, match="not a held-out constant"):
        holdout.load(2024, root=tmp_path)
    with pytest.raises(ValueError, match="needs why_not"):
        holdout.record(2024, "predict.TALENT_CV", None, command="x", root=tmp_path)


def test_the_committed_sets_cover_every_season_the_draft_gate_replays():
    """One file per replayed season, each naming its excluded season and a command for every
    key it holds; a key with no value carries why."""
    for season in (2022, 2023, 2024, 2025):
        got = holdout.load(season)
        assert got.season == season
        for key in holdout.HELD_OUT:
            assert key in got.values or key in got.missing, (season, key)
            if key in got.values:
                assert got.commands[key], (season, key)


def test_a_set_that_leaves_a_held_out_key_out_is_refused_and_names_it(tmp_path, record_holdout_set):
    """A file that omits a key used to load and be described as "shipped, not refitted" with
    no reason -- the shipped constant under the hold-out's name (#320)."""
    record_holdout_set(2024, {}, root=tmp_path)
    raw = json.loads((tmp_path / "2024.json").read_text())
    del raw["constants"]["predict.IMPUTE_CV"]
    (tmp_path / "2024.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"does not account for predict\.IMPUTE_CV"):
        holdout.load(2024, root=tmp_path)
    with pytest.raises(ValueError, match=r"does not account for predict\.IMPUTE_CV"):
        holdout.run_lines([2024], root=tmp_path)


def test_a_key_with_neither_a_value_nor_a_reason_is_refused(tmp_path, record_holdout_set):
    record_holdout_set(2024, {}, root=tmp_path)
    raw = json.loads((tmp_path / "2024.json").read_text())
    raw["constants"]["predict.WEEKLY_K"]["why_not"] = "  "
    (tmp_path / "2024.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"predict\.WEEKLY_K has no value and no why_not"):
        holdout.load(2024, root=tmp_path)


def test_scripts_recording_into_one_season_at_once_lose_no_key(tmp_path, monkeypatch):
    """The record is a read-modify-write. Widen the window between the read and the write
    (the read sleeps after it has read) and run every key's recorder at once: without the
    lock each one reads the same empty file and the last rename keeps only its own key."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    real_read = holdout._read

    def slow_read(path):
        got = real_read(path)
        time.sleep(0.05)
        return got
    monkeypatch.setattr(holdout, "_read", slow_read)
    gate = threading.Barrier(len(holdout.HELD_OUT))

    def rec(key):
        gate.wait()
        holdout.record(2024, key, None, command="x", root=tmp_path, why_not="a reason")
    with ThreadPoolExecutor(len(holdout.HELD_OUT)) as pool:
        list(pool.map(rec, holdout.HELD_OUT))
    kept = json.loads((tmp_path / "2024.json").read_text())["constants"]
    assert set(kept) == set(holdout.HELD_OUT)


def test_a_function_that_reads_a_held_out_constant_sees_the_set_inside_the_block_and_not_after(
        tmp_path, record_holdout_set):
    """The attribute and the digest moving is not the property: what matters is that code
    *reading* the constant sees it. Three functions, one per way a reader used to hold a
    stale copy -- `weekly_skew_for` (reads the declaring module), `calibrate._k_of` (took a
    copy of a copy through `hub.draft.season`) and the leverage exhibit's spread (an array
    computed at import). Each returns a different value inside the block than outside."""
    import numpy as np

    from hub.draft import calibrate
    from hub.exhibits import leverage

    pos = np.array(["QB", "WR"])

    def read():
        return (predict.weekly_skew_for(pos).tolist(), calibrate._k_of(pos),
                leverage.position_sd().tolist())
    before = read()
    record_holdout_set(2024, {
        "predict.WEEKLY_SKEW": {"QB": 0.01, "WR": 0.02},
        "predict.WEEKLY_K": {"QB": 9.0, "WR": 9.5, "RB": 9.0, "TE": 9.0},
        "predict.WEEKLY_K_POOLED": 8.0}, root=tmp_path)
    with holdout.applied(2024, root=tmp_path):
        inside = read()
        assert inside[0] == [0.01, 0.02]
        assert inside[1] == 8.0                  # mixed positions: the pooled k, rebound
        assert inside[2] != before[2] and inside[2][0] == pytest.approx(9.0 * np.sqrt(19.0))
    assert read() == before


def test_a_held_out_impute_cv_reaches_talent_cv_for_inside_the_block_and_not_outside(
        tmp_path, record_holdout_set):
    """Rule-18 control for #320's option B: `talent_cv_for` adds `IMPUTE_CV_BY_POS.get(pos,
    IMPUTE_CV)` to the talent spread of an imputed player, so a hold-out that binds those two
    must change what it returns -- inside the block, to the planted per-position values and
    to the planted pooled value for a position the table does not key, and back to shipped
    after. A hold-out that left them declined would return the shipped answer in all three."""
    import numpy as np

    pos = np.array(["QB", "RB", "WR", "TE", "K"])
    imputed = np.ones(5, dtype=bool)
    shipped = predict.talent_cv_for(pos, imputed)
    planted = {"QB": 0.91, "RB": 0.92, "WR": 0.93, "TE": 0.94}
    record_holdout_set(2024, {"predict.IMPUTE_CV": 0.95,
                              "predict.IMPUTE_CV_BY_POS": planted}, root=tmp_path)
    with holdout.applied(2024, root=tmp_path):
        inside = predict.talent_cv_for(pos, imputed)
        observed = predict.talent_cv_for(pos, np.zeros(5, dtype=bool))
    base = np.array([predict.TALENT_CV_BY_POS.get(str(p), predict.TALENT_CV) for p in pos])
    expect_extra = np.array([planted["QB"], planted["RB"], planted["WR"], planted["TE"], 0.95])
    assert not np.allclose(inside, shipped)
    assert inside == pytest.approx(np.hypot(base, expect_extra))
    assert observed == pytest.approx(base), "an observed player carries no imputation error"
    assert predict.talent_cv_for(pos, imputed) == pytest.approx(shipped)


def test_nothing_in_the_forward_arms_closure_reads_the_imputation_error():
    """Why binding `IMPUTE_CV` under hold-out (#320) leaves the weekly forward arm computing
    what it did (#456's pin). `talent_cv_for` reads `IMPUTE_CV_BY_POS` only when handed
    `imputed`, so the arm can see the binding only through a call that passes it. Walk the
    arm's pinned closure and find every call of `talent_cv_for` in it: none passes a second
    argument. Positive control: the walk does find the one call that does (the
    championship-equity exhibit, outside the closure), so a vacuous walk would fail here."""
    import ast

    from hub.closure import import_closure, module_source
    from hub.season.weekly_forward import ARM_ROOTS

    def imputed_calls(modules):
        found = []
        for mod in modules:
            for node in ast.walk(ast.parse(module_source(mod).read_text())):
                if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "talent_cv_for"
                        and (len(node.args) > 1 or any(k.arg == "imputed" for k in node.keywords))):
                    found.append(mod)
        return found

    assert imputed_calls(sorted(import_closure(ARM_ROOTS))) == []
    assert "hub.exhibits.championship_equity" not in import_closure(ARM_ROOTS)
    assert imputed_calls(["hub.exhibits.championship_equity"]) == [
        "hub.exhibits.championship_equity"], "the control: this walk sees such a call"


@pytest.mark.parametrize("key", ["predict.IMPUTE_CV", "predict.IMPUTE_CV_BY_POS"])
def test_a_set_missing_either_impute_key_is_refused_naming_exactly_that_key(
        tmp_path, record_holdout_set, key):
    """Rule-18 control: deleting one of the two keys alone fails `load` and names that key and
    not its prefix-sharing neighbour (the message ends the name with a colon)."""
    record_holdout_set(2024, {}, root=tmp_path)
    raw = json.loads((tmp_path / "2024.json").read_text())
    del raw["constants"][key]
    (tmp_path / "2024.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError) as e:
        holdout.load(2024, root=tmp_path)
    assert f"does not account for {key}: " in str(e.value)
