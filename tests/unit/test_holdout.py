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
    got = holdout.load(2024, root=tmp_path)
    assert got.season == 2024
    assert got.values["predict.WEEKLY_K"] == {"QB": 1.9, "RB": 2.1}
    assert got.values["predict.TEAMMATE_RHO"] == {("QB", "WR"): 0.21}   # tuple keys survive
    assert got.commands["predict.WEEKLY_K"].startswith("fit_weekly_spread.py")
    raw = json.loads((tmp_path / "2024.json").read_text())
    assert raw["excluded_season"] == 2024 and set(raw["constants"]) == {"predict.WEEKLY_K", "predict.TEAMMATE_RHO"}


def test_a_key_that_is_not_a_held_out_constant_is_refused(tmp_path):
    with pytest.raises(KeyError, match="not one of the held-out constants"):
        holdout.record(2024, "predict.MIN_SKEW", 0.1, command="x", root=tmp_path)


def test_a_set_is_applied_for_the_duration_and_the_shipped_value_comes_back(tmp_path):
    """The gate rebinds the module attribute inside `applied` and nothing outside it: the
    shipped constant, and with it the shipped digest, is what every other reader sees."""
    shipped, before = predict.TALENT_CV, fitted_digest()
    holdout.record(2024, "predict.TALENT_CV", 0.99, command="x", root=tmp_path)
    with holdout.applied(2024, root=tmp_path) as said:
        assert predict.TALENT_CV == 0.99
        assert fitted_digest() != before
        assert "season 2024" in said and "predict.TALENT_CV" in said and "refitted" in said
    assert predict.TALENT_CV == shipped
    assert fitted_digest() == before


def test_the_run_line_names_what_was_refitted_and_what_stayed_shipped(tmp_path):
    """A set that refits some of the eight and not the rest says which is which, with the
    reason a key was not refitted carried from the file."""
    holdout.record(2024, "predict.WEEKLY_K", {"QB": 1.9}, command="x", root=tmp_path)
    holdout.record(2024, "availability.PICK_NOISE_SLOPE", None, command="scripts/fit_pick_noise.py",
                   root=tmp_path, why_not="needs an ESPN session; the 2026-09-13 lane had none")
    got = holdout.load(2024, root=tmp_path)
    said = holdout.describe(got)
    assert "refitted: predict.WEEKLY_K" in said
    assert "shipped (not refitted): " in said and "availability.PICK_NOISE_SLOPE" in said
    assert "needs an ESPN session" in said
    # Every held-out constant the file never mentions is shipped too, and is listed.
    assert "predict.TALENT_CV" in said


def test_a_season_with_no_set_is_refused_rather_than_silently_shipped(tmp_path):
    """Replaying "under hold-out" on a season nobody fitted a set for would be the shipped
    run wearing the hold-out's name -- and a gate asks for every season's line before it
    plays a draft, so the refusal comes first."""
    with pytest.raises(FileNotFoundError, match="no hold-out constant set for 2019"):
        holdout.load(2019, root=tmp_path)
    holdout.record(2024, "predict.TALENT_CV", 0.91, command="x", root=tmp_path)
    with pytest.raises(FileNotFoundError, match="no hold-out constant set for 2025"):
        holdout.run_lines([2024, 2025], root=tmp_path)
    lines = holdout.run_lines([2024], root=tmp_path)
    assert len(lines) == 1 and "season 2024" in lines[0] and "predict.TALENT_CV" in lines[0]


def test_applied_restores_after_an_exception(tmp_path):
    shipped = predict.TALENT_CV
    holdout.record(2024, "predict.TALENT_CV", 0.5, command="x", root=tmp_path)
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
    got = holdout.load(2024)
    assert got.values == {"predict.WEEKLY_K_POOLED": 2.0} and got.missing == {"predict.TALENT_CV": "no session"}
    assert got.commands["predict.TALENT_CV"] == "uv run python scripts/fit_x.py --exclude-season 2024 --record"
    assert "recorded predict.WEEKLY_K_POOLED into" in capsys.readouterr().out


def test_a_set_that_lies_about_its_season_or_carries_a_stray_key_is_refused(tmp_path):
    holdout.record(2024, "predict.TALENT_CV", 0.5, command="x", root=tmp_path)
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
