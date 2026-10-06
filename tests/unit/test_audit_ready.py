"""`hub.audit_ready` (#417): is every game in a week's published predictions scored?

Rule 18: each control plants the condition the check exists to detect. The first is the
2026-10-03 shape that Audit V's prose test passed -- week 4 predicted, 1 of 16 scored -- and the
check must fail it and name the other fifteen. Fixtures are written into `tmp_path`; nothing
here reads `site/data` or `data/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hub import audit_ready

WEEK4 = [f"2026_04_G{i:02d}" for i in range(16)]
EARLIER = [f"2026_0{w}_G{i:02d}" for w in (1, 2, 3) for i in range(16)]


def _site(tmp_path: Path, scored, *, field=True, season=2026, preds=WEEK4) -> Path:
    (tmp_path / "preds_2026_wk04.json").write_text(json.dumps(
        {"name": "preds_2026_wk04", "rows": [{"game_id": g} for g in preds]}))
    entry = {"season": season, "n_scored": len(scored)}
    if field:
        entry["scored_game_ids"] = sorted(scored)
    (tmp_path / "track_record.json").write_text(json.dumps(
        {"name": "track_record", "n_scored": len(scored), "seasons": [entry]}))
    return tmp_path


def _run(site: Path, capsys) -> tuple[int, str]:
    code = audit_ready.main(["--season", "2026", "--week", "4", "--site", str(site)])
    seen = capsys.readouterr()
    return code, seen.out + seen.err


def test_a_10_03_shaped_record_fails_and_names_the_fifteen(tmp_path, capsys):
    """The failure Audit V's prose test passed: week 4 predicted, one game scored."""
    site = _site(tmp_path, EARLIER + WEEK4[:1])
    code, said = _run(site, capsys)
    assert code == audit_ready.UNSCORED
    for g in WEEK4[1:]:
        assert f"unscored: {g}" in said
    assert f"unscored: {WEEK4[0]}" not in said
    assert said.count("unscored: ") == 15


def test_a_fully_scored_week_passes(tmp_path, capsys):
    code, said = _run(_site(tmp_path, EARLIER + WEEK4), capsys)
    assert code == 0 and "READY" in said and "unscored" not in said


def test_a_record_that_predates_the_field_refuses_rather_than_passes(tmp_path, capsys):
    """Today's `track_record.json`: counts scored games per season, cannot say which."""
    code, said = _run(_site(tmp_path, EARLIER + WEEK4, field=False), capsys)
    assert code == audit_ready.REFUSED
    assert "predates `scored_game_ids`" in said


def test_a_season_with_no_entry_has_scored_none_of_its_games(tmp_path, capsys):
    code, said = _run(_site(tmp_path, EARLIER, season=2025), capsys)
    assert code == audit_ready.UNSCORED and said.count("unscored: ") == 16


def test_a_missing_preds_file_is_a_named_refusal(tmp_path, capsys):
    code, said = _run(tmp_path, capsys)
    assert code == audit_ready.REFUSED and "preds_2026_wk04.json" in said


def test_an_unreadable_record_is_a_named_refusal(tmp_path, capsys):
    site = _site(tmp_path, WEEK4)
    (site / "track_record.json").write_text("{not json")
    code, said = _run(site, capsys)
    assert code == audit_ready.REFUSED and "not valid JSON" in said


@pytest.mark.parametrize("preds_body", [{"rows": []}, {"rows": [{"home": "x"}]}, []])
def test_a_preds_file_that_names_no_games_is_a_refusal(tmp_path, capsys, preds_body):
    site = _site(tmp_path, WEEK4)
    (site / "preds_2026_wk04.json").write_text(json.dumps(preds_body))
    code, _ = _run(site, capsys)
    assert code == audit_ready.REFUSED


def test_a_record_with_no_season_entries_is_a_refusal(tmp_path, capsys):
    site = _site(tmp_path, WEEK4)
    (site / "track_record.json").write_text(json.dumps({"seasons": []}))
    code, _ = _run(site, capsys)
    assert code == audit_ready.REFUSED


def test_the_module_entry_point_is_the_cli_the_ticket_names(tmp_path, monkeypatch):
    """`python -m hub.audit_ready` exits with `main`'s code, so a shell can gate on it."""
    import runpy
    import sys
    monkeypatch.setattr(sys, "argv", ["hub.audit_ready", "--season", "2026", "--week", "4",
                                      "--site", str(_site(tmp_path, EARLIER + WEEK4[:1]))])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("hub.audit_ready", run_name="__main__")
    assert e.value.code == audit_ready.UNSCORED
