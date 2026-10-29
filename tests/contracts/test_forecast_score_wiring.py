"""The interim leaderboard workflow is gated and writes only the `forecasts` branch (#453).

As `test_forecast_close_wiring.py` does, these read the workflow file for the properties that
matter, since a workflow change has no live CI: gated by `vars.LLM_FORECASTS_ENABLED`, a
Tuesday cron, and every write through `commit-state.sh` pointed at `forecasts`, never main.
"""
from __future__ import annotations

from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "forecast-score.yml"
DOC = yaml.safe_load(WF.read_text())
JOB = DOC["jobs"]["score"]
STEPS = {s["name"]: s for s in JOB["steps"] if "name" in s}


def test_gated_by_the_studys_variable():
    assert JOB["if"] == "vars.LLM_FORECASTS_ENABLED == 'true'"


def test_tuesday_cron_and_manual_dispatch():
    on = DOC[True] if True in DOC else DOC["on"]               # YAML reads `on:` as True
    crons = [c["cron"] for c in on["schedule"]]
    assert len(crons) == 1 and crons[0].split()[-1] == "2"
    assert "workflow_dispatch" in on


def test_everything_is_written_to_forecasts_and_never_main():
    commit = STEPS["Commit to the forecasts branch"]
    assert commit["env"] == {"STATE_REPO": "forecasts-branch", "STATE_PATHS": "leaderboard",
                             "STATE_REF": "forecasts"}
    assert commit["run"].startswith(".github/scripts/commit-state.sh")
    text = WF.read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("origin main", "HEAD:main", "git push", "STATE_REF: main",
                      "GITHUB_REF_NAME"):
        assert forbidden not in code
    score_run = STEPS["Score the finished weeks"]["run"]
    assert "--out forecasts-branch/" in score_run and "--forecasts forecasts-branch" in score_run


def test_refusals_are_a_warning_not_a_red_run():
    run = STEPS["Score the finished weeks"]["run"]
    assert "3|4|5)" in run and "::warning::" in run and '*) exit "$rc"' in run
