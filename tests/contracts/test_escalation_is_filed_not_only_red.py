"""`escalate: true` files an incident before it fails the run (#498).

The red step alone went unseen for ten days: it fired once and the odds-only polls that followed
were green. This holds `slate.yml` to filing the incident (with the permission to), ahead of the
red step, and not soft-failing the red step.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SLATE = ROOT / ".github" / "workflows" / "slate.yml"


def _steps():
    wf = yaml.safe_load(SLATE.read_text())
    return wf["jobs"]["refresh"], wf["jobs"]["refresh"]["steps"]


def test_the_refresh_job_can_file_an_issue():
    job, _ = _steps()
    assert job["permissions"].get("issues") == "write" and job["permissions"].get("contents") == "write"


def test_the_incident_is_filed_before_the_red_step_and_both_run_always():
    _, steps = _steps()
    names = [s.get("name", "") for s in steps]
    file_i = next(i for i, n in enumerate(names) if n.startswith("File or close the escalation"))
    red_i = next(i for i, n in enumerate(names) if n.startswith("Escalate a CFBD source"))
    assert file_i < red_i
    assert steps[file_i]["if"] == "always()" and steps[red_i]["if"] == "always()"
    assert "hub.escalate" in steps[file_i]["run"] and "gh issue create" in steps[file_i]["run"]
    assert "gh issue close" in steps[file_i]["run"]
    assert "continue-on-error" not in steps[red_i]
