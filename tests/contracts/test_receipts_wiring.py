"""Timestamp receipts are stamped only by the gated forecast workflow, on `forecasts` (#449)."""
from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
NEEDLE = "hub.forecasters.receipts"


def test_only_the_gated_forecast_workflow_stamps():
    users = sorted(p.name for p in WORKFLOWS.glob("*.yml") if NEEDLE in p.read_text())
    assert users == ["forecast.yml"]


def test_the_stamp_step_is_gated_best_effort_and_before_the_commit():
    wf = (WORKFLOWS / "forecast.yml").read_text()
    step = wf[wf.index("- name: Stamp the forecast files"):]
    step = step[:step.index("- name: Save the output")]
    assert "steps.gate.outputs.run == 'true'" in step      # same window gate as the run
    assert "continue-on-error: true" in step               # an outage never fails the day
    assert "forecasts-branch" in step and "main" not in step.split("run:")[1]
    assert wf.index("hub.forecasters.cli mark-done") < wf.index(NEEDLE) \
        < wf.index("git -C forecasts-branch add -A")
