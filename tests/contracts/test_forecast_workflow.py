"""`.github/workflows/forecast.yml` holds to the prereg's *Forecast procedure*.

Every 15 minutes 13:30-16:30 UTC (the same list as `hub.forecasters.days.CRON_TRIGGERS_UTC`),
gated in code to the 09:30-11:30 ET window; a concurrency group; output saved as an artifact
*before* it is committed; committed only to the `forecasts` branch; and switched off until
`prereg-llm-v0` is tagged, by a repository variable rather than by commenting out the schedule
(a commented-out schedule is what `scripts/preflight_public.sh` warns about, and it would also
leave this test nothing to check).
"""
from __future__ import annotations

import re
from pathlib import Path

from hub.forecasters import days

ROOT = Path(__file__).resolve().parents[2]
WF = (ROOT / ".github" / "workflows" / "forecast.yml").read_text()


def _crons() -> list[str]:
    return re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', WF, re.M)


def _expand(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        if part.startswith("*/"):
            out |= set(range(lo, hi + 1, int(part[2:])))
        elif "-" in part:
            a, b = map(int, part.split("-"))
            out |= set(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def test_the_crons_are_exactly_the_declared_triggers():
    fired: set[tuple[int, int]] = set()
    for cron in _crons():
        minute, hour, dom, month, dow = cron.split()
        assert (dom, month, dow) == ("*", "*", "*"), cron
        fired |= {(h, m) for h in _expand(hour, 0, 23) for m in _expand(minute, 0, 59)}
    assert fired == set(days.CRON_TRIGGERS_UTC)


def test_every_trigger_goes_through_the_gate_before_anything_else_runs():
    gate = WF.index("hub.forecasters.cli should-run")
    run = WF.index("hub.forecasters.cli run")
    assert gate < run
    assert "steps.gate.outputs.run == 'true'" in WF


def test_one_run_at_a_time():
    assert re.search(r"^concurrency:\n\s+group: llm-forecast\n\s+cancel-in-progress: false", WF, re.M)


def test_switched_off_until_v0_by_a_variable():
    assert "vars.LLM_FORECASTS_ENABLED == 'true'" in WF


def test_the_artifact_is_saved_before_the_commit():
    assert WF.index("actions/upload-artifact") < WF.index("git -C forecasts-branch commit")


def test_it_commits_to_the_forecasts_branch_and_nowhere_else():
    pushes = re.findall(r"git[^\n]*\bpush\b[^\n]*", WF)
    assert pushes and all("forecasts-branch" in p and "origin forecasts" in p for p in pushes)
    assert "git add state" not in WF and "git -C . commit" not in WF


def test_the_day_is_marked_done_only_after_the_run_succeeds():
    assert WF.index("hub.forecasters.cli run") < WF.index("hub.forecasters.cli mark-done")


def test_no_key_value_is_written_in_the_file():
    for line in WF.splitlines():
        if re.search(r"_KEY\s*:", line):
            assert "${{ secrets." in line, line


def _steps() -> dict[str, dict]:
    import yaml
    jobs = yaml.safe_load(WF)["jobs"]
    return {s.get("name", s.get("uses", "")): s for s in jobs["forecast"]["steps"]}


def test_forecasting_and_marking_done_run_only_past_the_gate_and_only_on_success():
    steps = _steps()
    for name in ("Forecast the day's games", "Mark the day done"):
        assert steps[name]["if"] == "steps.gate.outputs.run == 'true'", name


def test_saving_and_committing_run_past_the_gate_even_after_a_failure():
    steps = _steps()
    for name in ("Save the output as an artifact", "Commit to the forecasts branch"):
        assert steps[name]["if"] == "always() && steps.gate.outputs.run == 'true'", name
