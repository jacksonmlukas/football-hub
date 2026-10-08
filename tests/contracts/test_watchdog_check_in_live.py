"""The watchdog check is a step of the `live` loop, so it cannot be dropped on its own (#457).

Two game windows (2026-10-02 and 2026-10-06) ran with a refresher and no check, after #391,
because `watchdog.yml` was a cron on the scheduler `live` shares and GitHub drops scheduled
runs under load. The fix is structural, so the contract is on the workflow file: `live.yml`
must carry the step, in the right place, in the right mode. Without the contract the step
could be deleted in a tidy-up and every window would be back to depending on a cron.

**Untested on Actions.** There is no live CI for a workflow change. The step's shell is
executed here, with `gh` replaced by a recorder and the published artifact by a `file://` URL,
the way `test_capture_workflows_say_why.py` executes the capture steps. What this proves is
the step's logic; that the runner has `curl`, `jq` and the token is what the first window
after merge confirms.

**Rule 18.** `_problems` is the check; each `test_planted_*` builds a `live.yml` with one
property taken away and requires `_problems` to say so. A check that cannot be seen to fail
is not a contract (the same shape as `test_guards_are_load_bearing.py`).
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
STEP = "Watchdog check"
HANDOVER = "Continue while a game is in progress"
LOOP = "Refresh the overlay through the window"
SCRIPT_NAME = "watchdog-check.sh"


def _live() -> dict:
    return yaml.safe_load((WORKFLOWS / "live.yml").read_text())


def _steps(wf: dict) -> list[dict]:
    return wf["jobs"]["refresh"]["steps"]


def _names(wf: dict) -> list[str]:
    return [s.get("name", "") for s in _steps(wf)]


def _problems(wf: dict) -> list[str]:
    """Everything wrong with a `live` workflow's watchdog check. Empty means it holds."""
    out: list[str] = []
    names = _names(wf)
    if STEP not in names:
        return [f"no step named {STEP!r}: live runs would carry no watchdog check"]
    step = _steps(wf)[names.index(STEP)]
    if SCRIPT_NAME not in step.get("run", ""):
        out.append(f"the step does not run {SCRIPT_NAME}")
    if step.get("continue-on-error") is not True:
        out.append("a failed check would fail the refresh (no continue-on-error)")
    if "always()" not in str(step.get("if", "")):
        out.append("no `if: always()`: the check is skipped when the loop step fails or the "
                   "run is cancelled")
    if LOOP not in names or HANDOVER not in names:
        out.append("the loop or the hand-over step is gone")
    elif not names.index(LOOP) < names.index(STEP) < names.index(HANDOVER):
        out.append("the check is not between the loop and the hand-over (the hand-over "
                   "starts the run that cancels this one)")
    handover = _steps(wf)[names.index(HANDOVER)] if HANDOVER in names else {}
    if handover.get("if") and "failure" in str(handover["if"]):
        out.append("the hand-over is gated on the check's outcome")
    perms = wf.get("permissions", {})
    if perms.get("issues") != "write":
        out.append("the job cannot file the incident (no `issues: write`)")
    return out


def test_every_live_run_performs_the_watchdog_check():
    assert _problems(_live()) == []


def test_planted_a_live_workflow_with_the_step_removed_fails_the_contract():
    """The Rule-18 control the ticket names: a copy of `live.yml` without the step."""
    wf = copy.deepcopy(_live())
    wf["jobs"]["refresh"]["steps"] = [s for s in _steps(wf) if s.get("name") != STEP]
    got = _problems(wf)
    assert got and "no step named" in got[0], got


def test_planted_a_step_that_can_fail_the_refresh_fails_the_contract():
    wf = copy.deepcopy(_live())
    del next(s for s in _steps(wf) if s.get("name") == STEP)["continue-on-error"]
    assert any("continue-on-error" in p for p in _problems(wf))


def test_planted_a_check_after_the_hand_over_fails_the_contract():
    wf = copy.deepcopy(_live())
    steps = _steps(wf)
    check = steps.pop(_names(wf).index(STEP))
    steps.append(check)
    assert any("between the loop and the hand-over" in p for p in _problems(wf))


def test_planted_a_check_that_the_loop_failing_would_skip_fails_the_contract():
    wf = copy.deepcopy(_live())
    del next(s for s in _steps(wf) if s.get("name") == STEP)["if"]
    assert any("always()" in p for p in _problems(wf))


def test_planted_a_job_that_cannot_file_fails_the_contract():
    wf = copy.deepcopy(_live())
    wf["permissions"] = {"contents": "read", "actions": "write"}
    assert any("issues: write" in p for p in _problems(wf))


def test_the_step_is_in_the_file_as_written_not_only_in_the_parsed_copy():
    """`_problems` works on parsed YAML; this ties it to the text a reviewer reads, so a
    parser that dropped a step could not make the contract pass on its own."""
    text = (WORKFLOWS / "live.yml").read_text()
    assert re.search(rf"^      - name: {STEP}$", text, flags=re.M)
    assert SCRIPT_NAME in text


def test_the_look_back_is_its_own_daily_workflow_with_no_window_cron():
    """The independent alarm for a window in which `live` never ran. Daily, so it does not
    depend on a window cron being delivered, and not inside `live.yml`'s window crons."""
    wf = yaml.safe_load((WORKFLOWS / "watchdog-lookback.yml").read_text())
    crons = [c["cron"] for c in wf[True]["schedule"]]       # `on:` parses as the key True
    assert len(crons) == 1
    minute, hour, dom, month, dow = crons[0].split()
    assert dom == month == dow == "*" and minute.isdigit() and hour.isdigit(), crons
    assert "gaps" in wf["jobs"]
    assert "gaps" not in yaml.safe_load((WORKFLOWS / "watchdog.yml").read_text())["jobs"], (
        "the look-back is back on the window crons, where it is dropped with the check")


def test_both_workflows_hold_the_same_season_opening():
    a = yaml.safe_load((WORKFLOWS / "watchdog.yml").read_text())["env"]["SEASON_OPENS"]
    b = yaml.safe_load((WORKFLOWS / "watchdog-lookback.yml").read_text())["env"]["SEASON_OPENS"]
    assert a == b


# --- the step executed ----------------------------------------------------------------------

FAKE_GH = """#!/bin/bash
echo "$*" >> "$GH_LOG"
case "$1 $2" in
  "issue list") cat "$GH_OPEN" 2>/dev/null ;;
  *) [ -n "${GH_FAIL:-}" ] && exit 1 ;;
esac
exit 0
"""


def _iso(seconds_ago: int) -> str:
    when = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=seconds_ago)
    return when.replace(microsecond=0).isoformat()


class Harness:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        (tmp / "bin").mkdir()
        gh = tmp / "bin" / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(0o755)
        self.log = tmp / "gh.log"
        self.open = tmp / "open.txt"
        self.artifact = tmp / "live.json"

    def run(self, *, age: int | None, open_incidents: str = "", fail: bool = False
            ) -> subprocess.CompletedProcess:
        if age is not None:
            self.artifact.write_text(json.dumps({"generated_at": _iso(age)}))
        self.open.write_text(open_incidents)
        step = next(s for s in _steps(_live()) if s.get("name") == STEP)
        env = {"PATH": f"{self.tmp / 'bin'}:{os.environ['PATH']}", "HOME": str(self.tmp),
               "GH_LOG": str(self.log), "GH_OPEN": str(self.open),
               "REPO": "o/r", "HEARTBEAT_URL": f"file://{self.artifact}"}
        if fail:
            env["GH_FAIL"] = "1"
        return subprocess.run(["bash", "-ec", step["run"]], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def test_a_stale_heartbeat_files_the_incident_and_the_step_still_exits_zero(harness):
    """The ticket's semantics control: a failed check reports and does not fail the refresh."""
    got = harness.run(age=4000)
    assert got.returncode == 0, got.stderr
    created = [c for c in harness.calls() if c.startswith("issue create")]
    assert created and "<!-- watchdog-heartbeat -->" in created[0]
    assert "Live poller down" in created[0]


def test_a_stale_heartbeat_updates_the_open_incident_instead_of_filing_another(harness):
    got = harness.run(age=4000, open_incidents="41\n")
    assert got.returncode == 0, got.stderr
    calls = harness.calls()
    assert any(c.startswith("issue comment 41") for c in calls), calls
    assert not any(c.startswith("issue create") for c in calls), calls


def test_a_healthy_heartbeat_closes_only_the_incident_it_marks(harness):
    got = harness.run(age=30, open_incidents="41\n")
    assert got.returncode == 0, got.stderr
    closed = [c for c in harness.calls() if c.startswith("issue close")]
    assert [c.split()[2] for c in closed] == ["41"]
    # the listing asks for the marker; a human's issue is not in that listing
    assert any("watchdog-heartbeat" in c for c in harness.calls() if c.startswith("issue list"))


def test_a_healthy_heartbeat_with_nothing_open_does_nothing(harness):
    got = harness.run(age=30)
    assert got.returncode == 0 and not any(
        c.startswith(("issue close", "issue create", "issue comment")) for c in harness.calls())


def test_an_unreachable_artifact_is_an_incident_not_a_crash(harness):
    got = harness.run(age=None)                     # no file at the URL
    assert got.returncode == 0, got.stderr
    text = harness.log.read_text()          # the body spans lines; the log is whole
    assert "issue create" in text
    assert "could not be fetched" in text and "unknown" in text


def test_a_check_that_cannot_file_is_red_but_the_job_is_not_failed(harness):
    """The step exits 1 -- visible -- and the workflow marks it `continue-on-error`, so the
    refresh and the hand-over after it are unaffected. Both halves are asserted."""
    got = harness.run(age=4000, fail=True)
    assert got.returncode == 1, "the failure to report must be visible, not swallowed"
    assert "could not be filed" in got.stdout
    steps = _steps(_live())
    check = next(s for s in steps if s.get("name") == STEP)
    handover = next(s for s in steps if s.get("name") == HANDOVER)
    assert check["continue-on-error"] is True
    assert "if" not in handover, "the hand-over runs on success(), which a tolerated step keeps"
