"""The watchdog's Big Ten half: a stalled capture files an incident the way a stalled
overlay does (#240).

`site/data/bigten.json` carries `generated_at` in the same envelope shape `live.json` does,
and until this the watchdog's window logic was the live overlay's alone -- a capture that
silently stopped (the cron disabled, the page's shape changed at every deadline, the job
failing before it commits) left nothing that anyone would be told about. The Big Ten job
below measures the stamp against the deadline it should follow, inside a window
`hub.fetch.bigten.watch` defines, and outside every window says so rather than measuring.

Tested the way `test_game_windows.py` tests the heartbeat job: the step's shell is executed
with its expressions bound, rather than grepped for. The live job's own tests are untouched
and still pass, which is the third acceptance criterion.
"""
import re
import subprocess
from pathlib import Path

import yaml

WATCHDOG = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "watchdog.yml"
JOB = "bigten"
VERDICT = "What this run can conclude about the capture"


def _steps() -> dict[str, dict]:
    jobs = yaml.safe_load(WATCHDOG.read_text())["jobs"]
    assert JOB in jobs, f"watchdog.yml has no `{JOB}` job; it watches live.json alone"
    return {s["name"]: s for s in jobs[JOB]["steps"] if "name" in s}


def _run_step(name: str, **expressions: str) -> dict[str, str]:
    script = _steps()[name]["run"]

    def bind(m: re.Match) -> str:
        key = m.group(1).strip()
        assert key in expressions, f"{name!r} reads {key!r}, which this test does not bind"
        return expressions[key]

    script = re.sub(r"\$\{\{([^}]+)\}\}", bind, script)
    out = Path(subprocess.run(["mktemp"], capture_output=True, text=True,
                              check=True).stdout.strip())
    got = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                         env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                              "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": "/dev/null"})
    assert got.returncode == 0, got.stderr
    parsed: dict[str, str] = {}
    lines = out.read_text().splitlines()
    while lines:
        key, _, value = lines.pop(0).partition("=")
        if "<<" in key:
            key, _, delim = key.partition("<<")
            value = ""
            while lines and lines[0] != delim:
                value += lines.pop(0) + "\n"
            if lines:
                lines.pop(0)
        parsed[key] = value.strip()
    return parsed


def _verdict(line: str) -> dict[str, str]:
    return _run_step(VERDICT, **{"steps.watch.outputs.line": line,
                                 "steps.watch.outputs.at": "18:11:07Z"})


def test_the_job_reads_the_committed_stamp_through_the_module_that_owns_the_deadlines():
    """The seam. The window is a fact about `bigten.DEADLINES`, so the check has to be the
    module's own rather than a second statement of the schedule in shell -- the drift
    `window_state.sh` reads the cron string to avoid."""
    steps = _steps()
    watch = next(s for n, s in steps.items() if n.startswith("Which Big Ten"))
    assert "hub.fetch.bigten --watch" in watch["run"], (
        "the job does not ask `hub.fetch.bigten` which window it is in")
    assert "site/data/bigten.json" in yaml.safe_dump(steps), (
        "the job never names the stamp it is watching")


def test_a_stale_stamp_inside_a_window_is_an_incident_and_a_fresh_one_closes_it():
    got = _verdict("inside 2026-10-03T1500Z stale 46860")
    assert got["verdict"] == "stale" and got["closes"] == "false", got
    assert "2026-10-03T1500Z" in got["why"] and "46860" in got["why"], got["why"]
    got = _verdict("inside 2026-10-03T1500Z ok 5700")
    assert got["verdict"] == "healthy" and got["closes"] == "true", got
    assert got["close_reason"] == "completed"
    assert "2026-10-03T1500Z" in got["close_note"]


def test_a_fresh_stamp_that_archived_no_report_is_an_incident_with_the_reason():
    """#278: written on time and kept nothing is the stall wearing a fresh date. It files
    like a stall and says so, rather than being read as healthy or as predating the deadline."""
    got = _verdict("inside 2026-10-03T1500Z empty 5700")
    assert got["verdict"] == "stale" and got["closes"] == "false", got
    assert "no report document" in got["why"] and "2026-10-03T1500Z" in got["why"], got
    assert "predates" not in got["why"], "the stamp is fresh; that is the point"


def test_outside_every_window_the_run_reports_that_rather_than_measuring():
    """The second acceptance criterion, read off the branch. A stamp that predates the
    deadline is not evidence while a late start may still be queued."""
    got = _verdict("outside 2026-10-03T1500Z 9540")
    assert got["verdict"] == "no-window" and got["closes"] == "false", got
    assert got["reason"] == "delayed"
    assert "9540" in got["detail"]


def test_before_the_first_report_a_standing_incident_is_unfounded_and_closes():
    got = _verdict("preseason the first availability report is due 2026-09-17T00:00:00+00:00")
    assert got["verdict"] == "no-window" and got["closes"] == "true", got
    assert got["close_reason"] == "not planned"
    assert "2026-09-17" in got["close_note"]


def test_an_absent_or_unreadable_stamp_is_an_incident_with_its_own_cause():
    """Different fixes: restore the file, or reconcile writer and monitor. Neither is a
    number, and neither is silence."""
    got = _verdict("unreachable")
    assert got["verdict"] == "stale" and "no site/data/bigten.json" in got["why"], got
    got = _verdict("unreadable")
    assert got["verdict"] == "stale" and "generated_at" in got["why"], got


def test_a_line_the_verdict_cannot_read_measures_rather_than_going_quiet():
    """A monitor that silences itself on a case it does not understand is the failure the
    live check paid for twice."""
    got = _verdict("something the module never said")
    assert got["verdict"] == "stale", got
    assert "could not be read" in got["why"]


def test_the_verdict_gates_filing_and_closing_and_the_incident_is_its_own():
    steps = _steps()
    filing = next(s for n, s in steps.items() if n.startswith("File or update"))
    assert filing["if"] == "steps.capture.outputs.verdict == 'stale'", filing["if"]
    closing = next(s for n, s in steps.items() if n.startswith("Close the capture incident"))
    assert closing["if"] == "steps.capture.outputs.closes == 'true'", closing["if"]
    assert "--reason" in closing["run"]
    # Its own marker, so the Big Ten branch never closes a live incident or a human's, and
    # the live branch never closes this one.
    markers = set(re.findall(r"<!--\s*(watchdog-[a-z-]+)\s*-->", filing["run"]))
    assert markers == {"watchdog-bigten"}, markers
    assert "watchdog-bigten" in closing["run"]
    live = yaml.safe_dump(yaml.safe_load(WATCHDOG.read_text())["jobs"]["heartbeat"])
    assert "watchdog-bigten" not in live, "the live job would close the capture's incident"
