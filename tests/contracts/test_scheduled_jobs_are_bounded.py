"""Every job in a scheduled workflow carries `timeout-minutes` (review 2026-09-12).

`bigten.yml` shares `slate.yml`'s concurrency group on purpose -- both commit the CFBD quota
counter, and two runs rebasing onto each other's counter is the race the group prevents --
so a wedged slate run holds the Big Ten capture behind it, and GitHub's default bound is six
hours. Seven Big Ten deadlines a week are unrecoverable once missed. A bound on every
scheduled job is the whole of the fix; this holds it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

_SCHEDULED = re.compile(r"^\s*schedule:\s*$", re.M)
_JOB = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$", re.M)        # two-space indent under `jobs:`
_TIMEOUT = re.compile(r"^\s+timeout-minutes:\s*\d+\s*$", re.M)


def _jobs(text: str) -> list[tuple[str, str]]:
    """(name, body) for each job, split on the two-space-indented keys under `jobs:`."""
    jobs = text[text.index("\njobs:") :]
    names = list(_JOB.finditer(jobs))
    return [(m.group(1), jobs[m.end(): names[i + 1].start() if i + 1 < len(names) else None])
            for i, m in enumerate(names)]


def test_every_job_in_a_scheduled_workflow_has_a_timeout():
    unbounded = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text()
        if not _SCHEDULED.search(text):
            continue
        for name, body in _jobs(text):
            if not _TIMEOUT.search(body):
                unbounded.append(f"{path.name}:{name}")
    assert not unbounded, (
        "scheduled jobs with no `timeout-minutes` -- a wedged run holds its concurrency "
        f"group for six hours by default: {unbounded}")


def test_the_scan_sees_the_jobs():
    """Not vacuous: at least the slate, live, watchdog and bigten jobs are scheduled."""
    seen = sum(len(_jobs(p.read_text())) for p in WORKFLOWS.glob("*.yml")
               if _SCHEDULED.search(p.read_text()))
    assert seen >= 4
