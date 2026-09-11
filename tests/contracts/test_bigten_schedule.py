"""`bigten.yml` fires on exactly the deadlines `hub.fetch.bigten` attributes captures to (#215).

The module decides which deadline a run was *for* by the most recent deadline before it; the
workflow decides when a run *starts*. Those are two statements of one schedule, and a cron
edited without the constant -- or the constant without the cron -- leaves the CLI recording a
deadline nobody scheduled a run for as missed, or a run that fires and is attributed to a
deadline three days earlier. `test_game_windows.py` holds `live.yml` and `watchdog.yml` to
each other for the same reason and this file is the same shape, one workflow smaller.

The other three properties are the ones the disposition asked for by name: the job runs the
contract subset the slate's step runs (#228) rather than the whole suite, it commits the
archive it captured, and it never spends a CFBD call it cannot keep the result of.
"""
from __future__ import annotations

import re
from pathlib import Path

from hub.fetch import bigten

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "bigten.yml"

_CRON = re.compile(r'^\s*-\s*cron:\s*"([^"]+)"', re.M)


def _crons() -> list[str]:
    """Live crons only: a commented-out block is not a schedule, and `test_game_windows.py`
    records the day two workflows carried a `SCHEDULE DISABLED` comment above a live one."""
    return [" ".join(c.split()) for c in _CRON.findall(WORKFLOW.read_text())]


def test_the_workflow_fires_on_exactly_the_modules_deadlines():
    assert sorted(_crons()) == sorted(s.cron() for s in bigten.DEADLINES), (
        f"the workflow's crons {_crons()} and hub.fetch.bigten.DEADLINES "
        f"{[s.cron() for s in bigten.DEADLINES]} disagree; a run would be attributed to a "
        f"deadline it was not scheduled for, or a deadline would be reported missed by "
        f"a run nobody scheduled")


def test_the_scan_sees_the_crons():
    """Non-vacuity. A regex that matched nothing would make the test above compare two
    empty lists and pass."""
    assert len(_crons()) >= 5


def test_the_evening_deadline_follows_the_deadline_in_both_daylight_saving_states():
    """8pm ET is 00:00 UTC in summer and 01:00 UTC in winter. A capture before 01:00 UTC
    runs before the deadline for half the season and archives the previous day's report
    under the wrong deadline; one after 05:00 UTC is the next morning ET, which is after the
    kickoff for nobody but is the wrong side of the night's news."""
    for s in bigten.DEADLINES:
        if s.name == "evening":
            assert (1, 0) <= (s.hour, s.minute) < (5, 0), f"{s} is not after 8pm ET"


def test_the_gameday_deadlines_are_saturday():
    for s in bigten.DEADLINES:
        if s.name.startswith("gameday"):
            assert s.weekday == 6, f"{s} fires on a day no Big Ten slate is played"


def test_the_job_runs_the_contract_subset_and_not_the_suite():
    """#228's argument, verbatim: seven minutes of coverage on a second runner to read a
    handful of JSON files. The subset is the modules that read `site/data` plus this file."""
    text = WORKFLOW.read_text()
    m = re.search(r"uv run pytest((?:[^\n]*\\\n)*[^\n]*)", text)
    assert m, "the workflow no longer runs any contract before committing"
    named = set(re.findall(r"tests/contracts/(\w+\.py)", m.group(1)))
    assert named == {"test_page_reads.py", "test_published_envelopes.py",
                     "test_bigten_schedule.py"}, named
    assert "tests/unit" not in m.group(1) and "tests/contracts -q" not in m.group(1)


def test_the_job_commits_the_archive_the_stamp_and_the_quota_state():
    """A capture a runner does not commit is a capture that did not happen, and a call
    spent against a counter that is not committed is a call the next run cannot see."""
    adds = [ln for ln in WORKFLOW.read_text().splitlines()
            if ln.strip().startswith("git add")]
    assert adds, "the workflow commits nothing"
    line = adds[0]
    for path in ("archive", "site/data/bigten.json", "state"):
        assert path in line, f"`{line.strip()}` does not commit {path}"


def test_no_call_is_spent_on_a_snapshot_that_cannot_be_kept():
    """The odds snapshot is a CFBD payload and cannot be committed in the clear. Without
    the key that lets it be committed encrypted, the CLI is told to skip it -- otherwise
    every run would spend a call on a parquet the runner then discards."""
    text = WORKFLOW.read_text()
    assert "BIGTEN_ARCHIVE_KEY" in text
    assert "--skip-lines" in text
    assert re.search(r'if \[ -z "\$BIGTEN_ARCHIVE_KEY" \]; then\s*\n\s*SKIP="--skip-lines"', text), (
        "the skip is no longer conditioned on the key being absent")


def test_the_job_never_fails_on_a_capture_that_fetched_nothing():
    """A deadline the conference did not publish for is a warning on the run and a stamp
    in the tree, never a red job that stops the record being committed."""
    text = WORKFLOW.read_text()
    step = text.split("Capture the current deadline", 1)[1].split("- name:", 1)[0]
    assert "set +e" in step and "exit 0" in step
