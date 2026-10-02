"""#391's two workflow changes, held to the constraints that made them the chosen options.

The decisions are in `hub.live_chain` and `hub.watchdog_gap`, which have their own tests. What
nothing else holds is that the workflows *call* them, and that the calls stay inside the rules
`live.yml`'s header sets: the crons are the only window definition, and the loop's re-dispatch
asks about game state rather than carrying a window of its own.
"""
import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
LIVE = (WORKFLOWS / "live.yml").read_text()
WATCHDOG = (WORKFLOWS / "watchdog.yml").read_text()


def _step(text: str, name: str) -> str:
    """The text of one named step, up to the next step or job."""
    m = re.search(rf"- name: {re.escape(name)}\n(.*?)(?=\n      - |\n  \w+:\n|\Z)", text,
                  flags=re.DOTALL)
    assert m, f"no step named {name!r}"
    return m.group(1)


def test_the_loop_re_dispatches_itself_on_game_state():
    step = _step(LIVE, "Continue while a game is in progress")
    assert "hub.live_chain" in step, "the decision is the module's, not shell's"
    assert re.search(r"gh workflow run live\.yml", step)
    assert '-f league="$LEAGUE"' in step, (
        "a dispatch with no league would be 'auto', which is Saturday-means-college and not "
        "the board this loop was refreshing")


def test_the_re_dispatch_carries_no_window_of_its_own():
    """The crons stay the only definition. Nothing in the step names an hour, a weekday or a
    clock -- the module has the same test from its own side."""
    step = _step(LIVE, "Continue while a game is in progress")
    code = "\n".join(line.split("#")[0] for line in step.splitlines())
    assert not re.search(r"\bdate\b|\bTZ=|%[uHMa]|\*/10", code), code


def test_the_continue_step_runs_after_the_loop_and_is_the_last_step():
    steps = re.findall(r"^      - (?:name|uses|run): (.*)$", LIVE, flags=re.MULTILINE)
    loop = next(i for i, s in enumerate(steps) if s.startswith("Refresh the overlay"))
    cont = next(i for i, s in enumerate(steps) if s.startswith("Continue while"))
    assert cont == loop + 1 == len(steps) - 1


def test_the_dispatch_has_the_permission_it_needs_and_the_group_still_hands_over():
    assert re.search(r"permissions:\n  contents: read\n  actions: write", LIVE)
    assert re.search(r"concurrency:\n  group: live-loop\n  cancel-in-progress: true", LIVE), (
        "the hand-over relies on the new run replacing this one; a group that queued instead "
        "would leave two loops, or the new one waiting out a loop that has already finished")


def test_the_watchdog_has_a_gap_job_and_no_new_cron():
    job = re.search(r"\n  gaps:\n(.*)", WATCHDOG, flags=re.DOTALL)
    assert job, "no gaps job"
    body = job.group(1)
    assert "hub.watchdog_gap" in body
    assert "<!-- watchdog-gap -->" in body, "its own marker, so it closes only its own incident"
    assert "actions: read" in body, "`gh run list` needs it, and a job-level block replaces the workflow's"
    assert "issues: write" in body and "contents: read" in body
    assert "cron:" not in body, "the window is the crons above, not a second schedule"


def test_a_failure_to_read_the_run_history_fails_the_job():
    job = re.search(r"\n  gaps:\n(.*)", WATCHDOG, flags=re.DOTALL)
    assert job, "no gaps job"
    body = job.group(1)
    assert re.search(r'\[ "\$CODE" -ne 0 \]', body) and "exit 1" in body, (
        "an unreadable run list must not read as 'no gaps'")
