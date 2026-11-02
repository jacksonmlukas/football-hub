"""The capture workflows commit the record of a miss, and say why when every deadline missed (#424).

Audit V finding V6: `bigten.yml` went red on 25 of 25 scheduled runs because its commit step
named `archive/`, a directory the first capture creates, so `git add` died with a pathspec
error (exit 128) on exactly the runs that most needed `site/data/bigten.json` -- the record of
the misses -- committed. One unset repository variable, `CFB_WEEK_ONE`, was the cause of
both, and no step said so.

**Untested on Actions.** There is no live CI for a workflow change; these tests execute each
step's shell, with its expressions bound, against a scratch git repository or a scratch
working tree, the way `test_watchdog_bigten.py` does. What they prove is the step's logic. That
the runner provides `jq`, the secrets and the push permission is what the first scheduled run
after merge confirms.

Each check has its positive control (method.md rule 18), in the same file: the pathspec
failure is planted by running the *old* commit step in the same harness and requiring it to
die, so a harness that could not see the defect would fail here and not pass quietly.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

COMMIT = "Commit what was captured"
NAMED = "Fail, naming the cause, when no deadline has ever been captured"
ESCALATE = "Escalate a CFBD source that has stayed unfetched"
OLD_ADD = "git add archive site/data/bigten.json state\n"

CAUSE = ("nothing was captured: CFB_WEEK_ONE is not set, so nothing here knows which "
         "college week it is.")


def _script(workflow: str, job: str, step: str) -> str:
    jobs = yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"]
    return next(s["run"] for s in jobs[job]["steps"] if s.get("name") == step)


def _sh(script: str, cwd: Path, *, extra_env: dict[str, str] | None = None
        ) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"], "HOME": str(cwd), "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_TERMINAL_PROMPT": "0", **(extra_env or {})}
    return subprocess.run(["bash", "-ec", script], cwd=cwd, capture_output=True, text=True,
                          timeout=60, env=env)


def _git(cwd: Path, *args: str) -> str:
    got = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
                         env={"PATH": os.environ["PATH"], "HOME": str(cwd),
                              "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                              "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                              "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    return got.stdout


def _stamp(missed: int, held: int, reason: str | None, **extra) -> dict:
    return {"name": "bigten", "generated_at": "2026-10-06T00:00:00+00:00", "fetched": False,
            "stale": True, "reason": reason, "deadline": {"id": None, "name": None},
            "missed": {"count": missed, "deadlines": []}, "archive_rows": held, **extra}


@pytest.fixture
def runner(tmp_path):
    """A clone of a scratch origin holding last run's `bigten.json` and a `state/`, and
    no `archive/` -- the repository as it stood for all 25 failed runs."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True,
                   capture_output=True)
    _git(work, "checkout", "-q", "-b", "main")
    (work / "site/data").mkdir(parents=True)
    (work / "state").mkdir()
    (work / "site/data/bigten.json").write_text(json.dumps(_stamp(0, 0, "before the first")))
    (work / "state/cfbd-quota.json").write_text("{}")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "seed")
    _git(work, "push", "-q", "-u", "origin", "main")
    assert not (work / "archive").exists()
    return work, origin


def _commit_step() -> str:
    return _script("bigten.yml", "capture", COMMIT)


def _record_a_run_that_captured_nothing(work: Path) -> None:
    (work / "site/data/bigten.json").write_text(json.dumps(_stamp(20, 0, CAUSE)))


def test_a_run_with_no_archive_directory_commits_the_miss_record(runner):
    """Rule-18 control for #424's second box: the run that captured nothing, in a tree with
    no `archive/`, must land `bigten.json` on the remote."""
    work, origin = runner
    _record_a_run_that_captured_nothing(work)
    got = _sh(_commit_step(), work)
    assert got.returncode == 0, got.stderr
    landed = json.loads(_git(origin, "show", "main:site/data/bigten.json"))
    assert landed["missed"]["count"] == 20, "the miss record did not reach the remote"
    assert "bigten:" in _git(origin, "log", "-1", "--format=%s", "main")


def test_the_plant_the_old_commit_step_dies_at_the_pathspec_in_this_harness(runner):
    """The plant. Put the old `git add` back and the same scenario must fail with exit 128;
    if it passes, the test above was not able to see the defect it is about."""
    work, _origin = runner
    _record_a_run_that_captured_nothing(work)
    new = _commit_step()
    old = re.sub(r"for p in archive site/data/bigten.json state; do.*?\n\s*done\n", OLD_ADD,
                 new, flags=re.S)
    assert old != new and OLD_ADD in old, "could not reconstruct the old step"
    got = _sh(old, work)
    assert got.returncode == 128 and "pathspec 'archive'" in got.stderr, got.stderr


def test_an_archive_that_exists_is_still_committed(runner):
    """The fix must not drop the thing it was written around: when a capture did create
    `archive/`, it is committed with the stamp."""
    work, origin = runner
    (work / "archive/bigten").mkdir(parents=True)
    (work / "archive/bigten/index.parquet").write_text("x")
    _record_a_run_that_captured_nothing(work)
    got = _sh(_commit_step(), work)
    assert got.returncode == 0, got.stderr
    assert "archive/bigten/index.parquet" in _git(origin, "ls-tree", "-r", "--name-only", "main")


def test_a_run_that_changed_nothing_commits_nothing(runner):
    work, origin = runner
    before = _git(origin, "rev-parse", "main")
    got = _sh(_commit_step(), work)
    assert got.returncode == 0 and "no commit" in got.stdout, got.stderr
    assert _git(origin, "rev-parse", "main") == before


# --- the named failure --------------------------------------------------------------------

def _named(tmp_path: Path, stamp: dict | None) -> subprocess.CompletedProcess:
    (tmp_path / "site/data").mkdir(parents=True, exist_ok=True)
    if stamp is not None:
        (tmp_path / "site/data/bigten.json").write_text(json.dumps(stamp))
    return _sh(_script("bigten.yml", "capture", NAMED), tmp_path)


def test_every_deadline_missed_fails_and_names_the_unset_variable(tmp_path):
    got = _named(tmp_path, _stamp(20, 0, CAUSE))
    assert got.returncode == 1, "a capture that has never landed must not be green"
    assert "CFB_WEEK_ONE is not set" in got.stdout
    assert "20" in got.stdout


def test_the_named_failure_does_not_fire_once_anything_was_captured(tmp_path):
    """The other arm of the control: misses with a non-empty archive are the watchdog's
    business (a capture that worked and stopped), not this step's."""
    assert _named(tmp_path, _stamp(3, 5, "a deadline went by")).returncode == 0


def test_before_the_first_deadline_nothing_is_missed_and_the_step_passes(tmp_path):
    assert _named(tmp_path, _stamp(0, 0, "before the first report")).returncode == 0


def test_no_stamp_at_all_is_left_to_the_report_step(tmp_path):
    """`Report what was captured` already errors on a missing stamp; this step adds no
    second, noisier message for the same fact."""
    assert _named(tmp_path, None).returncode == 0


def test_the_named_failure_runs_after_the_commit(tmp_path):
    """Order is the point: the miss record is published, then the run goes red."""
    steps = [s.get("name") for s in
             yaml.safe_load((WORKFLOWS / "bigten.yml").read_text())["jobs"]["capture"]["steps"]]
    assert steps.index(NAMED) > steps.index(COMMIT)


# --- the cfbd escalation ------------------------------------------------------------------

def _escalate(tmp_path: Path, stamp: dict | None) -> subprocess.CompletedProcess:
    (tmp_path / "site/data").mkdir(parents=True, exist_ok=True)
    if stamp is not None:
        (tmp_path / "site/data/cfbd.json").write_text(json.dumps(stamp))
    return _sh(_script("slate.yml", "refresh", ESCALATE), tmp_path)


def test_an_escalated_cfbd_record_turns_the_run_red_with_the_cause(tmp_path):
    got = _escalate(tmp_path, {"fetched": False, "escalate": True,
                               "unfetched_since": "2026-09-09", "reason": CAUSE})
    assert got.returncode == 1
    assert "2026-09-09" in got.stdout and "CFB_WEEK_ONE" in got.stdout


@pytest.mark.parametrize("stamp", [
    {"fetched": False, "escalate": False, "unfetched_since": "2026-10-03", "reason": CAUSE},
    {"fetched": True, "escalate": False, "unfetched_since": None, "reason": None},
    {"fetched": False, "reason": CAUSE},        # a record written before the field existed
    None,
], ids=["still-a-warning", "fetched", "legacy-record", "no-record"])
def test_a_cfbd_record_that_has_not_escalated_leaves_the_run_alone(tmp_path, stamp):
    assert _escalate(tmp_path, stamp).returncode == 0


def test_the_escalation_runs_even_when_the_gate_before_it_is_red():
    jobs = yaml.safe_load((WORKFLOWS / "slate.yml").read_text())["jobs"]
    steps = jobs["refresh"]["steps"]
    step = next(s for s in steps if s.get("name") == ESCALATE)
    assert step["if"] == "always()"
    assert steps.index(step) > next(i for i, s in enumerate(steps)
                                    if s.get("name") == "Commit what was published")


# --- consensus.yml (#432, #437) ------------------------------------------------------------
#
# The capture's commit step had no test: it is the step that makes a capture a record, and the
# shape of #424 -- a `git add` naming a directory the first run has not created -- is exactly
# what it could have carried.

CONSENSUS_COMMIT = "Commit what was captured"
CONSENSUS_CAPTURE = "Capture the next week's consensus page"
OLD_CONSENSUS_ADD = "git add state/consensus\n"


def _consensus_step(name: str) -> str:
    return _script("consensus.yml", "capture", name)


def test_a_capture_is_committed_and_reaches_the_remote(runner):
    work, origin = runner
    cap = work / "state/consensus/2026/wk05"
    cap.mkdir(parents=True)
    (cap / "cap-20261007T150000.json").write_text("{}")
    got = _sh(_consensus_step(CONSENSUS_COMMIT), work)
    assert got.returncode == 0, got.stderr
    assert "state/consensus/2026/wk05/cap-20261007T150000.json" in _git(
        origin, "ls-tree", "-r", "--name-only", "main")
    assert "consensus:" in _git(origin, "log", "-1", "--format=%s", "main")


def test_a_run_that_captured_nothing_in_a_tree_with_no_consensus_directory_passes(runner):
    """The plant, and its flip. `runner` has `state/` and no `state/consensus/` -- the tree of
    every run before the first capture. The step as written passes with no commit; the old
    `git add state/consensus` dies at the pathspec (exit 128), so this harness can see it."""
    work, origin = runner
    assert not (work / "state/consensus").exists()
    before = _git(origin, "rev-parse", "main")
    got = _sh(_consensus_step(CONSENSUS_COMMIT), work)
    assert got.returncode == 0 and "no commit" in got.stdout, got.stderr
    assert _git(origin, "rev-parse", "main") == before
    new = _consensus_step(CONSENSUS_COMMIT)
    old = re.sub(r"(?:[ \t]*#[^\n]*\n)*[ \t]*git add state\n", "          " + OLD_CONSENSUS_ADD, new)
    assert old != new and OLD_CONSENSUS_ADD in old, "could not reconstruct the old step"
    died = _sh(old, work)
    assert died.returncode == 128 and "pathspec 'state/consensus'" in died.stderr, died.stderr


def _capture_with_exit(tmp_path: Path, code: int) -> subprocess.CompletedProcess:
    """The capture step with `uv` replaced by a stub that exits `code`."""
    work = tmp_path / f"run{code}"
    bin_dir = work / "bin"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "uv"
    stub.write_text(f"#!/bin/sh\nexit {code}\n")
    stub.chmod(0o755)
    return _sh(_consensus_step(CONSENSUS_CAPTURE), work,
               extra_env={"PATH": f"{bin_dir}:{os.environ['PATH']}"})


def test_exit_three_is_a_notice_and_a_failed_fetch_turns_the_run_red(tmp_path):
    ok = _capture_with_exit(tmp_path, 3)
    assert ok.returncode == 0 and "nothing captured" in ok.stdout
    assert _capture_with_exit(tmp_path, 1).returncode == 1, "a failed fetch must not be green"
    assert _capture_with_exit(tmp_path, 0).returncode == 0


def _capture_run(tmp_path: Path, code: int, schedule: str = "", purpose: str = "",
                 step: str | None = None) -> subprocess.CompletedProcess:
    """The capture step (or `step`, a planted copy) with `uv` replaced by a stub that records
    its arguments and exits `code`, started as if by `schedule` (empty: a dispatch)."""
    work = tmp_path / f"run{code}-{abs(hash((schedule, purpose, step)))}"
    bin_dir = work / "bin"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "uv"
    stub.write_text(f'#!/bin/sh\necho "uv $*"\nexit {code}\n')
    stub.chmod(0o755)
    return _sh(step or _consensus_step(CONSENSUS_CAPTURE), work,
               extra_env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "SCHEDULE": schedule,
                          "PURPOSE": purpose})


def _lineup_crons() -> list[str]:
    """The crons the capture step routes to `--lineup`, read off its own `case` line."""
    line = next(ln for ln in _consensus_step(CONSENSUS_CAPTURE).splitlines()
                if "flag=--lineup ;;" in ln and '"0 ' in ln)
    return re.findall(r'"([^"]+)"', line.split(")")[0])


def test_an_empty_lineup_run_inside_a_day_of_sunday_is_red_and_says_so(tmp_path):
    red = _capture_run(tmp_path, 4, schedule="0 8 * * 6")
    assert red.returncode == 4 and "::error::" in red.stdout and "--lineup" in red.stdout
    notice = _capture_run(tmp_path, 3, schedule="0 13 * * 5")
    assert notice.returncode == 0 and "::notice::" in notice.stdout


def test_the_plant_folding_exit_four_into_the_notice_turns_the_red_run_green(tmp_path):
    """The plant: `-eq 3` widened to `-ge 3` is the old behaviour, every empty run a notice. If
    the test above could not tell that apart, it would not be holding the loud failure."""
    new = _consensus_step(CONSENSUS_CAPTURE)
    old = new.replace('if [ "$code" -eq 3 ]', 'if [ "$code" -ge 3 ]')
    assert old != new, "could not plant the defect"
    assert _capture_run(tmp_path, 4, schedule="0 8 * * 6", step=old).returncode == 0


def test_each_cron_takes_the_capture_it_is_for(tmp_path):
    """#515 retired the Tuesday and Wednesday runs: every scheduled run is the Friday capture,
    and the week capture is a dispatch's (`purpose: week`). A cron the step does not route is a
    schedule edited without it, and is red rather than a guess -- planted below."""
    wf = yaml.safe_load((WORKFLOWS / "consensus.yml").read_text())
    crons = [c["cron"] for c in wf[True]["schedule"]]          # `on:` loads as the key True
    lineup = _lineup_crons()
    assert set(lineup) == set(crons) and len(lineup) == 4
    for cron in crons:
        got = _capture_run(tmp_path, 0, schedule=cron)
        assert "hub.fetch.consensus --lineup" in got.stdout, (cron, got.stdout)
    assert "--lineup" in _capture_run(tmp_path, 0).stdout                   # a bare dispatch
    assert "--lineup" in _capture_run(tmp_path, 0, purpose="lineup").stdout
    assert "--capture" in _capture_run(tmp_path, 0, purpose="week").stdout
    stray = _capture_run(tmp_path, 0, schedule="0 23 * * 3")              # the retired Wednesday
    assert stray.returncode == 1 and "::error::" in stray.stdout and "uv " not in stray.stdout


def test_no_run_is_scheduled_before_thursday_since_the_archive_publishes_on_fridays():
    """The retired runs stay retired unless someone says why (#515): in six weeks of 2026 they
    never captured, and under ADR-0027 a pre-Thursday page adds only Thursday's players."""
    wf = yaml.safe_load((WORKFLOWS / "consensus.yml").read_text())
    for c in wf[True]["schedule"]:
        assert _week_minute(c["cron"]) >= 5 * 1440, c["cron"]
    text = (WORKFLOWS / "consensus.yml").read_text()
    assert "Tuesday and Wednesday runs are retired" in text and "#515" in text


def _week_minute(cron: str) -> int:
    """Minutes since Sunday 00:00 UTC of a `M H * * D` cron, Sunday = 0."""
    minute, hour, _, _, dow = cron.split()
    return int(dow) * 1440 + int(hour) * 60 + int(minute)


def test_every_lineup_run_falls_after_thursday_and_before_the_saturday_publish():
    """The lineup capture is only worth taking if the publish that writes `roster.json` can see
    it: `slate.yml`'s Saturday full run. Every lineup cron is after Thursday (the week's first
    game day, the strict deadline) and before that publish."""
    slate = yaml.safe_load((WORKFLOWS / "slate.yml").read_text())
    gate = slate["jobs"]["refresh"]["if"]
    full = re.findall(r'"(\d+ \d+ \* \* \d)"', gate)
    publish = next(c for c in full if c.endswith(" 6"))                     # the Saturday one
    thursday_ends = 5 * 1440
    for cron in _lineup_crons():
        assert thursday_ends <= _week_minute(cron) < _week_minute(publish), cron
    assert _week_minute(publish) - max(map(_week_minute, _lineup_crons())) >= 6 * 60


def test_the_commit_runs_after_the_capture_and_the_run_is_bounded():
    wf = yaml.safe_load((WORKFLOWS / "consensus.yml").read_text())
    steps = [s.get("name") for s in wf["jobs"]["capture"]["steps"]]
    assert steps.index(CONSENSUS_COMMIT) > steps.index(CONSENSUS_CAPTURE)
    assert wf["jobs"]["capture"]["timeout-minutes"]
    assert wf["concurrency"]["group"] == "slate"


# --- the odds poll (#468) -----------------------------------------------------------------
# GitHub starts `poll_odds` 4-8.5h after its cron slot (#468, 2026-10-14). Late is accepted;
# silent is not. These hold that when it does run, the capture reaches the remote.

POLL_COMMIT = "Commit the credit balance and the capture"
POLL_SNAP = "state/odds/2026/wk05/snap-20261008T174636.json"


def _poll_step() -> str:
    return _script("slate.yml", "poll_odds", POLL_COMMIT)


def _a_poll_ran(work: Path) -> None:
    snap = work / POLL_SNAP
    snap.parent.mkdir(parents=True)
    snap.write_text("{}")
    (work / "state/odds.json").write_text(json.dumps({"remaining": 400}))


def test_the_poll_commit_lands_the_capture_and_the_balance_on_the_remote(runner):
    work, origin = runner
    _a_poll_ran(work)
    got = _sh(_poll_step(), work)
    assert got.returncode == 0, got.stderr
    landed = _git(origin, "ls-tree", "-r", "--name-only", "main")
    assert POLL_SNAP in landed and "state/odds.json" in landed
    assert _git(origin, "log", "-1", "--format=%s", "main").startswith("odds: scheduled poll")


def test_the_plant_adding_only_the_balance_leaves_the_capture_off_the_remote(runner):
    """The plant: `git add state/odds.json` commits the credit balance and strands the capture
    on the runner that is then discarded. If this lands the capture, the test above could not
    tell the two steps apart."""
    work, origin = runner
    _a_poll_ran(work)
    new = _poll_step()
    old = re.sub(r"git add state\n", "git add state/odds.json\n", new)
    assert old != new and "git add state/odds.json\n" in old, "could not plant the defect"
    got = _sh(old, work)
    assert got.returncode == 0, got.stderr
    landed = _git(origin, "ls-tree", "-r", "--name-only", "main")
    assert "state/odds.json" in landed and POLL_SNAP not in landed


def test_the_poll_job_shares_the_slate_concurrency_group():
    wf = yaml.safe_load((WORKFLOWS / "slate.yml").read_text())
    assert wf["concurrency"]["group"] == "slate"
    assert "concurrency" not in wf["jobs"]["poll_odds"], (
        "a job-level group would replace the workflow's and let a poll race the refresh")
