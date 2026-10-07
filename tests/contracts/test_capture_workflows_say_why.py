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
