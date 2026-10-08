"""#423 box 4 and #460: the slate measures before it publishes, and a stamp mismatch turns the
run red without leaving the slate unpublished.

`make slate` runs `hub.publish`, which reads `state/interval_coverage.json` into the track
record. Audit V found the record embedding the *previous* run's measurement because the measure
step ran after the publish. The fix is an order, and an order held only by being true today is
not held: the first test below fails the old order, planted.

The check that the record carries the measurement the same run wrote runs *after* the commit
(#460): it is exercised here in the shape of `test_capture_workflows_say_why.py`, with a scratch
origin, so "the publish commit has already happened" is observed and not read off an index.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

SLATE = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "slate.yml"

MEASURE = "Measure the interval's coverage and the survivor price"
REFRESH = "Refresh the slate"
COMMIT = "Commit what was published"
CHECK = "Check the track record carries the measurement this run wrote"


def _steps() -> list[dict]:
    return yaml.safe_load(SLATE.read_text())["jobs"]["refresh"]["steps"]


def _names(steps: list[dict]) -> list[str]:
    return [s.get("name", "") for s in steps]


def measures_before_it_publishes(names: list[str]) -> bool:
    """The measure step is above the step that runs `make slate` (which publishes)."""
    return names.index(MEASURE) < names.index(REFRESH)


def checks_after_the_commit(names: list[str]) -> bool:
    return names.index(CHECK) > names.index(COMMIT)


def _script(name: str) -> str:
    return next(s["run"] for s in _steps() if s.get("name") == name)


def test_the_measure_step_runs_before_make_slate_publishes():
    """Mutation: put the old order back (measure after the refresh) and this is red."""
    names = _names(_steps())
    assert "make slate" in _script(REFRESH)
    assert measures_before_it_publishes(names)


def test_the_plant_the_old_order_is_seen_by_the_check():
    names = _names(_steps())
    old = [n for n in names if n != MEASURE]
    old.insert(old.index(REFRESH) + 1, MEASURE)          # the order before #423
    assert not measures_before_it_publishes(old), "the plant must be red or the test is blind"


def test_the_measure_step_is_still_soft_failed_and_still_precedes_the_commit():
    """It moved earlier, not out of the degradation rule: a failed fetch leaves last-good."""
    step = next(s for s in _steps() if s.get("name") == MEASURE)
    assert step.get("continue-on-error") is True
    assert _names(_steps()).index(MEASURE) < _names(_steps()).index(COMMIT)


def test_the_stamp_check_runs_after_the_commit_and_the_old_position_is_seen():
    names = _names(_steps())
    assert checks_after_the_commit(names)
    old = [n for n in names if n != CHECK]
    old.insert(old.index(COMMIT), CHECK)                 # as first built: before the commit
    assert not checks_after_the_commit(old)


# --- the check, exercised against a scratch origin ---------------------------------------


def _env(cwd: Path) -> dict[str, str]:
    return {"PATH": os.environ["PATH"], "HOME": str(cwd), "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _sh(script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-ec", script], cwd=cwd, capture_output=True, text=True,
                          timeout=60, env=_env(cwd))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
                          env=_env(cwd)).stdout


@pytest.fixture
def runner(tmp_path):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True,
                   capture_output=True)
    _git(work, "checkout", "-q", "-b", "main")
    (work / "site/data").mkdir(parents=True)
    (work / "state").mkdir()
    (work / "site/data/manifest.json").write_text(json.dumps({"week": 5}))
    (work / "state/interval_coverage.json").write_text(
        json.dumps({"generated_at": "2026-10-03T00:00:00+00:00"}))
    (work / "site/data/track_record.json").write_text(
        json.dumps({"interval_coverage": {"generated_at": "2026-10-03T00:00:00+00:00"}}))
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "seed")
    _git(work, "push", "-q", "-u", "origin", "main")
    return work, origin


def _publish(work: Path, *, published: str | None, wrote: str | None) -> None:
    """What a run leaves on disk: a measurement (`wrote`) and the record that embedded
    `published` (None: the record carries no coverage at all)."""
    state = {} if wrote is None else {"generated_at": wrote}
    (work / "state/interval_coverage.json").write_text(json.dumps(state))
    rec = {"interval_coverage": None if published is None else {"generated_at": published}}
    (work / "site/data/track_record.json").write_text(json.dumps(rec))
    (work / "site/data/manifest.json").write_text(json.dumps({"week": 6}))


def _run_steps(work: Path, order: list[str]) -> tuple[int, list[str]]:
    """Run the named steps' scripts in order, stopping at the first red -- as Actions does."""
    ran = []
    for name in order:
        got = _sh(_script(name), work)
        ran.append(name)
        if got.returncode != 0:
            return got.returncode, ran
    return 0, ran


def _landed(origin: Path) -> bool:
    return "slate: week 6" in _git(origin, "log", "-1", "--format=%s", "main")


def test_matching_stamps_pass_and_the_slate_is_committed(runner):
    work, origin = runner
    _publish(work, published="2026-10-07T01:00:00+00:00", wrote="2026-10-07T01:00:00+00:00")
    code, ran = _run_steps(work, [COMMIT, CHECK])
    assert code == 0 and ran == [COMMIT, CHECK]
    assert _landed(origin)


def test_a_mismatch_fails_the_check_but_the_slate_has_already_landed(runner):
    """Planted: the record embeds yesterday's stamp. The run is red, and the commit step has
    already put the slate on the remote -- a red check, not an unpublished slate."""
    work, origin = runner
    _publish(work, published="2026-10-03T17:53:10+00:00", wrote="2026-10-07T01:00:00+00:00")
    code, ran = _run_steps(work, [COMMIT, CHECK])
    assert code == 1 and ran == [COMMIT, CHECK]
    assert _landed(origin), "the slate must publish even when the stamp check is red"
    got = _sh(_script(CHECK), work)
    assert "::error::" in got.stdout and "2026-10-03T17:53:10" in got.stdout


def test_the_plant_the_check_before_the_commit_would_leave_the_slate_unpublished(runner):
    """The control that flips: the same mismatch with the check in its first position (before
    the commit) stops the run before the commit, and nothing reaches the remote."""
    work, origin = runner
    _publish(work, published="2026-10-03T17:53:10+00:00", wrote="2026-10-07T01:00:00+00:00")
    code, ran = _run_steps(work, [CHECK, COMMIT])
    assert code == 1 and ran == [CHECK]
    assert not _landed(origin)


def test_no_measurement_on_disk_or_none_ever_made_is_not_a_mismatch(runner):
    work, _ = runner
    (work / "state/interval_coverage.json").unlink()
    assert _sh(_script(CHECK), work).returncode == 0
    _publish(work, published=None, wrote=None)           # a file with no stamp: never measured
    assert _sh(_script(CHECK), work).returncode == 0
