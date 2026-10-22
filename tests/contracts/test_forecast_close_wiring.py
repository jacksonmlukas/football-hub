"""The study's closing line is wired so it can fire, off until v0, and writes only `forecasts` (#450).

Untested on Actions -- there is no live CI for a workflow change -- so, as
`test_close_capture_wiring.py` does for the hub's close, these execute the shell against stub
`uv` and dispatch commands (and `commit-state.sh` against scratch git repositories), and read
the workflow files for the properties that keep the spend bounded. Each "does not" case has
its positive control in the same file: the dispatcher is made to dispatch when the study is
on and due, so a harness that could not see a dispatch would fail the negatives.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SCRIPTS = ROOT / ".github" / "scripts"
STUDY_DISPATCH = SCRIPTS / "study-close-dispatch.sh"
HUB_DISPATCH = SCRIPTS / "close-dispatch.sh"
COMMIT = SCRIPTS / "commit-state.sh"
WF = WORKFLOWS / "forecast-close.yml"


def _doc(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _stub_bin(tmp_path: Path, due_exit: int) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> \"{tmp_path}/uv-args\"\nexit {due_exit}\n")
    uv.chmod(0o755)
    return bin_dir


def _recorder(tmp_path: Path, name: str) -> tuple[Path, Path]:
    log = tmp_path / f"{name}.log"
    rec = tmp_path / f"{name}.sh"
    rec.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log}"\n')
    rec.chmod(0o755)
    return rec, log


def _lines(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def _study(tmp_path: Path, *, due_exit: int, enabled: str | None) -> list[str]:
    rec, log = _recorder(tmp_path, "study")
    env = {"PATH": f"{_stub_bin(tmp_path, due_exit)}:{os.environ['PATH']}", "HOME": str(tmp_path)}
    if enabled is not None:
        env["LLM_FORECASTS_ENABLED"] = enabled
    subprocess.run([str(STUDY_DISPATCH), str(rec), "forecast-close.yml"], env=env, check=True,
                   capture_output=True, text=True, timeout=30)
    return _lines(log)


# --- the loop's dispatcher ----------------------------------------------------------------

def test_on_and_due_dispatches_once(tmp_path):
    assert _study(tmp_path, due_exit=0, enabled="true") == ["forecast-close.yml"]
    assert "hub.forecasters.close_line due --out" in (tmp_path / "uv-args").read_text()


@pytest.mark.parametrize("enabled", [None, "", "false", "TRUE"])
def test_switched_off_never_dispatches_even_when_due(tmp_path, enabled):
    assert _study(tmp_path, due_exit=0, enabled=enabled) == []
    assert not (tmp_path / "uv-args").exists()                     # not even asked


@pytest.mark.parametrize("due_exit", [1, 2])
def test_not_due_or_unknown_dispatches_nothing(tmp_path, due_exit):
    assert _study(tmp_path, due_exit=due_exit, enabled="true") == []


def _hub_hook(tmp_path: Path, league: str, hub_due: int) -> tuple[list[str], list[str]]:
    study_rec, study_log = _recorder(tmp_path, "study-hook")
    hub_rec, hub_log = _recorder(tmp_path, "hub")
    env = {"PATH": f"{_stub_bin(tmp_path, hub_due)}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "LIVE_LOOP_LEAGUE": league, "STUDY_CLOSE_HOOK": f"{study_rec} asked"}
    subprocess.run([str(HUB_DISPATCH), str(hub_rec), "odds-close.yml"], env=env, check=True,
                   capture_output=True, text=True, timeout=30)
    return _lines(study_log), _lines(hub_log)


@pytest.mark.parametrize(("league", "hub_due", "hub_dispatched"),
                         [("cfb", 0, []), ("nfl", 1, []), ("nfl", 0, ["odds-close.yml"])])
def test_the_study_hook_runs_in_every_leagues_loop_and_leaves_the_hubs_alone(
        tmp_path, league, hub_due, hub_dispatched):
    """A December Saturday NFL game is played in the college loop, so the study is asked in
    every loop; the hub's own dispatch is unchanged (its positive control is the last case)."""
    study, hub = _hub_hook(tmp_path, league, hub_due)
    assert study == ["asked"] and hub == hub_dispatched


def test_the_window_loop_passes_the_study_hook_switched_by_the_variable():
    doc = _doc(WORKFLOWS / "live.yml")
    step = next(s for s in doc["jobs"]["refresh"]["steps"]
                if s.get("name") == "Refresh the overlay through the window")
    env = step["env"]
    assert "study-close-dispatch.sh" in env["STUDY_CLOSE_HOOK"]
    assert "gh workflow run forecast-close.yml" in env["STUDY_CLOSE_HOOK"]
    assert env["LLM_FORECASTS_ENABLED"] == "${{ vars.LLM_FORECASTS_ENABLED }}"
    assert env["STUDY_SYNC_REF"] == "forecasts"
    assert doc["jobs"]["refresh"]["permissions"]["contents"] == "read"


# --- the workflow -------------------------------------------------------------------------

def test_switched_off_until_v0_by_the_studys_variable():
    job = _doc(WF)["jobs"]["capture"]
    assert job["if"] == "vars.LLM_FORECASTS_ENABLED == 'true'"


def test_its_own_group_serialised_never_cancelled_and_bounded():
    doc = _doc(WF)
    assert doc["concurrency"] == {"group": "llm-forecast-close", "cancel-in-progress": False}
    assert _doc(WORKFLOWS / "forecast.yml")["concurrency"]["group"] != doc["concurrency"]["group"]
    assert doc["jobs"]["capture"]["timeout-minutes"] <= 15


def _slots(crons: list[str]) -> set[tuple[int, int]]:
    """(day of week, UTC hour) each `*/N` window cron covers."""
    out: set[tuple[int, int]] = set()
    for c in crons:
        minute, hour, _, _, dow = c.split()
        if not minute.startswith("*/"):
            continue                                      # live.yml's backup starts (#470)
        lo, _, hi = hour.partition("-")
        out |= {(int(dow), h) for h in range(int(lo), int(hi or lo) + 1)}
    return out


def _crons(path: Path) -> list[str]:
    return re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', path.read_text(), re.M)


def test_the_backup_crons_ask_every_15_minutes_on_live_ymls_window_hours():
    crons = _crons(WF)
    assert crons and all(c.split()[0] == "*/15" for c in crons)
    assert _slots(crons) == _slots(_crons(WORKFLOWS / "live.yml"))


def test_the_attempt_is_committed_before_the_request_and_everything_goes_to_forecasts():
    steps = _doc(WF)["jobs"]["capture"]["steps"]
    cap = next(s for s in steps if "hub.forecasters.close_line capture" in s.get("run", ""))
    assert "--commit-attempt" in cap["run"] and "--out forecasts-branch" in cap["run"]
    assert cap["env"] == {"ODDS_API_KEY": "${{ secrets.ODDS_API_KEY }}"}
    commit = next(s for s in steps if "commit-state.sh" in s.get("run", ""))
    assert commit["if"] == "always()" and steps.index(commit) > steps.index(cap)
    assert commit["env"] == {"STATE_REPO": "forecasts-branch", "STATE_PATHS": "closing state",
                             "STATE_REF": "forecasts"}
    text = WF.read_text()
    assert "git push" not in text and "git add" not in text        # only through the script


def test_the_scripts_are_executable_and_parse():
    for script in (STUDY_DISPATCH, HUB_DISPATCH, COMMIT):
        assert os.access(script, os.X_OK)
        subprocess.run(["bash", "-n", str(script)], check=True)


# --- commit-state.sh, pointed at the study's branch ----------------------------------------

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null",
                               "GIT_CONFIG_NOSYSTEM": "1"}).stdout


@pytest.fixture
def checkout(tmp_path) -> tuple[Path, Path]:
    """A bare origin with `main` holding `state/odds.json`, and a clone of it."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    for k, v in (("user.name", "t"), ("user.email", "t@t")):
        _git(work, "config", k, v)
    (work / "state").mkdir()
    (work / "state" / "odds.json").write_text('{"remaining": 400}\n')
    _git(work, "add", "state")
    _git(work, "commit", "-q", "-m", "seed")
    _git(work, "push", "-q", "origin", "HEAD:main")
    return origin, work


def _commit_state(work: Path, msg: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(COMMIT), msg], cwd=work, capture_output=True, text=True,
                          timeout=60, env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null",
                                           "GIT_CONFIG_NOSYSTEM": "1",
                                           "GITHUB_REF_NAME": "main", **env})


def test_pointed_at_forecasts_it_pushes_there_and_never_to_main(checkout):
    origin, work = checkout
    main_before = _git(origin, "rev-parse", "main")
    # As forecast.yml's `worktree add --orphan`, spelled for a git older than 2.42.
    _git(work, "worktree", "add", "-q", "--detach", "forecasts-branch")
    _git(work / "forecasts-branch", "checkout", "-q", "--orphan", "forecasts")
    _git(work / "forecasts-branch", "rm", "-rfq", ".")
    (work / "forecasts-branch" / "closing").mkdir()
    (work / "forecasts-branch" / "closing" / "attempts.json").write_text('{"attempts": []}\n')
    (work / "state" / "odds.json").write_text('{"remaining": 1}\n')   # a hub change, unstaged
    r = _commit_state(work, "forecasts: attempt", STATE_REPO="forecasts-branch",
                      STATE_PATHS="closing state", STATE_REF="forecasts")
    assert r.returncode == 0, r.stderr                 # `state` absent there: skipped, no error
    assert _git(origin, "rev-parse", "main") == main_before
    assert _git(origin, "ls-tree", "-r", "--name-only", "forecasts").split() == [
        "closing/attempts.json"]


def test_unpointed_it_is_still_the_hubs_state_commit_to_the_running_ref(checkout):
    origin, work = checkout
    (work / "state" / "odds.json").write_text('{"remaining": 399}\n')
    r = _commit_state(work, "odds - close capture")
    assert r.returncode == 0, r.stderr
    assert _git(origin, "show", "main:state/odds.json") == '{"remaining": 399}\n'
