"""The near-kickoff close capture is wired so it can fire, and cannot fire by accident (#495).

Scheduled odds polls start 4-8.5h late (#468), so the close is taken by a workflow the
game-window loop dispatches. Untested on Actions -- there is no live CI for a workflow change --
so these tests execute the dispatcher's shell against stub `uv` and dispatch commands, and read
the workflow files for the properties that keep the spend bounded. Each has its positive
control in the same file: the dispatcher is made to dispatch when a capture is due, so a harness
that could not see a dispatch would fail the "does not dispatch" cases rather than pass them.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
DISPATCH = ROOT / ".github" / "scripts" / "close-dispatch.sh"
LOOP = ROOT / ".github" / "scripts" / "live-loop.sh"


def _run(tmp_path: Path, *, due_exit: int, league: str) -> list[str]:
    """Run the dispatcher with `hub.closing --due` answering `due_exit`; return dispatches."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f"#!/usr/bin/env bash\nexit {due_exit}\n")
    uv.chmod(0o755)
    log = tmp_path / "dispatched"
    recorder = tmp_path / "record.sh"
    recorder.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log}"\n')
    recorder.chmod(0o755)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "LIVE_LOOP_LEAGUE": league}
    subprocess.run([str(DISPATCH), str(recorder), "odds-close.yml"], env=env, check=True,
                   capture_output=True, text=True, timeout=30)
    return log.read_text().splitlines() if log.exists() else []


def test_a_due_capture_in_the_nfl_window_dispatches_once(tmp_path):
    assert _run(tmp_path, due_exit=0, league="nfl") == ["odds-close.yml"]


def test_nothing_due_dispatches_nothing(tmp_path):
    assert _run(tmp_path, due_exit=1, league="nfl") == []


def test_an_unknown_is_not_a_reason_to_spend(tmp_path):
    assert _run(tmp_path, due_exit=2, league="nfl") == []


def test_a_due_capture_in_the_college_window_dispatches_too(tmp_path):
    """A December Saturday NFL game is played inside the college loop (#532)."""
    assert _run(tmp_path, due_exit=0, league="cfb") == ["odds-close.yml"]


def test_a_college_window_with_nothing_due_dispatches_nothing(tmp_path):
    assert _run(tmp_path, due_exit=1, league="cfb") == []


# --- the two-loop case (#532) ------------------------------------------------------------
#
# The stub `uv` answers `hub.closing --due` with the REAL `hub.closing.is_due` over one NFL
# Saturday kickoff and an attempt file in a scratch directory, and the dispatch command runs the
# REAL `hub.closing.capture` against the same file with a counting fetch. So what keeps a
# second hook from capturing again is the production attempt record and the weekly cap, not a
# rule of the harness.

_KICK = "2026-12-12T20:00:00"                         # a Saturday, inside the college window
_NOW = "2026-12-12T19:50:00"                          # ten minutes before kickoff

_HARNESS = """
import sys
from datetime import datetime
from pathlib import Path
import polars as pl
from hub import closing

work = Path(sys.argv[1])
mode = sys.argv[2]
now = datetime.fromisoformat("NOW")
kicks = pl.DataFrame({"game_id": ["2026_14_AAA_BBB"], "kickoff": [datetime.fromisoformat("KICK")]})
closing.ATTEMPTS_FILE = work / "attempts.json"
if mode == "due":
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.replace(tzinfo=tz)
    closing.datetime = _Clock
    closing.kickoff_table = lambda season: kicks
    closing.store.lines = lambda season, base=None: pl.DataFrame()
    sys.exit(0 if closing.is_due() else 1)
calls = work / "fetch-calls"
def fetch(*a, **k):
    with calls.open("a") as f:
        f.write("call\\n")
closing.capture(now=now, kickoffs=kicks, archive=pl.DataFrame(), fetch=fetch,
                state_path=work / "quota.json", attempts_path=work / "attempts.json")
"""


def _two_loop_env(tmp_path: Path) -> dict[str, str]:
    harness = tmp_path / "harness.py"
    harness.write_text(_HARNESS.replace("NOW", _NOW).replace("KICK", _KICK))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{harness}" "{tmp_path}" due\n')
    uv.chmod(0o755)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "src")}


def _capture_cmd(tmp_path: Path) -> Path:
    """What the dispatched `odds-close.yml` run does: `hub.closing.capture`, for real."""
    cmd = tmp_path / "capture.sh"
    cmd.write_text(f'#!/usr/bin/env bash\necho run >> "{tmp_path}/runs"\nexec "{sys.executable}" "{tmp_path}/harness.py" '
                   f'"{tmp_path}" capture\n')
    cmd.chmod(0o755)
    return cmd


def _runs(tmp_path: Path) -> int:
    f = tmp_path / "runs"
    return len(f.read_text().splitlines()) if f.exists() else 0


def _fetches(tmp_path: Path) -> int:
    f = tmp_path / "fetch-calls"
    return len(f.read_text().splitlines()) if f.exists() else 0


def _hook(env: dict[str, str], league: str, dispatch: Path) -> None:
    subprocess.run([str(DISPATCH), str(dispatch)], env={**env, "LIVE_LOOP_LEAGUE": league},
                   check=True, capture_output=True, text=True, timeout=60)


def test_a_saturday_nfl_kickoff_in_the_college_loop_gets_one_capture(tmp_path):
    _hook(_two_loop_env(tmp_path), "cfb", _capture_cmd(tmp_path))
    assert _fetches(tmp_path) == 1


@pytest.mark.parametrize("order", [("cfb", "nfl"), ("nfl", "cfb")])
def test_both_loops_hooks_for_the_same_kickoff_capture_once(tmp_path, order):
    """Each hook fires after the other's dispatched run has landed its attempt."""
    env = _two_loop_env(tmp_path)
    for league in order:
        _hook(env, league, _capture_cmd(tmp_path))
    assert _runs(tmp_path) == 1, "the second hook dispatched a window already spent"
    assert _fetches(tmp_path) == 1


def test_two_hooks_that_both_saw_it_due_still_capture_once(tmp_path):
    """The hand-over overlap: both ask before either run lands, so both dispatch. The runs are
    serialised (`odds-close.yml` never cancels); the second finds the window spent."""
    env = _two_loop_env(tmp_path)
    queue = tmp_path / "queue.sh"
    queued = tmp_path / "queued"
    queue.write_text(f'#!/usr/bin/env bash\necho dispatched >> "{queued}"\n')
    queue.chmod(0o755)
    for league in ("cfb", "nfl"):
        _hook(env, league, queue)
    runs = queued.read_text().splitlines()
    assert len(runs) == 2, "both hooks should have dispatched before either run landed"
    for _ in runs:
        subprocess.run([str(_capture_cmd(tmp_path))], env=env, check=True,
                       capture_output=True, text=True, timeout=60)
    assert _fetches(tmp_path) == 1


def test_the_loop_runs_the_hook_before_the_refresh_so_an_espn_outage_cannot_skip_it():
    text = LOOP.read_text()
    assert text.index("CYCLE_HOOK") < text.index("uv run python -m hub.publish --live"), (
        "the hook runs after the ESPN refresh, which can be skipped or fail")
    assert 'LIVE_LOOP_LEAGUE="$league" $CYCLE_HOOK' in text


def test_the_window_loop_dispatches_the_close_workflow_and_stays_read_only():
    doc = yaml.safe_load((WORKFLOWS / "live.yml").read_text())
    step = next(s for s in doc["jobs"]["refresh"]["steps"]
                if s.get("name") == "Refresh the overlay through the window")
    hook = step["env"]["CYCLE_HOOK"]
    assert "close-dispatch.sh" in hook and "gh workflow run odds-close.yml" in hook
    assert doc["jobs"]["refresh"]["permissions"]["contents"] == "read"


def test_the_close_workflow_is_dispatch_only_and_never_cancelled_mid_call():
    doc = yaml.safe_load((WORKFLOWS / "odds-close.yml").read_text())
    on = doc.get(True, doc.get("on"))              # YAML reads a bare `on` as True
    assert set(on) == {"workflow_dispatch"}, "a schedule here would be a second, late poller"
    assert doc["concurrency"]["cancel-in-progress"] is False
    job = doc["jobs"]["capture"]
    assert job["timeout-minutes"] <= 15
    runs = [s.get("run", "") for s in job["steps"]]
    assert any("hub.closing --capture" in r for r in runs)
    keyed = next(s for s in job["steps"] if "hub.closing --capture" in s.get("run", ""))
    assert "ODDS_API_KEY" in keyed["env"]


def test_a_scheduled_odds_poller_is_still_the_only_cron_that_spends_credits():
    for path in WORKFLOWS.glob("*.yml"):
        if path.name in {"slate.yml", "odds-close.yml"}:
            continue
        jobs = yaml.safe_load(path.read_text()).get("jobs") or {}
        commands = " ".join(st.get("run", "") for j in jobs.values() for st in j["steps"])
        assert "hub.fetch.odds" not in commands and "hub.closing" not in commands, (
            f"{path.name} calls the odds poller; the credit budget accounts for slate.yml and "
            f"odds-close.yml only")


@pytest.mark.parametrize("script", [DISPATCH, LOOP])
def test_the_scripts_are_executable_and_parse(script):
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_the_dispatch_names_the_same_branch_the_state_is_synced_from():
    doc = yaml.safe_load((WORKFLOWS / "live.yml").read_text())
    step = next(s for s in doc["jobs"]["refresh"]["steps"]
                if s.get("name") == "Refresh the overlay through the window")
    assert step["env"]["CLOSE_SYNC_REF"] == "${{ github.ref_name }}"
    assert step["env"]["CYCLE_HOOK"].endswith("--ref ${{ github.ref_name }}"), (
        "the dispatched run and the state sync could land on different branches")


def test_the_attempt_is_committed_before_the_request_and_the_capture_always_commits():
    doc = yaml.safe_load((WORKFLOWS / "odds-close.yml").read_text())
    steps = doc["jobs"]["capture"]["steps"]
    cap = next(s for s in steps if "hub.closing --capture" in s.get("run", ""))
    assert "--commit-attempt" in cap["run"], "the marker is not published before the request"
    commit = next(s for s in steps if "commit-state.sh" in s.get("run", ""))
    assert commit["if"] == "always()"
    assert steps.index(commit) > steps.index(cap)
    assert os.access(ROOT / ".github" / "scripts" / "commit-state.sh", os.X_OK)
