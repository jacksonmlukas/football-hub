"""The per-edit quality gate.

The gate was silently dead for every agent working in a git worktree, and stayed dead
long enough to wave through four tickets' worth of edits. `pyrefly check` with no
arguments resolves its own file set in project mode, where it honours `.git/info/exclude`
-- which ignores `.claude/worktrees/`. Inside a worktree it therefore matched zero files
and exited non-zero; the hook called that a type error and returned; the pytest step
never ran at all.

That is the failure mode this repo keeps rediscovering, in its worst form. Not a check
that cannot fail -- a check that cannot *pass*, whose only effect was to skip the tests
behind it while reporting something plausible.

So the gate now reports three outcomes, not two: passed, failed, and could not run. These
tests hold the third one open. They drive the hook as a subprocess with a stub `uv` on
PATH, because the real one would re-enter the very suite this file lives in.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".claude" / "hooks" / "tdd_gate.sh"

# What pyrefly actually printed inside a worktree on 2026-09-06, trimmed to the line the
# gate has to recognise. Kept verbatim: if pyrefly reworded it, this fixture is the thing
# that should go red, not the gate in production.
NOTHING_MATCHED = (
    "No Python files matched patterns `/repo/src`, `/repo/tests`"
)

STUB = """#!/usr/bin/env bash
echo "$*" >> "$UV_CALL_LOG"
case "$*" in
  *"pyrefly --version"*)
    if [ -n "${STUB_NO_PYREFLY:-}" ]; then
      echo "error: Failed to spawn: \\`pyrefly\\`" >&2
      exit 2
    fi
    echo "pyrefly 1.2.0"; exit 0 ;;
  *"pyrefly check"*)
    printf '%s\\n' "${STUB_PYREFLY_OUT:-}"; exit "${STUB_PYREFLY_EXIT:-0}" ;;
  *pytest*)
    printf '%s\\n' "${STUB_PYTEST_OUT:-}"; exit "${STUB_PYTEST_EXIT:-0}" ;;
esac
echo "stub uv: unexpected call: $*" >&2
exit 99
"""


@pytest.fixture
def gate(tmp_path):
    """Run the real hook with a scripted `uv`, and report what it was asked to do."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(STUB)
    (bin_dir / "uv").chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")

    def run(**env) -> tuple[subprocess.CompletedProcess, str]:
        e = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "UV_CALL_LOG": str(log), **{k: str(v) for k, v in env.items()}}
        proc = subprocess.run([str(HOOK)], capture_output=True, text=True, env=e)
        return proc, log.read_text()
    return run


def test_the_gate_checks_the_files_it_names_rather_than_the_ones_pyrefly_infers(gate):
    """The worktree fix. Naming the paths is what makes the gate portable."""
    _, calls = gate()
    check = [c for c in calls.splitlines() if "pyrefly check" in c]
    assert check, f"pyrefly was never invoked; calls were: {calls!r}"
    assert "src" in check[0] and "tests" in check[0], (
        f"pyrefly was invoked without explicit paths: {check[0]!r}. In project mode it "
        "honours .git/info/exclude, which ignores .claude/worktrees/, so inside an "
        "agent's worktree it resolves zero files."
    )


def test_checking_nothing_is_reported_as_the_gate_failing_not_the_code(gate):
    """The defect itself: zero files matched must not read as a type error."""
    proc, calls = gate(STUB_PYREFLY_EXIT=1, STUB_PYREFLY_OUT=NOTHING_MATCHED)
    assert proc.returncode == 2
    assert "could not run" in proc.stderr.lower(), (
        f"a gate that checked nothing reported: {proc.stderr!r}"
    )
    assert "type check failed" not in proc.stderr.lower(), (
        "checking zero files was reported as a type error, which is how this gate "
        "stayed dead through four tickets"
    )
    assert "pytest" not in calls, "the gate ran tests it could not vouch for"


def test_a_missing_pyrefly_names_itself_rather_than_the_code(gate):
    """A worktree venv without the dev extras has no pyrefly at all."""
    proc, calls = gate(STUB_NO_PYREFLY=1)
    assert proc.returncode == 2
    assert "could not run" in proc.stderr.lower()
    assert "type check failed" not in proc.stderr.lower()
    assert "uv sync" in proc.stderr, "the message should say how to fix the environment"
    assert "pyrefly check" not in calls


def test_a_real_type_error_still_fails_the_gate_and_holds_back_the_tests(gate):
    """The gate's original job, unchanged. Cheapest signal first, and it short-circuits."""
    proc, calls = gate(STUB_PYREFLY_EXIT=1,
                       STUB_PYREFLY_OUT="src/hub/x.py:4:1 bad-return [bad-return]")
    assert proc.returncode == 2
    assert "type check failed" in proc.stderr.lower()
    assert "could not run" not in proc.stderr.lower()
    assert "bad-return" in proc.stderr, "the finding itself should reach the reader"
    assert "pytest" not in calls


def test_a_failing_test_fails_the_gate(gate):
    proc, _ = gate(STUB_PYTEST_EXIT=1, STUB_PYTEST_OUT="1 failed, 2309 passed")
    assert proc.returncode == 2
    assert "tdd gate failed" in proc.stderr.lower()
    assert "1 failed" in proc.stderr


def test_a_clean_run_says_both_halves_ran(gate):
    """`Reaching the pytest step is observable`. Silence is what hid the defect."""
    proc, calls = gate(STUB_PYTEST_OUT="2310 passed, 1 skipped in 400.00s")
    assert proc.returncode == 0, f"stderr: {proc.stderr!r}"
    assert "pytest" in calls, "the gate passed without running the suite"
    assert "2310 passed" in proc.stdout
    assert "pyrefly" in proc.stdout.lower(), (
        "a passing gate should name the check it ran, so a gate that silently stops "
        "running one is visible in the transcript"
    )
