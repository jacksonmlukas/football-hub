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
import json
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

    def run(edited: str | None = None, **env) -> tuple[subprocess.CompletedProcess, str]:
        e = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "UV_CALL_LOG": str(log), **{k: str(v) for k, v in env.items()}}
        # The hook is a PostToolUse hook: the harness hands it the tool event on stdin, the
        # same shape `guard_data_reads.py` already reads. No event means no named file.
        event = "" if edited is None else json.dumps(
            {"tool_name": "Edit", "tool_input": {"file_path": edited}})
        proc = subprocess.run([str(HOOK)], input=event,
                              capture_output=True, text=True, env=e)
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


def test_a_reworded_refusal_still_downgrades_to_could_not_run(gate):
    """The gate must not depend on pyrefly's exact sentence.

    The first version of this fix matched the literal `No Python files matched`, and the
    test above hands the stub that same string -- so between them they proved only that the
    hook recognises its own fixture. A pyrefly release that reworded the line would have put
    the original defect straight back, silently, with the suite green.

    What separates the two cases structurally is that a *finding* names a file and a line
    and a refusal to start does not. This is the reworded message, and it must still be
    read as the gate failing rather than the code.
    """
    proc, calls = gate(STUB_PYREFLY_EXIT=1,
                       STUB_PYREFLY_OUT="warning: nothing to do; include patterns matched 0 targets")
    assert proc.returncode == 2
    assert "could not run" in proc.stderr.lower(), (
        f"a reworded refusal was read as a type error: {proc.stderr!r}"
    )
    assert "pytest" not in calls


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


def test_an_edit_runs_the_tests_that_bear_on_it_and_not_the_whole_suite(gate):
    """The cost of a per-edit gate has to fit an edit.

    This ran `pytest tests/unit tests/contracts` -- about 2,380 tests, four minutes -- after
    every Edit and Write. It was invisible for the session in which it was written, because
    the type check above it was failing first and the suite never ran. Once it did run, the
    arithmetic was twenty edits to eighty minutes, and agents began routing around it by
    making their edits through Bash heredocs, which the hook does not see. A gate people
    learn to avoid is worse than one that is merely slow: it still costs everyone who does
    not know the trick, and it has stopped covering the ones who do.
    """
    _, calls = gate(edited="src/hub/draft/board.py")
    ran = [c for c in calls.splitlines() if "pytest" in c]
    assert ran, f"no tests ran for an edit to a covered module; calls were {calls!r}"
    assert "tests/unit tests/contracts" not in ran[0], (
        "the whole suite ran for a single-file edit"
    )
    assert "test_board" in ran[0], (
        f"the tests that reference the edited module were not the ones run: {ran[0]!r}"
    )


def test_a_module_no_test_references_says_so_rather_than_passing_quietly(gate):
    """Nothing to run is a fact about the coverage, not a pass.

    The failure this repo keeps rediscovering is a check that cannot fail. A gate that finds
    no tests for a module and prints its usual success line is exactly that, and it would
    report most loudly on precisely the modules that need tests most.
    """
    # Assembled from parts: spelling the module name here would put it in this very file,
    # and the scan would dutifully find its own test as the one that references it.
    unreferenced = "src/hub/" + "zqx" + "_absent" + ".py"
    proc, calls = gate(edited=unreferenced)
    assert "pytest" not in calls, "a test run was invented for a module nothing references"
    assert "no test" in proc.stdout.lower() or "no test" in proc.stderr.lower(), (
        f"the gate passed silently on an unreferenced module: {proc.stdout!r} {proc.stderr!r}"
    )


def test_editing_a_test_file_runs_that_file(gate):
    """The obvious case, and the one a module-to-test mapping alone would miss."""
    _, calls = gate(edited="tests/unit/test_roster.py")
    ran = [c for c in calls.splitlines() if "pytest" in c]
    assert ran and "tests/unit/test_roster.py" in ran[0], f"got {ran!r}"


def test_an_edit_with_no_python_in_it_type_checks_and_says_it_ran_no_tests(gate):
    """Editing a document is not a reason to run the suite, nor to claim it passed."""
    proc, calls = gate(edited="docs/method.md")
    assert "pyrefly check" in calls, "the type check was skipped"
    assert "pytest" not in calls, "the suite ran for a documentation edit"
    assert proc.returncode == 0
    assert "no test" in proc.stdout.lower()


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
