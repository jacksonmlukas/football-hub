#!/usr/bin/env bash
# Quality gate on every edit. Ordered cheapest-first so Claude sees the fastest signal first.
# Pyrefly runs in well under a second on a repo this size, which is the whole reason it
# belongs in a per-edit hook and mypy did not.
#
# Every step reports three outcomes, not two: passed, failed, and *could not run*. The
# third used to be folded into the second, and that folding cost four tickets' worth of
# unchecked edits. `pyrefly check` with no arguments resolves its own file set in project
# mode, where it honours `.git/info/exclude` -- which ignores `.claude/worktrees/`. Inside
# an agent's worktree it matched zero files and exited non-zero; this script called that a
# type error and returned; the pytest step below never ran. A gate that cannot say "I did
# not run" reports something plausible instead, which is worse than reporting nothing.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TARGETS=(src tests)
cd "$ROOT" || { echo "TDD gate could not run: no repo root at $ROOT." >&2; exit 2; }

fail_hook() {  # the gate itself is broken; say so in its own words, not the code's
  echo "TDD gate could not run: $1" >&2
  shift
  [ $# -gt 0 ] && printf '%s\n' "$@" >&2
  exit 2
}

# The environment is a precondition, not a finding, and "not installed" must not read as
# "your code is wrong". #142 made this branch unreachable in the case that produced it -- the
# toolchain is a default `[dependency-groups]` group, so the venv uv builds for a fresh
# worktree already has pyrefly. The branch stays: it is what would report a venv broken some
# other way, and the hint is now the plain sync (there is no `dev` extra to name any more).
if ! probe=$(uv run pyrefly --version 2>&1); then
  fail_hook "pyrefly is not installed in this environment." \
            "  $probe" \
            "  Rebuild the environment: uv sync"
fi

# Naming the paths is what makes the gate portable: an explicit path bypasses pyrefly's
# `project-excludes` and its ignore-file handling, so a worktree checks what the primary
# checkout checks.
# There are files to check. Asked of the tree rather than of pyrefly, so that "the checker
# found nothing wrong" and "the checker was pointed at nothing" cannot arrive as one answer.
expected=$(find "${TARGETS[@]}" -name '*.py' -not -path '*/.venv/*' 2>/dev/null | wc -l)
if [ "$expected" -eq 0 ]; then
  fail_hook "no Python files under ${TARGETS[*]} in $ROOT, so there was nothing to check."
fi

types=$(uv run pyrefly check "${TARGETS[@]}" 2>&1)
status=$?
# A non-zero exit means a finding *or* a refusal to start, and only one of those is about
# the code. A finding names a file and a line; nothing else does. Matching that shape rather
# than pyrefly's "No Python files matched" sentence means a reworded message downgrades to
# "could not run" instead of quietly restoring the defect this hook was written for.
if [ $status -ne 0 ] && ! grep -qE '[^ ]+\.py:[0-9]+' <<<"$types"; then
  fail_hook "pyrefly exited $status without naming a file, and $expected Python files exist \
under ${TARGETS[*]}. It checked nothing." \
            "$(printf '%s\n' "$types" | tail -5)"
fi
if [ $status -ne 0 ]; then
  echo "Type check failed. Fix before continuing:" >&2
  printf '%s\n' "$types" | tail -20 >&2
  exit 2
fi

out=$(uv run pytest tests/unit tests/contracts -q --no-header -x 2>&1)
if [ $? -ne 0 ]; then
  echo "TDD gate failed. Fix before continuing:" >&2
  printf '%s\n' "$out" | tail -30 >&2
  exit 2
fi
# Name both halves. The silence of the old success line is what let a dead type check pass
# for a session without anyone noticing which of the two had stopped running.
echo "pyrefly clean (${TARGETS[*]}); $(printf '%s\n' "$out" | tail -1)"
