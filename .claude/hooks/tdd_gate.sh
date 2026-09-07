#!/usr/bin/env bash
# Quality gate on every edit, costed to fit one. Ordered cheapest-first so Claude sees the
# fastest signal first: a whole-project type check, which runs in well under a second on a
# repo this size and is the reason pyrefly belongs in a per-edit hook and mypy did not, and
# then only the tests that reference the file just edited.
#
# **What this no longer runs, and what covers it instead.** It used to run
# `pytest tests/unit tests/contracts` -- about 2,380 tests, four minutes -- after every edit.
# That is not a per-edit cost: twenty edits bought eighty minutes of suite, several agents
# editing at once multiplied it, and this repo has an incident where concurrent suites
# produced 800-second runs and failures that were contention rather than defects. Agents had
# begun making their edits through Bash heredocs, which this hook never sees, so the gate was
# already not covering the people who had worked out how to avoid it. A gate that is switched
# off is worse than a cheaper one, and this repo has that incident too, in
# `preflight_public.sh`'s exemption budget.
#
# The full suite runs in CI, on every push, together with the coverage floor and the per-module
# ratchet. That is the gate that must stay whole. This one exists to catch the mistake while
# the file is still open, and it is honest about being narrower: an edit whose module no test
# references is reported as such rather than passed, because "nothing to run" is a fact about
# the coverage and not a success.
#
# Measured on 2026-09-07, against the 240s the old shape cost on every edit regardless:
#   src/hub/draft/board.py   -> 12 files,   ~20s
#   src/hub/season/roster.py ->  6 files,   ~50s   (guard excision is slow and relevant)
#   docs/method.md           ->  no tests,   ~2s   (the type check alone)
# A central module still costs real time, because a central module genuinely has more
# bearing on it. What has gone is paying for `tests/contracts` in full to change a
# docstring.
#
# The trade it accepts: a change that breaks a *different* module's tests is not caught
# here. The type check catches the signature-shaped half of that, and CI catches the rest.
#
# Every step reports three outcomes, not two: passed, failed, and *could not run*. The
# third used to be folded into the second, and that folding cost four tickets' worth of
# unchecked edits. `pyrefly check` with no arguments resolves its own file set in project
# mode, where it honours `.git/info/exclude` -- which ignores `.claude/worktrees/`. Inside
# an agent's worktree it matched zero files and exited non-zero; this script called that a
# type error and returned; the pytest step below never ran. A gate that cannot say "I did
# not run" reports something plausible instead, which is worse than reporting nothing.
set -uo pipefail

# Drained once, up front: stdin is the tool event and a later read would find it empty.
EVENT=$(cat 2>/dev/null || true)

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

# Which tests bear on the file just edited. The harness hands this hook the tool event on
# stdin, the same shape `guard_data_reads.py` reads; no event means no named file, and then
# there is nothing to select on.
# Pulled out with sed rather than a JSON parser: this runs on every edit, and starting a
# Python interpreter to read one string would put back a slice of the cost being removed.
edited=$(printf '%s' "$EVENT" \
  | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
edited=${edited#"$ROOT"/}

select_tests() {
  case "$1" in
    "")            return 1 ;;                      # no event: nothing to select on
    tests/*)       [ -f "$1" ] && { echo "$1"; return 0; }; return 1 ;;
    *.py)          ;;                               # fall through to the reference scan
    *)             return 1 ;;                      # documentation, config, a shell script
  esac
  # The module as it is imported, e.g. src/hub/draft/board.py -> hub.draft.board. Tests are
  # selected by *referencing* that path rather than by a name-mangling convention, because
  # this repo has modules whose tests do not share their name -- board.py is covered by
  # test_board_build.py and test_board_main.py, and a convention would have found neither.
  local mod=${1#src/}; mod=${mod%.py}; mod=${mod//\//.}
  # `test_*.py` only: `tests/` also holds helper modules -- coveragefloor, panelarchive, the
  # guard library -- which reference plenty of modules and contain no tests to run.
  grep -rl --include='test_*.py' -e "$mod" -e "${mod##*.} import" tests/ 2>/dev/null \
    | sort | head -20
}

# `mapfile` is bash 4; macOS ships 3.2, and a hook that only works on the maintainer's Linux
# runner is a hook that does not run where the edits happen.
bearing=""
while IFS= read -r line; do
  [ -n "$line" ] && bearing="$bearing $line"
done < <(select_tests "$edited")
bearing=${bearing# }

if [ -z "$edited" ]; then
  # No file named -- a missing or unreadable event, not a narrow edit. Fail closed and run
  # everything: a hook that quietly tests nothing when its input surprises it is the defect
  # this file already carries one scar from.
  echo "  the edit named no file, so the whole suite ran rather than none of it"
  bearing="tests/unit tests/contracts"
elif [ -z "$bearing" ]; then
  # Not a pass. "Nothing to run" is a fact about the coverage, and the modules it is truest
  # of are the ones that most need saying so.
  echo "pyrefly clean (${TARGETS[*]}); no test file references $edited, so no test ran"
  exit 0
fi

# Unquoted on purpose: `bearing` is a space-separated list of paths this repo
# controls, and it must split into one argument per file.
# shellcheck disable=SC2086
out=$(uv run pytest $bearing -q --no-header -x 2>&1)
if [ $? -ne 0 ]; then
  echo "TDD gate failed. Fix before continuing:" >&2
  printf '%s\n' "$out" | tail -30 >&2
  exit 2
fi
# Name both halves, and name what was run. The silence of the old success line is what let a
# dead type check pass for a session without anyone noticing which half had stopped running.
echo "pyrefly clean (${TARGETS[*]}); $(printf '%s\n' "$out" | tail -1) [$bearing]"
