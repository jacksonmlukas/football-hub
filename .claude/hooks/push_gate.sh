#!/usr/bin/env bash
# The contracts run before a push, and a push that would go red is refused.
#
# **Why this exists, and why it is a hook and not a note.** On 2026-09-18 a doc commit was
# pushed to `main` with `test_every_avoided_term_in_the_glossary_is_classified` failing. The
# chain that pushed it was `pytest ...; echo "exit=$?"; git add ... && git commit && git push`:
# the `echo` succeeded, and that is what the `&&` saw. CI caught it minutes later and a fix-up
# commit turned `main` green, so the cost was a red interval rather than a red main -- but it
# was the fourth time in this repo's record that a verdict rested on someone reading an exit
# code correctly, and the remedy each earlier time was a sentence in an agent's memory file.
# Everything else settled that week went the other way -- ordering into native dependencies,
# the freeze into an edge, adoption into a grep-able write -- each time because prose in one
# place does not bind a reader somewhere else. A memory file is that shape. This is not.
#
# **Why a push, and not an edit.** `tdd_gate.sh` deliberately stopped running the contracts
# per edit: four minutes, twenty times a session, multiplied by concurrent agents, and it
# measured why. A push is rare -- a few a session -- and it is the moment the property has to
# hold, because CI's own verdict arrives after the commit is already on `main`. Two minutes
# here is the price of never pushing red; it is paid once per push, not once per edit.
#
# **Why `.claude/hooks` and not a git `pre-push`.** Two reasons, and the second is the one
# that is not obvious. A git hook is not cloned; this directory is, and the harness runs it on
# every agent tool call in every worktree. And a git `pre-push` is exactly the thing
# `git push --no-verify` exists to skip -- whereas this intercepts the *tool call*, before git
# sees any flag, so `--no-verify` does not bypass it. For a gate whose whole job is to bind the
# population that keeps `&&`-ing past exit codes, a bypass flag one word long is not a small
# difference. The session that pushed red was an agent session. A maintainer at a terminal is
# not bound by this file and knows it; `ci.yml` is that reader's net.
#
# **Three outcomes, not two**, as every gate in this repo reports: passed, failed, and *could
# not run*. The third refuses too. A push gate that cannot run the contracts and lets the
# push through has reported something plausible instead of nothing, which is the defect
# `tdd_gate.sh`'s header records losing four tickets to. **And the two refusals are worded
# apart, deliberately.** An agent that cannot tell a failing contract from a venv that will
# not start will begin fixing lint against an environment that is the actual problem -- the
# `cached.py` defect in miniature, where `--status` on a corrupt cache printed what a fresh
# pull prints. So `FAILED` names the test and says *fix it*; `COULD NOT RUN` names the command
# it tried and says *fix the environment*; nothing here prints one for the other.
#
# Matches only a Bash tool call whose command drives `git push`. Everything else passes at
# once; this hook must never be the thing that slows an edit.

EVENT="$(cat 2>/dev/null || true)"
: "${EVENT:=}"

# Two refusals, worded apart (see the header): `failed` is about the tree, `could_not_run`
# is about the environment and names the command it tried.
failed        () { printf 'push gate -- FAILED (the push would go red in CI; fix the tree, then push):\n%s\n' "$1" >&2; exit 2; }
could_not_run () { printf 'push gate -- COULD NOT RUN (this is the environment, not the tree; the push is refused rather than passed unchecked):\n  tried: %s\n  %s\n' "$1" "$2" >&2; exit 2; }

printf '%s' "$EVENT" | grep -Eq '"tool_name"[[:space:]]*:[[:space:]]*"Bash"' || exit 0
printf '%s' "$EVENT" | grep -Eq '\bgit([[:space:]]+-[A-Za-z-]+([[:space:]=][^[:space:]]+)?)*[[:space:]]+push\b' || exit 0
# A dry run pushes nothing and is how a caller asks what *would* be pushed.
printf '%s' "$EVENT" | grep -Eq '\bgit\b[^&|;]*\bpush\b[^&|;]*--dry-run' && exit 0

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || could_not_run "git rev-parse --show-toplevel" "not inside a git repository, so the contracts cannot be located"
cd "$ROOT" || could_not_run "cd $ROOT" "cannot enter the repository root"
[ -d tests/contracts ] || could_not_run "ls $ROOT/tests/contracts" "the contracts directory does not exist in this checkout"
command -v uv >/dev/null 2>&1 || could_not_run "uv" "uv is not on PATH; the contracts run under uv and nothing else"

OUT="$(uv run ruff check src tests 2>&1)"; RC=$?
if [ "$RC" -ne 0 ]; then
  # ruff exits 1 on findings and 2 on its own failure to run; only the first is about the tree.
  if [ "$RC" -eq 1 ]; then failed "ruff (exit 1):
$(printf '%s\n' "$OUT" | tail -20)"; else could_not_run "uv run ruff check src tests" "ruff exited $RC before checking anything:
$(printf '%s\n' "$OUT" | tail -10)"; fi
fi

OUT="$(uv run pytest tests/contracts -q -p no:cacheprovider 2>&1)"; RC=$?
case "$RC" in
  0) exit 0 ;;
  1) failed "the contracts (pytest exit 1). Fix-up as a new commit -- main is never amended -- then push:
$(printf '%s\n' "$OUT" | grep -E 'FAILED|Error|assert|passed|failed' | tail -20)" ;;
  # pytest: 2 interrupted, 3 internal error, 4 usage error, 5 no tests collected -- none is a
  # verdict on the tree. A ModuleNotFoundError on a first-party import is the venv, not the code.
  *) could_not_run "uv run pytest tests/contracts -q" "pytest exited $RC without a verdict (2 interrupted, 3 internal error, 4 usage, 5 nothing collected):
$(printf '%s\n' "$OUT" | tail -10)" ;;
esac
