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
# **Why `.claude/hooks` and not a git `pre-push`.** A git hook is not cloned; this directory
# is, and the harness runs it on every agent tool call in every worktree. The session that
# pushed red was an agent session. A maintainer at a terminal is not bound by this file and
# knows it; `ci.yml` is that reader's net.
#
# **Three outcomes, not two**, as every gate in this repo reports: passed, failed, and *could
# not run*. The third refuses too. A push gate that cannot run the contracts and lets the
# push through has reported something plausible instead of nothing, which is the defect
# `tdd_gate.sh`'s header records losing four tickets to.
#
# Matches only a Bash tool call whose command drives `git push`. Everything else passes at
# once; this hook must never be the thing that slows an edit.

EVENT="$(cat 2>/dev/null || true)"
: "${EVENT:=}"

refuse () { printf '%s\n' "$1" >&2; exit 2; }

printf '%s' "$EVENT" | grep -Eq '"tool_name"[[:space:]]*:[[:space:]]*"Bash"' || exit 0
printf '%s' "$EVENT" | grep -Eq '\bgit([[:space:]]+-[A-Za-z-]+([[:space:]=][^[:space:]]+)?)*[[:space:]]+push\b' || exit 0
# A dry run pushes nothing and is how a caller asks what *would* be pushed.
printf '%s' "$EVENT" | grep -Eq '\bgit\b[^&|;]*\bpush\b[^&|;]*--dry-run' && exit 0

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || refuse "push gate: not inside a git repository, so the contracts cannot be located; refusing the push rather than guessing"
cd "$ROOT" || refuse "push gate: cannot enter $ROOT"
[ -d tests/contracts ] || refuse "push gate: $ROOT/tests/contracts does not exist here, so the gate cannot run; refusing the push rather than passing it unchecked"
command -v uv >/dev/null 2>&1 || refuse "push gate: uv is not on PATH, so the contracts cannot run; refusing the push rather than passing it unchecked"

OUT="$(uv run ruff check src tests 2>&1)"; RC=$?
[ "$RC" -eq 0 ] || refuse "push gate: ruff failed (exit $RC), so this push would go red in CI. Fix it, then push.
$(printf '%s\n' "$OUT" | tail -20)"

OUT="$(uv run pytest tests/contracts -q -p no:cacheprovider 2>&1)"; RC=$?
if [ "$RC" -eq 0 ]; then
  exit 0
elif [ "$RC" -eq 5 ]; then
  refuse "push gate: pytest collected no contract tests (exit 5) -- the gate could not run. Refusing the push rather than passing it unchecked."
else
  refuse "push gate: the contracts failed (pytest exit $RC), so this push would go red in CI. Fix-up as a new commit -- main is never amended -- then push.
$(printf '%s\n' "$OUT" | grep -E 'FAILED|Error|assert|passed|failed' | tail -20)"
fi
