#!/usr/bin/env bash
# One cycle's close check for the game-window loop (#495): if a kickoff is imminent and no
# capture is inside its window yet, ask `odds-close.yml` to take one. Spends no credits itself.
#
#   LIVE_LOOP_LEAGUE=nfl close-dispatch.sh <dispatch-command...>
#
# Run as `live-loop.sh`'s `CYCLE_HOOK`, which exports the league the loop is for.
#
# The dispatch command is an argument for the reason `live-loop.sh` takes its deploy command:
# it is the one thing a test cannot let happen for real.
#
# `hub.closing --due` exits 0 due, 1 not due, 2 cannot tell. Only 0 dispatches: an unknown is
# not a reason to spend. Syncing `state/` first keeps "due" honest -- the dispatched run commits its
# capture, and a stale checkout would call the same window due again every cycle. It is
# best-effort and read-only (no `contents: write` here), and a failure is an ordinary cycle.
#
# The professional league only: the odds market polled is the NFL's.
set -uo pipefail

league=${LIVE_LOOP_LEAGUE:-}
[ "$league" = "nfl" ] || exit 0
[ "$#" -gt 0 ] || { echo "close-dispatch.sh: no dispatch command given" >&2; exit 2; }

cd "$(dirname "$0")/../.." || exit 2

# Only `state/` is brought up to date: the loop has the overlay's files modified in its tree, and
# a pull would refuse over them. `CLOSE_SYNC_REF` is the branch (unset: no sync, as in a test).
if [ -n "${CLOSE_SYNC_REF:-}" ]; then
  git fetch -q --depth=1 origin "$CLOSE_SYNC_REF" 2>/dev/null \
    && git checkout -q FETCH_HEAD -- state 2>/dev/null || true
fi
uv run python -m hub.closing --due
case "$?" in
  0) echo "  close capture due; dispatching"; "$@" || echo "  could not dispatch the close capture" ;;
  1) ;;
  *) echo "  could not tell whether a close capture is due" ;;
esac
exit 0
