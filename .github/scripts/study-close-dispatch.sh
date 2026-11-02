#!/usr/bin/env bash
# One cycle's check for the LLM study's closing line (#450): if an NFL kickoff is within 30
# minutes and no study attempt covers its window, ask `forecast-close.yml` to take one. Spends
# no credits itself.
#
#   LLM_FORECASTS_ENABLED=true STUDY_SYNC_REF=forecasts study-close-dispatch.sh <dispatch-command...>
#
# Run from `close-dispatch.sh` (the game-window loop's `CYCLE_HOOK`) through `STUDY_CLOSE_HOOK`,
# because the loop is the one thing awake at kickoff: scheduled starts arrive 4-8.5h late
# (#468). **Every league's loop asks**, not only the professional one's: the check reads the NFL
# schedule, and a December Saturday NFL game is played inside the college window's loop.
#
# Off unless `LLM_FORECASTS_ENABLED` is `true` -- the same switch as the study's jobs, which
# also refuse on their own `if:`. The attempt record lives on the study's branch, never main,
# so it is read from `STUDY_SYNC_REF` (unset: no sync, as in a test) into a scratch directory;
# without it every cycle in the window would dispatch. `close_line due` exits 0 due, 1 not due,
# 2 cannot tell; only 0 dispatches -- an unknown is not a reason to spend. A failure here is
# an ordinary cycle.
set -uo pipefail

[ "${LLM_FORECASTS_ENABLED:-}" = "true" ] || exit 0
[ "$#" -gt 0 ] || { echo "study-close-dispatch.sh: no dispatch command given" >&2; exit 2; }

cd "$(dirname "$0")/../.." || exit 2

scratch=$(mktemp -d) || exit 0
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/closing"
if [ -n "${STUDY_SYNC_REF:-}" ]; then
  if git fetch -q --depth=1 origin "$STUDY_SYNC_REF" 2>/dev/null; then
    git show FETCH_HEAD:closing/attempts.json > "$scratch/closing/attempts.json" 2>/dev/null \
      || rm -f "$scratch/closing/attempts.json"
  fi
fi
uv run python -m hub.forecasters.close_line due --out "$scratch"
case "$?" in
  0) echo "  study closing line due; dispatching"; "$@" || echo "  could not dispatch the study's closing line" ;;
  1) ;;
  *) echo "  could not tell whether the study's closing line is due" ;;
esac
exit 0
