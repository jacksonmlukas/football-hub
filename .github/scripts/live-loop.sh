#!/usr/bin/env bash
# Refresh the published overlay on a cadence GitHub actually delivers, for one window.
#
# A scheduled workflow is not a clock. Measured on 2026-09-05 against a `*/10` cron, runs
# landed 97-126 minutes apart, median 106, each completing in under a minute -- so the
# ten-minute window refresh was a ninety-minute one, and the watchdog's 1500s threshold made
# the published overlay stale by its own definition for most of every game window. Issue #91.
# The fix is not a tighter cron: it is one long-lived job that loops, started by the same
# cron and by the slate's completion, whose *start* being late costs coverage once rather
# than every ten minutes.
#
#   live-loop.sh <seconds-to-run> <seconds-between-cycles> <league> <out-dir> <deploy-command...>
#
# The deploy command is an argument for the reason `heartbeat.sh` takes a URL: it is the one
# thing a test cannot let happen for real, and passing it in is what lets the loop itself be
# run and watched rather than reasoned about. `tests/contracts/test_live_loop.py` runs this
# script against a recorder and an ESPN that is made to fail on demand.
#
# **The three outcomes, which are the whole of this script.** `hub.publish --live` reports
# them as exit codes, because a caller branching on the wording of a sentence breaks the day
# the sentence is reworded:
#
#   0  ESPN answered -- with games, or with an empty board, which under ADR-0018 is ESPN
#      saying there are no games and is worth publishing. Write and deploy.
#   3  ESPN was not reached. `hub.fetch.espn._get` degrades to a last-good cache for the
#      dashboard's sake, so `hub.publish.live` refuses that cache and writes nothing at all.
#      **Do not deploy, and do not let anything advance the stamp.** A deploy here would
#      republish the carried-forward overlay under a fresh `generated_at`, so the heartbeat
#      would report the page fresh while it showed frozen scores -- and the watchdog reads
#      that stamp, so it could never fire again. The failure would erase its own evidence.
#
# Deploying on every answer, changed payload or not, is deliberate: the stamp means "we asked
# ESPN at this time", which is true of an unchanged scoreboard, and skipping unchanged
# payloads would make a quiet stretch indistinguishable from an outage.
#
# No `set -e`. A refusing fetch and a failed dispatch are both ordinary cycles here, and a
# loop that exits on the first of them is the unattended refresher stopping silently, which
# is the defect this replaces rather than a stricter version of it.
set -uo pipefail

duration=${1:?usage: live-loop.sh <seconds-to-run> <seconds-between-cycles> <league> <out-dir> <deploy-command...>}
interval=${2:?usage: live-loop.sh <seconds-to-run> <seconds-between-cycles> <league> <out-dir> <deploy-command...>}
league=${3:?usage: live-loop.sh <seconds-to-run> <seconds-between-cycles> <league> <out-dir> <deploy-command...>}
out=${4:?usage: live-loop.sh <seconds-to-run> <seconds-between-cycles> <league> <out-dir> <deploy-command...>}
shift 4
if [ "$#" -eq 0 ]; then
  echo "live-loop.sh: no deploy command given; a loop that refreshes and never publishes" >&2
  echo "  would leave the page exactly as frozen as it is now" >&2
  exit 2
fi

cd "$(dirname "$0")/../.." || exit 2

started_at=$(date +%s)
deadline=$(( started_at + duration ))
cycles=0
deploys=0
held=0

echo "live-loop: ${league} board into ${out}, a cycle every ${interval}s, for up to ${duration}s"

while :; do
  cycles=$(( cycles + 1 ))
  cycle_at=$(date +%s)
  echo "--- cycle ${cycles} at +$(( cycle_at - started_at ))s"

  uv run python -m hub.publish --live --league "$league" --out "$out"
  refreshed=$?

  case "$refreshed" in
    0)
      if "$@"; then
        deploys=$(( deploys + 1 ))
      else
        # The overlay is written and the stamp is honest; only the publish of it failed.
        # Reported and carried, because the next cycle is minutes away and a loop that dies
        # here stops refreshing for the rest of the window.
        echo "  the deploy command failed; the next cycle will publish this or something newer"
      fi
      ;;
    # 3 is `hub.publish.NOTHING_FRESH`. The number is written here rather than read out of
    # Python -- one interpreter start per loop to learn a constant -- and
    # `tests/contracts/test_live_loop.py` fails if the two ever disagree.
    3)
      held=$(( held + 1 ))
      echo "  ESPN was not reached: no deploy, and generated_at stands where it was"
      ;;
    *)
      # Not one of the outcomes `--live` declares, so it is this program being broken rather
      # than ESPN being down. Still not a deploy: the overlay's state is unknown.
      held=$(( held + 1 ))
      echo "  the refresh exited ${refreshed}, which it does not declare; not deploying"
      ;;
  esac

  now=$(date +%s)
  if [ $(( now + interval )) -ge "$deadline" ]; then
    break
  fi
  # Measured from the start of the cycle, so the cadence is the interval rather than the
  # interval plus however long ESPN and the dispatch took. That is what makes the cadence a
  # number this can be held to.
  nap=$(( interval - (now - cycle_at) ))
  [ "$nap" -lt 0 ] && nap=0
  sleep "$nap"
done

echo "live-loop: ${cycles} cycles over $(( $(date +%s) - started_at ))s -- ${deploys} deployed, ${held} held"
