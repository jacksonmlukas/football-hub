#!/usr/bin/env bash
# The watchdog's heartbeat check, run by `live.yml` at the end of its own loop (#457).
#
# **Why this lives in the loop.** `watchdog.yml` is a cron and `live.yml` is a cron, and
# GitHub drops scheduled runs under load, so the two were dropped together: two game
# windows (2026-10-02 and 2026-10-06) ran with a refresher but no check, after #391's fix
# (#416). A check that is a step of the refresher cannot be dropped without the refresher.
#
# What it does is what `watchdog.yml`'s heartbeat job does, minus the window question: a
# loop that has just finished is by construction inside a window, so there is nothing to be
# "late" for. It measures the published `live.json` (via heartbeat.sh) and then
#   * healthy -> closes the open `<!-- watchdog-heartbeat -->` incidents, and only those;
#   * stale / unreachable / unreadable -> files or updates the one open incident.
# The marker and title are the heartbeat job's own, so the two share one incident rather
# than racing to file two.
#
# **It never fails the refresh.** The exit status is 0 unless it found a problem AND could
# not file it, in which case it exits 1 so the step is visibly red; the workflow marks the
# step `continue-on-error` so that red never stops the loop's hand-over or fails the job.
#
#   env: HEARTBEAT_URL  published live.json
#        REPO           owner/name, for `gh`
#        THRESHOLD      seconds (default 1500, the heartbeat job's)
set -uo pipefail

url=${HEARTBEAT_URL:?HEARTBEAT_URL}
repo=${REPO:?REPO}
threshold=${THRESHOLD:-1500}
here=$(cd "$(dirname "$0")" && pwd)
MARKER="<!-- watchdog-heartbeat -->"

result=$("$here/heartbeat.sh" "$url" "$threshold")
age=$(echo "$result" | cut -d' ' -f2)
echo "heartbeat: $result"

existing() {
  gh issue list --repo "$repo" --label incident --state open --json number,body \
    --jq ".[] | select(.body | contains(\"$MARKER\")) | .number" 2>/dev/null
}

case "$result" in
  ok*)
    for n in $(existing); do
      gh issue close "$n" --repo "$repo" --reason completed \
        --comment "Heartbeat healthy again (age ${age}s), measured by the check at the end of the live loop." \
        || echo "::warning::could not close incident #$n"
    done
    exit 0 ;;
  stale*)       why="the published live.json has not been refreshed" ;;
  unreachable*) why="the published live.json could not be fetched"; age=unknown ;;
  *)            why="live.json carries no readable generated_at, so its writer and this check disagree about the schema"; age=unknown ;;
esac

BODY="$MARKER
Heartbeat age: ${age}s (threshold ${threshold}s).

Cause: ${why}.

Measured at $(date -u +%H:%M:%SZ) by the check at the end of the \`live\` loop, which has just
spent its run refreshing the overlay: the deploys it dispatched are not landing, or ESPN
has been unreachable throughout. The dashboard serves last-good state, so nothing is
broken -- it is just not moving. This issue closes on the next healthy check."

first=$(existing | head -1)
if [ -n "$first" ]; then
  gh issue comment "$first" --repo "$repo" --body "$BODY"
else
  gh issue create --repo "$repo" --title "Live poller down" --label incident --body "$BODY" \
    || gh issue create --repo "$repo" --title "Live poller down" --body "$BODY"
fi
if [ $? -ne 0 ]; then
  echo "::error::heartbeat is stale AND the incident could not be filed"
  exit 1
fi
exit 0
