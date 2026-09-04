#!/usr/bin/env bash
# How stale a published artifact is, and which of three different things is wrong.
#
# This existed inline in watchdog.yml and read `.ts`, a field only `hub.fetch.espn.poll`
# writes -- while `hub.publish.live`, which is what actually writes the page's live.json,
# has never written it. `jq -r '.ts // 0'` then made the age `now - 0`, so the watchdog
# reported 1788536723s stale (56.7 years), filed an incident on every run, and could never
# reach the healthy branch that closes one. A monitor that cannot pass is not a monitor.
#
# The `// 0` was the real defect and it is the shape this repo keeps finding: one sentinel
# standing for three causes. Unreachable, unreadable and stale are different failures with
# different fixes, and reporting them as one number told the reader the wrong thing about
# all three.
#
# Extracted so it can be tested. The watchdog's own history is that it passed every run
# only because it never reached the branch that was broken.
#
#   heartbeat.sh <url> <threshold-seconds>
#
# Prints one of: "ok <age>" | "stale <age>" | "unreachable" | "unreadable"
set -uo pipefail

url=${1:?usage: heartbeat.sh <url> <threshold-seconds>}
threshold=${2:?usage: heartbeat.sh <url> <threshold-seconds>}

body=$(curl -sf --max-time 15 "$url" 2>/dev/null)
if [ -z "$body" ]; then
  echo "unreachable"
  exit 0
fi

# `generated_at` is what every artifact carries and what the page itself reads; ISO-8601
# with a +00:00 offset, which `fromdateiso8601` wants as Z.
ts=$(printf '%s' "$body" \
  | jq -r '(.generated_at // empty | sub("\\+00:00$"; "Z") | fromdateiso8601) // empty' \
    2>/dev/null)
case "$ts" in
  ''|*[!0-9]*) echo "unreadable"; exit 0 ;;
esac

age=$(( $(date +%s) - ts ))
if [ "$age" -gt "$threshold" ]; then echo "stale $age"; else echo "ok $age"; fi
