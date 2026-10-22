#!/usr/bin/env bash
# Commit and push `state/`, retrying, for the close capture (#495).
#
#   commit-state.sh <commit message>
#
# Used twice in `odds-close.yml`: by `hub.closing --commit-attempt` to put the attempt marker on
# the branch BEFORE the odds request (exit non-zero means no request is made), and after the
# request to commit the capture. A rebase conflict keeps our commit (`-X theirs` is ours during
# a rebase): the credit balance is the only file two jobs can both write, and ours is newer.
# Exits 0 if there was nothing to commit.
#
# **The LLM study's closing line (#450) uses it too, pointed elsewhere.** Three optional
# variables, each defaulting to the hub's behaviour above: `STATE_REPO` (the checkout to commit
# in; default `.`), `STATE_PATHS` (space-separated paths to add; default `state`) and
# `STATE_REF` (the branch to push; default `$GITHUB_REF_NAME`). `forecast-close.yml` sets them to
# the `forecasts` worktree, `closing state` and `forecasts`, so the study's writes never reach
# main. A named path that does not exist yet is skipped rather than handed to `git add`, whose
# pathspec error is #424's failure.
set -uo pipefail

msg=${1:?usage: commit-state.sh <commit message>}
repo=${STATE_REPO:-.}
paths=${STATE_PATHS:-state}
ref=${STATE_REF:-${GITHUB_REF_NAME:?GITHUB_REF_NAME is not set}}

g() { git -C "$repo" "$@"; }

g config user.name  "github-actions[bot]"
g config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for p in $paths; do
  [ -e "$repo/$p" ] && g add -- "$p"
done
if g diff --cached --quiet; then
  echo "nothing in ${paths} changed; no commit"
  exit 0
fi
g commit -q -m "$msg ($(date -u +%Y-%m-%dT%H:%MZ))"
for attempt in 1 2 3; do
  g pull --rebase -X theirs origin "$ref" || { g rebase --abort || true; }
  g push origin "HEAD:${ref}" && exit 0
  sleep 5
done
echo "::warning::${paths} could not be pushed"
exit 1
