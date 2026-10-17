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
set -uo pipefail

msg=${1:?usage: commit-state.sh <commit message>}
ref=${GITHUB_REF_NAME:?GITHUB_REF_NAME is not set}

git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add state
if git diff --cached --quiet; then
  echo "nothing in state/ changed; no commit"
  exit 0
fi
git commit -q -m "$msg ($(date -u +%Y-%m-%dT%H:%MZ))"
for attempt in 1 2 3; do
  git pull --rebase -X theirs origin "$ref" || { git rebase --abort || true; }
  git push origin "HEAD:${ref}" && exit 0
  sleep 5
done
echo "::warning::state/ could not be pushed"
exit 1
