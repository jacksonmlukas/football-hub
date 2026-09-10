#!/usr/bin/env bash
# Refuse to work in an agent worktree that is behind the branch it will merge into.
#
# **The defect this exists for.** On the night of 2026-09-06/07, three agents each reported an
# isolated worktree created from a stale base -- 13 commits behind in one case, 17 in two
# others. All three noticed and fast-forwarded, and two said the ticket was *unimplementable*
# at that base, because the change it followed on from had landed on `main` in the meantime.
#
# They caught it. Nothing caught it for them. An agent that did not notice would have written
# against code that no longer exists, run its targeted tests green on that stale tree, reported
# clean gates honestly, and produced a merge that silently reverts whatever landed in between.
# Every safeguard in this repo -- pytest, pyrefly, the guard excisions, the mutation checks, the
# review -- runs *inside* the worktree and would agree with it. The failure is invisible from
# the only vantage point the agent has, which is the shape this repo keeps finding: a check
# whose validity depends on context it cannot see. This comment is the single copy of that
# story; everything else that needs it points here.
#
# **Why it refuses rather than warns.** A warning printed into a transcript fires only if
# somebody reads it, which is the same defect one level up. Refusing is fixable by fixing: the
# moment the worktree merges the integration branch, `behind` is zero and this hook goes quiet
# for the rest of the session. It clears itself, so it costs a current agent nothing and a
# stale one one command.
#
# **Wired to `Read|Bash|Edit|Write`, not to the edit tools alone.** The first draft policed
# `Edit|Write` only, and review caught that this reproduces a defect this repo has already
# documented: `tdd_gate.sh`'s own header records that agents "had begun making their edits
# through Bash heredocs, which this hook never sees, so the gate was already not covering the
# people who had worked out how to avoid it." A staleness check with that hole is worse than
# none, because the agents most likely to slip past it are the ones moving fastest. Covering
# `Read` as well is what makes the refusal arrive *before* the agent has read a dozen stale
# files and built a plan on them.
#
# So this script does not filter by tool. What it is asked about is settings.json's business;
# what it answers is whether this tree is current.
#
# **Local `main`, not `origin/main`, and no network.** A linked worktree shares the object store
# with the primary checkout, so the local ref *is* the primary checkout's -- which is the branch
# the work merges back into and therefore the only one staleness is measured against. Fetching
# would be slower, would fail offline, and would answer a different question.
#
# The event is read with `sed` rather than a JSON parser for the same reason `tdd_gate.sh` does
# it: a hook that needs an interpreter to start is a hook that fails differently on a machine
# without one.
set -uo pipefail

# The one branch this repo integrates on. A worktree stacked on a feature branch is still
# measured against this, which is correct -- it is where the work lands either way -- and the
# refusal names the branch so the comparison is never implicit. If this repo ever grows a second
# integration branch, that belongs here rather than in a workaround.
INTEGRATION_BRANCH=main

refuse () { printf '%s\n' "$1" >&2; exit 2; }

if ! command -v git >/dev/null 2>&1; then
  refuse "No git on PATH, so whether this worktree is current cannot be established.
This hook does not report 'up to date' on a question it could not ask -- see the header."
fi

# Fail closed on an unreadable event, the way tdd_gate.sh does. An empty read here means the
# event shape changed under us, and a hook that quietly checks nothing when its input surprises
# it is the defect this repo already carries scars from. There is no tool filter to fall
# through: whatever arrived, the tree is either current or it is not.
EVENT="$(cat 2>/dev/null || true)"
: "${EVENT:=}"

# A linked worktree's git dir sits under the primary's; in the primary checkout the two paths
# are identical. Compared as git prints them, which works on older git than `--path-format`
# does. A failure here means this is not a repository at all -- genuinely none of our business,
# and distinct from the "cannot answer" cases above, which refuse.
GIT_DIR="$(git rev-parse --git-dir 2>/dev/null)" || exit 0
COMMON_DIR="$(git rev-parse --git-common-dir 2>/dev/null)" || exit 0
[ "$GIT_DIR" = "$COMMON_DIR" ] && exit 0

# `git rev-list` against a ref that does not exist exits non-zero and prints nothing, and a hook
# that reads that as "zero commits behind" passes hardest exactly when it is least able to
# answer. That failure mode has cost this repo three separate wrong verdicts, so the count's
# exit code is read before the number is believed.
BEHIND="$(git rev-list --count "HEAD..$INTEGRATION_BRANCH" 2>/dev/null)"
STATUS=$?

if [ $STATUS -ne 0 ] || [ -z "$BEHIND" ]; then
  refuse "Cannot tell whether this worktree is current: there is no '$INTEGRATION_BRANCH' here.

Establish what this work merges back into before going further -- 'git branch -a'. If the
branch is genuinely named something else, that name belongs in
.claude/hooks/worktree_base.sh, not in a workaround."
fi

[ "$BEHIND" -eq 0 ] && exit 0

AHEAD="$(git rev-list --count "$INTEGRATION_BRANCH..HEAD" 2>/dev/null || echo 0)"
refuse "This worktree is $BEHIND commit(s) behind '$INTEGRATION_BRANCH' (and $AHEAD ahead).

    git merge --no-edit $INTEGRATION_BRANCH

Then re-read the files your ticket names: what you have read so far may already be gone. Three
agents hit this on 2026-09-06/07 and two found their ticket unimplementable at the base they
were given -- see the header of this file for why nothing else in the repo can catch it.

This message stops once '$INTEGRATION_BRANCH' is merged in."
