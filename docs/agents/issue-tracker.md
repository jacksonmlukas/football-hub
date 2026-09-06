# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --reason completed --comment "..."`, or
  `--reason "not planned"` when the work did not land. **Never close without a reason** — a bare
  close silently records `completed`, which is how "finished", "abandoned" and `wontfix` ended up
  indistinguishable. The reason *is* this tracker's done-state; see `triage-labels.md`.
- **Read the outcome of a closed issue**: `gh issue view <number> --json state,stateReason`, or in
  bulk `gh issue list --state closed --json number,title,stateReason`. `stateReason` is upper snake
  case (`COMPLETED` / `NOT_PLANNED`); the `reason:` search qualifier and `--reason` are not.

Infer the repo from `git remote -v`; `gh` does this automatically when run inside a clone.

## Commit trailers, and the sweep they need

A commit that finishes a ticket ends with `Closes #<n>`. GitHub acts on it: pushed to the
default branch, the ticket closes itself with reason `completed`. `Fixes` and `Resolves` work
the same way. The trailer is the mechanism, not decoration on top of one.

A commit that lands a coherent *part* of a ticket ends with `Advances #<n>` instead, and says
in its body what is not done. That is the honest trailer and it should keep being used — but
it is inert, and that is the defect it carries:

> **`Advances` leaves an obligation on a future commit that has no way to know it inherited
> one.** Measured 2026-09-05: four tickets had ever carried `Advances`, and none ever received
> a `Closes`. Two of them (#34, #53) were finished and still open. Both were finished by
> commits about something else — #34's last criterion landed under a message calling it *"the
> catch on the way through"*, #53's inside two commits about review findings. Neither had any
> reason to think about a ticket it was silently completing. #34 was the head of the dependency
> graph, blocking 23 tickets while being done.

So the escalation cannot be a rule about trailers. The commit that would have to obey it does
not know the rule applies to it. It has to be a sweep:

```sh
scripts/partial_work.sh
```

Every open ticket carrying an `Advances` that no later commit resolved. Everything on it is
either still partial — leave it — or finished and open, which is the case no trailer catches.
The queue is short by construction and a ticket leaves it the moment something closes it. Run
it when picking work off the frontier: a ticket on this list may already be built.

`scripts/partial_work.sh --numbers` is the same query without the network, for a checkout with
no `gh` credentials.


## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies**, the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only, the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n> --reason completed` (a research ticket that reached an answer *has* landed, even when the answer is "no" and no production code changed — close it `not planned` only if it was dropped unanswered), then append a context pointer (gist + link) to the map's Decisions-so-far.
