# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual
label strings used in this repo's issue tracker — and records the one state that is deliberately
**not** a label.

## The five routing roles

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label
string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

All five answer one question — **who picks this up next** — and they only mean anything while the
issue is open. None of them says the work is finished, and none of them should. Labels are not
cleared on close: they stay as a record of how the ticket was routed, which is why every frontier
query must be state-scoped (`gh issue list` defaults to `--state open`; `--label ready-for-agent`
alone will hand you thirty finished tickets).

## The sixth state is not a label

*Done* is a ticket's outcome, not a routing decision about it, and the tracker already stores it.
GitHub records a **close reason** alongside the closed state, so the sixth state costs no new
vocabulary:

| Outcome                    | State  | Close reason           | Label it keeps            |
| -------------------------- | ------ | ---------------------- | ------------------------- |
| The work landed            | closed | `completed`            | whichever it was routed by |
| Will not be actioned       | closed | `not planned`          | `wontfix`                 |
| Already filed elsewhere    | closed | `duplicate` (API only) | `duplicate`               |

Those three are the whole enum, checked against this tracker's live schema on 2026-09-05 rather
than assumed:

```sh
gh api graphql -f query='{ __type(name: "IssueClosedStateReason") { enumValues { name } } }'
# {"COMPLETED","NOT_PLANNED","DUPLICATE"}
```

**Why not a sixth label.** A `done` label would be a second copy of a fact the tracker already
holds, and two copies drift: someone closes without adding it, or adds it to something still open,
and then the label and the state disagree with no way to tell which is lying. A close reason is
written by the same call that closes the ticket — there is no second write to forget and no state
where the two can disagree. This is the same trade `wayfinder` already made when it put blocking
edges in GitHub's native dependencies instead of a `Blocked by:` line in the body (see
`issue-tracker.md`). It also keeps the label vocabulary at five, which is what `CLAUDE.md` and the
skills claim it is.

**The trap, and the half of this convention that does the work.** A `gh issue close` that passes no
`--reason` sets `completed`. Silently. So the "work landed" signal is also the default, and on its
own it proves nothing: as of 2026-09-05 all thirty closed issues in this repo read `COMPLETED`,
every one of them by default rather than by anyone asserting it. That they happen to all be
genuinely complete is the point — the value has never once had to discriminate, so it has never
been evidence.

The reason only carries information once the *other* value is in use. So the load-bearing rule is
the negative one:

> A ticket closed as `wontfix`, abandoned, or superseded **must** be closed with
> `--reason "not planned"`. A close with no reason is a bug, not a shorthand for "done".

`tests/contracts/test_close_reason_is_documented.py` fails if any `gh issue close` shown in
`docs/agents/` loses its `--reason`, because a reason-less example is how the default quietly
becomes the convention again.

## Writing it

```sh
# The work landed.
gh issue close <n> --reason completed --comment "Done in <sha>. ..."

# It will not be actioned. The label says why; the reason says it did not land.
gh issue edit  <n> --add-label wontfix
gh issue close <n> --reason "not planned" --comment "..."

# Already filed elsewhere. `gh issue close` does not expose this one; the API does.
gh api --method PATCH repos/jacksonmlukas/football-hub/issues/<n> \
  -f state=closed -f state_reason=duplicate

# Correcting an already-closed ticket. `gh issue close` refuses a closed issue; PATCH is idempotent.
gh api --method PATCH repos/jacksonmlukas/football-hub/issues/<n> \
  -f state=closed -f state_reason=not_planned
```

Two spellings of the same value, and mixing them up fails quietly rather than loudly:
`gh issue close --reason` takes **`"not planned"`** (a space), the REST API takes **`not_planned`**
(an underscore).

## Reading it back, through `gh`

```sh
# One ticket, outcome included — no comment to read.
gh issue view 27 --json number,title,state,stateReason
# {"number":27,"state":"CLOSED","stateReason":"COMPLETED", ...}

# Every ticket whose work landed.
gh issue list --state closed --json number,title,stateReason \
  --jq '.[] | select(.stateReason == "COMPLETED") | "\(.number)\t\(.title)"'

# Everything closed *without* landing — the query that used to require reading close comments.
gh issue list --state closed --search 'reason:"not planned"' --json number,title
```

A third spelling, so check the case before comparing: `--json stateReason` comes from GraphQL and
returns **`COMPLETED`** / **`NOT_PLANNED`** in upper snake case, while `gh api` returns
**`completed`** / **`not_planned`** and the `reason:` search qualifier takes **`"not planned"`**. An
agent grepping for `completed` against `--json` output matches nothing and sees an empty list, not
an error.

## Not covered here

Several tickets were closed with a comment naming the ticket that carries their unfinished
remainder. That is a relationship, not a state, so a close reason cannot hold it; it is still
prose. GitHub's native issue dependencies (`issue-tracker.md`, *Wayfinding operations*) are the
same edge in reverse and are the obvious home for it.

There is also a window this cannot describe: a ticket whose work is written and committed but not
yet pushed stays open, because its `Closes #<n>` line has not fired. #54, #55, #59 and #61 were in
exactly that state on 2026-09-05. No label was added for it — the window is transient and closes
itself on the next push, and a label for it would have to be removed by hand afterwards, which is
the drift this whole section is avoiding. During that window the tracker genuinely does not know;
`git log --grep 'Closes #<n>'` does.
