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
| Answered elsewhere         | closed | `not planned`          | whichever it was routed by |
| Already filed elsewhere    | closed | `duplicate` (API only) | `duplicate`               |

The close reasons are the whole enum, checked against this tracker's live schema on 2026-09-05
rather than assumed (the fourth row reuses the second's reason — see below):

```sh
gh api graphql -f query='{ __type(name: "IssueClosedStateReason") { enumValues { name } } }'
# {"COMPLETED","NOT_PLANNED","DUPLICATE"}
```

**The fourth row was added 2026-09-16, from #222 and #223.** The table originally bound
`not planned` to `wontfix` as if they were one fact. They are two: `wontfix` is a *routing*
decision ("nobody picks this up"), `not planned` is an *outcome* ("this ticket's work did not
land"). #222 was declined on its merits — routing, and it carries `wontfix`. #223 was not declined:
its pre-registered bar was relocated into ADR-0025, so its own work (a regression run) never landed
while the question it existed to ask was answered elsewhere. That is `not planned` with no
`wontfix`, and the close comment names where the answer went. Without the row, a `not planned`
close without `wontfix` reads as a mistake, and the next reader "fixes" it.

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

## `ready-for-human` carries a drafted decision (adopted 2026-09-07)

> **Amended 2026-09-16** — see [*Adoption is a write, not a state*](#adoption-is-a-write-not-a-state)
> below. Nothing here is withdrawn; the section below says what actually happened to the two
> tickets this one names as permanently human, and states the rule that describes it.

A ticket earns `ready-for-human` when it contains an unmade decision, not when nobody has
picked it up. The failure mode the label had developed was the second one: #136 and #138 sat
labelled for weeks while several others were labelled for a choice among three named options
that nobody had written down.

**So whenever a ticket's human part is a choice among a small number of named options, the
decision is drafted into the ticket rather than left to be written from scratch.** The draft
is a *pre-registration*: it names the candidates, recommends one, states what each outcome
would mean, and keeps the reasoning separate from the conclusion so the reasoning can be
rejected on its own. It is marked `PROPOSED` and the label does not change until it is
adopted.

This is not a way around `docs/method.md` rule 1. Rule 1 requires the decision to precede the
numbers; it does not require a particular person to make it. A drafted pre-registration
written before the run satisfies it. A number measured first and a rule chosen after does not,
whoever writes either.

**Two kinds of ticket stay human however they are drafted**, and their tickets say which they
are:

- **The decision is the point.** #138 asks whether to amend a pre-registered rule after the
  effect sizes are known. Delegating that launders the decision through a process that looks
  like work; the rule it produced would carry no more authority than the numbers that prompted
  it.
- **The decision is waiting on evidence, not on a person.** #136 reopens ADR-0002 and the
  modelling programme may still supply the adapter that answers it. Deciding now is deciding
  early.

For both, the decision-free half is split out where one exists — #205 and #207 are those
splits — so the ticket that stays human is only the part that has to.

## Adoption is a write, not a state (2026-09-16)

The section above says two kinds of ticket "stay human however they are drafted", and names
#138 and #136. Both are closed `COMPLETED` and both carry `ready-for-agent`. That is not a
mislabel. #138's actual life was: a drafted decision → the maintainer commented *"Option 1"* →
an agent amended the document *before any run* and closed the ticket. #136 went the same way
once the modelling programme supplied the evidence it was waiting on. Neither *never converted*;
each converted **after one human write**, and everything downstream of that write was agent work.
So the rule the two examples demonstrate is not the one the section states, and this section
states the one they demonstrate.

**One rule, stated twice.** [ADR-0024](../adr/0024-a-modelling-decision-becomes-agent-work-when-its-alternatives-can-be-measured.md)
asks *can the alternatives be placed on one axis and reported in one table?* — yes is agent work
and the table is the deliverable; different objects is a decision. The section above's two
permanent-human kinds are exactly ADR-0024's exclusion classes: #138 was *amending a
pre-registered rule after the effect sizes are known* and #136 was *waiting on evidence, not on
a person*. The drafted-decision convention was always scoped to the different-objects class; the
section above just did not say so. These were never two rules that could disagree.

**The procedure**, applied to the audit-IV queue on 2026-09-16 and to every `ready-for-human`
ticket after it:

1. **Apply ADR-0024's test first.** One axis → the ticket converts now, no human write needed;
   the drafted recommendation becomes the arm the table is read against, not a choice to make.
   Different objects → it is a decision, and it needs the write in step 3.
2. **Split the decision-free half** where one exists (#206 → #207, #322 → #327): a new
   `ready-for-agent` ticket for the measurement, the human ticket `blocked_by` it, so the
   ticket that stays human is the smallest one that has to.
3. **Adoption is one line, and it is the maintainer's.** A comment beginning `ADOPTED:` naming
   the option (or the number, for a re-taken constant). It is grep-able — a sweep can find
   adopted-but-unconverted tickets — and it is the write that carries the authority, so it is
   written by the maintainer, not recorded on their behalf. The attributed-AI form (*"Recorded by
   AI during a grilling session; the disposition is the maintainer's"*) is for closes and
   dispositions quoted from a session the maintainer was in; it is not the `ADOPTED:` line.
4. **Everything after the write is agent work:** the body edit, the label flip, the amendment
   or the run. Converting a ticket without updating its body is the defect ADR-0024 was written
   about; the body block is: *converted <date> under ADR-0024; the axis (or the pre-registered
   gate, where there is one option and a gate); the deliverable; what would reopen it as a
   decision.* Prior wording stays visible. **No calendar line** ("do not start before …") goes in
   a body — that is a second copy of a fact that goes stale and that nobody sweeps.
5. **Ordering is an edge.** Anything that must land before something else is a native
   `blocked_by`, never a prose obligation on a ticket that will be closed by the time the
   obligation is actionable. A freeze is a ticket that everything frozen is `blocked_by`, closed
   once by a human; a restatement obligation sits on the open ticket whose work discharges it.
   (`issue-tracker.md` makes the same trade for `Blocked by:` lines; the sixth-state section
   above makes it for `done`.)

**What stays human, then, is one thing:** a ticket whose alternatives are different objects
*and* whose `ADOPTED:` line has not been written. Both the audit-IV examples of that class —
#300 (which estimand licenses a shipped component) and #310 (a claim re-decided once #309's
measurement exists) — are on the tracker with their drafts, waiting on the line, not on a
person to start writing from scratch.
