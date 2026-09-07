# A refused injury report nulls three columns, not the Panel

**Status:** accepted 2026-09-06. Investigated under issue #139.

**Decision.** `hub.models.panel.injury_severity` refuses every injury report whose columns are
not the ones `INJURIES` declares — a renamed `report_status`, a renamed `full_name`, or a
missing `practice_status` alike. `build_panel` does **not** propagate that refusal: it catches
it, carries `inj_sev`, `status` and `practice` as **nulls**, and prints the contract's own
message to stderr. The refusal is stated once, in the contract; the coping is stated once, in
`hub.models.panel.injury_columns`.

Explicitly rejected: filling the missing practice column with `"None"`, which is what the code
did before #132, and filling the degraded columns with `0.0` / `"Healthy"`, which is what the
join below them already does for a player with no injury row.

## Why it was proposed

#132 routed `injury_severity` through `INJURIES.conform(...)` and deleted the hand-rolled
lookup it had been using. Most of what went was a duplicated schema, which was the point. But
three of the deleted lines were tolerances rather than duplication:

| Deleted | What it did |
|---|---|
| `cols.get("report_status") or cols.get("game_status")` | took whichever status name arrived |
| `cols.get("full_name") or cols.get("player_name")` | took whichever name column arrived |
| `practice_key(prac) if prac else pl.lit("None")` | treated `practice_status` as optional |

All three are now refusals. This function loads through `nflreadpy` directly rather than
through `hub.fetch`, so no contract had ever seen the frame — the tolerances were the only
thing standing there, and nothing tested any of them in either direction.

## The two renamed columns: the refusal is right

A fallback that quietly accepts `game_status` is precisely how a silent rename goes unnoticed
— the Week 7 failure this repo names, and the failure `hub.contracts` exists for. The old code
was not *no* refusal, only a later one: it raised when neither spelling was present. So what
the deletion actually changed is which frames it refuses, and the frames it newly refuses are
the ones where a column moved underneath us and the ordinal would otherwise have been computed
from something the declaration never promised.

Reading a second name would also put a second schema back beside `INJURIES`, which is the
duplication #132 removed. If nflverse renames the column for good, the fix is to change the
declaration — one line, in the file that owns the answer — not to accept both spellings
forever in the one consumer that noticed.

## The practice column: the refusal is right for a different reason

This one looks like a plain narrowing, and it is the case where the tolerance was most
defensible: a frame with no `practice_status` used to yield a working Panel and now raises.
Restoring it still loses, because of what the filled value means.

`"None"` is **already a legal designation** in this vocabulary. `hub.names.practice_key` fills
a null practice cell with `"None"`, and `INJURIES` deliberately leaves `practice_status` out of
`non_null` because a player on the report with no practice designation is the ordinary case —
45 rows of 40,204 over 2019-25. So a whole column of `"None"` is indistinguishable from a real
week in which nobody was designated. The `(status, practice)` pair is what
`hub.models.injury` keys its retention table on, the instrument measured at +0.170 MAE and 3.8
se; filling the column silently narrows that pair to `status` alone and leaves nothing saying
so. A tolerance whose failure mode is a silently weaker model is not a graceful one.

## Where the degradation goes instead, and why null

CLAUDE.md's rule is real and it points at exactly this function: every module must produce a
usable answer with zero attention, and the Sunday panel is the live path it is written for.
`build_panel` calls `injury_severity` unconditionally, so a refusal there would cost the whole
Panel — 40-odd columns and three consumers — over one feature.

So the coping lives in the caller, where it can serve the other 40 columns, and it degrades to
null:

* **Null is not `Healthy`.** The fill under the join means "this player has no row on the
  injury report". A refused report is a different statement — nobody has one and nothing is
  known — and spelling it `Healthy` would publish a league in perfect health on the week the
  source broke, with `inj_sev` reaching `hub.models.weekly_screen` as a measured zero on every
  row. That is the same silent-narrowing failure as filling `"None"`, one level up.
* **Null is loud enough.** `weekly_screen` reports an all-null feature as *nothing measured*
  rather than as a result, so the loss is visible where the feature is read, and the
  `ContractViolation` goes to stderr in the contract's own words, naming the column.
* **Only a `ContractViolation` degrades.** A source that changed shape is what the Panel can
  serve around. A `TypeError` inside the ordinal is this repo being wrong, and swallowing it
  would hide a defect behind a column of nulls that looks exactly like a quiet week.

## What holds this

`tests/unit/test_panel.py` covers all three shapes in the direction chosen — each rename
refuses and the message names the column `INJURIES` declares, the missing practice column
refuses rather than filling `"None"` — and covers what `build_panel` does around each of them:
same rows, three null columns, no `Healthy`, and a message on stderr. Reversing any part of
this decision turns one of those red.

## What would reopen this

nflverse renaming a column for good, or dropping `practice_status` from the weekly report for
good. Either is a change to `INJURIES` — the declaration is the thing that would be wrong —
and neither is a reason for a consumer to accept two spellings of one column.
