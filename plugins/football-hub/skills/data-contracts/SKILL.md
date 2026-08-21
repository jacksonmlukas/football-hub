---
name: data-contracts
description: Use when adding a new data source, changing an ingest path, or debugging suspicious upstream data — writing schema assertions, freezing golden fixtures, or diagnosing a contract violation. Triggers on "new data source", "contract violation", "schema changed", "the numbers look wrong", "golden fixture".
---

# Data Contracts

The failure mode for a seasonal pipeline is not a crash. It is a third-party silently renaming a
field while projections quietly degrade for three weeks. Contracts exist to make that loud.

## Rule

**Every ingest function validates before returning.** No exceptions, including one-off scripts.
An unvalidated fetch is how bad data reaches a model.

```python
from hub.contracts import DRAFT_BOARD
return DRAFT_BOARD.validate(df)
```

## Adding a source

1. Fetch once by hand. Save the raw response to `tests/golden/fixtures/<source>.json`.
2. Write the `Contract` in `src/hub/contracts.py`. Specify all five: `required`, `non_null`,
   `unique`, `ranges`, `min_rows`. A contract without `ranges` catches structural breakage but
   misses the dangerous case, which is plausible-looking wrong numbers.
3. Write the contract test against the fixture, in `tests/contracts/`. Red first.
4. Write the parser. Green.
5. Add a golden test in `tests/golden/` that hits the live API and diffs against the fixture.
   Nightly only, never on PRs. A third-party outage must not block a docs commit.

## Choosing ranges

Set them from observed history plus a margin, not from theory. Pull three prior seasons, take
min and max, widen 20%. Too tight means weekly false alarms and you start ignoring the alerts,
which is worse than having none.

Ranges that have already earned their place:

| Field | Range | Catches |
|---|---|---|
| `total_fantasy_points_exp` | -10 to 80 | A scoring-rule change silently doubling projections |
| `ecr` | 1 to 1000 | Rank returned as percentile instead of rank |
| `win_prob` | 0 to 1 | Percentage-vs-proportion swap, the classic |

## Diagnosing a violation

Read the message before touching code. It names the field and the observed value.

- **Missing column** usually means the upstream renamed it. Diff against the golden fixture, then
  update parser and fixture together in one commit so they never drift.
- **Out of range** is the dangerous one. Structure held but the meaning changed. Do not widen the
  range to make it pass. Find out what changed first.
- **Not unique** typically means a join fanned out. The bug is upstream of the contract.

## When not to use

Not for model validation. Contracts check that data is what it claims to be, never that a model
is any good. That is `model-eval`.
