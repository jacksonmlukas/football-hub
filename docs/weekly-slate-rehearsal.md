# Rehearsing `make slate`, and the bug it found

**Run 2026-08-29.** The draft-night runbook has been rehearsed repeatedly. The **weekly**
path never had been — and it is the one that runs unattended from Week 1 (2026-09-10) and
publishes to a repo that goes public on 2026-09-04.

## It was broken

`uv run python -m hub.publish --all` **crashed**:

```
_duckdb.InvalidInputException: schema mismatch in glob: column "cfg_digest" was read from
"preds/league=nfl/season=2025/week=18/market-d89c2c70-281b7b7a.parquet", but could not be
found in "preds/league=nfl/season=2026/week=01/market-20140835.parquet"
```

The store holds `preds` partitions written under two schemas — twelve columns before
`cfg_digest` existed, fourteen after — and `read_parquet` over the glob refuses rather than
filling nulls.

**This has been broken the whole time and could not be seen**, because
[improvements.md #16](improvements.md) had `publish._scored` wrapped in a bare
`except Exception: return empty`. The weekly page would have reported *"no scored
predictions"* every Sunday, and the track record — the artifact whose entire purpose is to
prove predictions were committed before kickoff — would have quietly shown nothing.

Fixing #16 earlier the same day is what turned it from a silent wrong answer into a crash,
and the rehearsal is what found the crash. Neither on its own would have.

## The fix

`store.connect` builds its views with `union_by_name := true`.

Schema evolution is this store's *design*: `LAYOUT`'s docstring is "immutable dated
partitions; corrections write a new file, nothing is overwritten". A reader that cannot span
two schemas cannot read a store built on that promise. There is a test that writes a
twelve-column partition and a fourteen-column one and asserts both read, with the older
filling null for the column it predates.

## Now

```
  published 5 artifacts for 2025 week 18
    preds_wk18       ok
    track_record     ok
    live             ok
    draft_board      ok
    survivor         ok
```

## And then the page could not read the board

Opening `site/index.html` — which nobody had done either — the draft-board panel said
**"No draft board. Run `make draft`"** against a present, fresh, 199KB `draft_board.json`, with
the manifest reporting it not stale.

The file contained **135 bare `NaN`s**. Python's `json` emits them by default and reads them
back happily; **JSON has no such literal and neither does JavaScript**, so `JSON.parse` throws
on the whole document, `load()` catches it, and the panel renders as absent.

That artifact is what [draft-night.md](draft-night.md) names as the **last-resort fallback**:
*"serve `site/data/draft_board.json` — the top 300 by ECR, which covers all 192 picks"*, for
the case where the board itself will not build. **The safety net was unreadable by the only
thing that reads it**, five days before the draft.

`hub/jsonio.py` is now the one writer — `finite()` scrubs non-finite floats at any depth and
`allow_nan=False` makes a future one raise here rather than ship a document that fails to parse
in the field. Both `hub.publish` and `hub.draft.board` go through it, which a test pins, and
the panel now renders 300 players.

**The draft path was re-verified end to end afterwards**, since this touched `board.py`: board
builds 449 players, THE PICK renders, 192 picks replay at 16ms against a 2000ms budget,
`--taken` records and `--reset` restores, and the no-network path still prints BUILD FAILED →
last good → THE PICK.

## What is still unrehearsed

`make slate` also runs `hub.fetch.cfbd` and `hub.fetch.odds`, both of which need keys this
repo deliberately does not require, and both are prefixed `-` in the Makefile so a failure
does not halt the slate. That design is right — *"a weekly refresh that halts on an
unconfigured extra is a system that needs an operator, and those die in October"* — and it
means neither has been exercised here. They report their own status; the fatal pair,
`nflverse --refresh` and `ratings --fit`, both ran.

**Worth doing before Week 1:** one `make slate` end to end, with whatever keys are configured,
on a Sunday-like state. This rehearsal covered the two fatal steps and the publish, which is
where the bug was.
