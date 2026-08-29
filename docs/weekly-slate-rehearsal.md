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
