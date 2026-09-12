# Architecture decisions

The decisions themselves live one per file in [`docs/adr/`](adr/), which is where `CLAUDE.md`
and `docs/agents/domain.md` have always said to look. This file is the index.

They were a single document until 2026-08-24. Splitting them made each one addressable — an
architecture review had cited "ADR-004 in `docs/architecture.md`" for want of a path — and let
a superseded decision keep its original text instead of being edited in place.

**There are twenty-five decision records in [`docs/adr/`](adr/).** That count is the
directory's, and it is the only one that can be right; see
[How many there are](#how-many-there-are) at the foot of this file for the three numbers this
repo was carrying instead, and why they drifted.

## The index

**Rests on** is the re-scoring: what the decision would fall over without. A decision resting on
an argument stands or falls on the argument; one resting on a number stands or falls on the
number, and this repo re-runs numbers. Where a row says **moved**, the evidence has changed
under the decision since it was accepted — the row names the ticket that owns it, and *nothing
in this index re-decides anything*.

| ADR | Decision | Status | Rests on |
|---|---|---|---|
| [0001](adr/0001-duckdb-query-layer.md) | DuckDB as the query layer, parquet stays the format | accepted | ASOF JOIN existing natively, and an as-of match being the thing a backtest needs |
| [0002](adr/0002-one-forecaster-protocol.md) | One `Forecaster` protocol, league as a field | accepted, **amended 2026-09-11** | a head-to-head comparison — carried by the prediction schema and the store's model column, not by the protocol, which is a placeholder until a second track writes through it (**#136**, decided) |
| [0003](adr/0003-make-now-dagster-in-october.md) | `make` now, Dagster in October, written so the port is a decorator | accepted, partly addressed | a draft thirteen days out, and the asset boundary coinciding with the function boundary |
| [0004](adr/0004-hydra-config-digest.md) | Hydra structured config, digest folded into the model version | accepted, amended by 0006 | a prediction being traceable to the code and constants that made it |
| [0005](adr/0005-tiered-sync-polling.md) | Tiered sync polling, no async | accepted | one scoreboard call returning every game; ~13 requests per three minutes |
| [0006](adr/0006-fitted-constants-live-with-their-provenance.md) | Fitted constants live with their provenance, not in the config | accepted, **limitation reopened** | coverage by the digest, not location — the split holds; its stated escape hatch **moved**, from a stray threshold to shape constants (**#201**) |
| [0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) | A measurement that steers the product must be committed code | accepted | P0's unreproducible −19.66 and its two unrecorded departures; **silent on where such code lives** (**#198**) |
| [0008](adr/0008-the-simulator-indexes-the-board.md) | The draft simulator indexes the board | accepted | held players having to be inside the frame the indices point at |
| [0009](adr/0009-championship-equity-does-not-pick.md) | Championship equity does not pick | accepted | Lost all four seasons, n=80; −19.66 pts/team-game as first measured, re-measured on the fixed bracket at −17.30 and on the repaired harness at −12.48 / −11.59 — **restated 2026-09-11 (#52)**: season-clustered CI [−26.68, −11.45], no magnitude quotable, verdict unchanged. **Silent on where the removed arm lives** (**#198**, since resolved) |
| [0010](adr/0010-edge-is-displayed-but-never-ranked-on.md) | `edge` is displayed but never ranked on | accepted | ESPN retaining no historical ADP — a displayed number and a sort order making different claims |
| [0011](adr/0011-the-pick-ranks-on-corrected-adp.md) | THE PICK ranks on corrected ADP | accepted | three fitted coefficients, a curve read off the board, and a 20% clamp bounding an assembly that cannot be scored |
| [0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md) | The lineup optimiser waits for real variance | accepted, amended twice | +0.00, CI [−0.00, +0.00] — a structural zero; its own forecast measured 2026-08-25 and did not hold, and the amendment of 2026-08-29 found parameter uncertainty worth +0.18 and not significant |
| [0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md) | The snap trend is shown, and never ranked on | accepted, amended 2026-08-27 | partial r = +0.236, twelve of twelve cells, against a decision gate at −3.81 needing ~44 seasons to resolve |
| [0014](adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md) | A provisional rule may act where no gate can run | accepted, **amended 2026-09-07** | five requirements, one qualifying candidate. Criterion 2 is now closed against a NOT-RUNNABLE reclassification (**#51**); requirement 4's log **is written by no code** (**#204**) |
| [0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md) | The weekly gate is a decision, not an accuracy test | accepted | `r2p_pts` being null on every historical weekly row, so the only incumbent worth beating exists as a ranking |
| [0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) | The Weekly projection is shown and never ranked on | accepted, **re-scored 2026-09-07** | its own gate at −0.304, CI [−1.043, +0.415] — unmoved. Its *closing* figure **moved**: +0.215 → −1.004, 0 of 4, branch REMOVE (**#44**) |
| [0017](adr/0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md) | The market/Usage blend is a model, and gets one clean gate | accepted; resolved SHOW, **superseded 2026-09-07** | frozen +0.215, CI [−0.242, +0.684], 3 of 4 seasons — **moved** to −1.004, CI [−1.391, −0.621], 0 of 4. Pre-registration re-scored against the re-run in the ADR (**#44**) |
| [0018](adr/0018-live-scores-are-not-part-of-the-record.md) | Live scores are not part of the record | accepted | rule 1 pinning claims this repo makes, and a live score being someone else's fact |
| [0019](adr/0019-a-gate-requires-every-season.md) | A Gate requires every season, not only the interval | accepted, **amended 2026-09-07** | one rule with three homes that had diverged; #45 added the NOT-RUNNABLE precondition ahead of both halves. Its evidence table's third row is superseded and its conclusion survives (**#44**) |
| [0020](adr/0020-the-board-is-not-split-into-a-reader-and-a-builder.md) | The Board is not split into a reader and a builder | accepted | exactly one cross-package reader, 55 ms marginal on the live path, and `last_good` being Board knowledge |
| [0021](adr/0021-a-correction-is-a-shape-not-a-module.md) | A Correction is a shape, not a module | accepted | two `correct_projection` bodies that differ in kind, and `fitted_digest` keying every constant by its module |
| [0022](adr/0022-the-boards-xfp-stays-the-published-total.md) | The Board's xFP stays the published total, with the rebuild beside it | accepted | the rebuild reproducing the total to 0.017 points a player-week, which makes a gate on it vacuous |
| [0023](adr/0023-a-refused-injury-report-nulls-three-columns.md) | A refused injury report nulls three columns, not the Panel | accepted | `"None"` being a legal designation, so a filled column is indistinguishable from a real week |
| [0024](adr/0024-a-modelling-decision-becomes-agent-work-when-its-alternatives-can-be-measured.md) | A modelling decision becomes agent work when its alternatives can be measured side by side | accepted | nine issues labelled `ready-for-agent` over bodies saying they were not |
| [0025](adr/0025-systematic-bias-bounds-the-bottom-up-plan-and-seven-angles-are-declined.md) | Systematic bias bounds the bottom-up plan, and seven angles are declined | accepted | an arithmetic bound — a 3% bias in Usage is ≈0.75 points a team and ≈1.5 on the total, against a band of about one point — and seven declines, two of which (the offensive-line study, west-coast travel) are recorded thinner than the other five and say so |

## Re-scored 2026-09-07

Issue #51. **Nine** decisions were absent from this index — the issue says eight, and was
written before [ADR-0023](adr/0023-a-refused-injury-report-nulls-three-columns.md) landed on
2026-09-06. Among them was [ADR-0019](adr/0019-a-gate-requires-every-season.md), which changed
the adoption rule for every gate in the repo, so the index has been silent about the rule the
rest of it is scored under since 2026-09-04. Every ADR above now carries what it rests on, and the seven whose evidence has
moved carry a dated note in their own file.

**Re-scoring is not re-deciding.** Each note says what the decision rests on and whether that
still holds. Where it does not, the note names the ticket that owns the question and stops. No
decision below is reopened, adopted, withdrawn or reversed by this pass.

| ADR | What moved | Owned by |
|---|---|---|
| [0002](adr/0002-one-forecaster-protocol.md) | The stated payoff. One adapter implements the protocol; the comparison is carried by the prediction schema and the store's model column, which have two writers. Separately, the writer bypassed the interface — so *"leakage is enforced at the type boundary"* named a boundary with no code on it — **closed by #205 on 2026-09-07**, which puts code on the boundary without adding a second adapter | **#136** — **decided 2026-09-11 by amendment**: the schema carries the payoff, the protocol is a placeholder, retired by a second track writing through it |
| [0006](adr/0006-fitted-constants-live-with-their-provenance.md) | The size of its own stated limitation. It framed the coverage gap as a stray *threshold*; the live instances are *shape* constants setting the extent of a random draw, so moving one re-prices every published Gate interval and moves no digest | **#201** |
| [0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) + [0009](adr/0009-championship-equity-does-not-pick.md) | Nothing in either verdict. The two pull opposite ways over **where** a removed measurement should live — 0009 removed it from the product, 0007 keeps it as code, neither names a home — and ~1,000 lines sit in `hub.draft` downstream of no product decision | **#198** — **landed 2026-09-11**: the home is `hub.exhibits`, read by the harness and by nothing the product ships, and a contract test holds that rather than a census |
| [0014](adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md) | Two things. #45 made NOT-RUNNABLE a computed branch, which opened a second door into criterion 2 — closed here by amendment. And requirement 4's log, the only mechanism by which a provisional rule can stop being provisional, is written by no code | **#51** (the amendment, landed), **#204** (the log) |
| [0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) | The figure it was **closed** on, not the one it was decided on. Its own gate at −0.304 is unmoved; the market/Usage blend's +0.215 became −1.004 | **#44** |
| [0017](adr/0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md) | The result. Frozen +0.215 → −1.004, 3 of 4 seasons → 0 of 4, pre-registered branch SHOW → REMOVE. The rules were fixed before the run and are re-scored against the interval the re-run produced | **#44** |
| [0019](adr/0019-a-gate-requires-every-season.md) | One row of its evidence table, which cites the superseded [−0.242, +0.684] at 3 of 4. Re-scored, the row reads REMOVE rather than SHOW — and the claim the table makes, that the unification moved no published verdict, holds either way | **#44** |

**What is not in that table is as much of the answer as what is.** Fifteen of the twenty-three
rest on arguments, constraints, or measurements that have not moved: 0001, 0003, 0004, 0005,
0008, 0010, 0011, 0012, 0013, 0015, 0018, 0020, 0021, 0022, 0023. And 0019 appears above only
for one row of an evidence table — its decision is intact and its amendment strengthened it. A
re-scoring pass that found everything moved would be measuring itself.

**One thing was found broken rather than moved.** ADR-0016's superseding note had been spliced
into the middle of a sentence, leaving a dangling *"The"* on either side of the blockquote and
breaking the paragraph that reports the season decay. The note and the sentence are both intact
now; neither's words changed.

## How many there are

Three places in this repo gave three different answers, and none of them was twenty-three:

| Source | Said | Was it ever right? |
|---|---|---|
| `README.md` | *"index of twelve decision records"* | No. It has never matched this file, which listed fourteen when the line was written |
| `docs/method.md` | *"Fourteen decisions"*, linking `docs/adr/` | Right until ADR-0015 landed on 2026-08-27, then never re-derived |
| This index | listed 0001–0014 | Right until 2026-08-27; nine ADRs have landed since |
| `docs/adr/` | 25 files | Guarded by `test_no_document_states_a_count_that_contradicts_the_directory`, which is what caught this line drifting |

**The fourteen is a coincidence worth naming, because it is what made the drift invisible.**
`docs/method.md` separately reports that *"Fourteen things have been measured properly"* — a
count of **measurements**, not of decision records, and a genuinely different fourteen that
was correct on this page's date (fifteen since 2026-09-11; the table in `docs/method.md` is
the count). The two sat in one file agreeing with each other for eleven days while both
descriptions of `docs/adr/` were going wrong, and a reader checking one against the other would
have been reassured.

The count in the header of this file is now derived from the directory, and
`tests/contracts/test_adr_index_is_complete.py` fails if this index and `docs/adr/` disagree in
either direction, or if a count stated in `README.md` or `docs/method.md` contradicts the number
of files. That is the mechanism; the numbers above are why it exists rather than a convention.

Related, and deliberately not ADRs:

- [`decisions.md`](decisions.md) is the running journal — corrections issued during design,
  standing directions, open questions. Decisions of record are here; the working out is there.
- [`CONTEXT.md`](../CONTEXT.md) is the glossary. An ADR says what was decided; the glossary
  says what the words mean.
