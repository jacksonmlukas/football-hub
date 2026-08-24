# Make now, Dagster in October, written so the port is a decorator

**Status:** accepted. Partly addressed 2026-08-24 — see the note at the end.

**Decision.** Keep `make` through the draft. Migrate to Dagster after Week 4. Write every
pipeline step now as a pure function taking explicit inputs and a partition key.

**Why not Dagster today.** The asset model takes real ramp-up time, and the draft is in 13 days.
Learning an orchestrator is not on the critical path to a draft board.

**Why Dagster eventually rather than Prefect.** This pipeline is asset-shaped, not workflow-
shaped. "Week 7 NFL ratings" is a partitioned asset with upstream dependencies and a freshness
requirement, which is exactly Dagster's model, and its partitioned backfills are the strongest
implementation available. Prefect is the better pick when the work is arbitrary Python
orchestration; that is not this. (Note that much of the Dagster-vs-Prefect writing online is
published by one vendor about the other. Read accordingly.)

Run Dagster OSS locally and self-hosted. Dagster+ moved to per-materialization pricing on its
Solo and Starter plans in May 2026, which is the wrong cost model for a pipeline that
materializes one asset per team per week.

**What make cannot do, and when it starts to hurt:**

| Gap | When it bites |
|---|---|
| No retries | First transient ESPN 403 mid-slate |
| No partition awareness | Week 5's data is corrected and you need to re-run 5 through 9 |
| No lineage | "Which ratings version produced this prediction?" |
| No observability | Tuesday refit half-fails and you notice Sunday |

**Making the port cheap.** Every step is a pure function with the partition key in its
signature:

```python
def build_ratings(league: str, season: int, week: int) -> pl.DataFrame: ...
```

Adding `@asset(partitions_def=WeeklyPartitions)` then wraps it. No restructuring, because the
asset boundary and the function boundary already coincide. The `Makefile` targets stay as thin
wrappers so the CLI keeps working either way.

**Trigger to migrate.** The first time you need to backfill more than two weeks by hand. That is
the moment make stops being cheaper than the alternative.

> **Note, 2026-08-24.** `hub.draft.board.build` is the one step that does not match the
> "pure function taking explicit inputs" rule, and it is the module with the most churn in
> the repo. It fetches, prints, and degrades in four independent `except Exception` handlers.
>
> Partly addressed rather than fixed. `build` now returns a `BuildReport` alongside the
> frame, so the report layer asks which optional stages ran instead of inferring it from
> which columns exist — a stage that ran and returned an all-null column used to be
> indistinguishable from one that never ran. The decisions that had been living inside print
> blocks (`held_positions`, `pick_notes`) are pure functions with tests, `replacement_levels`
> takes its three inputs instead of a frame with three magic column names, and `main` takes
> `argv` like the other sixteen CLIs. Coverage went 25% → 46%.
>
> Still true: `build` prints and fetches. Splitting it properly means separating the fetch
> from the assembly, which is the Dagster port's job — the asset boundary wants to be
> between them anyway. Not worth doing twice before a draft that is ten days out.
