# Architecture decisions

Three structural calls, with the reasoning kept so future-you can tell whether a decision was
considered or inherited.

---

## ADR-001: DuckDB as the query layer, parquet stays the format

**Decision.** Keep writing parquet. Add a DuckDB catalog (`data/processed/hub.duckdb`) holding
views over a Hive-partitioned tree. Nothing migrates.

**Why not flat files alone.** They work fine until roughly Week 6, then "join every prediction
to the closing line that was live when it was made, grouped by model version" becomes a
100-line polars script that nobody trusts.

**Why not SQLite.** Wrong shape. SQLite is row-oriented and single-threaded per query, built for
transactional point lookups. This workload is scans and aggregations over 20 weeks of snapshots,
which is OLAP. DuckDB is columnar and vectorized, and reads parquet directly with no import step.

**The deciding feature is ASOF JOIN.** Matching a prediction to the line that was live at the
moment it was made is an as-of problem. Approximating it with a normal join on week is exactly
how lookahead sneaks into a backtest, and it produces results that look *better*, which is why
it goes unnoticed. DuckDB has ASOF JOIN natively; SQLite does not.

Secondary wins: `QUALIFY` for per-model top-N, zero-copy Arrow so polars frames query without a
copy, and reading polars DataFrames directly.

**Why not Postgres.** Single user, no concurrent writers, no network. Nothing to buy.

**Layout.**

```
data/processed/preds/league=nfl/season=2026/week=07/part.parquet
```

Hive partitioning means DuckDB infers `league`, `season`, `week` as columns from the path and
prunes entire directories on a `WHERE`, rather than opening every file.

**Partitions are immutable.** A correction writes a new file; nothing is overwritten. Backfills
stay reconstructible and auditable, which matters because the public track record's whole claim
is that predictions were not edited after the fact.

**Cost.** One dependency. The catalog is disposable and rebuilds from parquet in seconds, so it
is gitignored.

---

## ADR-002: One Forecaster protocol, league as a field

**Decision.** Every track implements `hub.models.base.Forecaster`. NFL and CFB stay separate at
the *instance* level (a fitted model is always league-scoped) but share the type.

**Why a protocol rather than an ABC.** Structural typing means the market baseline, a NumPyro
model, and a torch model conform without importing a shared base or inheriting behavior they do
not want. `@runtime_checkable` keeps `isinstance` available for tests.

**Why league is a field, not a subclass.** Subclassing would duplicate the entire hierarchy for
zero behavioral difference. What differs between NFL and CFB is data and priors, not interface.

**Why a fixed prediction schema.** This is the actual point. `model-eval` can compare any two
forecasters head to head without knowing what either one is, so "does the Bayesian model beat
the market" is a comparison of two objects rather than a bespoke script per model.

**Leakage is enforced at the type boundary.** `validate_predictions` rejects any prediction for
a week at or before `fit_through_week`. Leakage is the highest-value check in the codebase
because it masquerades as success. A leaky backtest does not error, it just looks good.

**Conformal is composition, not inheritance.** `Conformalized(BayesianRatings(...))` is itself a
`Forecaster`. Track C therefore applies to every other track for free, and you can compare a
model against its own conformalized version.

**Cost.** Every model carries fields it might not use (`margin_lo`/`margin_hi` equal the mean
when a model has no uncertainty estimate). Cheap, and it keeps the comparison table rectangular.

---

## ADR-003: Make now, Dagster in October, written so the port is a decorator

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

---

## ADR-004: Hydra structured config, digest folded into the model version

**Decision.** Every number that changes a prediction lives in `src/hub/config.py` as a
dataclass, registered with Hydra's `ConfigStore`. YAML overrides in `conf/`.

**Why structured configs rather than plain YAML.** `mypy strict` is already on. Plain
`OmegaConf` gives you `Any` everywhere, so a typo in `conf/config.yaml` surfaces at runtime in
the middle of a Sunday refit. Dataclass schemas make it a startup error.

**The actual reason for Hydra, though, is not config management.** It is that
`config_digest(cfg)` folds the resolved config into `FitSpec.digest`, which becomes the model
version written to every prediction row. Two runs that differ only in `projection_lambda` can
therefore never collide in the track record. The public claim is that a prediction came from a
*specific* model, not from "the Bayesian model" as a category, and without config hashing that
claim is unverifiable.

**Operational settings are excluded from the digest** (`poll`, `quota`). Changing the poll
interval must not invalidate a model version, or version bumps become noise and you stop
reading them.

**Bonus that matters in practice.** Multirun sweeps come free, which is exactly what tuning
lambda against a 2024-to-2025 holdout needs:

```bash
python -m hub.draft.board -m draft.projection_lambda=0.02,0.04,0.08,0.16,0.32
```

Each run lands in its own `outputs/multirun/` directory with its config snapshot beside it.

**Rule.** If you are typing a float into a module, it belongs in `config.py`.

---

## ADR-005: Tiered sync polling, no async

**Decision.** Plain synchronous loop with two tiers. No asyncio, no thread pool.

**The concurrency problem was overstated, and the fix is endpoint selection.** One scoreboard
request returns score, status, possession, and down/distance for *every* game in the league --
13 NFL games or 60+ CFB games in a single response. That is everything the dashboard renders.
The per-game cost only appears on the `summary` endpoint, which adds win probability and box
score.

| Tier | Endpoint | Cadence | Requests |
|---|---|---|---|
| 1 | `scoreboard` | every 45s | 1 per league |
| 2 | `summary` | every 4th tick | 1 per game of interest, capped at 12 |

Worst case is about 13 requests per three minutes. Sync is comfortably sufficient.

**Why not add the thread pool anyway.** These endpoints are undocumented and ESPN publishes no
rate limits, with the standing community guidance being to keep volume low and cache
aggressively. Concurrency here buys nothing and raises the odds of getting blocked. If tier 2
ever needs to grow, `ThreadPoolExecutor(max_workers=8)` is a four-line change -- but the right
first move is to shrink the watch list instead.

**Games of interest** are your fantasy starters' teams plus your survivor pick, not the full
slate. If that list exceeds 12 you are watching games you have no stake in.
