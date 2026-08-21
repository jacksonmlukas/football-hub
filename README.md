# football-hub

Calibrated probabilistic forecasting for NFL and FBS football, built as evaluation
infrastructure first and a prediction product second.

The interesting problem here is not predicting football. It is knowing whether your
predictions are any good, in a domain with ~285 NFL games a season, a highly efficient
market as the benchmark, and no ability to run an A/B test. This repo is organized
around that measurement problem.

## What it does

| Component | Method |
|---|---|
| Team strength | Hierarchical Bayesian state-space ratings, partial pooling across 136 FBS teams (NumPyro) |
| Game forecasts | Market prior from closing lines, model fit to the residual |
| Uncertainty | Split and adaptive conformal prediction; Mondrian by position group (MAPIE) |
| Fantasy | Expected-points decomposition (ffopportunity) with full-PPR VOR and ADP arbitrage |
| Pool strategy | Integer-programming survivor solver with a pool-aware `P(finish first)` objective |
| Evaluation | Log-loss, Brier, reliability diagrams, CLV; leakage-aware temporal splits |

## Evaluation methodology

Every claim in this repo is scored against a held-out benchmark, not against intuition.

- **Benchmark is the closing line, not the opening line.** Backtesting against openers
  measures stale-line capture you could not have executed.
- **Metrics are log-loss and CLV.** Never win rate. Win rate over 17 weeks is noise.
- **Temporal splits only.** No random k-fold. Week *t* is never predicted using week *t+1*.
- **Coverage is monitored, not assumed.** Conformal intervals report empirical coverage
  weekly; drift triggers recalibration.
- **Every model writes predictions to a versioned parquet with a timestamp and model hash**,
  so December can audit August.

## Reliability

The failure mode for a seasonal pipeline is not a crash. It is a third-party schema change
that goes unnoticed for three weeks while projections quietly degrade.

- **Data contracts** (`src/hub/contracts.py`) assert schema, nullability, uniqueness, and
  plausible ranges at every ingest boundary. Violations raise; the pipeline serves last-good
  state rather than propagating bad data.
- **Golden-file tests** run nightly against live APIs and diff against frozen fixtures.
  Unit and contract tests use fixtures only, so CI never flakes on someone else's outage.
- **Graceful degradation** is a requirement, not a nicety. Every module produces a usable
  answer with zero operator attention.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env      # add ESPN cookies; .env is gitignored
make draft                # build the draft board
pytest -q
```

Strict TDD. Tests precede implementation, enforced by a `PostToolUse` hook
(`.claude/hooks/tdd_gate.sh`) rather than by discipline alone.

## Data sources and licensing

All free-tier. nflverse (CC-BY 4.0; FTN charting data CC-BY-SA 4.0, attributed to FTN Data
via nflverse), CollegeFootballData API, ESPN's undocumented public endpoints.

**No raw third-party payloads are committed.** CFBD prohibits redistribution, so `data/raw/`
is gitignored and only code and derived model outputs live in version control.

## License

MIT (code). Data remains under its respective source licenses.
