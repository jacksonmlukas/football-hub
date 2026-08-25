# Architecture decisions

The decisions themselves live one per file in [`docs/adr/`](adr/), which is where `CLAUDE.md`
and `docs/agents/domain.md` have always said to look. This file is the index.

They were a single document until 2026-08-24. Splitting them made each one addressable — an
architecture review had cited "ADR-004 in `docs/architecture.md`" for want of a path — and let
a superseded decision keep its original text instead of being edited in place.

| ADR | Decision | Status |
|---|---|---|
| [0001](adr/0001-duckdb-query-layer.md) | DuckDB as the query layer, parquet stays the format | accepted |
| [0002](adr/0002-one-forecaster-protocol.md) | One `Forecaster` protocol, league as a field | accepted |
| [0003](adr/0003-make-now-dagster-in-october.md) | `make` now, Dagster in October, written so the port is a decorator | accepted, partly addressed |
| [0004](adr/0004-hydra-config-digest.md) | Hydra structured config, digest folded into the model version | amended by 0006 |
| [0005](adr/0005-tiered-sync-polling.md) | Tiered sync polling, no async | accepted |
| [0006](adr/0006-fitted-constants-live-with-their-provenance.md) | Fitted constants live with their provenance, not in the config | accepted |
| [0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) | A measurement that steers the product must be committed code | accepted |
| [0008](adr/0008-the-simulator-indexes-the-board.md) | The draft simulator indexes the board | accepted |
| [0009](adr/0009-championship-equity-does-not-pick.md) | Championship equity does not pick | accepted |

Related, and deliberately not ADRs:

- [`decisions.md`](decisions.md) is the running journal — corrections issued during design,
  standing directions, open questions. Decisions of record are here; the working out is there.
- [`CONTEXT.md`](../CONTEXT.md) is the glossary. An ADR says what was decided; the glossary
  says what the words mean.
