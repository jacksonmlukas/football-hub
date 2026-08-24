# Hydra structured config, digest folded into the model version

**Status:** accepted 2026-08-14; amended by [ADR-0006](0006-fitted-constants-live-with-their-provenance.md), which carries the part that reversed.

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
