# Fitted constants live with their provenance, not in the config

**Status:** accepted 2026-08-24. Supersedes the "every number lives in `config.py`" rule of
[ADR-0004](0004-hydra-config-digest.md); the rest of 0004 stands.

**Decision.** Split the two kinds of number that ADR-0004 conflated, and hash both.

* A **setting** is a choice — `projection_lambda`, `conformal_alpha`, the roster shape. It
  lives in `src/hub/config.py`, where Hydra can override it from a command line.
* A **fitted constant** is a measurement with a confidence interval and a write-up in `docs/`.
  It lives beside its provenance in a module named by `config.FITTED_MODULES`, and
  `config_digest` hashes it from there.

`config.NOT_FITTED` names modules whose measured floats are deliberately excluded, with a
reason for each. `config.FITTED_EXTRA` names individual constants in modules that cannot be
registered wholesale.

## Why ADR-0004's rule had to go

It was never followed. Seventeen fitted constants sat outside `config.py`, and
`config_digest`'s docstring claimed to hash "everything that can change a prediction" while
hashing only `HubConfig`. Refitting `TALENT_CV` from 0.35 to 0.42 on 2026-08-23 changed every
prediction in the repo and left the model version byte-identical — so the track record claimed
one model had produced both sets of rows. That is precisely the failure ADR-0004 exists to
prevent, committed by ADR-0004's own file.

The obvious repair — move the seventeen constants into `HubConfig` — is worse than the
disease, for two reasons:

1. **It would make measurements overridable.** `draft.talent_cv=0.9` on a command line would
   silently replace a measured quantity with a preference. A fitted constant has a confidence
   interval; a config field invites a guess.
2. **It would strand the provenance.** `TALENT_CV` carries thirty lines explaining the fit,
   the sample, and why the previous value sat 4.6 standard errors low. That belongs next to
   the number, not in a config schema.

So the rule is not relaxed, it is made true: coverage by the digest is what matters, and
*where* a number lives follows from what kind of number it is.

## Consequences

- `FITTED_MODULES` is a list of **modules**, not of constants, so a new fitted number is
  covered the day it lands rather than the day someone remembers to register it.
- A test walks `hub/models` and `hub/draft` asserting every module holding a measured float is
  in `FITTED_MODULES` or `NOT_FITTED`. Known limitation, stated rather than hidden: it scans
  for floats, so an integer threshold in an unregistered module still slips through — which is
  how `board.MIN_GAMES` escaped until `FITTED_EXTRA` picked it up by hand.
- Refitting anything now moves the model version, which is the point.
