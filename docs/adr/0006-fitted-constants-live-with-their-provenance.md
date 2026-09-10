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

  > **Re-scored 2026-09-07 (issue #51): this limitation is understated, and #201 reopens it.**
  > The sentence above frames the gap as a stray *threshold* — one number, one hand-repair, a
  > rounding error on a rule that otherwise holds. The live instances are not thresholds. They
  > are **shape** constants that set the extent of a random draw: `REG_SEASON_WEEKS = 14`,
  > `PLAYOFF_ROUNDS = 3`, `PLAYOFF_TEAMS = 6`, `DEFAULT_ROUNDS = 14`, `cohort.ROUNDS` /
  > `cohort.DRAFTS`, and the `n_draft_sims` / `n_season_sims` defaults.
  >
  > By the finding in #196 each of those fixes the seed-to-outcome map, so moving any one of
  > them re-prices every published Gate interval **while leaving both digests byte-identical**.
  > That is the failure this ADR exists to prevent, reached through the type system rather than
  > through a missing registration — and it is a materially larger claim than the one weighed
  > here, which is why it is a re-scoring and not a footnote.
  >
  > **This ADR's decision is not disturbed.** The split between a setting and a fitted constant
  > holds, and so does *coverage by the digest is what matters*. What is wrong is the size this
  > file assigns to its own escape hatch. **Owned by #201**, whose criteria require that a
  > digest move when any of these move, or that their exclusion become a recorded decision
  > rather than a consequence of being written as `int`.
  >
  > **Settled 2026-09-10 (#201).** `config_digest` moved `6bdcb663` → `3f96c0c2` as a coverage
  > correction — no constant was refitted and no prediction differs across it. Two things
  > changed. Six of the shape constants above are now registered individually in
  > `FITTED_EXTRA`, so moving any one of them moves the model version, asserted one constant at
  > a time in `test_moving_a_simulation_extent_moves_the_model_version`. And the sweep inside a
  > registered module stopped inferring coverage at all: it takes every name the module
  > *assigns*, and a constant leaves the digest only by being named in `config.NOT_IN_DIGEST`
  > with the argument for it. That closes the same hole from the other two directions it had
  > also been open in — a name excluded for being capitalised (#187's `_FACTOR_CACHE_MAX`) or
  > included for being a number (#199's `TYPE_CHECKING`).
  >
  > **One escape is left open and named**: `n_draft_sims` and `n_season_sims` are function
  > signature defaults, so there is no module-level name to register. Covering them means
  > giving them names in `hub.draft.optimize` and `hub.draft.backtest` first.
  >
  > The float scan this bullet describes still scans for floats, and that is now a narrower
  > claim than it was: it answers "has a whole module fallen off the registry", not "is this
  > constant covered", because a module that has never declared anything has no declaration to
  > read. Widening it to `int` flags fifty names under `src/hub` — cache sizes, API tiers,
  > filesystem roots, print widths — which is why the constants known to matter are registered
  > rather than left to it.
- Refitting anything now moves the model version, which is the point.
