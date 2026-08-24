# One Forecaster protocol, league as a field

**Status:** accepted.

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
