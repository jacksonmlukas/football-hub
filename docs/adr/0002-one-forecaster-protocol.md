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

**That sentence described nothing until 2026-09-07, and issue #205 is why.** The boundary had
no code on it. `hub.models.ratings` — the module that writes the published predictions — named
`MarketBaseline` concretely, fitted it, predicted, and then *remembered* to call
`validate_predictions` on the way past. So the highest-value check in the codebase was
reachable from exactly one function nothing typed, and Track A replacing "the middle of that
function" would have carried the obligation to remember it again. A check nothing can reach
reports nothing and cannot fail.

`hub.models.base.forecast(model: Forecaster, spec, games)` is now the one path from a
`Forecaster` to a row anything publishes: it fits, predicts and checks against a single
`FitSpec`, so the three cannot drift apart in a caller. `ratings.forecaster()` is the seam
Track A arrives at, and its return type is this protocol — replacing the passthrough is a
substitution rather than an edit to a writer's middle. The tripwire's reachability is a
`# GUARD` proved by `tests/contracts/test_guards_are_load_bearing.py`: delete the call and
both `unit/test_ratings.py` and `unit/test_models_base.py` go red.

None of that settles **#136**, which stays `ready-for-human`. #136 asks where this protocol's
payoff lands and whether a second adapter changes the answer. The two things above were wrong
under either answer, which is why they were split out.

**Conformal is composition, not inheritance.** `Conformalized(BayesianRatings(...))` is itself a
`Forecaster`. Track C therefore applies to every other track for free, and you can compare a
model against its own conformalized version.

**Cost.** Every model carries fields it might not use (`margin_lo`/`margin_hi` equal the mean
when a model has no uncertainty estimate). Cheap, and it keeps the comparison table rectangular.
