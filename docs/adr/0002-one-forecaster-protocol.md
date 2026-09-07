# One Forecaster protocol, league as a field

**Status:** accepted. **Re-scored 2026-09-07** — the stated payoff is under question; see below.

> **Re-scoring, 2026-09-07 (issue #51).** This ADR is accepted on a payoff — *"`model-eval` can
> compare any two forecasters head to head without knowing what either one is"* — and the
> architecture review of 2026-09-06 measured that payoff against the tree rather than against
> the argument. What it found:
>
> * the protocol has **one** implementation;
> * the conformal wrapper this file cites as its composition argument is defined in
>   `hub.models.base` and instantiated nowhere;
> * the ratings track is not a second implementation — `ratings.forecaster()` returns this
>   protocol, but it is the passthrough #205 named, and a seam Track A has not yet arrived at;
> * `eval.compare` never receives a forecaster at all. Its two arguments are frames, read back
>   from the store by model name.
>
> **Re-checked against the tree on 2026-09-07, after #205 merged**, and all four still hold.
> #205 put code on the type boundary — a real repair, recorded above — but routing the writer
> *through* an interface is not a second implementation *of* it, so it moves none of the four.
>
> **The comparison itself is real and shipping.** The open question is which seam carries it:
> the protocol, or the prediction schema plus the store's model column — which has two writers
> and a reader indifferent to which wrote. One adapter is a hypothetical seam; two is a real
> one.
>
> **Nothing here is decided by this note, deliberately.** #136 is `ready-for-human` precisely
> because settling it either builds a second adapter or amends this ADR, and because the
> modelling programme may yet supply the adapter that changes the answer. What a future reader
> must not do is conclude the polymorphism works because this ADR says so.
>
> One half needed no decision and was split out as **#205** — the writer bypassing this
> interface, and the leakage tripwire reachable only from a function nothing typed. **It landed
> on 2026-09-07**, and its own account is the paragraph beginning *"That sentence described
> nothing until 2026-09-07"* below, which is the authority on what changed.
>
> **Which leaves #136 as the whole of the open question.** #205 does not settle it either way,
> and this ADR still rests on a payoff no adapter yet demonstrates. **Owned by #136**
> (which seam carries it — `ready-for-human`).

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
