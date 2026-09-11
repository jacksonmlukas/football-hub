# One Forecaster protocol, league as a field

**Status:** accepted. **Re-scored 2026-09-07** — the stated payoff was put under question; see
below. **Amended 2026-09-11** (issue #136) — the payoff is carried by the prediction schema and the
store's model column, not by the protocol; the protocol is a **placeholder**, and what retires it is
a second track writing predictions through it. The amendment is set out in full at the foot of this
file.

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
>
> **Settled 2026-09-11.** The question this note left open is answered by the amendment at the
> foot of this file; nothing in the note above is withdrawn, and the four findings still hold.

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

None of that settled **#136**, which stayed `ready-for-human` until 2026-09-11. #136 asked where
this protocol's payoff lands and whether a second adapter changes the answer. The two things above
were wrong under either answer, which is why they were split out. The answer is the amendment
below.

**Conformal is composition, not inheritance.** `Conformalized(BayesianRatings(...))` is itself a
`Forecaster`. Track C therefore applies to every other track for free, and you can compare a
model against its own conformalized version.

**Cost.** Every model carries fields it might not use (`margin_lo`/`margin_hi` equal the mean
when a model has no uncertainty estimate). Cheap, and it keeps the comparison table rectangular.

---

# Amendment, 2026-09-11: the schema carries the payoff; the protocol is a placeholder

Issue #136, decided by the maintainer on 2026-09-11 after the modelling programme this ADR was
waiting on closed without producing what it was waiting for.

**Amended rather than replaced, and that is the point.** Nothing above is withdrawn. The
decision — every track implements `hub.models.base.Forecaster`, league is a field, the
prediction schema is fixed — stands exactly as accepted. What changes is the *account of where
the payoff lands*, because the tree has now been measured against it twice and the answer did
not move.

## Why the answer arrived

The re-scoring of 2026-09-07 held this open on one condition: the modelling programme
(#45–#52) might supply a second adapter, and a second adapter changes the answer. It did not.
#42, #43, #45, #47, #48 and #51 closed and none of them produced a second forecaster; #49, #50
and #52 remain open and none of them will, because none of them is a model. So the condition
the note was waiting on has resolved, in the direction of *no second adapter*, and continuing
to wait would be waiting for nothing.

## Which seam carries the payoff

The head-to-head comparison this ADR promised — *"`model-eval` can compare any two forecasters
head to head without knowing what either one is"* — is real and shipping. It is carried by
**`PREDICTION_SCHEMA` and the store's `model` column**, a row-shaped seam:

* rows conforming to the schema reach the store from more than one writer, each stamping its
  own `model` name — `hub.models.ratings.fit` under the baseline's name, and `hub.store.verify`'s
  market-implied stand-in under `market_implied`;
* the reader does not care which wrote. `hub.store.predictions(model=...)` selects by name, and
  `hub.models.eval.compare(a, b)` takes two *frames* read back by name and never receives a
  forecaster at all.

One adapter is a hypothetical seam; two is a real one. The protocol has one; the schema has two.
That is the whole of the finding, and it is a finding about the tree rather than about the
argument: the argument for a row-shaped comparison was correct, and the row-shaped comparison
is what shipped. The protocol is not what carries it.

## The protocol is a placeholder

`Forecaster` has **one** implementation, `hub.models.market.MarketBaseline`. `Conformalized` is
defined in `hub.models.base` and instantiated nowhere. `ratings.forecaster()` returns the
protocol and is the seam Track A would arrive at, but what it returns today is the passthrough
#205 named. **The protocol stays, as a placeholder** — its type sits on the one path a published
row can take, so the substitution the seam was cut for is a substitution and not an edit — and
this amendment says so in words, so that a future reader cannot conclude the polymorphism works
because an ADR says it does. It has not been demonstrated. Nothing in this repo has ever put
two implementations of this protocol side by side.

## What retires the placeholder

**A second track that writes published predictions through it.** Concretely: a second
`Forecaster` whose rows reach the store by way of `hub.models.base.forecast(model, spec, games)`
— the one path from a `Forecaster` to a row anything publishes — under its own `model` name,
so that `eval.compare` can be pointed at the two names and the comparison the protocol was
accepted for is finally a comparison *of two objects implementing it*. Not a wrapper around the
baseline, not a re-fit of it under another name, and not a passthrough. The day that lands, the
word *placeholder* comes out of this file and the status line records the second adapter by
name and date. Until it lands, this section is the authority on what the protocol is.

## The placeholder's cost, and what holds it

A placeholder interface is cheap only while it is not also a lie about where the checks are.
Two things the original ticket named must not persist under either answer, and both are
recorded here as the placeholder's cost so that neither can regress quietly:

1. **The module that writes the published predictions must not bypass the interface** this
   ADR says every track implements. Until 2026-09-07 `hub.models.ratings` named
   `MarketBaseline` concretely and went around the protocol; #205 routed it through
   `forecaster()` and `base.forecast`, whose parameter is typed as the protocol.
2. **The leakage tripwire must not be reachable only from a function nothing types.**
   `validate_predictions` — the check this ADR calls the highest-value one in the codebase —
   was reachable from exactly one untyped call in the writer's middle; #205 put it inside
   `base.forecast`, where it sits on the typed path as a `# GUARD` that
   `tests/contracts/test_guards_are_load_bearing.py` proves load-bearing.

Both landed on 2026-09-07 and the paragraph above beginning *"That sentence described nothing
until 2026-09-07"* is their account. They are repeated here because a placeholder is exactly
the kind of thing a later hand strips as unused — and stripping the protocol from
`base.forecast`'s signature, or the guard from its body, would reopen both.

## What this amendment does not do

**It changes no behaviour.** No module, constant, test or published artifact moves. It is a
record of which seam carries a promise, written so the promise is not misread.

**It does not recommend building the second adapter.** Whether a second track is worth
building is a modelling question with its own ceiling to compute first (`docs/method.md`
rule 8); this amendment only says what such a track would have to do to retire the placeholder.
