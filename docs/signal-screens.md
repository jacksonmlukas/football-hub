# Signal screens

Six hypotheses for beating consensus. Five null, **one not** — the snap-share trend, screened
2026-08-25, which is the first thing in this repo to survive a control for ECR. This is the
index; the ones with longer stories have their own documents.

Kept because a record of what does not work is worth more than a fifth attempt at the same
thing, and because two of these produced errors worth not repeating.

| signal | horizon | result | detail |
|---|---|---|---|
| Expected-vs-actual points | next season | **null** — r = 0.21 self-persistence, ~4% shared variance | [lambda-sweep.md](lambda-sweep.md) |
| Recency-weighted expected-vs-actual | next season | **null, and worse** — shorter windows strictly degrade it | [lambda-sweep.md](lambda-sweep.md) |
| Depth-chart climb | next season | **null** — partial r = +0.008 beyond ECR | [depth-chart-signal.md](depth-chart-signal.md) |
| Depth-chart climb | rest of season | **null** — within 1 se of zero at weeks 6/8/10/12 | [depth-chart-signal.md](depth-chart-signal.md) |
| Age | next season | **null** — see below | this document |
| **Snap-share trend** | next 3 weeks, in season | **POSITIVE** — partial r **+0.24** beyond PPG *and* ECR, 12/12 season-anchor cells, placebo-clean | [snap-trend-signal.md](snap-trend-signal.md) |
| Snap-share trend | next 3 weeks, before week 8 | **null** — sign flips between seasons | [snap-trend-signal.md](snap-trend-signal.md) |
| **Snap-share trend** | **next week**, week ≥ 8 | **POSITIVE** — partial r **+0.043** beyond PPG *and* weekly ECR, joint-screened, 5/5 seasons | [weekly-screen.md](weekly-screen.md) |
| **Prior TD rate per yard** | next week | **POSITIVE, negative sign** — **−0.040**, 5/5 seasons. TD-carried scoring predicts *less* next week | [weekly-screen.md](weekly-screen.md) |
| **Defence vs position** | next week | **POSITIVE** — **+0.028** joint, 5/5 seasons. Clears against points, not against Usage | [weekly-screen.md](weekly-screen.md) |
| **Injury severity** | next week | **POSITIVE, negative sign** — **−0.023** joint, 5/5 seasons | [weekly-screen.md](weekly-screen.md) |
| Implied team total | next week | **null on the joint screen** — +0.048 alone, 4/5 once own spread is controlled for | [weekly-screen.md](weekly-screen.md) |
| Own spread | next week | **null on the joint screen** — one signal with the implied total, two hats (r = +0.83) | [weekly-screen.md](weekly-screen.md) |
| Target-share trend | next week | **null** — +0.006, 1/5 seasons | [weekly-screen.md](weekly-screen.md) |
| Wind | next week | **null** — 4/5 seasons | [weekly-screen.md](weekly-screen.md) |
| Rest days | next week | **null** — −0.014, 3/5 seasons | [weekly-screen.md](weekly-screen.md) |

## Where the weekly work landed

Four features cleared the weekly screen; two carried into a model; the model beat the flat
projection on accuracy and lost the lineup decision to a free consensus ranking, harder with
waivers than without, and a pre-registered shrinkage did not rescue it.
[weekly-screen.md](weekly-screen.md) → [weekly-projection.md](weekly-projection.md) →
[weekly-gate.md](weekly-gate.md) → [weekly-shrinkage.md](weekly-shrinkage.md).

## Age

**Hypothesis.** Consensus anchors on production and underweights aging. Running backs fall
off a cliff around 27-28, receivers around 30; if rankers are slow to mark that down, age
is a free adjustment.

**Screen 1, linear.** Does age at the season opener predict production beyond ECR? Partial
correlation controlling for preseason ECR, one row per player-season, no leakage risk since
age is known in advance:

| board season | n | partial r |
|---|---|---|
| 2021 | 424 | -0.012 |
| 2022 | 417 | +0.037 |
| 2023 | 415 | -0.085 |
| 2024 | 455 | +0.010 |
| 2025 | 429 | +0.088 |

Pooled **r = +0.0076**, se ~0.022, sign flipping in two of five.

**Screen 2, nonlinear.** A linear correlation is the wrong instrument for a cliff, and
would cancel a peaked curve entirely, so this needed the bucketed version too. Mean
ECR-residual by age bucket, pooled 2021-2025, residual taken from a quadratic fit on ECR
within position:

| pos | <24 | 24-26 | 26-28 | 28-30 | 30+ |
|---|---|---|---|---|---|
| RB | -6±5 | +2±4 | +4±6 | -1±7 | +3±8 |
| WR | -0±4 | +3±4 | +2±5 | +1±6 | -10±6 |
| TE | +1±5 | +6±5 | -7±4 | -1±6 | +1±7 |
| QB | -5±14 | +4±10 | -4±13 | +7±15 | -1±9 |

No cliff at any position. Two cells sit near 1.7 sigma -- WR 30+ and TE 26-28 -- and with
twenty buckets that is what chance produces, not a finding. There is no RB decline visible
at 27-28, which is where the hypothesis says it should be loudest.

**What is not claimed.** The obvious story is "consensus already prices age", and it is
probably true, but this screen does not demonstrate it. Correlation between age and ECR
within position is near zero (RB +0.02, WR -0.04, QB +0.01, TE -0.11), which sounds like
the opposite -- except the board is heavily survivorship-selected. Only older players who
are still good remain on it at all, so raw age-versus-rank says little about pricing.

What the screen does establish is the actionable part: **age adds nothing to ECR.** Why is
a separate question it was not designed to answer, and after getting a mechanism wrong on
the depth-chart signal, asserting one here would be repeating that.

## What was built

Nothing, for any of the five screens. No module, no lambda, no adjustment in `src/`. Each
cost a few queries.

## The screening protocol, as it now stands

Assembled from the errors these produced, not from foresight.

1. **Screen before building.** A partial correlation against the outcome, controlling for
   the board you would otherwise use. Minutes, not sessions. The lambda investigation took
   five rounds of increasingly careful measurement to reach a null that this finds at once.
2. **Measure the predictor strictly before the outcome window.** The depth-chart signal
   appeared real at 7.4 sigma because the climb was measured inside the window it was
   predicting. That is the failure `hub.store` exists to prevent, made by hand in an
   analysis that did not route through it.
3. **Repeated measures are not independent.** Eight weeks of the same player is not eight
   observations, and pooling them turned noise into an apparent 4-sigma result.
4. **A significant effect whose sign flips between seasons is a bug.** That was the tell in
   both errors above, and it is the cheapest diagnostic available.
5. **Use the right instrument for the shape.** A linear correlation cannot see a cliff.
   Age needed the bucketed screen as well, and would have been dismissed too early without
   it.
6. **Do not assert a mechanism the screen did not test.** "Already priced" is a satisfying
   explanation and was wrong once here already.

## Where this leaves the modelling

Consensus survived every *preseason* attempt, which is the expected outcome and the reason
`MarketBaseline` exists as the benchmark. It did **not** survive the in-season snap-share
screen, and the difference between those two facts is the most useful thing in this file:
every null here asked consensus about information it had all summer to price. The one
positive asked about something published on a Monday. `docs/decisions.md` sets the order: win the
league first, understand the techniques second, edge last. Four null screens are consistent
with that ordering rather than a setback to it.

The durable output is `hub.draft.evaluate`, which scores any board by the roster it drafts,
and this protocol. Both outlive the hypotheses that produced them.
