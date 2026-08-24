# The depth-chart signal: not predictive, and a leakage bug of my own making

Run 2026-08-23. **This document contains a correction to its own first version.** The
correction is the more useful half.

**Result: rejected as a draft signal and as a waiver signal. The earlier claim that it was
"real within a season but already priced" was wrong, and wrong because of lookahead.**

## The hypothesis

Consensus anchors on season totals. A back who took over the job in November and one who
lost it in October can post identical totals and be entirely different assets. Depth-chart
movement should separate them.

Signal: **climb** = mean depth-chart rank over weeks 1-4 minus mean rank over a later
window. Positive means the player moved up. From `nflreadpy.load_depth_charts()`,
restricted to the offensive formation, which gives one clean row per player-week.

## Screen 1: as a draft signal

Does climb in season N predict season N+1 production, beyond what ECR already says?
Partial correlation controlling for preseason ECR:

| pair | n | partial r |
|---|---|---|
| 2021-22 | 283 | +0.007 |
| 2022-23 | 282 | +0.074 |
| 2023-24 | 313 | +0.025 |
| 2024-25 | 287 | -0.072 |

Pooled **r = +0.0085**, about 0.3 sigma, sign flipping in one of four. With ~1,150
observations this detects |r| > 0.06 at two sigma. It does not. **This result stands.**

## The correction

The first version of this document then reported that the signal was nonetheless *real*,
because climb predicted second-half production at r = +0.19, 7.4 sigma, across four
seasons. From that it concluded the signal existed but was already priced by preseason --
a satisfying story, and false.

The climb in that test was measured over **the last four weeks of the season**. The outcome
was **points from week 10 onward**. Those windows overlap. A player ranked RB1 in week 17
scored points in week 17, which is inside the thing being predicted. The test was using the
future to predict the past.

Same outcome window, same control, only the measurement date changed:

| climb measured | 2021 | 2022 | 2023 | 2024 | mean |
|---|---|---|---|---|---|
| weeks 16-19, **inside** the outcome window | +0.152 | +0.220 | +0.221 | +0.182 | **+0.194** |
| through week 9 only, **before** it | -0.000 | +0.072 | -0.069 | -0.023 | **-0.005** |

The entire effect was lookahead. Measured honestly the signal predicts nothing within a
season either.

**This is the exact failure `hub.store` exists to prevent.** `AS_OF_LINES` and its tests
were built precisely so a prediction can only ever see the line that was live when it was
made, and `CLAUDE.md` says leakage looks like success rather than failure. Having built
that apparatus, I then made the error by hand in an ad-hoc analysis that did not go through
it -- and it produced exactly the confident, plausible, wrong result the machinery was
designed to catch.

## Screen 2: as a waiver signal

The corrected picture removes the reason to expect anything, but the waiver question is
genuinely different and worth its own screen: at week W, does climb predict rest-of-season
points beyond season-to-date scoring? Only information available at week W is used.

`docs/championship-leverage.md` also constrains this. Reverse-standings waivers with no
FAAB mean a good team picks 10th-12th every week, so contested adds are effectively
unavailable and only uncontested ones matter. The "uncontested" rows below restrict to
players in the bottom half of season-to-date scoring -- the ones nobody is claiming yet.

| week | group | 2021 | 2022 | 2023 | 2024 | mean | se |
|---|---|---|---|---|---|---|---|
| 6 | all | -0.062 | -0.028 | -0.032 | -0.077 | -0.050 | 0.052 |
| 6 | uncontested | -0.142 | +0.029 | -0.085 | -0.150 | -0.087 | 0.073 |
| 8 | all | -0.031 | +0.047 | -0.042 | +0.030 | +0.001 | 0.052 |
| 10 | all | -0.040 | +0.100 | -0.049 | +0.035 | +0.012 | 0.052 |
| 10 | uncontested | -0.051 | +0.180 | -0.055 | +0.015 | +0.022 | 0.074 |
| 12 | all | -0.025 | +0.072 | -0.005 | +0.056 | +0.024 | 0.053 |

Every mean is within about one standard error of zero and the sign flips across seasons in
every row. Nothing.

### A second methodological error, also caught

The first version of this screen pooled weeks 6-13 into one correlation, giving n ~ 2,900
per season and apparently significant results at 3-4 sigma -- with the sign flipping between
seasons, which should never happen at that significance. The cause: **the same player
appears at eight different weeks**, so the observations are anything but independent and
the effective n is roughly eight times smaller. The table above uses one week per season so
each player appears once.

Two errors in one investigation. Both were caught by a result looking wrong rather than by
a test, which is worth noting given how much of this repo's testing exists to make that
unnecessary.

## What this leaves

Depth-chart climb, honestly measured, predicts nothing at any horizon tested: not next
season beyond ECR, not rest-of-season beyond season-to-date, not among uncontested players
specifically.

That is a cleaner and less interesting answer than "real but priced", and it is the correct
one. The comparison with `docs/lambda-sweep.md` also collapses: both signals are simply
null. There is no instructive contrast between "does not exist" and "exists but is not
yours", because the second case was never demonstrated.

## What was not built

No board adjustment, no lambda, no sweep, no waiver module, nothing in `src/`. The screens
cost a few queries. Building first would have produced a carefully-engineered pipeline
consuming a signal that does not exist -- and, given the leakage, one whose validation would
have looked excellent.

## The transferable part

1. **Check whether a signal survives the horizon you need it over, before building
   anything.** Both screens here cost minutes and settled the question.
2. **Measure the predictor strictly before the outcome window.** Obvious, embarrassing to
   restate, and it is what went wrong here anyway.
3. **Repeated measures are not independent observations.** Eight weeks of the same player
   is not eight data points.
4. **A significant result whose sign flips between seasons is a bug, not a finding.** That
   pattern was the tell in both errors.
