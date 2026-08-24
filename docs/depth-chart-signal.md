# The depth-chart signal: real, and already priced

Run 2026-08-23, immediately after `docs/lambda-sweep.md` concluded that the
expected-versus-actual signal had nothing in it.

**Result: rejected before building anything.** The signal is genuinely real -- it predicts
production strongly *within* a season -- and carries essentially nothing into the next one.
That is a different failure from the previous investigation and the difference is the
interesting part.

## The hypothesis

Consensus anchors on last season's totals. A back who took over the job in November posts
modest totals and is a different asset going forward from one who lost the job in October
with the same totals. Depth-chart movement should separate them.

Signal: **climb** = mean depth-chart rank over weeks 1-4 minus mean rank over the last four
weeks. Positive means the player moved up. Built from `nflreadpy.load_depth_charts()`,
restricted to the offensive formation, which gives one clean row per player-week. Bucky
Irving in 2024 reads RB3 through week 5 and RB2 from week 6 -- the pattern the signal is
meant to catch.

## The screen, run first

The lesson from the lambda investigation was that five rounds of increasingly careful
measurement could have been short-circuited by one cheap precondition check. So this time
the check came first, before any sweep, board adjustment, or draft simulation.

**Does climb in season N predict season N+1 production, beyond what ECR already says?**
Partial correlation, controlling for preseason ECR:

| pair | n | raw r | partial r |
|---|---|---|---|
| 2021-22 | 283 | +0.003 | +0.007 |
| 2022-23 | 282 | +0.042 | +0.074 |
| 2023-24 | 313 | +0.083 | +0.025 |
| 2024-25 | 287 | -0.033 | **-0.072** |

Pooled: **r = +0.0085**, about 0.3 sigma, with the sign flipping in one of four seasons.

With roughly 1,150 observations the screen can detect |r| > 0.06 at two sigma. It does not.
Anything the signal contributes beyond ECR is smaller than that, which is under half a
percent of variance.

## But the signal is not broken

The obvious worry with a null is that the construction is wrong rather than the hypothesis.
So the same signal was checked against **same-season** production: does climb predict
second-half points, controlling for first-half points?

| season | n | partial r |
|---|---|---|
| 2021 | 371 | +0.152 |
| 2022 | 357 | +0.220 |
| 2023 | 365 | +0.217 |
| 2024 | 359 | +0.187 |

Pooled **r = +0.194, 7.4 sigma, no sign flips.** The signal is real, sizeable and
consistent. A player who climbs the depth chart genuinely scores more in the second half
than his first half predicts.

It simply does not survive the offseason.

## Two different ways to fail

This is worth separating from the previous result, because they look the same from the
outside and are not.

| | expected-vs-actual | depth-chart climb |
|---|---|---|
| real within season | weakly | **yes, r = +0.19 at 7.4 sigma** |
| persists to next season | r = 0.21 self-correlation | not tested -- irrelevant |
| adds to ECR next season | no | **no, r = +0.01** |
| why | **too noisy to begin with** | **already priced** |

The expected-versus-actual gap barely reproduces itself year to year -- about 4% shared
variance -- so there was never enough there to exploit. Depth-chart climb is the opposite:
a strong, stable, real effect that consensus has fully absorbed by the time it publishes a
preseason board. A player who took over in November is not a secret in August.

The first is a signal that does not exist. The second is a signal that exists and is not
yours. Only measurement tells them apart, and the remedy is different in each case: the
first is unfixable, the second would need to be observed *earlier* than the market rather
than measured better.

## What was not built

No board adjustment, no lambda, no sweep, no draft evaluation. The screen costs one query
and answers the question, and building the apparatus first would have produced six seasons
of carefully-measured noise -- which is exactly what the previous investigation did before
arriving at the same place.

That is the transferable part: **check whether a signal survives the horizon you need it
over, before building anything that consumes it.**

## Reproduce

The screen is thirty lines and lives in this document's history rather than in `src/`,
deliberately -- nothing imports it, because nothing should. If a future signal needs the
same treatment, the shape to copy is: partial correlation against next-season outcome
controlling for the board you would otherwise use, plus a same-season control to prove the
construction works.
