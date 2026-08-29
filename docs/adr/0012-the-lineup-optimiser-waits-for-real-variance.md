# The lineup optimiser waits for real variance

**Status:** accepted 2026-08-24. Decision unchanged; its forecast was measured 2026-08-25 and did not hold — see *What would change the verdict*, below.

**Decision.** Start your highest projections in Week 1. `hub.season.lineup.optimize` stays in
the repo and does not set lineups, until per-player variance exists that is not a function of
the projection.

## The measurement

`hub.season.lineup_gate`, four seasons, 80 drafted rosters, both arms seeing only projections:

    n=80   optimiser - projections = +0.00 points per game
           95% CI [-0.00, +0.00]      P(optimiser better) 72.7%

Not a small effect. A structural zero.

## Why it is zero, which is the useful part

The optimiser's only advantage over sorting on projection is that it also reads `sd`: it
maximises P(beat the opponent) rather than expected points, so it can start a lower-projected,
higher-variance player when the matchup calls for upside. `test_the_optimiser_can_prefer_upside_over_projection`
shows it doing exactly that, given the right inputs.

It never gets the right inputs. `hub.models.predict.moments` sets

    sd = WEEKLY_K[pos] * sqrt(mu)

so within a position `sd` is a **deterministic increasing function of `mu`** — measured
correlation 0.985 across the projection range, and the residual is only curvature. Ranking by
`mu` and ranking by any increasing function of `(mu, sd)` therefore produce the same order.
Across positions `WEEKLY_K` spans just 1.88 to 2.13, so even the cross-position channel is
nearly closed.

The optimiser is not wrong. It is being handed no information it does not already have.

## What would change the verdict

Per-player variance that varies *independently of the mean*. Two players projected at 12
points a game are not equally volatile in reality — a boom-bust deep threat and a
target-hogging possession receiver are not the same asset — and the square-root law cannot
say so because it only knows the mean.

That is exactly what the component and usage layers produce: `hub.models.components` already
samples receptions, carries and touchdowns from measured dispersions, and a usage model
anchored on snap and target share would give each player his own volume distribution rather
than a positional constant. When `sd` stops being a function of `mu`, re-run this gate.

Note this is a *stronger* reason to build that layer than "it would be more accurate". It
would make an existing, working, well-tested piece of the system do something for the first
time.

### Measured 2026-08-25: it would not

[player-spread.md](../player-spread.md) tested exactly this claim and it does not survive.
Per-player weekly volatility beyond the positional constant is **real but tiny and not
estimable**: the true spread is ±9.3% in `sd`, a single season recovers 15% of it, and the
shipped `K[position]·sqrt(mu)` already sits within 0.085 MAE of the *irreducible* floor set by
sampling noise in the outcome itself. A player's own prior-season `k` beat the constant in all
five held-out seasons and still missed the significance bar at 1.8 se, gaining 0.0065.

So the sentence above — "two players projected at 12 points a game are not equally volatile in
reality" — is true and much weaker than it reads. They differ by about ±9%, and nothing
observable tells you reliably which is which.

**The decision stands, and now stands on firmer ground.** Start your highest projections. The
instruction to "re-run this gate when `sd` stops being a function of `mu`" is withdrawn: there
is no known route to that, and the headroom would not justify one.

### Amended 2026-08-29: there was a route, and it was one line

The withdrawal above is wrong on its own terms, and the decision it defends is not.

`player-spread.md` measured per-player **outcome volatility** — how much a player's weeks vary
around his own mean — and correctly found it real, tiny and not estimable. **How well we know
that mean is a different quantity**, it is `sigma_pos/√games`, and it is **+36% at one game
played against twelve**. It is not a function of `mu`; it is a function of `n`. That is the
condition this ADR named.

Re-run with `sd = √((k·√mu)² + se²)`:

    as shipped                    +0.00  CI [-0.00, +0.00]   P(better) 72.7%
    with parameter uncertainty    +0.18  CI [-0.28, +0.79]   P(better) 75.5%

**The zero stops being structural.** The interval was [−0.00, +0.00] because the optimiser was
choosing among orderings that could not differ; it now spans a point in each direction because
the optimiser finally has variance to read. Reading it is worth +0.18 points a game and is not
significant.

**So: start your highest projections, unchanged — but because it was measured, not because no
route existed.** `hub.season.lineup_gate --parameter-uncertainty` reproduces it. See
[parameter-uncertainty.md](../parameter-uncertainty.md).

## Also recorded: the gate's first version could not fail

The optimiser arm originally chose each week's lineup from **realised** scores while the
baseline used projections. It returned +31.15 points per game, CI [+29.35, +32.93], and would
have licensed trusting an optimiser that had never been tested — the number was the value of
perfect foresight, not of `lineup.optimize`.

A gate whose treatment arm has information its control arm lacks cannot fail. That is the same
defect class as the draft tripwire earlier the same day, and the reason both are written down
is that neither was obvious while being built.
