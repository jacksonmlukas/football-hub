# Opponent correlation: the shootout is real, and nothing consumes it yet

**Measured 2026-08-25**, `hub/models/correlate.py`. Fills the hole
[correlation.md](correlation.md) names explicitly:

> **Opponent correlation is not modelled.** [...] Shared game exposure shrinks *margin*
> variance, which helps whoever is favoured. Nobody prices it here either.

Now it has a number. It is still not priced, and the last section says why that is the right
call today.

## Method, and the check that it is the right one

Identical to the teammate measurement, so the two are comparable: standardise each player's
weekly PPR points within his own season — at least 8 weeks, non-zero spread — then pair
players who appeared in the same game and correlate the standardised values by position pair.
2022-25 regular season, 20,200 player-weeks.

**The check:** run the same function over *same-team* pairs and it should reproduce
`TEAMMATE_RHO`, which was fitted independently.

| pair | this module | `TEAMMATE_RHO` |
|---|---|---|
| QB-WR | +0.2219 | 0.232 |
| QB-TE | +0.2045 | 0.225 |
| QB-RB | +0.0519 | 0.054 |

Within about 0.02 on all three, and teammate WR-WR comes back at +0.009 (1.1 se), matching
correlation.md's observation that two receivers on one team are *not* the folklore's stack.
The method is the method.

## What opponents do

| pair | n | rho | se | se away |
|---|---|---|---|---|
| **QB-QB** | 933 | **+0.1484** | 0.0328 | **4.5** |
| **QB-TE** | 3,897 | **+0.0660** | 0.0160 | **4.1** |
| **QB-WR** | 7,985 | **+0.0550** | 0.0112 | **4.9** |
| WR-WR | 17,181 | +0.0216 | 0.0076 | 2.8 |
| TE-WR | 16,767 | +0.0192 | 0.0077 | 2.5 |
| RB-RB | 6,606 | **−0.0236** | 0.0123 | 1.9 |
| QB-RB | 4,945 | +0.0213 | 0.0142 | 1.5 |
| RB-WR | 21,269 | +0.0081 | 0.0069 | 1.2 |
| TE-TE | 4,038 | +0.0075 | 0.0157 | 0.5 |
| RB-TE | 10,404 | +0.0046 | 0.0098 | 0.5 |

**Two quarterbacks in one game correlate at +0.148** — larger than the *teammate* QB-RB edge
of +0.054 that the simulator already models. The passing game travels: QB-TE and QB-WR against
opponents both clear four standard errors.

**RB-RB goes negative.** −0.024 at 1.9 se, so suggestive rather than established, but the sign
is what game script implies: the team that is ahead runs out the clock while the team that is
behind throws, so two opposing backfields split rather than share.

Everything that is not a quarterback pairing is small. The effect is concentrated exactly where
game total lives.

## Why this is not wired into anything

It is a measurement, not a model, deliberately. The two things that would consume opponent
correlation are both currently inert:

* **`lineup.optimize`** uses `group_sd`, which reads teammate correlation. But
  [ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md) measured the optimiser
  at +0.00 points a game, because `sd = k·sqrt(mu)` gives it no variance to exploit. Adding a
  correlation term to an objective that cannot use variance changes nothing.
* **`season.simulate_weeks`** correlates teammates by Cholesky. Its consumer was championship
  equity, removed by [ADR-0009](adr/0009-championship-equity-does-not-pick.md); the only caller
  left is the backtest harness.

There is also plumbing missing: `group_sd` takes `(sd, position, nfl_team)` and would need the
*game*, which the board does not carry.

So wiring it now would add a parameter, and plumbing to feed it, to buy nothing measurable.
That is the shape of change this repo has spent a week removing.

**When it becomes worth wiring:** when per-player variance stops being a function of the
projection — the usage and component layers — the lineup optimiser starts having something to
optimise, and at that point correlation is what makes a two-players-from-one-game lineup price
correctly. The gate is the one correlation.md already uses: does adding it move interval
coverage toward nominal?

## Reproduce

```bash
uv run python -m hub.models.correlate                # opponents
uv run python -m hub.models.correlate --teammates    # the method check above
```

23 offline tests in `tests/unit/test_correlate.py`, including that a negative relationship
survives the pipeline with its sign intact — this repo has shipped a sign bug before.
