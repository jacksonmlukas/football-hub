# Championship leverage

Optimizing for P(finish 1st) rather than expected points, and the one object that also
produces player props and game probabilities.

## Declared objective

**Maximize P(finish 1st).** Explicitly do not hedge toward 2nd or 3rd, even if they pay.
This is Jackson's stated preference, not a modeling simplification. A future session must not
quietly substitute E[payout].

## League facts that drive everything

| Fact | Consequence |
|---|---|
| 12 teams, **8 make playoffs**, 3 weeks (15-17), no byes | Regular season is nearly a formality |
| Pure head-to-head | Surplus points above your opponent's score are wasted |
| **Reverse-standings waivers**, weekly | A good team picks 10th-12th on the wire all season |
| **6 bench, 1 IR** | ~2 speculative stashes at a time, after byes and a handcuff |
| Snake, slot 3, 12-team, full PPR, 3WR + flex | See `decisions.md` |

## The core reframe: one object, three consumers

The primitive is a **joint distribution over player-week scores** — not point projections, the
full correlated joint. Everything reads from it:

| Consumer | How it reads the joint |
|---|---|
| Player props | Marginal quantiles per player-week |
| Game probabilities | Aggregate players to team scores, compare |
| P(championship) | Aggregate your roster, simulate the league |

Three roadmap items collapse into one object plus three readers. This is the main structural
argument for building it.

## Why P(champ) is not max points

1. **Head-to-head wastes surplus.** 180 against an opponent's 90 is worth exactly what 91 is.
2. **Playoffs are a different game than the regular season.** P(champ) = P(seed) x P(win 3
   single-elimination games). The first rewards consistency, the second rewards variance.
3. **Variance preference is state-dependent and sign-flips.** Favorites want low variance,
   underdogs want high. The same player therefore has different value on different rosters,
   which VOR cannot express.
4. **Correlation matters twice.** Within your roster (stacks raise weekly variance) and
   against your opponent's roster (shared game exposure shrinks *margin* variance, which helps
   whoever is favored). Nobody prices the second one.

## What 8-of-12 does to the objective

Two-thirds of the league advances and there are no byes, so all eight survivors play the same
three rounds. Seeding buys marginally easier matchups and nothing else. For a competent roster,
P(make playoffs) is comfortably north of 85%, so **dP(champ)/d(regular-season win) is close to
zero**.

Therefore: **optimize the roster for weeks 15 through 17 and pay nothing for regular-season
consistency.**

### The uncomfortable corollary

Reverse-standings waivers mean winning regular-season games *costs* you wire access while buying
almost nothing. The marginal value of your 7th through 10th win is plausibly negative.

**This is not a license to tank.** Throwing games is bad faith and likely against league rules.
The legitimate, narrower version: **never sacrifice ceiling for a marginal regular-season win.**
Start the boom-bust player. Take the upside stash over the safe veteran. In this league a
ceiling-for-floor trade is simply a bad trade.

## The largest underexploited edge

**Playoff-schedule-aware roster construction.** Weeks 15-17 NFL matchups are known today
(`nflreadpy.load_schedules()`). A player facing three soft defenses in the fantasy playoffs is
worth materially more than his season-long projection implies.

In a 4-team-playoff league this is secondary to securing a bye. Here it is close to primary, and
no casual ESPN drafter prices it. Build the weeks 15-17 strength-of-schedule table before Sep 3
and carry it as a tiebreaker column on the draft board.

## Structural constraints that push weight back onto the draft

Three facts compound:

- Reverse-standings waivers + a good team = last priority every week. **Contested adds are
  effectively unavailable.** Only uncontested adds work.
- 6 bench spots in a 12-team league is tight. After bye coverage and one handcuff, realistic
  speculative capacity is **about two players**.
- No FAAB means there is no budget to allocate. Your priority is a deterministic function of
  your record, known in advance.

So the in-season acquisition channel is weak *and* storage is small. **Whatever shape you want
in weeks 15-17, you largely have to draft.** You will not fix it in October.

### Three consequences

**The IR slot is unusually valuable here.** A player who misses eight weeks and returns for the
stretch costs almost nothing: early losses are nearly free under 8-of-12, and IR does not consume
bench space. It is close to a free option on a discounted asset. Worth one deliberate pick.

**You cannot stream K or DST.** Streaming assumes waiver access you do not have. Draft ones you
will hold all season, or confirm via `espn-api` whether the league even uses those slots (it also
feeds replacement level).

**Cap speculative stashes at two.** Each must survive the question "would I rather have bye-week
insurance in this slot?"

## Architecture

**L1 — Joint sampler.** Marginals from the projection layer. Correlation from nflverse
play-by-play as a copula over (QB, WR1, WR2, TE, RB, opposing DST) within game, conditioned on
game total and spread.

Published priors to shrink toward, because per-offense estimates are extremely noisy:

| Pair | Approx correlation |
|---|---|
| QB - WR1 | +0.3 to +0.5 |
| QB - TE1 | second strongest positive |
| WR1 - WR2 | ~+0.16 at ceiling outcomes |
| RB - WR | slightly negative (~-0.07) |
| QB - opposing DST | ~-0.45 |

Note these are single-game correlations. Season-long correlation (e.g. QB-RB) is stronger and is
a different quantity — do not mix them.

**L2 — League simulator.** Schedule, weekly lineup-setting for you *and* opponents, head-to-head,
seeding, 8-team bracket. Output: P(champ) for a roster configuration.

Opponent lineup model: start from "they start their highest ESPN projection," then validate
against actual historical lineups pulled from `espn-api`.

**L3 — Evaluator.** delta-P(champ) for a candidate action. Draft, waiver, and trade all call this.

**L4 — Decision layers.** Waiver and trade are direct calls. Draft requires rolling out the
remaining draft with an opponent model, which `hub/draft/availability.py` already provides.

## Compute

Nested Monte Carlo (rollouts x seasons x weeks) is expensive, and worse, the P(champ) gap between
two candidates is often smaller than simulation noise. Two fixes, both already in your toolkit:

- **Common random numbers.** Evaluate every candidate against the same sampled futures. The
  *difference* then has far lower variance than either estimate.
- **Distillation**, the pattern from the World Cup sim. Slow realism engine generates training
  data; a fast surrogate maps roster features to P(champ) and answers in milliseconds during a
  live draft.

## Baseline you must beat

`ffsimulator` (R, ffverse) already does a version of L2: bootstrap resampling of historical weekly
finishes, optimal lineups, connects to ESPN private leagues through `ffscrapr`. Run it as a
**baseline, not an implementation** — it resamples marginals and will not carry the correlation
structure that is the entire point. If your correlated simulator cannot beat it, you have learned
something real.

## Sequencing (revised)

| Priority | Layer | Why |
|---|---|---|
| 1 | Playoff-schedule roster construction | Largest edge, computable today, no simulator needed |
| 2 | Weekly lineup optimizer with ceiling preference | Happens 14 times, reuses existing code |
| 3 | Waiver availability under known priority | Reuses `availability.py`, different priors |
| 4 | Trade evaluator | **Gated** on measuring whether this league trades |

Trades are last because the league's trade activity is unknown. Do not guess: `espn-api` exposes
transaction history. Pull two or three prior seasons and count actual trades. A twenty-minute
spike that decides whether L4 is worth building.

## Gates

**Week 6, the load-bearing one:** does the correlated joint beat independent marginals on
held-out weekly *team-score* distributions? Score with log-loss and calibration, not correlation
recovery. If it does not, everything above L1 is unsupported and the honest move is to say so in
`docs/` and fall back to independent marginals.

**Pre-draft:** weeks 15-17 SoS table exists and is a column on the board.

**Week 10:** does delta-P(champ) exceed the projection error on the underlying players? If not,
the apparatus is decorating noise.

## What would kill this

Named in advance so they are findings rather than surprises.

- Per-offense correlation estimates are noisy enough that the copula loses to independent
  marginals. **First thing to test.**
- Opponent lineup prediction is a modeling problem that has not been scoped.
- delta-P(champ) between candidate actions is smaller than upstream projection error.
- 8-of-12 flattens P(champ) so much that most decisions barely move it, making the whole
  objective insensitive. If so, pivot to optimizing P(win the weeks 15-17 bracket | qualified),
  which is the same math with the formality removed.

## Inputs still needed

1. Weeks 15-17 NFL schedule mapped to your rostered players (`nflreadpy.load_schedules()`)
2. League transaction history count (does this league trade?)
3. Exact starter composition including whether K and DST slots exist (`espn-api`)
4. Payout structure (recorded for completeness; objective stays P(1st) regardless)
