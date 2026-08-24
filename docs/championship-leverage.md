# Championship leverage

Optimizing for P(finish 1st) rather than expected points, and the one object that also
produces player props and game probabilities.

> **Corrected 2026-08-23.** This doc was written against "8 make playoffs, no byes". The
> live league is **6 of 12 with byes for seeds 1-2**, and the sections below marked
> *(corrected)* have been rewritten against measurements in
> [six-of-twelve.md](six-of-twelve.md). The original draft-time argument concluded the
> regular season was nearly a formality. It is not: a marginal win is worth 4-6 points of
> championship equity. The ceiling advice survives in half — buy season-long upside, do not
> buy weekly volatility — and the two were previously treated as one thing.

## Declared objective

**Maximize P(finish 1st).** Explicitly do not hedge toward 2nd or 3rd, even if they pay.
This is Jackson's stated preference, not a modeling simplification. A future session must not
quietly substitute E[payout].

## League facts that drive everything

| Fact | Consequence |
|---|---|
| 12 teams, **6 make playoffs**, 3 one-week rounds (15-17), **byes for seeds 1-2** | Regular season decides everything; half the league misses |
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

## What 6-of-12 with two byes does to the objective *(corrected)*

Half the league misses, and the two teams that finish top of the pile skip a
single-elimination round. Measured over 20,000 simulated seasons in
[six-of-twelve.md](six-of-twelve.md):

- A league-average roster makes the playoffs **50%** of the time, not "comfortably north of
  85%". Reaching 85% takes roughly +15% of projected points.
- **A marginal regular-season win is worth 4-6 percentage points of championship equity**
  around the median, rising past 9 for a strong roster. Against a baseline title chance of
  8.4%, one win is worth about half your equity.
- The gradient *steepens* with roster strength, so wins matter most precisely where the
  original argument said they stopped mattering.
- P(title | seed) runs 37.9% / 26.9% for the bye seeds against 12.5% / 9.4% / 7.3% / 6.0%
  for seeds 3-6. Seed 2 is worth **2.2x** seed 3 across a single place.

Therefore: **seeding is the prize, and the whole regular season buys it.** Weeks 15-17 still
decide the title, but you reach them from a position the previous fourteen weeks set.

### The corollary is dead

The old version argued that reverse-standings waivers make the marginal value of a win
"plausibly negative". A win is worth 4-6 points of title equity; waiver priority in a league
that runs about 23 acquisitions per team per season
([trade-spike.md](trade-spike.md)) cannot be worth a fraction of that. Wins are the scarce
good. There was never a case for tanking and there is now not even an argument to refuse.

## Ceiling, split into the two things it was conflating *(corrected)*

"Never sacrifice ceiling" ran together two different quantities. Held at a fixed mean, they
point opposite ways:

- **Season-long outcome spread** — how uncertain it is what a player *becomes* — is worth
  paying for at every roster strength. Worth 10x to a weak roster and +39% to a strong one,
  *even though it costs the strong roster 15 points of playoff probability*. Trading berths
  for byes is a good trade because the seeding payoff is convex.
- **Weekly boom-bust** — spread in what he does on a Sunday, given his talent — is not. Flat
  for a weak roster, clearly negative for a strong one. Head-to-head wastes surplus, so a
  spikier distribution with the same mean has a lower median and loses more weeks.

> **Buy season-long upside. Do not buy weekly volatility.** The upside stash is right; the
> boom-bust starter is not.

"Start the boom-bust player" was also the wrong *kind* of rule: it is a lineup decision, and
once you know your opponent it is state-dependent. `hub/season/lineup.py` optimises P(win
this matchup), so a favourite declines volatility and an underdog takes it. That is
consistent with the unconditional draft-time rule above — at draft time you do not know your
matchups and the convexity of seeding dominates.

## The largest underexploited edge

**Playoff-schedule-aware roster construction.** Weeks 15-17 NFL matchups are known today
(`nflreadpy.load_schedules()`). A player facing three soft defenses in the fantasy playoffs is
worth materially more than his season-long projection implies.

**Downgraded from "close to primary" (corrected).** That ranking rested on seeding being
worthless, which it is not: this league *does* have byes, and they are worth 2.2x a seed.
Weeks 15-17 SoS still decides the title once you are there, but it now competes with
full-season strength rather than dominating it. It stays a tiebreaker column on the board,
not a primary sort. No casual ESPN drafter prices it either way. Build the weeks 15-17 strength-of-schedule table before Sep 3
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

**The IR slot is worth less than this doc originally claimed *(corrected)*.** The argument was
that "early losses are nearly free under 8-of-12". They are not free at 6-of-12: each win is
worth 4-6 points of title equity, so eight weeks of a hole is expensive. IR still does not
consume bench space, which is real, and a stash that *returns* in time to lift your weeks
15-17 roster can pay — but it is a priced option now, not a free one, and probably not worth
a deliberate early pick. (Inferred from the measured win gradient, not directly simulated.)

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
seeding, **6-team bracket with byes for seeds 1-2**. Output: P(champ) for a roster
configuration. Implemented in `hub/draft/season.py`; `hub/draft/leverage.py` is the harness
that measures the structure itself.

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
- ~~8-of-12 flattens P(champ) so much that most decisions barely move it.~~ **Tested and
  dead.** At 6-of-12 with byes, P(champ) is highly sensitive: 0.6% to 31.1% across a +/-15%
  band of roster strength. Insensitivity is not the risk; the risk is the opposite, that the
  apparatus reads more into small roster differences than the projections support.

## Inputs still needed

1. Weeks 15-17 NFL schedule mapped to your rostered players (`nflreadpy.load_schedules()`)
2. ~~League transaction history count~~ — answered: no. 3.75 trades/season, see
   [trade-spike.md](trade-spike.md).
3. Exact starter composition including whether K and DST slots exist (`espn-api`)
4. Payout structure (recorded for completeness; objective stays P(1st) regardless)
