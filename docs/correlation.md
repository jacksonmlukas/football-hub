# Teammate correlation: the L1 gate

**Measured 2026-08-24.** `championship-leverage.md` calls this L1 and gates everything above
it on one question:

> does the correlated joint beat independent marginals on held-out weekly *team-score*
> distributions? Score with log-loss and calibration, not correlation recovery.

**It passes, narrowly and specifically.** Only the quarterback's edges matter. Everything
else can stay independent, which makes L1 far smaller than the doc envisioned.

## What correlates

Within-game correlation of standardised weekly points between teammates, 2022-25:

| pair | n | r | doc's prior |
|---|---|---|---|
| QB–WR1 | 1,860 | **+0.323** | +0.3 to +0.5 ✓ |
| QB–WR2 | 1,693 | +0.293 | |
| QB–TE1 | 1,724 | +0.252 | "second strongest positive" |
| QB–RB1 | 1,834 | +0.060 | |
| WR1–WR2 | 1,727 | +0.025 | ~+0.16 at ceiling |
| RB1–WR1 | 1,846 | −0.024 | ~−0.07 ✓ |
| WR1–TE1 | 1,736 | +0.013 | |
| RB1–RB2 | 1,630 | +0.013 | |

The published priors hold where the doc gave them. Two corrections: the doc ranks QB–TE1 as
the second strongest positive and it is third, behind QB–WR2. And WR1–WR2 is **+0.025**, not
the +0.16 the doc cites — that figure is described there as holding "at ceiling outcomes",
so the two are not necessarily in conflict, but the average-case number is essentially zero
and that is what a lineup is priced on.

At position level, which is all a roster actually knows:

| pair | n | r |
|---|---|---|
| QB–WR | 6,352 | **+0.232** |
| QB–TE | 3,057 | **+0.225** |
| QB–RB | 4,077 | +0.054 |
| everything else | — | within ±0.03 of zero |

## The gate

Correlations fitted on 2022-24, evaluated on held-out 2025. An 80% interval should cover 80%
of outcomes; below that the model is overconfident.

| grouping | model | n | 80% coverage | mean log score |
|---|---|---|---|---|
| QB + his pass catchers | independent | 542 | **72.9%** | −4.122 |
| QB + his pass catchers | correlated | 542 | **80.4%** | −4.094 |
| QB + WR1 only | independent | 418 | 75.4% | −3.923 |
| QB + WR1 only | correlated | 418 | 81.3% | −3.910 |
| no QB (WR1/WR2/RB1/TE1) | independent | 541 | 79.3% | −4.001 |
| no QB (WR1/WR2/RB1/TE1) | correlated | 541 | 79.1% | −4.001 |

**Independence is materially overconfident for a stack** — a nominal 80% interval covers
72.9% — and correlation fixes the calibration almost exactly while improving the log score.
**Without a quarterback, independence is already right** and correlation changes nothing.

That is a much narrower result than "build a copula over six positions conditioned on game
total and spread". The honest version of L1 is three numbers.

## What it changes

`hub/season/lineup.py` prices a lineup's spread with `components.group_sd`, which adds the
covariance terms for teammates. Nothing else changes: players on different NFL teams
contribute no covariance, and a roster with no `nfl_team` column behaves exactly as before.

The consequence is the same state-dependence as everywhere else in this repo. Correlation is
volatility, so **stacking your quarterback with his own receiver helps when you are an
underdog and hurts when you are favoured**, and the optimizer now works that out per matchup
rather than taking a view on stacking in general.

## In the simulator too

`simulate_weeks` takes an `nfl_team` argument and correlates teammates by Cholesky on each
team's own small block. `champion_probability` passes it through and `hub.draft.optimize`
supplies it from the board's existing `team` column, so the draft optimizer now prices a
stacked roster as the more volatile thing it is.

That required one change to how a week is drawn. Weekly points were drawn from a shifted
gamma matched to (mean, spread, skew); a gamma cannot easily be correlated, so the draw is
now Cornish-Fisher on a Gaussian latent — the latent correlates trivially, and the quadratic
term supplies the skew with the variance it adds divided back out. Mean and spread are
unchanged by construction.

One approximation worth naming: correlating through a shared quarterback implies a small
positive correlation between two pass catchers on the same team, around +0.05 where the
measured figure is +0.014. The star topology the data actually shows — the quarterback
correlated with each catcher, catchers not with each other — is not exactly representable
this way, because catchers compete for the same targets and would need a slightly negative
conditional correlation. The overstatement is small and in the conservative direction for a
roster holding two catchers from one team.

## What this does not cover

**Opponent correlation is not modelled.** `championship-leverage.md` makes the point that
correlation matters twice: within your roster, and against your opponent's. Shared game
exposure shrinks *margin* variance, which helps whoever is favoured. Nobody prices it here
either.

> **Measured 2026-08-25** — [opponent-correlation.md](opponent-correlation.md). It is real and
> concentrated on the passing game: two opposing quarterbacks correlate at **+0.148** (4.5 se),
> which is larger than the QB-RB *teammate* edge this document fits. QB-TE +0.066 and QB-WR
> +0.055 both clear four standard errors, and RB-RB goes *negative* at −0.024, which is what
> game script implies. Still not priced, and deliberately: the two consumers of a correlation
> term — `lineup.optimize` and `simulate_weeks` — are both inert (ADR-0012, ADR-0009), so
> wiring it would add a parameter and its plumbing to buy nothing measurable.

**QB–opposing-DST (~−0.45 per the doc) is unmeasured**, since this league's DST handling was
never in scope for the component layer.

## A block that will not factor

> **Measured 2026-09-07**, issue #171. `correlated_normal` used to catch the Cholesky failure
> and `continue`, so a block that is not positive semi-definite was simulated **independently**
> — the exact model the three numbers above exist to replace — with no counter, no warning and
> nothing recorded. The draw it produces has the shape and dtype of a correlated one, so there
> was no evidence of it anywhere.
>
> It is counted now (`CorrelationReport`), and above a pre-registered share of blocks the draw
> refuses rather than reporting a correlated simulation it did not perform.
>
> **On the live 457-player board of 2026-09-07, four of thirty-three blocks will not factor**
> — LAR (−0.085 smallest eigenvalue), ATL (−0.067), SF (−0.045), WAS (−0.042) — which is
> 12.1%, against the 5% floor. So `win_probability` now refuses on that board. That is the
> correct reading and not a regression: those four teams' players have been drawn
> independently since correlation landed, and the only thing that changed is that it says so.
>
> **The cause is a second quarterback.** The block is a star while a team has one — PSD
> exactly while its pass catchers' squared correlations sum below one, which at +0.232 turns
> over around nineteen receivers and never happens. Two quarterbacks make it bipartite
> instead: each is correlated with every catcher and with the other at zero, and the bound
> halves. All four failing teams carry two or three quarterbacks and seven or more catchers;
> every single-quarterback team on the board factors comfortably. A board lists a depth chart,
> so this is a property of real boards rather than an edge case.
>
> Fixing it is a modelling question and not this one's: two quarterbacks on a team are not
> independent of each other, and `TEAMMATE_RHO` has no entry saying so.

## The block is repaired rather than dropped

> **Measured 2026-09-07**, issue #187, under [method.md rule 13](method.md). **Nothing above
> is edited.** The four blocks and their eigenvalues are re-measured below on the same live
> board and agree with the section above to the digit it quotes.

A block that will not factor is now repaired to its **nearest valid correlation matrix**
(Higham's alternating projections) and the repair is recorded per team, rather than the
team being dropped to an independent draw. `CorrelationReport.independent` goes to zero by
repair; `repaired` and `repairs` carry what it cost.

The fit is **not** constrained to stay PSD. Constraining it would mean constraining the
estimate to be representable, and it would make #171's counter permanently zero — a check
that cannot fire. The counter still can: a block that will not factor *after* repair (a NaN
out of a bad refit is the case that gets there) is still counted and still voids past the
floor. The floor is unchanged at 5%.

**On the live board of 2026-09-07** — the same board as the section above, 4 of 33 blocks,
12.5% of factorisations:

| team | block | smallest eigenvalue before | after | moved (Frobenius) | largest single pairing |
|---|---|---|---|---|---|
| LAR | 16x16 | −0.0845 | +0.0000 | 0.0915 | +0.0250 |
| ATL | 13x13 | −0.0665 | +0.0000 | 0.0708 | +0.0126 |
| SF | 17x17 | −0.0454 | +0.0000 | 0.0492 | +0.0134 |
| WAS | 17x17 | −0.0423 | +0.0000 | 0.0459 | +0.0125 |

No single pairing moves more than **+0.025**, against fitted edges of +0.232 and +0.225. The
repair is spread thinly across a whole block rather than concentrated on one edge, which is
what makes it a numerical repair and not a refit.

**The 2024 backtest board is worse and this is where the gate actually runs.** `--seasons 2024
--drafts 2` repairs **14 of 32** blocks (43.8% of factorisations), with blocks of 16x16 to
22x22 against the live board's 13x13 to 17x17, and moves up to 0.354 (NE, smallest eigenvalue
−0.335). The blocks are board-sized rather than roster-sized — issue #187's remaining
criterion, untouched here — so a backtest board carries every listed player on a team at once
and fails far more often than a live draft board does.

### Free agents are not a team

Found while measuring the above and fixed beside it. `"FA"` is a value in the team column,
not a team, and `correlated_normal` was correlating every free agent with every other as
though they shared a quarterback. On the 2024 board that is one 43-player block with a
smallest eigenvalue of **−0.914**, worse than the worst real team by an order of magnitude.

It mattered only once repair existed. Before it, that block failed to factor and fell back to
independence — the right answer for a free agent, reached by accident. Repairing it instead
would have turned an accidentally-correct independent draw into a confidently-wrong
correlated one, so the repair made this *worse* until the exclusion landed with it.
`playoff_sos._canon_team` has excluded `"FA"` since it was written.

### The pre-registered committee direction is falsified

#187 pre-registers that *two players in a committee produce a wider combined distribution
than two independent players with the same marginals*. **It cannot hold as written, and no
implementation can satisfy it.** Holding the marginals fixed, the combined variance is
`s1² + s2² + 2ρ·s1·s2`, so the sign of the change *is* the sign of ρ. A committee is
negatively correlated by the ticket's own definition — "one rises when the other falls" — and
a negative ρ makes the combined distribution strictly **narrower**. The same holds for the
best-lineup max, where `Var(max) = 1 − (1−ρ)/π` for a standard pair and also falls with ρ.

The two halves of the pre-registration point in opposite directions: it asks for a negative
correlation and for the effect of a positive one. Which half survives is a modelling question
and is left open here — see the note below on #213, where the mechanism now lives.

Measured rather than assumed: `test_a_committee_is_narrower_than_two_independent_backs_not_wider`
asserts both directions, so the fixture cannot pass by accident.

### Nothing negative is shipped

The machinery now carries a negative within-team pairing end to end — it is fitted freely,
survives the repair with its sign intact, and shows up in the draw. **No negative value is
added to `TEAMMATE_RHO`**, because there is no fitted one to add: the position-level table
above puts every non-quarterback pairing within ±0.03 of zero, and RB1–RB2 at **+0.013**.
Shipping a hand-set negative number would be inventing the precision this document exists to
avoid.

#187's amendment of 2026-09-07 puts the mechanism in **#213** — team-level attempt counts
with player shares as Dirichlet-multinomial, where negative teammate correlation falls out of
the shared simplex rather than a fitted term. That is where a real negative number should
come from, and it is also the reading under which "wider" could be true: two players on one
team share the team's volume, so a positive net covariance from the shared count can outweigh
the negative share split. That is a claim about #213's fitted output, not something this
change can settle.

## Reproduce

The measurement and gate run from `hub.fetch.nflverse` weekly player stats; see
`docs/component-projection.md` for the data layer.
