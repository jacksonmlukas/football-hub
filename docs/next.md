# What to do next, and why

**Written 2026-08-24, 10 days before the draft.** Ordered by what would change a decision,
not by what is interesting.

## The finding that reorders everything

**The P(win) recommendation reproduces VOR, and VOR loses to the market.**

Rank correlation between the championship-equity lift and the quantity it is scored on:

| pick | vs `vor_proj` | vs `proj_blend` | vs market (−ADP) |
|---|---|---|---|
| 3 | **0.927** | 0.879 | 0.515 |
| 22 | **0.964** | 0.964 | 0.818 |

(Picks beyond 3 reuse the same candidate set, so treat pick 3 as the independent observation.)

That matters because [market-value.md](market-value.md) measured VOR ordering **losing** to
simply drafting the market by 5.06 points per team game on realised outcomes, P(better) 0.0%.
If the lift is 0.93 correlated with VOR, the board's headline recommendation may be actively
worse than "take the best available by ADP".

This is not proven — correlation with VOR is not the same as sharing VOR's fate, and the lift
does carry roster-construction information VOR does not. But it is the most important open
question in the repo, and everything below is ordered around it.

---

## P0 — Settle whether the objective helps or hurts

**Blocking. Nothing else matters as much.**

Extend the realised-outcome backtest from *strategies* to the actual optimiser. At each of my
picks in 2022/2024/2025, compare what `win_probability` recommends against what the market
recommends, play the draft out, and score the final roster on realised points.

Three outcomes, each with a clear action:

- **P(win) beats the market** → keep it, and fix the circularity (P1).
- **P(win) ties the market** → keep it for the roster-construction information, drop the
  claim that it beats ADP, and lead with the market.
- **P(win) loses** → lead with the market plus the measured corrections, and demote
  championship equity to a tiebreaker.

Cost: expensive (nested simulation inside a backtest) but tractable with reduced sims. This
is the one thing worth being slow about.

## P1 — Break the circularity *(only if P0 says keep it)*

`optimize.py` scores the season on `proj_blend` and ranks candidates on `proj_blend`. The
size of that bias is now measured: 17.25 pp of apparent championship equity, which is larger
than any real effect in the repo.

Options, cheapest first: score seasons on a market-implied mu while ranking on `proj_blend`;
or bootstrap-perturb the scoring mu per simulation so no candidate is scored on exactly the
number it was ranked on. Then check whether the lift ordering survives.

## P2 — Calibrate the opponent model

`simulate_remaining_draft` models opponents as ADP plus assumed noise. Measured against three
real drafts, actual picks deviate from the market's implied order with a standard deviation of
**29–42 picks**, which is far more than the model assumes.

Caveat that makes this a project rather than a number: that spread was measured against
*projection* rank, not ADP, so part of it is projection≠ADP rather than room noise. Measuring
it properly needs historical ADP, which ESPN returns as a 169 sentinel for 69-78% of players
([decisions.md](decisions.md)). Worth doing because a wrong opponent model changes who is
available at your next pick, which is one of the few things the market cannot price for you.

## P3 — Draft-night rehearsal

The deadline is real and untested end to end under failure.

- Time every command against a 90-second clock, including the slowest path.
- Kill the network mid-build and confirm it serves last-good state rather than a traceback,
  which `CLAUDE.md` requires and the draft path has never been tested for.
- Feed deliberately bad input: a mistyped pick (now warns), a duplicate pick, an out-of-order
  sync, a pick of someone already gone.
- Write the runbook: exact commands, in order, with the fallback for each.

Cheap, and the only item here whose value is certain.

## P4 — Playoff schedule into the objective

`championship-leverage.md` calls weeks 15-17 strength of schedule "the largest underexploited
edge", and [six-of-twelve.md](six-of-twelve.md) showed seeding is worth 4.4x under 6-of-12.
The simulator models neither: `_round_robin` has no opponent quality and playoff weeks reuse
regular-season draws.

Today SoS is a tiebreaker column the drafter reads by eye. Putting it in the objective is what
would let it price a pick.

## P5 — Market prices for weekly lineups *(after the draft)*

The odds key now works, and the API carries the **full component set including volume** —
`player_receptions`, `player_rush_attempts`, and the yardage and touchdown markets. Volume is
the half that persists (targets 0.805, carries 0.791) and touchdowns are the noise, so a
market pricing receptions and carries is pricing exactly the part worth having.

`hub/season/lineup.py` currently takes mu and sd from ESPN's projection. Taking them from
money-backed prices is the strongest input upgrade available anywhere in the repo — and it is
for the seventeen weeks *after* the draft, so it does not compete with P0-P3.

Quota is the constraint: 64 credits a week for a full slate against 500 a month. Pulling only
your roster and your opponent's fits comfortably, and is all a lineup decision needs.

## P6 — Validate the `edge` column

`edge` (expert consensus rank against ADP) is the repo's original draft edge and has never
been tested against outcomes. The P0 harness tests it for free once it exists.

---

## Explicitly not doing

- **More projection modelling.** Four documented nulls and a standing decision
  ([decisions.md](decisions.md)): do not backtest for edges, backtest to audit.
- **The draft-room extension.** Researched, filed as a nice-to-have at Jackson's direction.
  More moving parts than anything in the repo, ten days out, with `--taken` already tested
  end to end against 192 real picks.
- **Season-long player props.** Settled: The Odds API does not carry them.
