# THE PICK ranks on corrected ADP

**Status:** accepted 2026-08-24.

**Decision.** THE PICK ranks on ADP adjusted for the corrections this repo has measured and
the draft market has not priced — touchdown luck, durability, current injury designation. The
adjustment is converted from points to picks using the board's own ADP-to-projection curve and
clamped at **20% of the player's own ADP** (`config.DraftConfig.correction_clamp_frac`).

## Why

Until now every fitted correction modified `proj_blend` — and therefore VOR, the context
table, and the season simulator — while THE PICK ranked on raw ADP. Stated plainly that was an
inconsistency: *we measured that the market is wrong about this player, printed the correction
beside his name, and then ordered the board by the market anyway.*

Each input carries a fitted coefficient with an interval:

| correction | coefficient |
|---|---|
| touchdown luck | −0.540 QB, −0.286 WR |
| durability (games missed) | −0.457 QB, −0.151 WR |
| current designation | −1.631 OUT / DOUBTFUL / IR |

## Why the conversion is interpolated, not fitted

Points and picks are different units, and the exchange rate is steeply non-linear: near the
top of the board players are separated by small projection gaps and few picks, so a point of
projection is worth a handful of picks; by round twelve the same point spans dozens. A single
fitted slope would over-move the top of the board and barely move the bottom — the opposite of
where a correction can be trusted.

So `optimize.market_curve` reads the exchange rate off the board itself: smoothed with a
rolling median, forced monotone so a better projection never maps to a later pick. Both the
corrected and uncorrected projections are looked up on that curve, and the *difference* is the
shift — which means a player whose own ADP already differs from the curve keeps that
difference. It is a re-pricing of the correction, not of the player.

## What is not validated, and the clamp that bounds it

The **assembly** cannot be tested against outcomes. Scoring a ranking needs historical ADP,
which ESPN does not retain — the same wall that closed P6 ([ADR-0010](0010-edge-is-displayed-but-never-ranked-on.md))
and blocks P2. So the pieces are measured and the combination is bounded.

The clamp is proportional rather than absolute, fixed before anything was fitted. An absolute
clamp is least protective exactly where damage is worst: twelve picks at ADP 150 is noise,
twelve picks at ADP 3 is catastrophic. At 20% the bound is 0.6 picks at the top of round one
and about one round at ADP 60 — a correction may say *"a round cheaper than the market
thinks"*, never *"a different tier"*.

## The gate, fixed before the numbers

`backtest.correction_tripwire` fires only on impossibilities: a player carrying **no** fitted
correction moving at all, or any move **exceeding the clamp**. Deliberately *not* "did the
recommendation change" — it is supposed to change, and gating on that would reject the change
for working. That mistake was made with the seeding tripwire on the morning of the same day
and corrected on the record; this is the corrected version applied first time.

## Result on the live 2026 board

54 of 452 players move, all within the clamp, tripwire clear. **47 move later and 7 move
earlier** — players who ran cold on touchdowns are marked up, which is the sign symmetry
working and is worth stating because this repo has shipped a sign bug before.

THE PICK changes at **one of the first six picks** (pick 46, Davante Adams → Emeka Egbuka).
The top of round one is unchanged, which is what a proportional clamp is for.

## Reopening

If a source retaining historical ADP appears, this becomes measurable: add an arm to
`hub.draft.backtest` that ranks on corrected ADP and score it against consensus the way P0b
scored championship equity. Until then the honest description is *bounded application of
validated components*, not a validated rule.
