# Gate B: the Weekly projection loses, and loses harder with waivers

**Run 2026-08-28.** `hub.season.weekly_gate` decides whether the Weekly projection sets
lineups. It does not.

| variant | weekly − consensus | 95% CI | seasons won | verdict |
|---|---|---|---|---|
| **frozen rosters** (the pre-registered gate) | **−0.304** | [−1.043, +0.415] | 1/3 | **SHOW, NEVER RANK ON** |
| with waiver churn | **−3.790** | [−5.243, −2.382] | 0/3 | REMOVE *on that variant* |
| churn, unrestricted pool *(not the gate)* | −3.661 | [−5.128, −2.188] | 0/3 | — |

680–740 roster-weeks over 60 rosters, paired by roster-week, bootstrapped by roster, on the
weeks consensus covers. Join failures **0.0%** against a 2% floor.

## Churn was supposed to be the rescue and it is the opposite

[ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md) named waiver churn
as the one experiment that could flip the frozen result, on the reasoning that streaming is
where a week-specific number should beat a season-long rank. The pre-registration written
before the run said: *"churn helps the Weekly arm more than the incumbent … expected outcome:
still SHOW, with a smaller gap."*

**Wrong, and by a factor of twelve in the wrong direction.** The gap goes from −0.30 to −3.79.

The restriction I worried most about turned out not to matter: with the pool **unrestricted** —
the version I pre-registered as *not the gate* because it lets the Weekly arm add ~600 players
the incumbent cannot score — the answer is −3.661, statistically indistinguishable from the
masked −3.790. The masking decision was the right one and it was not load-bearing.

## Why: a winner's curse

The projection is **unbiased at every sample size**:

| games played before the week | n | mean projection | mean actual | bias |
|---|---|---|---|---|
| 1 | 2,591 | 6.25 | 6.61 | −0.36 |
| 2 | 2,382 | 6.77 | 6.94 | −0.18 |
| 3–4 | 4,260 | 7.31 | 7.67 | −0.36 |
| 5–8 | 6,692 | 8.28 | 8.40 | −0.12 |
| 9+ | 3,716 | 9.76 | 9.85 | −0.09 |

**And badly biased exactly where a waiver decision reads it.** Among players with ≤4 games, the
top 400 *by projection* project **23.00** and score **17.12** — **+5.88**.

A waiver pick is the **maximum** of a noisy estimator over a pool of hundreds, and the maximum
of a noisy estimator is biased upward even when its mean is perfect. A rostered player has
history, so the frozen gate never touched this. The waiver pool is precisely the thin-sample
population.

Consensus rank suffers the same effect far less: it is human-curated and coarse, so its top is
not the argmax of a per-player regression fitted on four games.

**This is not being fixed to rescue the number.** Shrinking each projection toward its
positional mean by sample size is the obvious remedy and it is a *model* change made after
seeing a gate fail — which is the tripwire lesson [ADR-0009](adr/0009-championship-equity-does-not-pick.md)
records: *do not touch it after you have seen what it caught.* It is written down as the next
pre-registered experiment, not applied tonight.

## Two defects found while running it

**The waiver rule added backup quarterbacks.** The first version added the highest-scoring free
agent and dropped the lowest-scoring bench player — on *absolute* weekly points, which are much
larger at quarterback. It picked up Russell Wilson at a projected 24.4 (he scored 5.1), Marcus
Mariota at 16.5 (−2.1), Jayden Daniels at 18.5 (2.7): a mean add of 19.6 projected against 13.7
realised. You start one quarterback. Consensus rank does not make that mistake because a
*ranking* already prices scarcity, so the naive rule handed the incumbent a free win owing
nothing to either arm's forecasting.

The swap is now scored by **what it does to the projected starting lineup**. Note that fixing
it made the Weekly arm *worse* (−2.41 → −3.79), which is coherent: the naive rule made mostly
inert moves — a backup QB never starts — while the corrected rule makes consequential ones, and
the projection is at its worst exactly where the consequences are.

**`board_as_of` is not reproducible.** Two calls in one process return the same 1,103 players in
a different row order, with `wk15_17_sos` differing below 1e-6. The draft indexes the board by
row, so the gate wobbled by ~0.04 points a team-week between identical runs — the previously
published **−0.684 was one draw from an unstable process**. The gate now sorts the board before
drafting and returns −0.304 on three consecutive runs. Sorted in the gate rather than in
`board.build`, which is draft-path code six days from a live draft. Recorded as
[improvements.md #18](improvements.md).

## What this leaves

The Weekly projection beats the flat projection on accuracy — **+0.074 MAE at 5.9 se, 4/4
seasons** — and loses the lineup decision on frozen rosters, and loses it much harder when
asked to evaluate unfamiliar players. The pre-registered verdict on the pre-registered gate
stands: **SHOW, NEVER RANK ON**.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run
uv run python -m hub.season.weekly_gate --run --churn
```
