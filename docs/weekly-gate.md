# Gate B: VOID on the first run

**Run 2026-08-28.** `hub.season.weekly_gate` asks the question that decides whether the Weekly
projection ships: does a lineup set off it beat a lineup set off weekly consensus rank?

**It has not been answered.** The first run returned **+11.14 points per team-week, 95% CI
[+10.08, +12.25], P(weekly better) 100.0%**, and was **VOID**.

## Why void

Pre-registered in [weekly-projection-plan.md](weekly-projection-plan.md) before any of this ran:

> Void the run above 2% of roster-weeks unmatched … Tight because the error is *directional*:
> a player missing from consensus is ranked last, so a join failure does not add noise, it
> forces a bench on the incumbent's arm.

Measured: **6.2%**, against a 2% floor.

The guard is worth the words it took. +11 points a week is a tenth of a lineup's score, from
*selection alone*, against a strong public incumbent — which is the repo's own rule that a
result too large to believe is a bug, and it was.

## Two separate defects, and only one of them is a defect

**Whole weeks have no incumbent.** Historical `weekly-op` coverage is not complete:

| season | weeks 1–14 present |
|---|---|
| 2021 | 13 (no week 1) |
| 2022 | 12 (no weeks 1–2) |
| 2023 | 12 (no weeks 1–2) |
| **2024** | **10 (nothing before week 5)** |
| 2025 | 12 (no weeks 1–2) |

On a week with no scrape, *every* rostered player is unranked and the consensus arm picks an
arbitrary lineup while the weekly arm has projections. That is not a hard comparison, it is no
comparison. `compare` now takes the set of covered weeks and the sample is restricted to them —
stated as a sample definition rather than applied quietly.

**And a name join that misses.** Restricted to covered weeks, 15.7% of roster-weeks are
unranked. Most of that is correct: FantasyPros drops players who are **out**, and the absence
*is* the incumbent saying "do not start him" — the pre-registered treatment. The 2024 week-5
unmatched list reads Christian McCaffrey, Cooper Kupp, Puka Nacua, Isiah Pacheco, Austin
Ekeler: injured, every one.

So the two are separated by what the player actually did:

| | share of roster-weeks | what it is |
|---|---|---|
| unranked | 15.7% | mostly players who were out, correctly benched |
| **unranked *and scored*** | **6.2%** | consensus would have ranked a player who played. **A defect.** |

Only the second biases the comparison, and it is the one `VOID_FLOOR` is measured against.

## What has to happen before this number means anything

1. **Fix the join.** 6.2% → under 2%. Board names come from FantasyPros' draft pages and weekly
   names from its weekly pages, so `player_key` should already agree; that it does not is the
   thing to find. Report the unmatched **by name**, per the plan, not as a count.
2. **Decide what weeks 1–2 are.** Four of five seasons have no week-1 consensus and 2024 has
   none before week 5. The pre-registered window is weeks 1–14; the honest options are to
   report the covered subset as the sample, or to find an incumbent that covers week 1.
3. **Only then read the result.**

## What is already known to be right

The machinery under it is tested: the lineup rule, the per-week re-selection that is the whole
subject, the separation of an absence from a defect, the cluster bootstrap over rosters rather
than rows, and all four verdict branches including the losing ones. 15 tests, offline.

The pre-registered rules held where it counted. The floor was written down hours before a
number existed, and it caught one that would otherwise have been reported as an enormous win.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run
```
