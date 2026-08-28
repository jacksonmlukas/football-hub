# Gate B: the Weekly projection loses to consensus rank

**Run 2026-08-28.** `hub.season.weekly_gate` asks the question that decides whether the Weekly
projection ships — does a lineup set off it beat a lineup set off weekly consensus rank? — and
this is the answer.

    680 roster-weeks over 60 rosters, on the weeks consensus covers
    unranked 16.3%, of which a join failure 0.0% (floor 2%)

    season   gain
      2023  -0.709
      2024  -0.383
      2025  -0.961

    weekly - consensus = -0.684 points per team-week
    95% CI [-1.519, +0.159]   P(weekly better) 5.8%

**Verdict: SHOW, NEVER RANK ON.** Lost in every held-out season, and the interval contains
zero. Printed beside consensus, never sorted on — the pre-registered middle branch, and the
same disposition the snap trend got in
[ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md).

## The first run said +11.1 and was void

Before the join was fixed this returned **+11.14 points per team-week, CI [+10.08, +12.25],
P(weekly better) 100.0%** — a tenth of a lineup's score from selection alone, against a strong
public incumbent. It was reported **VOID**, not adopted, because 6.2% of roster-weeks were a
join failure against a floor of 2% pre-registered in
[weekly-projection-plan.md](weekly-projection-plan.md) hours earlier:

> Tight because the error is *directional*: a player missing from consensus is ranked last, so
> a join failure does not add noise, it forces a bench on the incumbent's arm.

That is exactly what it was doing, and the guard is the only reason the number was not believed.

**The cause was an off-by-one in the as-of join.** Consensus scrapes were being attached to the
week *after* the one they ranked, so a page that correctly omits a bye-week player was benching
him the following week instead. The tell: Saquon Barkley, CeeDee Lamb and Patrick Mahomes were
each missing from exactly one week, and each was the week after their team's bye. Joining on the
week's **last** kickoff rather than its first took join failures from **6.2% to 0.0%** — and
turned +11.14 into −0.68. See [weekly-screen.md](weekly-screen.md#the-off-by-one).

## What the sample is

**Weeks the incumbent actually covers.** Historical `weekly-op` scrapes miss whole weeks — 2024
has nothing before week 4 — and on a week with no scrape *every* rostered player is unranked, so
the consensus arm picks an arbitrary lineup. That is not a hard comparison, it is no comparison,
and `compare` takes the covered set explicitly rather than applying the restriction quietly.

**16.3% of roster-weeks remain unranked and that is correct.** FantasyPros drops players who are
out or on bye, and the absence *is* the incumbent saying "do not start him" — the pre-registered
treatment. What counts against the floor is only *unranked **and** scored*, which is now zero.

**Paired by roster-week, bootstrapped by roster.** A roster's weeks share its players, its bye
and its draft, so they are one observation with fourteen readings.

## What this means

The Weekly projection beats the flat projection on accuracy — **+0.074 MAE at 5.9 se, 4/4
seasons** ([weekly-projection.md](weekly-projection.md)) — and still loses the lineup decision
to a free public ranking. That is the screen/gate distinction at its sharpest, and it is the
fifteenth measurement in a repo where consensus has now won fourteen of them.

It is also exactly what was written down before the run:

> Gate B fails even where Gate A passes, because a lineup is a max over a roster and most
> projection error never reaches the decision. This is now the *primary* gate, so this
> prediction is the one that decides whether anything ships.

**One thing keeps it from being a clean null.** The interval touches zero at +0.159 and the loss
is −0.68, so this is "absence of evidence" — but the *direction* is consistent across all three
seasons, and the honest reading is that consensus is probably better, not that the two are
equivalent.

## The limitation that would change it

The rosters are **static all season**. A large part of what a weekly projection is worth is
deciding who to stream into the flex, and a frozen roster deletes that decision — which the plan
said before this ran, and which makes this a *conservative* design rather than a clean negative.
A version with waiver churn is the one experiment that could plausibly flip it.

## Reproduce

```bash
uv run python -m hub.season.weekly_gate --run
```
