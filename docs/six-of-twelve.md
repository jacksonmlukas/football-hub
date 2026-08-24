# What 6-of-12 with two byes rewards

**Measured 2026-08-23** with `hub.draft.leverage`. This replaces the draft-time argument in
[championship-leverage.md](championship-leverage.md), which was derived from a playoff
structure this league does not have.

## What was wrong

That doc reasons from *"12 teams, 8 make playoffs, 3 weeks (15-17), no byes"*. The live
league, from raw `mSettings.scheduleSettings` and recorded in
[decisions.md](decisions.md), is `playoffTeamCount: 6` with one-week rounds, so **seeds 1-2
get byes**. Three claims follow from the false premise and all three are false:

> "Regular season is nearly a formality" · "P(make playoffs) is comfortably north of 85%" ·
> "dP(champ)/d(regular-season win) is close to zero"

## What the structure actually pays

Twelve teams, one synthetic roster archetype replicated across all of them, 14-week
head-to-head, top 6, byes for 1-2, three single-elimination rounds. 20,000 seasons.

| roster (mu x) | E[wins] | P(playoff) | P(bye) | P(title) | marginal |
|---|---|---|---|---|---|
| 0.85 | 4.12 | 10.9% | 1.2% | 0.6% | — |
| 0.95 | 6.02 | 34.8% | 8.6% | 4.3% | +2.0 pp/win |
| 1.00 | 7.00 | 50.0% | 16.7% | 8.4% | +4.1 pp/win |
| 1.05 | 7.98 | 64.1% | 27.5% | 14.2% | +6.0 pp/win |
| 1.15 | 9.77 | 85.4% | 54.1% | 31.1% | +9.4 pp/win |

A league-average roster makes the playoffs **50%** of the time, not 85%. You need roughly
+15% of projected points — about ten wins — to reach the 85% the doc assumed as the
baseline case.

**A marginal regular-season win is worth 4 to 6 percentage points of championship equity**
around the median, rising to 9+ for a strong roster. Against a baseline title chance of
8.4%, one extra win is worth about half your title equity. It is not close to zero, and the
gradient *steepens* as you get better, so wins matter most exactly when the doc says they
stop mattering.

### Where that equity lives: the bye

P(title | seed), twelve identical rosters, so none of this is the seeded team being better:

| seed | P(title) |
|---|---|
| 1 (bye) | 37.9% |
| 2 (bye) | 26.9% |
| 3 | 12.5% |
| 4 | 9.4% |
| 5 | 7.3% |
| 6 | 6.0% |

Seed 2 is worth **2.2x** seed 3 across a one-place gap, because that gap is the bye. The bye
seeds average 32.4% against 8.8% for seeds 3-6 — **3.7x**. "Seeding buys marginally easier
matchups and nothing else" is the single most wrong sentence in the old doc.

## The ceiling argument, repaired

The doc's advice is *"never sacrifice ceiling for a marginal regular-season win. Start the
boom-bust player. Take the upside stash over the safe veteran."* That treats two different
quantities as one. They point in opposite directions.

Both sweeps below hold the team's **mean weekly score fixed**, so each moves variance
alone. See the trap section — this control is the whole measurement.

| roster | variance kind | multiplier | P(playoff) | P(bye) | P(title) |
|---|---|---|---|---|---|
| weak | weekly | 0.7 | 20.3% | 4.0% | 1.9% |
| weak | weekly | 1.8 | 24.3% | 3.5% | **1.9%** |
| weak | season-long | 0.5 | 15.2% | 1.1% | 0.5% |
| weak | season-long | 2.0 | 28.2% | 10.4% | **5.1%** |
| strong | weekly | 0.7 | 75.7% | 42.4% | **23.1%** |
| strong | weekly | 1.8 | 76.4% | 35.4% | 18.8% |
| strong | season-long | 0.5 | 83.5% | 37.9% | 19.1% |
| strong | season-long | 2.0 | 68.1% | 42.9% | **26.6%** |

**Season-long outcome spread — how uncertain it is what a player *becomes* — is worth
paying for, at every roster strength.** It is worth 10x to a weak roster (0.5% → 5.1%) and
still worth +39% to a strong one (19.1% → 26.6%), *even though it costs the strong roster
15 points of playoff probability* (83.5% → 68.1%). Trading berths for byes is a good trade
because the seeding payoff is convex: 37.9% at the top against 6.0% at the bottom.

**Weekly boom-bust is not worth paying for.** At a fixed mean it is flat for a weak roster
(1.9% → 1.9%) and clearly negative for a strong one (23.1% → 18.8%). Head-to-head wastes
surplus: a spikier weekly distribution with the same mean has a lower median, so it loses
more weeks than it wins. It does buy the weak roster a few more berths (20.3% → 24.3%) —
but berths without byes are worth 6-12% each, so they do not convert into titles.

So the corrected rule is:

> **Buy season-long upside. Do not buy weekly volatility.** A player who might be a WR1 or
> might be nothing is worth a premium. A known-quantity player who happens to score in
> lumps is not — and for a strong roster he is a small negative.

Note this is the *draft-time* rule and it is unconditional, unlike the weekly lineup rule.
Once the season starts and you know who you are playing, [lineup.py](../src/hub/season/lineup.py)
optimises P(win this matchup) and the ceiling-or-floor choice becomes state-dependent —
a favourite should decline volatility, an underdog should take it. Those two answers are
consistent: at draft time you do not know your matchups, and the convexity of *seeding*
dominates; in a given week you do know, and the convexity of *that game* dominates.

## The trap this measurement nearly fell into

The first version of this sweep found that variance helped enormously at every roster
strength — weak rosters going from 0.6% to 15.2%, strong from 15.8% to 44.9%. That result
was an artifact and it is worth recording because it is not obvious.

**A starting lineup is the best legal subset of a roster, so it is a max.** Raising player
spread raises the expected maximum. Scaling every player's weekly spread by 1.8x raises the
team's mean weekly score from 116.3 to 133.9 — **+15%** — before any variance effect at all.
The sweep was quietly handing the test roster more points and then crediting variance.

`hub.draft.leverage.calibrate` rescales projections to restore the mean, and
`test_raising_spread_alone_also_raises_the_mean` keeps the confound pinned so the
uncalibrated version cannot come back.

The correction reversed the sign of the answer. There is also a real effect hiding inside
the artifact — start/sit optionality is genuinely worth something, since you can bench a
bust week — but it is a *mean* effect and belongs in the projection, not in an argument
about variance.

A second confound, caught the same way: holding the mean fixed requires scaling projections
down, which also shrinks the *absolute* season-long talent spread, because that spread is
modelled as a fraction of projected points. Sweeping weekly spread without correcting for
it silently sweeps both. The weekly rows above set `cv_mult = k0/k` to hold the season-long
spread constant.

## What this does not establish

Inside the model only. Opponents are twelve copies of one archetype rather than real
rosters; players are independent, so the correlation layer that
[championship-leverage.md](championship-leverage.md) calls L1 is absent and stacks are
mispriced; and `TALENT_CV = 0.35` was never fitted — `hub/draft/season.py` says so in the
constant's own comment. The directions are the finding. The decimals are not.

The one number that would change the conclusions if it were wrong is `TALENT_CV`, since the
season-long sweep is a sweep in exactly that quantity. It is the obvious thing to fit.

## Reproduce

```bash
uv run python -m hub.draft.leverage --sims 20000
```
