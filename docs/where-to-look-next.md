# Where to look next, and what twelve measurements say about it

**Written 2026-08-25**, after the per-player spread null. This is a research note, not a plan:
nothing here is committed to, and every proposal carries the gate it would have to clear.

## The record

Twelve things have now been measured properly in this repo. One worked.

| # | attempt | horizon | result |
|---|---|---|---|
| 1 | Expected-vs-actual points | next season | null — r = 0.21 self-persistence |
| 2 | Recency-weighted expected-vs-actual | next season | null, and strictly worse |
| 3 | Depth-chart climb | next season | null — partial r = +0.008 beyond ECR |
| 4 | Depth-chart climb | rest of season | null — within 1 se of zero |
| 5 | Age | next season | null |
| 6 | Championship equity as the objective | draft | **−19.66** pts/team-game vs consensus |
| 7 | VOR ordering | draft | **−5.06** pts/team-game vs the market |
| 8 | `edge` | draft | unvalidatable — needs historical ADP |
| 9 | Volume model beating the market's mean | draft | null — market mean reproduced instead |
| 10 | Lineup optimiser | weekly | **+0.00** — a structural zero |
| 11 | Per-player weekly spread | weekly | null — 0.085 MAE of headroom exists at all |
| 12 | **Weekly injury retention** | weekly | **+0.170 MAE at 3.8 se, wins 5/5 seasons** |

## What separates #12 from the other eleven

Not sophistication — the winner is a nine-cell lookup table of ratios. The difference is
**what information it uses**.

Attempts 1–11 all tried to beat a market or a well-fitted incumbent using information that
market already had. Age, depth charts, prior-season efficiency, ADP itself: all of it has been
sitting in front of every analyst who built the consensus, all summer.

Attempt 12 used **Wednesday's practice report to decide Sunday's lineup**. That information is
not in any preseason ranking because it did not exist when the ranking was written, and it is
mechanically linked to production: a player who did not practise plays less.

The thesis the record supports:

> **Edge came from timeliness and granularity, not from better processing of shared
> information.** Every attempt to out-think a market with the market's own inputs failed.

That is not a claim about what is true in general. It is what these twelve measurements say,
and it is a much better prior for the thirteenth than "try a fancier model".

## It also says something about the calendar

The draft is the hardest decision this repo makes, and it is the one that got the most work.
The opponents there are eleven managers anchored on a consensus that hundreds of analysts spent
a summer refining. After 2026-09-03 that decision is gone, and every remaining one has a
weaker opponent:

| decision | who you are beating | difficulty |
|---|---|---|
| the draft | a summer-refined public consensus | hardest — and now over |
| start/sit | one manager, once a week | moderate |
| waivers / FAAB | eleven managers deciding fast on Tuesday night, with no consensus to anchor on | **weakest opponent in the league** |
| trades | one manager, negotiating | weakest, but low volume |

The waiver wire has no ADP. There is no expert consensus published in time to matter. That is
the structural opposite of the draft, and it is where #12's kind of information lives.

## Proposals, ranked, each with the gate it must clear

### M1. Price the fantasy week off the betting market, not the fantasy consensus

**The idea.** `hub.fetch.odds` already pulls 272 events, and `MARGIN_SD = 12.741` is already
fitted against closing spreads. A game's implied **team total** is a real-money price on how
much offence there will be. Weekly fantasy rankings are not real-money prices.

**Why this is different from #1–#11.** It is not this repo out-thinking a market. It is using a
*sharper* market to correct a *softer* one. Betting lines are set by people with money at risk
and moved by sharp money; fantasy consensus is set by analysts with reputational stakes and a
publication deadline. Those are different populations with different incentives, and there is no
reason the second fully reflects the first by Sunday morning.

**Gate.** Both arms project weekly PPR points for the same players. Baseline: consensus weekly
rank alone. Candidate: consensus plus implied team total. Held-out MAE, walking forward by
week, paired, must clear 2 se and win most weeks. **Nothing is adopted on the season aggregate
alone** — the failure mode is a handful of blowout weeks carrying it.

**Cost — checked 2026-08-25, so this is a decision rather than an unknown.** The Odds API does
sell history, from 2020-06-06 onward, via `/v4/historical/sports/{sport}/odds`. Two things
follow:

* It is **paid-plan only**. `make slate` reports 489 credits remaining, consistent with the
  500/month free tier, so this needs an upgrade — not a code change.
* Historical odds cost **10 credits per region per market**. A team total needs both `spreads`
  and `totals`, so 20 credits a snapshot. One snapshot per game week over four seasons is
  roughly **1,400 credits** — trivial on a paid plan, impossible on the free one.

So M1 splits cleanly in two, and the second half is free:

1. **The backtest** needs a paid plan. That is a spending decision, and the number above is
   what it costs. Nobody should upgrade before deciding the question is worth it.
2. **Forward archiving is free and should start regardless.** `make slate` already pulls
   current odds every week and keeps only the snapshot. This is the ADP mistake in a second
   costume — the repo just spent forty lines fixing that one, and the same argument applies
   here with the same deadline logic: data not kept on the day is not recoverable later,
   except here it *is* recoverable, for money.

**This is the strongest untested idea in the repo**, and the only one whose blocker is a
purchase rather than a measurement.

### M2. Extend the one thing that worked, using the column it ignores — MEASURED 2026-08-25, not adopted

**The idea.** #12 prices a designation by `report_status` × `practice_status` and throws away
*what is actually wrong with the player*. `load_injuries` carries `report_primary_injury` and
`report_secondary_injury`, and nothing in this repo reads either.

A hamstring is not an ankle is not a concussion. Hamstrings re-injure; concussions are
protocol-governed and resolve on a schedule that has little to do with how the player felt on
Friday. The current table cannot express any of that.

**Availability, checked rather than assumed.** 2022–25, 23,564 injury rows:

    Knee 1,971   Ankle 1,604   Hamstring 1,293   Concussion 660
    Calf 459     Foot 454      Groin 450         Back 394   Hip 355

with 821 rows carrying a *secondary* injury as well — itself a plausible severity proxy, and
also unused.

**Why worth trying.** The base rate on extending a demonstrated winner beats that of a new
hypothesis, and the mechanism is the one already shown to work.

**Gate.** The same one #12 cleared, and note the incumbent has moved: the arm to beat is now
`retention`, not `out_zero`. Held-out MAE, every season, 2 se paired.

**Result: KEEP `retention`.** +0.0317 MAE at 3.1 se after fixing a feature-extraction bug
(the raw type field held 110 categories for 74 injuries, laterality splitting each one up to
three ways), but 2/3 seasons — it loses 2025, the most recent, both before and after the fix. Clears the significance half and fails the every-season half, the mirror image of
the spread null. Hamstring came out at 0.691, the largest multiplier and the one folk wisdom
predicts. Full write-up in [weekly-injury.md](weekly-injury.md), including why an effect that reverses
in the most recent season is the exact shape this repo already has a written rule about.

**Two things that will bite.** 53% of rows have a null primary injury, so any encoding needs an
explicit unknown category rather than a drop — dropping them would measure the cost of an
injury among players whose injury was reported, which is not the population being priced.
And the nine-cell table already has cells at n=63; crossing it with even eight injury types
empties most of them, so this wants pooling or an additive term, not a bigger lookup.

**Corrected 2026-08-25:** an earlier draft of this note proposed using Wednesday/Thursday/Friday
practice trajectories. nflverse does not carry them — there is one `practice_status` per row and
at most two rows per player-week. The proposal above is what the data actually supports.

### M3. Snap-share *trend* as a waiver signal — SCREENED 2026-08-25, POSITIVE

**The idea.** The trade press claims snap-share jumps precede production by one to two weeks.
That is a testable claim and this repo can test it in an afternoon.

**Why it is not just #3 again.** Depth-chart climb was null at both horizons, and this is close
enough that the prior should be low. But depth chart is a coarse ordinal that nflverse derives;
snap share is a continuous measure of what a coach actually did last Sunday. Being null on the
first is weak evidence about the second.

**Gate.** Does Δ snap share through week *w* predict points in *w+1…w+3*, **beyond** season-to-
date PPG? Partial correlation, walk-forward.

**Result: it survives.** Partial r **+0.236** at anchor week 10, controlling for season-to-date
PPG *and* rest-of-season ECR — and adding the ECR control changes it by less than 0.01, so
consensus prices none of it. Twelve of twelve season-anchor cells positive, placebo-clean,
stronger in the waiver-relevant tail (ECR outside the top 100, pooled +0.254). A >15pp snap
jump there is worth 1.2–2.5 points a game over the next three weeks. Null before week 8. Full
write-up in [snap-trend-signal.md](snap-trend-signal.md).

**This is the first screen in the repo to come back positive**, and it is the thesis above
paying out: snap counts are published Monday, and the consensus that gets rebuilt around them
is not doing so fast enough to price them.

**Read `docs/depth-chart-signal.md` first.** It documents two methodology errors made on exactly
this shape of question — a leakage bug, and treating eight weeks of one player as eight
observations. Both would recur here.

### M4. Not recommended: wiring opponent correlation

`docs/opponent-correlation.md` measured opposing quarterbacks at **+0.148 (4.5 se)** — larger
than the teammate QB-RB effect the simulator already models. Its stated consumer was the lineup
optimiser once per-player variance became real.

[player-spread.md](player-spread.md) has now measured that variance and it does not become real.
So the case for wiring correlation is weaker today than when it was measured, not stronger. It
stays a measurement.

## The codebase

### C1. Test the CLIs against reality, not against their own fixtures

Three bugs of one shape were found on 2026-08-25, all in code at or above 80% coverage:

* `hub.draft.board` raised `ConnectionError` and printed a traceback with no network, rather
  than serving the board sitting on disk.
* `hub.models.conformal --recalibrate` died on `BinderException: "margin_actual" not found`.
  The column has never existed. Every test handed the function its own frame.
* `hub.season.lineup` raised a bare `FileNotFoundError` for a roster file nothing writes.

**Coverage did not catch any of them, because coverage measures lines executed, not whether the
seam between a module and the real world exists.** All three lived at that seam.

**Done 2026-08-25.** `tests/contracts/test_cli_surface.py` runs every module CLI against a temp
data root and asserts a message rather than a traceback, and rescans `src/hub` so the list
cannot silently drift.

Writing it found a fourth bug of the same family. Threading `--store` through the store-backed
CLIs so they *could* be tested revealed that against a genuinely empty store both
`conformal --recalibrate` and `eval --compare` raised

    _duckdb.CatalogException: Table with name preds does not exist!

An empty store does not have an empty `preds` table; it has no `preds` view at all, because
`store.connect` builds one per directory that exists. **That is the state of a fresh clone** —
what the public gets on 2026-09-04. Fixed with `store.tables()`, discovered the same way
`connect` discovers views, with a test that the two agree: a caller checking the wrong thing
before querying is worse than not checking, because it looks safe.

### C2. `board.py` is still the hot spot

Improvement #7, deferred to the Dagster port, and that deferral still looks right. Noting only
that it grew again today: the last-good fallback and the ADP archive both landed in `main()`.

### C3. `hub.store` is bypassed by the board

Improvement #8. Unchanged, still low urgency, and worth doing *with* C2 rather than before it.

## What is blocked, and on what

| item | blocked on | unblocks |
|---|---|---|
| `fit_espn_weight` | historical ESPN ADP | **2027** — archiving started 2026-08-25 |
| P2 opponent model | same | 2027 |
| ADR-0010 `edge` validation | same | 2027 |
| Conformal's consumer | a scored prediction; the season has not started | after Week 1 |
| M1's historical arm | whether the odds vendor sells history | one API question |
| Real secret scan | `brew install gitleaks` | a command the user runs |

Three of those six were one missing input, and it is now being captured. That is the single
highest-leverage thing done today, and it took forty lines.
