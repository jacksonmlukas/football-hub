# Screening week-level features

**Run 2026-08-27**, ahead of the schedule [weekly-projection-plan.md](weekly-projection-plan.md)
set, at the user's instruction. Phase 1 only: a screen, not a gate. Nothing is adopted here and
no `src/` was written, per the screening protocol.

## Design, as pre-registered

Partial correlation of each week-level feature against **player-week PPR points**, controlling
for **season-to-date PPG** (strictly before week *w*) and **that week's consensus ECR**.

One correlation per **(season, week) cell**, so no player appears twice inside a correlation —
protocol item 3, which turned noise into an apparent 4-sigma result once already. Cells are
aggregated by season and the sign has to hold in **every** season.

Sample: 2021–25, weeks 1–14, QB/RB/WR/TE, ≥3 games played before the outcome week, and present
in that week's `weekly-op` consensus. **12,852 player-weeks, 844 players, 54 cells.** Snap and
target trends are restricted to week ≥ 8, where [snap-trend-signal.md](snap-trend-signal.md)
establishes the trend exists at all.

Every feature is measured strictly before its outcome week. Pre-kickoff facts published *for*
week *w* — the line, the injury report, the opponent — count as week-*w* information; anything
derived from play uses weeks < *w* only.

## Result

| feature | pre-stated | partial r | t | seasons with the stated sign | verdict |
|---|---|---|---|---|---|
| **snap-share trend** | + | **+0.070** | 4.8 | **5/5** | **clears** |
| **prior TD rate per yard** | *null* | **−0.045** | −6.0 | **5/5** | **clears, sign as the component work predicts** |
| implied team total | + | +0.044 | 4.6 | 5/5 | **confounded — see below** |
| own spread | ? | +0.036 | 3.9 | 5/5 | **killed**: it is the implied total in disguise |
| defence vs position | + | +0.038 | 4.8 | 4/5 (2024 −0.004) | killed on the every-season half |
| target-share trend | + | +0.023 | 2.0 | 4/5 | killed on the every-season half |
| injury severity | − | −0.030 | −3.2 | 4/5 (2025 +0.001) | killed on the every-season half |
| wind | − | −0.024 | −2.1 | 2/5, sign flips | killed — protocol item 4 |
| rest days | ? | −0.003 | −0.3 | 2/5 | null |

### The controls are real

Within-cell raw correlation with same-week points: **ECR −0.603**, **season-to-date PPG
+0.611**. ECR spans 1–479, median 151. These are not weak controls being beaten by a strong
feature; they are the two best things available and they are doing their job.

### The placebos are clean

Each surviving feature permuted within its own cell:

    implied_total   +0.0043   t +0.45   3/5 seasons
    snap_trend      -0.0022   t -0.18   2/5
    td_rate_prior   -0.0127   t -1.43   1/5

### Own spread is not a separate signal

`implied_total = total_line/2 + own_spread/2`, and the two correlate at **+0.83**. Controlling
own spread for implied total leaves **−0.001, t −0.13, 2/5 seasons** — it vanishes. The
surviving piece is the *offensive environment*, not who is favoured.

### The three survivors are independent of each other

    implied_total | + snap_trend     +0.043   t +2.84   5/5
    snap_trend    | + implied_total  +0.071   t +4.94   5/5
    td_rate_prior | + implied_total  -0.046   t -5.99   5/5

## The confound that changes the reading

**The consensus control is scraped a median of six days before kickoff.**

| lead time, scrape → first kickoff | scrapes |
|---|---|
| 6 days | 72 |
| 8 days | 4 |
| 1, 2, 5 days | 5 |

`weekly-op` is FantasyPros' **Monday/Tuesday** ranking. So the control is a *stale* one, and any
feature carrying information published between Tuesday and Sunday will beat it for that reason
alone. That splits the survivors in two:

**Not explained by staleness.** The **snap-share trend** and **prior TD rate** are both computed
entirely from games already played when the ranking was scraped. Consensus had every input on
Monday and did not price them. These are the honest positives.

**Confounded.** The **implied team total** is the *closing* line, which by construction absorbs
six days of news the control never saw. The screen as run cannot separate "adds beyond
consensus" from "is newer than consensus", and **there is no fresher historical consensus in
nflverse to control with**, so the confound is not removable with available data.

Weak evidence against staleness being the whole story: injury severity is also post-scrape
information and it did *not* clear (4/5). But that encoding is crude — a max-severity ordinal,
not the fitted retention table — so it is a hint, not a control.

## Against the pre-registration

Recorded because the predictions were written down before the run and two of them are wrong.

1. *"The model beats the flat incumbent by a wide margin."* Not tested here — this screen
   controls for consensus, not for the flat projection.
2. *"It does not beat weekly consensus on market-derived features alone."* **Wrong as stated.**
   The implied total clears at 5/5 beyond consensus ECR. **Right in substance**: it beats a
   six-day-stale consensus, which is the timeliness thesis restated, not a claim to process
   shared information better.
3. *"If anything survives it is the snap-share trend and the injury designation."* **Half
   right.** The snap trend survives — and this doubles as the one-week re-screen the plan
   required before it could enter Phase 2, since its established result was a three-week
   horizon. The injury designation does not, on this encoding.
4. *"Week-over-week TD rate comes back null."* **Wrong, and informatively.** It is a strong
   **negative** in every season: given the same season-to-date scoring and the same consensus
   rank, a player whose points came from touchdowns rather than yards scores *less* next week.
   That is [component-projection.md](component-projection.md)'s "regress touchdowns, leave
   volume alone" appearing at weekly grain by an independent route — the season-level version
   is r = −0.004 for receiving and −0.030 for rushing TD rate.

## What this does and does not license

It licenses **Phase 2 with three features**: snap-share trend (week ≥ 8), prior TD rate, and
the implied team total carried with its confound written down. Plus the injury designation,
which enters unscreened by the plan's own pre-registered rule, having been measured at
player-week grain at +0.170 MAE and 3.8 se.

It licenses **nothing about lineups**. A partial correlation says a quantity adds to the board
and never that it should be the board
([ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md)). Gate B decides that, and
[ADR-0015](adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md) records why it is
the gate.

**And it sharpens a problem with Gate B's incumbent.** If `weekly-op` is a Monday ranking, then
a lineup set off it is a *Monday* lineup, and beating it with Sunday information is a lower bar
than the plan assumed. The gate should say so, and should report the same split: how much of
any win comes from features consensus could have priced, and how much from being newer than the
snapshot we are measuring against.

## Reproduce

Ad hoc, in the scratchpad, per the protocol — exploration may be. **Not yet citable**:
[ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) requires a
committed harness the moment a number becomes a reason, and these numbers are about to steer
Phase 2.
