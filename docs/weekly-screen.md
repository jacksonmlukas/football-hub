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

### The joint screen: which of these are separate signals?

Four features are a signal on their own. Re-screening each with the others added to its
controls — **each keeping its own week range**, controlled only for survivors that exist over
it — is what separates findings from shadows:

| feature | alone | controlled for the others | |
|---|---|---|---|
| snap-share trend | +0.070, 5/5 | **+0.074, t +5.05, 5/5** | survives everything |
| prior TD rate | −0.045, 5/5 | **−0.047, t −5.93, 5/5** | survives everything |
| implied team total | +0.044, 5/5 | +0.026, t +2.31, **3/5** | dies |
| own spread | +0.036, 5/5 | −0.002, t −0.19, **3/5** | dies |

**`implied_total = total_line/2 + own_spread/2`, and the two correlate at +0.83.** They are one
finding wearing two hats. Neither *residual* clears once the other is controlled for, which is
the honest statement: the market's game-level forecast predicts a player's week, and the split
between "how many points are in this game" and "who is favoured" is not separable at this n.

**Independent signals: the snap-share trend and the prior TD rate.** Both are computed entirely
from games already played.

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

**Confounded.** The **implied team total** is the *closing* line, which by construction
absorbs six days of news the control never saw. The screen as run cannot separate "adds beyond
consensus" from "is newer than consensus", and **there is no fresher historical consensus in
nflverse to control with**, so the confound is not removable with available data. It is also
the feature that did not survive the joint screen, so nothing rests on it either way.

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

## Against Usage, not points — the premise of the multiplier form

The plan applies these features as a multiplier on **Usage**, so they were screened again
against the counts themselves. Controls are that count's **season-to-date** mean, its **last
three weeks**, and consensus ECR.

The second control matters and is not decoration. A season-to-date mean *lags*: by week 12 it
is eleven games against which three weeks of new form barely register, so a feature that is
really "he has been busier lately" clears against it while adding nothing a person watching
could not see. Against the recent mean, `snap_trend` on carries falls from **+0.127 to +0.049**
— so most of that effect *was* form — and on targets from +0.124 to +0.087. Five of five
seasons either way.

Under the stronger control:

| | targets | receptions | carries | attempts | **touchdowns** |
|---|---|---|---|---|---|
| **snap-share trend** | +0.094 (8.5) | +0.077 (6.3) | +0.062 (4.6) | +0.032 (2.7) | **+0.018 (1.5) — killed** |
| **prior TD rate** | +0.008 (1.0) | −0.000 (0.0) | −0.005 (0.7) | +0.005 (1.8) | **−0.120 (−14.2)** |

**The two survivors are orthogonal, and each is exactly the thing it was supposed to be.** The
snap trend moves every volume count and does not move touchdowns. The prior TD rate moves
touchdowns, hard, and moves no volume count at all.

That is the structural claim `weekly-projection-plan.md` made before any of this was measured —
*week-level information moves Usage and barely touches efficiency* — confirmed by a screen that
could have refuted it. It also fixes the form of the model: a **Usage multiplier** carrying the
snap trend, and a **touchdown regression** carrying the prior TD rate, with nothing crossing
between them.

Note that the snap trend is roughly twice as strong against volume (+0.094 on targets) as
against points (+0.070). Volume is the persistent part and points add touchdown noise on top —
[component-projection.md](component-projection.md)'s year-over-year table (targets 0.805 against
points 0.775) at weekly grain.

## What this does and does not license

It licenses **Phase 2 with two features, in two separate places**: the **snap-share trend**
as a Usage multiplier (week ≥ 8), and the **prior TD rate** as a touchdown regression. The
Usage screen says they do not cross. Plus the injury designation, which enters unscreened by the plan's own
pre-registered rule, having been measured at player-week grain at +0.170 MAE and 3.8 se.

The implied team total is **not** carried. It fails the joint screen on the every-season half,
and even the part that survives is confounded with a six-day-stale control. Two independent
reasons to leave it out, and either alone would be enough.

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

```bash
uv run python -m hub.models.weekly_screen --run
```

`src/hub/models/weekly_screen.py`, committed 2026-08-27 because these numbers steer Phase 2 and
[ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md)'s trigger is
citation. The statistics, the cell structure, the pre-registered verdict — every branch,
including the losing ones — and the joint screen are unit-tested offline; only the nflverse
assembly needs the network.

Two defects the harness found that the scratchpad version had:

* **The player control was career-to-date, not season-to-date.** Routing the expanding
  aggregate through `experiment.expanding_weeks` widened the past to every earlier season,
  which is a *stronger* control than the one pre-registered and made weeks 1–3 eligible where
  they had not been. Narrowed back, and the scope choice is now explicit at the call site.
* **`partial_r` correlated rounding error.** A feature that is an exact linear function of a
  control residualises to ~1e-16 rather than to 0, and `rx.std() == 0` does not catch that.
  It now returns NaN below a relative tolerance, so the cell is dropped rather than reported.
