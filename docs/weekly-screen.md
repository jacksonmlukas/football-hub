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

**Corrected 2026-08-28.** The first run of this screen was wrong: the as-of join attached each
consensus scrape to the week *after* the one it ranked. See *The off-by-one*, below. Every
number here is post-fix.

| feature | pre-stated | partial r | t | seasons with the stated sign | verdict |
|---|---|---|---|---|---|
| implied team total | + | +0.048 | 5.7 | 5/5 | clears alone |
| **snap-share trend** | + | **+0.038** | 2.8 | **5/5** | **clears** |
| own spread | ? | +0.037 | 4.4 | 5/5 | clears alone |
| **defence vs position** | + | **+0.033** | 4.4 | **5/5** | **clears** |
| **injury severity** | − | **−0.025** | −3.0 | **5/5** | **clears** |
| **prior TD rate per yard** | *null* | **−0.038** | −5.4 | **5/5** | **pre-stated null broken** |
| wind | − | −0.024 | −2.2 | 4/5 | killed |
| rest days | ? | −0.014 | −1.7 | 3/5 | killed |
| target-share trend | + | +0.006 | 0.6 | 1/5 | killed |

Sample: **14,370 player-weeks, 847 players, 55 cells.**

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

Six features are a signal on their own. Re-screening each with the others added to its
controls — **each keeping its own week range**, controlled only for survivors that exist over
it — separates findings from shadows:

| feature | alone | controlled for the others | |
|---|---|---|---|
| prior TD rate | −0.038, 5/5 | **−0.040, t −5.47, 5/5** | survives |
| snap-share trend | +0.038, 5/5 | **+0.043, t +3.04, 5/5** | survives |
| defence vs position | +0.033, 5/5 | **+0.028, t +3.74, 5/5** | survives |
| injury severity | −0.025, 5/5 | **−0.023, t −2.78, 5/5** | survives |
| implied team total | +0.048, 5/5 | +0.028, t +2.77, **4/5** | dies |
| own spread | +0.037, 5/5 | −0.006, t −0.55, **4/5** | dies |

**`implied_total = total_line/2 + own_spread/2`, and the two correlate at +0.83.** They are one
finding wearing two hats, and neither *residual* clears once the other is controlled for. The
market's game-level forecast predicts a player's week; the split between "how many points are
in this game" and "who is favoured" is not separable at this n.

**Four independent signals: the prior TD rate, the snap-share trend, defence vs position, and
the injury designation.**

## The staleness question, and why it is smaller than first reported

The consensus control is FantasyPros' `weekly-op` page, and the first version of this document
said it was scraped **six days before kickoff** — making it a stale control that any
Tuesday-to-Sunday news would beat for that reason alone.

**That was the off-by-one talking.** Measured against the week it actually ranks, the median
lead is **3 days**: a Friday scrape for that Sunday's games. It is a *fresh* control, not a
stale one, and the confound is correspondingly smaller.

What survives of the concern: a Friday scrape is after its own week's **Thursday** game, so for
a Thursday-night player the ranking is not strictly pre-kickoff. That hands the *incumbent* one
game of hindsight per team-week, which biases against the arm being tested — conservative
rather than dangerous, and it is stated in `assign_weeks` where the rule lives.

The feature the staleness story was about — the implied team total, which is a closing line —
**fails the joint screen anyway**, so nothing carried forward rests on it either way.

## The off-by-one

An NFL week runs Thursday to Monday, and the scrapes land mid-week. `assign_weeks` mapped each
scrape to the week whose **first** kickoff came next. So 2024-10-04, a Friday *inside* week 5
(Oct 3–7), ranking week 5's Sunday games, was attached to **week 6**.

The tell came from the gate, not the screen: Saquon Barkley, CeeDee Lamb and Patrick Mahomes
were each missing from exactly one week, and it was the week **after** their team's bye — a
page that correctly omits a bye-week player, attached to the following week.

It moved real numbers in both directions:

| | before | after |
|---|---|---|
| snap-share trend, joint | +0.074 | **+0.043** |
| defence vs position | 4/5, killed | **5/5, clears** |
| injury severity | 4/5, killed | **5/5, clears** |
| consensus lead time | "6 days" | **3 days** |
| Gate B join failures | 6.2% (VOID) | **0.0%** |

The rule now joins on the week's **last** kickoff — the first week whose games are not all
played — and there are five tests on it.

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
| **snap-share trend** | +0.088 (8.1) | +0.072 (6.1) | +0.055 (4.1) | +0.030 (2.6) | **+0.018 (1.5) — killed** |
| **prior TD rate** | +0.005 (0.8) | −0.004 (0.6) | −0.003 (0.5) | +0.005 (1.9) | **−0.118 (−14.4)** |
| defence vs position | +0.027 killed | +0.029 killed | +0.020 clears | +0.019 killed | +0.020 killed |
| injury severity | −0.004 killed | −0.011 killed | −0.025 clears | +0.004 killed | −0.024 clears |

**The two that carry into the model are orthogonal, and each is exactly the thing it was
supposed to be.** The snap trend moves every volume count and does not move touchdowns. The
prior TD rate moves touchdowns, hard, and moves no volume count at all.

That is the structural claim `weekly-projection-plan.md` made before any of this was measured —
*week-level information moves Usage and barely touches efficiency* — confirmed by a screen that
could have refuted it. It fixes the form of the model: a **Usage multiplier** carrying the snap
trend, and a **touchdown regression** carrying the prior TD rate, with nothing crossing between
them.

**Defence-vs-position and injury severity clear against points but not against Usage.** Only
one count each survives — carries for both — so neither has a clean home in the multiplier
form, and neither is carried into `hub.models.weekly`. They are real and unused, which is worth
saying plainly rather than quietly widening the model to accommodate them.

Note that the snap trend is more than twice as strong against volume (+0.088 on targets) as
against points (+0.038). Volume is the persistent part and points add touchdown noise on top —
[component-projection.md](component-projection.md)'s year-over-year table (targets 0.805 against
points 0.775) at weekly grain.

## What this does and does not license

It licenses **Phase 2 with two features, in two separate places**: the **snap-share trend**
as a Usage multiplier (week ≥ 8), and the **prior TD rate** as a touchdown regression. The
Usage screen says they do not cross, and it says defence-vs-position and the injury designation
do not belong in either. Plus the injury designation, which enters unscreened by the plan's own
pre-registered rule, having been measured at player-week grain at +0.170 MAE and 3.8 se.

The implied team total is **not** carried: it fails the joint screen on the every-season half,
being the own-spread finding in another hat.

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

**Split 2026-08-30.** The screen is now the statistic and the verdict; the **Panel** it measures
on — one row per player-week, every feature measured before its outcome — is
`src/hub/models/panel.py`, because the Weekly projection and the weekly gate read the same frame
and were reaching into the screen with function-local imports to get it. The CLI above is
unchanged, and the numbers on this page re-run identically.

Two defects the harness found that the scratchpad version had:

* **The player control was career-to-date, not season-to-date.** Routing the expanding
  aggregate through `experiment.expanding_weeks` widened the past to every earlier season,
  which is a *stronger* control than the one pre-registered and made weeks 1–3 eligible where
  they had not been. Narrowed back, and the scope choice is now explicit at the call site.
* **`partial_r` correlated rounding error.** A feature that is an exact linear function of a
  control residualises to ~1e-16 rather than to 0, and `rx.std() == 0` does not catch that.
  It now returns NaN below a relative tolerance, so the cell is dropped rather than reported.
