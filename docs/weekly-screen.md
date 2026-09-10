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

> **Restated 2026-09-10 — that restriction is now a sweep, not a value.** Issue #178. Week 8
> was the earliest of the anchors 4, 6, 8, 10, 12 that held when they were tested *against the
> outcome*, so screening features on rows it selected screened them on a sample chosen by a
> number fitted to the outcome on the same data. The screen is now run at all five anchors and
> reports the sensitivity: see *The screen's minimum week*, below. **One status moves** — the
> snap-share trend clears at 4, 6, 8 and 10 and is killed at 12.

Every feature is measured strictly before its outcome week. Pre-kickoff facts published *for*
week *w* — the line, the injury report, the opponent — count as week-*w* information; anything
derived from play uses weeks < *w* only.

## Result

> **Every `t` on this page restated 2026-09-07 — the standard error was over the cells, and
> the verdict is over the seasons.** Issue #169, under [method.md rule 13](method.md). The
> tables below keep their original text; the `t` column in each is superseded by the re-run
> here, and the `r` column is not — see *The unit of the standard error*, below, for the
> corrected figures, the one status that changes, and the finding that the correction
> **narrowed** most of these intervals rather than widening them.

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

## The unit of the standard error

**Re-run 2026-09-07**, issue #169, under [method.md rule 13](method.md). The tables above keep
their original text; what follows supersedes their `t` columns and comes from a re-run rather
than an argument.

`summarise` took the standard error across the **season-week cells** — 55 of them — while
`verdict` requires the pre-stated sign to hold across the **per-season means**, and reads
nothing else. The precision came from dozens of cells and the decision from five seasons. A
`t` built on one unit printed beside a rule built on another is two claims about different
quantities on one line, and the reported interval was not the interval the rule was reading.

That is [method.md rule 3](method.md) — *repeated measures are not independent observations* —
broken by the screen that exists to enforce it. This module already refuses to pool
player-weeks, and says in its own docstring that pooling would inflate every `t` by about
√14. Taking the season-weeks as independent is the same error one level up, and the
superseded docstring knew it: it called the cells *"not fully independent either"* and kept
them anyway, on the grounds that they were less wrong than pooling. Less wrong is not the
standard.

The se is now over the seasons, and so is `r` — the mean *of the season means*, so that
numerator and denominator come from the same unit.

### What moved

Every figure below was reproduced under the superseded estimator on the same panel first, so
the movement is attributable to the unit and to nothing else. **The `r` column does not move**
except for wind, and the sample is unchanged at 14,370 player-weeks, 847 players, 55 cells.

| feature | r | published t | re-run t |
|---|---|---|---|
| implied team total | +0.048 | 5.7 | **11.4** |
| snap-share trend | +0.038 | 2.8 | **3.4** |
| own spread | +0.037 | 4.4 | **5.0** |
| defence vs position | +0.033 | 4.4 | **3.9** |
| injury severity | −0.025 | −3.0 | **−4.4** |
| prior TD rate per yard | −0.038 | −5.4 | **−7.6** |
| wind | −0.024 → **−0.020** | −2.2 | **−1.2** |
| rest days | −0.014 | −1.7 | **−1.6** |
| target-share trend | +0.006 | 0.6 | **0.5** |

And the joint screen:

| feature | r | published t | re-run t |
|---|---|---|---|
| prior TD rate | −0.040 | −5.47 | **−7.18** |
| snap-share trend | +0.043 | +3.04 | **+3.74** |
| defence vs position | +0.028 | +3.74 | **+3.92** |
| injury severity | −0.023 | −2.78 | **−4.53** |
| implied team total | +0.028 | +2.77 | **+2.70** |
| own spread | −0.006 | −0.55 | **−0.44** |

**Wind is the only `r` that moves, and it moves because its cells are unbalanced**: 2022 has 7
cells where every other season has 11, so the mean of the season means is not the mean of the
cells. Every other feature is balanced at 11 and the two coincide exactly.

**No verdict changes in either table.** Nothing sits near enough to the 2-se bar to cross it —
the three killed features are killed on the every-season half, which no standard error can
rescue, and the six findings clear it by a margin either way.

### One status does change, in the Usage screen

**`prior TD rate` against passing attempts.** Published as **+0.005 (1.9)**, which clears as a
pre-stated null by being null. Over the seasons it is **+0.005 at +2.7 se**, positive in
**5/5** seasons — 2021 +0.0066, 2022 +0.0005, 2023 +0.0025, 2024 +0.0100, 2025 +0.0031 — and a
pre-stated null that is significant and consistent is a **PRE-STATED NULL BROKEN**, the
screen's third verdict.

> **What this contradicts.** *"The prior TD rate moves touchdowns, hard, and moves no volume
> count at all"* — the sentence below the Usage table — is now false as written. It moves
> passing attempts, against its own pre-registration. What survives is the *magnitude*
> argument, and it survives easily: **+0.005 against −0.118 on touchdowns**, a twenty-fold
> difference, and attempts is the one volume count that is almost entirely quarterbacks. The
> structural claim the plan made — week-level information moves Usage and barely touches
> efficiency, with the two features orthogonal — is dented rather than refuted, and the
> licence in *What this does and does not license* is unchanged: the prior TD rate still
> enters as a touchdown regression and nothing crosses into the Usage multiplier.

The other nineteen Usage cells keep their verdicts. Their `t` values move with everything
else: snap-share trend on targets 8.1 → **9.7**, receptions 6.1 → **10.8**, carries 4.1 →
**4.2**, attempts 2.6 → **2.4**, touchdowns 1.5 → **1.9** (still killed); prior TD rate on
touchdowns −14.4 → **−11.7**.

### The correction narrowed most of these intervals, which is the finding

`docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md` names this in advance: *"A
narrower interval after clustering would be a signal, not a win. Widening depends on
between-season variance exceeding within-season variance."* Five of the nine features
narrowed and four widened, so this is that case and not the one #169 assumed.

The mechanism is measurable. Clustering widens an interval only when cells inside a season
are *positively* correlated; if they were independent draws the two standard errors would be
equal, because sd over 55 cells ÷ √55 is sd over 5 season means ÷ √5 whenever the season
means are just averages of 11 independent cells:

| feature | sd over cells | sd over 5 season means | what independence predicts | ratio |
|---|---|---|---|---|
| implied team total | 0.0623 | 0.0094 | 0.0188 | **0.50** |
| injury severity | 0.0612 | 0.0123 | 0.0185 | **0.67** |
| prior TD rate | 0.0517 | 0.0111 | 0.0156 | **0.71** |
| snap-share trend | 0.0805 | 0.0252 | 0.0304 | **0.83** |
| defence vs position | 0.0546 | 0.0188 | 0.0165 | **1.14** |

A ratio under 1 means the season means are *more* stable than independent cells would be.
Week-to-week scatter inside a season is large — matchups, weather, who was hurt that Sunday —
and it averages out; what is left between seasons is small. Only defence-vs-position behaves
the way the clustering argument assumes.

**And a consequence worth naming rather than discovering later.** The two halves of the
pre-registered rule are no longer independent of each other. A feature whose sign holds in
5/5 seasons has, by construction, season means that agree — which is exactly the condition
that makes the season-clustered se small. So the significance half now partly re-reads the
every-season half, and a feature that passes one is likelier to pass the other than it was.
[method.md rule 4](method.md) put the every-season half there to catch sign flips that
significance alone would miss; it still does that, but the two are no longer the independent
hurdles the pre-registration treated them as. That is a property of clustering on the unit
the verdict uses, it is not fixable by choosing a different estimator for the same rule, and
it is the open question this correction leaves behind.

**What was not adopted through.** The estimator was not chosen for the direction it moved the
intervals. Selecting between two units by which gives the answer you prefer is the failure
this repo's whole method exists to prevent; the season is the unit because the verdict reads
seasons, and the narrowing is a fact reported, not a result banked.

## The control basis, and what the prior TD rate is conditional on

**Re-run 2026-09-10**, issue #179, under [method.md rule 13](method.md). The basis below was
written into the ticket before it was run. Nothing above this section moves; what follows adds
the condition the surviving claim carries.

**The circularity, stated exactly.** The controls are `ppg_before` and `ecr`. `ppg_before` is
season-to-date **PPR points** a game — and PPR points *contain touchdowns*. The feature is
`tds_prior / yds_prior`. So the control set contains the feature's own numerator, and holding
the points total fixed, a higher touchdown rate is arithmetically **fewer yards**. The recorded
−0.040 could therefore have been a yardage effect wearing an efficiency label, and the number
alone cannot say which. That matters because two modules cite it for the *efficiency* reading.

**The alternative basis.** `ppg_before` is split into a touchdown component and a
non-touchdown component and both are controlled on. The touchdown half prices the three
touchdown columns from `components.SCORING` — a passing touchdown is **four** points, not six —
and the non-touchdown half is the remainder by subtraction, so the two sum to `ppg_before`
exactly on every row. That identity is the point: the new set **spans** the old one, so the only
thing relaxed is the constraint that a point of touchdown scoring and a point of everything else
carry the same slope. A coefficient that moves has one cause, not two.

Sample unchanged: **14,370 player-weeks, 847 players, 55 cells**, the same rows under both
bases. Every other figure on this page was reproduced under the pooled basis in the same run.

### What moved

| | r | t | seasons | verdict |
|---|---|---|---|---|
| **alone**, pooled *(published −0.038, −7.6)* | −0.0375 | −7.58 | 5/5 | null broken |
| **alone**, decomposed | **−0.0426** | **−6.16** | **5/5** | **null broken** |
| **joint**, pooled *(published −0.040, −7.18)* | −0.0403 | −7.18 | 5/5 | null broken |
| **joint**, decomposed | **−0.0434** | **−5.64** | **5/5** | **null broken** |

**The pre-stated null stays broken, and the coefficient gets slightly larger rather than
smaller.** The sign does not move, the season count does not move, and the `t` falls only
because a redundant control costs precision — the se widens 0.0056 → 0.0077, and −5.64 clears
`MIN_SE` by a wide margin. The placebo is clean under the new basis too: **−0.0007, t −0.15**,
permuted within cell, against −0.0008 under the old one.

Per season, joint: 2021 −0.0497, 2022 −0.0425, 2023 −0.0210, 2024 −0.0676, 2025 −0.0362.

**Why the coefficient barely moves.** The constraint that was relaxed turns out not to bind.
Regressing week-*w* points on the two halves and ECR within each cell, the touchdown half
carries **+0.328** and the non-touchdown half **+0.289** — a difference of +0.039 at **t +1.75**,
with two of five seasons on the other side. The halves do not demonstrably want different
slopes, so the added degree of freedom is nearly unused, and what the decomposition mostly buys
is a wider interval.

### What the decomposition does not settle, and this is the part that matters

**It does not remove the yardage confound. It concentrates it.** Measure the coupling the
ticket names — the partial correlation between `td_rate_prior` and `yds_prior` inside a cell:

| controlling for | partial r |
|---|---|
| `ecr` only | −0.036 |
| `ppg_before`, `ecr` *(published basis)* | −0.114 |
| `td_ppg_before`, `nontd_ppg_before`, `ecr` *(this basis)* | **−0.401** |

Holding the touchdown half fixed pins the feature's **numerator**, so what is left varying in
`tds_prior / yds_prior` is very nearly the denominator alone. The decomposed basis therefore
makes the surviving −0.043 *more* yardage-loaded than the −0.040 it was run to check, not less.
It answers the question it was pre-registered to answer — "efficiency regresses" against "high
scorers regress" — and it answers it in the feature's favour. It does not answer "efficiency"
against "yardage".

**Controlled for prior yardage directly, the finding does not survive.** `yds_prior` is on the
Panel and is a legitimate control — it is a `_prior`, measured strictly before week *w*:

| basis | r | t | seasons | verdict |
|---|---|---|---|---|
| `yds_prior`, `ecr` | −0.0122 | −2.49 | **4/5** | **not broken** — noisy, not a signal |
| decomposed + `yds_prior` | −0.0238 | −2.46 | **4/5** | **not broken** |
| joint, decomposed + `yds_prior` | −0.0218 | −2.21 | **4/5** | **not broken** |

**2023 is the season that turns positive** (+0.0059 on the joint row) once prior yardage is
held directly, and the every-season half is what fails — which no standard error can rescue,
and which [method.md rule 4](method.md) put there precisely to catch.

The first of those three rows is not an improvisation: **controls of consensus rank and prior
yardage only** is the basis this ticket's own issue body pre-registered and the maintainer
adopted on 2026-09-07, before a later disposition replaced it with the decomposition. Two
pre-registered bases, run on one panel, give opposite verdicts. Which of them the published
claim should rest on is a decision, not a measurement, and it is not settled here.

### What this page now claims

**The −0.040 stands as published, and is conditional on its controls.** It is a partial
correlation beyond **season-to-date PPR points a game and weekly consensus ECR** — jointly,
beyond the implied total, own spread, defence-vs-position and injury severity — and it survives
splitting the first of those into its touchdown and non-touchdown halves. It is **not** shown to
survive a direct control for prior yardage, and it was never claimed to be. Anything citing it
should name that.

## The screen's minimum week, and the one status that is conditional on it

**Re-run 2026-09-10**, issue #178, under [method.md rule 13](method.md). The sweep range and
step were fixed in the commit that ran them, before the run, which is what the disposition on
#178 pre-registered.

**The circularity, stated exactly.** `panel.TREND_MIN_WEEK = 8` did two jobs. It is a *model*
threshold — `hub.models.weekly` fits on it and forces its multiplier to identity below it — and
it was also the screen's `min_week` for `snap_trend`, `tgt_trend` and the optional trend
features, which is what `cell_correlations` filters rows on and what `screen_joint` reads to
decide which survivors may act as controls. Its value is 8 because 8 is the **earliest anchor
that held when 4, 6, 8, 10 and 12 were tested against the outcome**
([snap-trend-signal.md](snap-trend-signal.md)). So features were screened on rows selected by a
value fitted to the outcome on the same data, and the page did not disclose the value.

The threshold keeps its measured value; the model's behaviour does not move, and nothing here
re-fits it. What changed is that the *screen* stopped borrowing it. `SCREEN_TREND_ANCHORS` is
the screen's own constant, it is swept rather than chosen, and
`test_the_screen_does_not_read_the_model_s_trend_threshold` is an AST guard that stops the two
being re-coupled by an import.

### The surviving feature set at each anchor

Basis **`pooled`** — the pre-registration, and the basis every figure above rests on. Sample
unchanged at **14,370 player-weeks, 847 players**; every figure above was reproduced exactly in
the same run before the sweep was taken.

| trend features from | cells | surviving feature set |
|---|---|---|
| week ≥ 4 | 45 | dvp, inj_sev, **snap_trend**, td_rate_prior |
| week ≥ 6 | 45 | dvp, inj_sev, **snap_trend**, td_rate_prior |
| week ≥ 8 *(published)* | 35 | dvp, inj_sev, **snap_trend**, td_rate_prior |
| week ≥ 10 | 25 | dvp, inj_sev, **snap_trend**, td_rate_prior |
| week ≥ 12 | 15 | dvp, inj_sev, td_rate_prior |

**It is not stable, and the instability is one feature at one anchor.** Everything except the
snap-share trend returns the identical verdict at every anchor, and the six week-1 features
return the identical `r` to four decimals — which is structural rather than reassuring: the
anchor is a floor on the *trend* features' weeks, and `screen_joint` admits a survivor as a
control only where its own minimum week is no later, so a week-1 feature is never controlled
for a trend feature at any anchor in this sweep.
`test_the_anchor_moves_only_the_trend_features_numbers` pins that, so a week-1 verdict that
ever did move would be a bug and not the anchor.

**Anchors 4 and 6 are one measurement, not two.** `snap_trend` is null before week 6 — it needs
six prior weeks to form both windows — so the week-4 filter and the week-6 filter select the
same 45 cells. The sweep has **four** distinct samples across five anchors, and the shallow end
of the declared range is not reachable. Reported because the range was fixed before the run and
this is what fixing it in advance bought.

### The snap-share trend, per season

| trend from | cells | r | t | 2021 | 2022 | 2023 | 2024 | 2025 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| week ≥ 4 | 45 | +0.0302 | +2.38 | +0.0091 | +0.0271 | +0.0790 | +0.0122 | +0.0235 | clears |
| week ≥ 6 | 45 | +0.0302 | +2.38 | +0.0091 | +0.0271 | +0.0790 | +0.0122 | +0.0235 | clears |
| week ≥ 8 | 35 | **+0.0382** | **+3.39** | +0.0277 | +0.0244 | +0.0828 | +0.0234 | +0.0326 | **clears** |
| week ≥ 10 | 25 | +0.0506 | +4.46 | +0.0428 | +0.0343 | +0.0771 | +0.0217 | +0.0773 | clears |
| week ≥ 12 | 15 | +0.0356 | +2.03 | +0.0324 | +0.0853 | +0.0300 | **−0.0225** | +0.0529 | **killed** |

Joint, controlled for the other survivors: +0.0347 (4 and 6), **+0.0425** (8, the published
figure), +0.0550 (10). At week ≥ 12 it never reaches the joint screen, because it is killed
alone.

**It dies on the every-season half, not on significance, and it dies in 2024.** The point
estimate at week ≥ 12 is +0.0356, which is not smaller than the published +0.0382 — what fails
is that 2024 comes back negative. [method.md rule 4](method.md) put that half there exactly to
catch a sign that flips between seasons, and it is the half no standard error can rescue.

**And the honest caveat in the other direction, which does not rescue it.** At week ≥ 12 a
season mean is built from **three cells** — weeks 12, 13 and 14 — against eleven at week ≥ 4.
The every-season half is being asked of a much noisier estimate of each season, so a single
season crossing zero is a likelier accident there than anywhere else in the sweep. That is a
reason the week-12 result is weak evidence *against* the trend; it is **not** a reason to
report the trend as clearing, and the sweep was not run to find a licence to keep the tidiest
anchor. The rule is the rule at every anchor or it is a rule at none.

### It is not an artefact of the control basis

The same sweep on **`decomposed`** — #179's alternative set, `td_ppg_before`,
`nontd_ppg_before`, `ecr`, on the same 14,370 rows — returns the same five verdicts and the
same surviving sets: snap_trend +0.0286 (4 and 6), +0.0365 (8), +0.0502 (10), and at week ≥ 12
+0.0363 at +2.21 with 2024 at **−0.0188**, killed. So the sensitivity reported here is a
property of the row filter and not of the basis.

**This does not decide #229** — which basis a surviving claim is conditional on is that
ticket's question, and running both here says only that #178's answer does not depend on how it
is settled.

### What this page now claims about the snap-share trend

**The +0.038 alone and the +0.043 joint stand as published, and they are conditional on a
minimum week of 8.** The finding holds at every anchor from 4 to 10 and fails at 12, so it is
reported with that range rather than as a single verdict. Anything citing the snap-share trend
as a screen result should name the range.

**What does not move.** `dvp`, `inj_sev` and `td_rate_prior` are unconditional across the
sweep; the three killed features stay killed at every anchor; `tgt_trend` is killed at every
anchor, reaching 3/5 seasons at its best. And the **model** threshold is untouched: the Usage
multiplier is still dark before week 8, for the reason
[snap-trend-signal.md](snap-trend-signal.md) measured and not for this one.

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

> **Restated 2026-09-07 — the last sentence above is no longer true.** Issue #169. The
> table's `t` values were taken over cells and the verdicts over seasons; corrected to the
> season, **prior TD rate against passing attempts becomes a broken pre-stated null** —
> +0.005 at **+2.7 se**, 5/5 seasons, against the **1.9** printed above. It is the one
> status on this page that changes. The magnitude argument holds regardless (+0.005 against
> −0.118 on touchdowns) and nothing downstream moves; the claim of a clean orthogonality
> does not. See *The unit of the standard error*, above.

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

> **Qualified 2026-09-10 — the snap-share trend's licence is conditional on the anchor.**
> Issue #178. The screen result behind it holds at a minimum week of 4, 6, 8 and 10 and is
> killed at 12; see *The screen's minimum week*, above. The licence itself is unchanged,
> because the Usage multiplier's own threshold of week ≥ 8 is inside the range over which the
> screen result holds — but it is a licence with a stated range now, not an unqualified one,
> and a later re-run that moved the range would move this sentence with it.

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
uv run python -m hub.models.weekly_screen --run                        # the sweep -- #178
uv run python -m hub.models.weekly_screen --run --trend-min-week 8     # the published tables
uv run python -m hub.models.weekly_screen --run --basis decomposed     # #179
```

`--basis` names the control set. `pooled` is the pre-registration and every figure above rests
on it; `decomposed` holds the two halves of `ppg_before` apart. Either run drops nulls over the
**union** of the two, so the two are taken on the same rows and a difference between them is
attributable to the basis.

`--trend-min-week` names the anchor the trend features are screened from. **There is no
default**, which is #178: the screen used to borrow `panel.TREND_MIN_WEEK`, a value fitted to
the outcome on these rows, and a plain `--run` now sweeps all five anchors and prints the
sensitivity rather than picking one. Pass `8` to reproduce the tables above; the tables
elsewhere on this page are unchanged by the sweep because only the trend features move with it.

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
