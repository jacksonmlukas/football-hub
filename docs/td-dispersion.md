# Refit `TD_DISPERSION`: pre-registration (#472)

Written 2026-10-15, **before any estimate of the constant on a held-out season, before the gate has
run, and before the module that runs it exists.** Nothing below carries a measured number: the
only figures this document leans on are the ones the issue already published
(`docs/td-count-sampler.md`, section 6), and they are used only as the thing being tested. Under
`docs/method.md` rule 1 the rule is fixed first, in words here and in code in the next commit;
under the sibling of rule 16 this text may be edited only until the gate has run, and every
edit after the implementation lands and before the run is dated in section 10.

## 0. The question

`components.TD_DISPERSION` (`pass` 0.85, `rush` 0.86, `rec` 0.83, declared `fitted`) is the
variance over mean that `components.draw_counts` holds *exactly* for an under-dispersed
touchdown count since #307. #307 measured the within-player-season variance over mean on
2022-25 player-weeks as 0.962 (rec), 0.957 (rush) and 0.842 (pass), and published that the new
sampler's 30+/35+/40+ point weeks overshoot the observed by 16-23%. **Does replacing the three
constants with the same measurement, made only from strictly earlier seasons, predict the next
season's weekly touchdown counts better than the constants in the file?**

The incumbent is the three constants as shipped. The challenger is the estimator in section 1,
refit on every season before the one scored. This is one gate on the three constants *jointly*
(one ledger entry, one verdict); the per-phase figures are reported and not separately gated,
because gating three phases would be three tests answering one question and the constants are
adopted or kept together.

## 1. The estimator

Per phase (`rec`, `rush`, `pass`), over the **qualifying player-seasons** of the training seasons:

* regular season, `QB/RB/WR/TE`, at least **8 games** (rows) in the season, and real volume in the
  phase: at least **2 receptions**, **4 carries** or **10 attempts** a game. (These replicate the
  sample #307 measured on to within a few player-seasons per phase; they are the thresholds that
  make "this player's receiving line" a line.)
* for each qualifying player-season, `SS_i` is the sum of squared deviations of the weekly touchdown
  counts from the player-season's own mean, and `n_i` its games;
* **`phi_hat = (sum SS_i / sum (n_i - 1)) / (sum x_i / sum n_i)`**: the pooled within-player-season
  variance (unbiased, `n_i - 1` degrees of freedom) over the pooled mean.

**Pooled across positions within a phase, not by position.** `TD_DISPERSION` is keyed by phase
and nothing downstream (`draw_counts`, `phase_draws`, `hub.models.props`) takes a position into
it, so a by-position estimate would need a new key and a new place for every reader to look it up;
and the by-position cells (a quarterback's rushing touchdowns, a tight end's carries) are far
thinner than the phase's. The by-position figures are computed and **reported as a diagnostic only**.

**Why the unbiased pooled form, and why no disattenuation (`docs/method.md`, "noted twice").** That
note is about a constant that is a *true-rate* variance read through few events. This is the
variance of the *count* itself around the player-season's own mean, which is the quantity the
sampler draws; the within-player `n_i - 1` divisor is the whole correction there is. The mean
being the player's own season average is the same construction the sampler is used with (a
season-average line).

The estimator's interval is a **player-cluster bootstrap, 1,000 resamples** (rule 3: a player's
weeks are not independent), reported with the figure. The gate does not read it.

**The value adopted on an ADOPT** is this estimator on **every season loaded** (2019-2025),
rounded to two decimals, per phase; the 2022-25 figure (the issue's) is printed beside it. Rounded
to two decimals because that is the grain of the constants it replaces.

## 2. Scoring: strictly earlier seasons, one metric

**Primary metric: the held-out log score of the weekly touchdown count, per player-week.** For each
held-out season `s` (`experiment.expanding_seasons`: `phi_hat` fit on seasons `< s`, never on `s`),
each qualifying player-season of `s` and phase, and each of its weeks `w` with count `x_w`:

* the predictive law is the family `draw_counts` draws -- Conway-Maxwell-Poisson below 1, Poisson at
  1, negative binomial above -- at mean `m_w` and the arm's dispersion, so a refit above 1 is scored
  as the sampler would draw it, not as a family the sampler does not have;
* **the mean `m_w` is leave-one-out with one pseudo-game**: `m_w = (S_i - x_w + r) / n_i`, where
  `S_i` is the player-season's touchdowns, `n_i` its games and `r` the pooled touchdowns a game of the
  phase over the training seasons (strictly earlier). The week scored never enters its own mean (the
  sampler is used on a line built without the outcome it prices); the pseudo-game keeps the mean
  positive for a player with no other touchdown, where a mean of zero would make a touchdown
  impossible and the log score minus infinity. **The mean is the same for both arms**, so the
  comparison isolates the dispersion and nothing else.
* a **player-week's score is the sum of its phases' log pmfs** over the phases that player-season
  qualifies in; the paired difference is `score(refit) - score(shipped)`, nats per player-week,
  positive when the refit predicts better.

**Biases, named before they can be read as a result.** (1) The shipped constants were measured on
data that includes 2022-25, so on those held-out seasons the incumbent has seen what it is scored
on and the challenger has not: this tilts *against* ADOPT. (2) The leave-one-out mean carries its
own estimation noise, which a log score reads as extra dispersion (the score's optimum sits above
the in-sample variance over mean by roughly the factor `1 + 1/(n-1)`): this tilts *toward* the
higher dispersion. The second is the honest out-of-sample cost of pricing a week from a line that
was estimated, and it is the reason the metric is the log score and not a comparison of two
variances. Neither is tuned away.

**Secondary, descriptive, not gated:** (a) the sampler's calibration at 30+/35+/40+ fantasy-point
weeks against the observed, at the incumbent and at the refit, on the player-seasons of #307's
table (this is what the issue is ultimately about, and it is *not* the gate, because it is an
aggregate of counts with no paired unit and it is confounded with `PER_UNIT_CV`,
`COUNT_DISPERSION` and `TD_YARDS_RHO`, none of which this refits); (b) the held-out calibration
of P(2+ touchdowns) per phase.

## 3. The unit, the ceiling, the rule

* **Unit of replication: the season** (`experiment.SEASON_CLUSTER`). Rows are player-weeks;
  `within = ("player_id",)` is the repeated-measure unit of a season's own tie test (rule 3).
* **Held-out seasons: 2020-2025** (2019 is training only; every season the repo holds `player_stats`
  for). Six clusters, so the t interval's critical value is `t(0.975, 5)`.
* **The ceiling arm: a dispersion fitted to the held-out season itself** -- per phase, the value
  that maximises that season's total log score under the same laws, found by a bounded
  one-dimensional search. It is the most any constant could gain over the incumbent on these rows,
  rule 8's ceiling. The gate's NOT-RUNNABLE branch compares the run's own MDE to it.
* **The rule is `experiment.gate`, unmodified**: ADOPT iff the season-clustered t interval of the
  mean paired difference excludes zero *above* and no resolved season lost, with at least one
  resolved season (ADR-0019, #381); REMOVE the mirror; NOT-RUNNABLE if the MDE exceeds the
  ceiling; SHOW otherwise. Its verdict function is `experiment._verdict`, tested branch by branch
  in `tests/unit/test_experiment.py`; this module adds a test of the branch where the refit loses
  (rule 1) and does not restate the rule.
* **Degeneracy (rule 17).** Under `SEASON_CLUSTER` the percentile bootstrap would be the sign test
  read twice; the rule reads the t interval, which is not implied by the seasons' signs
  (`experiment.t_interval`), so the two halves are not one. The size of this gate is checked by
  simulation in section 5 and not argued.

## 4. Power and MDE (rule 16), computed before the run

By simulation, through this module's own pipeline end to end, not by a formula that could
disagree with it: a synthetic player-week table with the *shape* of the real one (player-seasons
per phase and season at the qualifying counts, 8-17 games each, per-game touchdown means drawn
from a Gamma with a mean and spread typical of the phase) whose counts are drawn by
`components.draw_counts` at a **planted** truth. Two planted truths:

* **N (null): the truth is the shipped constants.** The refit has nothing to find. The ADOPT rate
  is the gate's size here; the rule is not a sign test of size `2**-6` and the simulation shows
  what it is.
* **P (the issue's claim): the truth is the three figures the issue measured** (0.96, 0.96, 0.84).
  The ADOPT rate is the gate's power against exactly the improvement #472 alleges.

Each is replicated enough times to put a standard error under the rate, each replicate a full
walk-forward over six held-out seasons and `experiment.gate`. The ceiling arm in the simulation is
the planted truth (it is known there; the real run searches). What is reported: the ADOPT rate
under N and P, the simulated MDE (`experiment.minimum_detectable_effect` off the gate's own
standard error) against the simulated ceiling, and the NOT-RUNNABLE rate.

**Which branches are reachable, named now.** The gate is run once, on the six seasons that
exist, and a second look is a year away. If the simulated power at P is below one half, the
expected verdict is SHOW and this document says so here rather than after: SHOW and
NOT-RUNNABLE are *not* a ruling that 0.83 is right. They keep the constant and record the
measurement (section 6), and the figure 0.96 stays published as a measurement with its interval.
The run goes ahead either way: it is cheap, and a run that cannot adopt can still remove.

## 5. The controls (rule 18)

Each is a test that plants the condition the check exists to detect, and each is shown to fail
on the defect before it is trusted:

1. **Recovery.** A planted dataset with a known dispersion is recovered by the estimator: a
   Poisson one (variance over mean 1.0) and a binomial-thinned one (0.8), the latter drawn by a
   mechanism other than the sampler's own so the estimator is not being tested against itself.
   A planted estimator that ignores the `n_i - 1` divisor (divides by `n_i`) is shown to miss the
   same data, so the test can tell.
2. **The log pmf is the sampler's law.** `count_logpmf` is checked against the empirical
   frequencies of `components.draw_counts` at the same mean and dispersion in all three
   families, and against `components.cmp_pmf`; a pmf in the wrong family is shown to fail.
3. **The gate adopts a planted real improvement and not a null.** Data drawn at a truth the
   incumbent misses by a planted margin: ADOPT. Data drawn at the incumbent's own value: not
   ADOPT. A planted *worse* challenger (a constant far from the truth) is REMOVE.
4. **No look-ahead.** The estimator is shown to read only seasons strictly before the scored one:
   a planted held-out season whose counts are altered leaves the training fit unchanged; and the
   week scored is shown not to enter its own mean.
5. **The guards removed one at a time**, each red.

## 6. The action per verdict

* **ADOPT.** Change `components.TD_DISPERSION` to the estimator on all seasons (section 1), still
  declared `fitted`. Restate `docs/td-count-sampler.md`'s section 3 tail table and section 5 prop
  figures as dated amendments with the prior text kept; record the move of `fitted_digest` and
  `config_digest(HubConfig())`, measured before and after, in `tests/unit/test_config.py`'s
  table; prove the forward arm does not read the constant (grep, #430's identity test, a
  tripwire like #307's) and re-pin `hub.models.components` in `docs/weekly-forward.md` with a
  dated amendment. If the arm reads it, stop. `TD_YARDS_RHO` is **not** refit here (it is its own
  fitted constant with its own interval); the implied within-player correlation at the new
  dispersion is measured and recorded as an open note.
* **REMOVE, SHOW or NOT-RUNNABLE.** Keep the constants. Record the measurement (the estimator's
  figures with their intervals, the held-out scores, the verdict line) in this document and in
  `docs/td-count-sampler.md`'s section 6, and leave `components.py` as it is.

## 7. What this does not do

It does not refit `COUNT_DISPERSION`, `PER_UNIT_CV` or `TD_YARDS_RHO`, does not change the
family (#307's), and does not read any 2026 outcome. The overshoot at 30+ points is a question
about all four constants together; this answers it for one.

## 8. Ledger

Exactly one entry is written, by the run in section 9, to `state/gate-width.json`. Every test and
every simulation uses an in-memory ledger.

## 9. The run

`uv run python -m hub.models.td_dispersion --fit` (written in the next commit). It prints the
verdict line, the per-season table, the estimator's per-phase figures with intervals and the
secondary tables, and writes the one ledger entry.

## 10. Amendments (dated; added only before the run)

**Amendment 1, 2026-10-15, after the module was written and its first planted-data smoke test,
before any real held-out season was scored: the log score is floored at one in a thousand.** The
section 2 metric as first written is the unfloored log pmf. On a *planted* table (counts drawn at
a known dispersion; no real outcome read) it was dominated by a handful of weeks: where the mean is
below `1 - phi` the Conway-Maxwell-Poisson at the constant is the narrowest law there is (#307,
section 1), a two-touchdown week at a mean of 0.05 has probability about 1e-30 under it, and that
week scores minus seventy nats. The paired difference between two constants was then mostly the
question of which one makes such a week impossible, a property of the unreachable region and not of
the level of the dispersion, and the season mean of it was a few rows. **The metric is now the log
pmf floored at 1e-3** (`td_dispersion.SCORE_FLOOR`): a count the law calls rarer than one in a
thousand is scored as one in a thousand. It is applied to both arms and to the ceiling arm
identically. The unfloored mean difference is printed beside the verdict as a diagnostic and
is not read by the rule. Everything else in sections 1-9 stands.

**Amendment 2, 2026-10-15, the power simulation of section 4, run on the module as committed in
amendment 1 and before any real held-out season was scored.** `uv run python -m hub.models.td_dispersion
--power --reps 24` (6 workers): 24 planted walk-forwards a scenario, seven planted seasons
(six held out), 165 / 78 / 36 player-seasons a season in rec / rush / pass, 8-17 games each,
per-game means from a Gamma typical of the phase, counts drawn by `components.draw_counts`, the
ceiling arm the planted truth, `experiment.gate` at 200 bootstrap resamples.

| scenario | ADOPT | NOT-RUNNABLE | REMOVE | mean gain | se | MDE | ceiling |
|---|---|---|---|---|---|---|---|
| N: truth = the shipped constants | 0/24 | 24/24 | 0/24 | -0.00023 | 0.00024 | 0.00082 | +0.00000 |
| P: truth = 0.96 / 0.96 / 0.84 (the issue's figures) | 24/24 | 0/24 | 0/24 | +0.00729 | 0.00076 | 0.00258 | +0.00777 |

Read: **the gate has power against the improvement the issue alleges** -- 24 of 24, so a lower
95% bound of about 0.86 -- and an MDE (0.0026 nats a player-week) a third of the ceiling it is
alleged to reach. **Under the null the only reachable branch is NOT-RUNNABLE**, because the
ceiling is zero when the incumbent is the truth: there is nothing to adopt, and the gate says so
rather than printing a SHOW. Size is zero in 24 (an upper bound of about 0.12). **The honesty
caveat:** the planted counts have a fixed mean for a player-season, so a planted truth is exactly
the law that is scored; real counts also move their mean week to week, which the scored law does
not carry, so the real effect will be smaller than a planted one at the same dispersion. The
simulation sizes the instrument, not the result. The expected verdict under the issue's claim is
therefore ADOPT if the claim is right, and not SHOW for want of power.

## 11. The result (run once, 2026-10-15; one Ledger entry, `state/gate-width.json`)

`uv run python -m hub.models.td_dispersion --fit` at commit `3a3d611c`, config digest `4fad9f78`,
data digest `17123a2e`, code digest `e290352b`, 1,987 qualifying player-season-phases over
2019-2025, six held-out seasons (2020-2025), 20,367 held-out player-weeks.

**The verdict line, verbatim:**

> ADOPT: TD_DISPERSION is replaced by the refit (docs/td-dispersion.md section 6). It won in every resolved season (6 resolved of 6, 0 abstained; won 6, tied 0, lost 0 of 6 seasons) and the t interval [+0.011, +0.014] excludes zero.

| | refit minus shipped, nats per player-week |
|---|---|
| mean paired difference | **+0.01233** (95% bootstrap [+0.01128, +0.01344]; t [+0.01087, +0.01380]) |
| MDE at 80% power | +0.00195 |
| ceiling (a dispersion fitted to the held-out season itself) | +0.01367 |
| the refit's share of the ceiling | 90% |

| season | gain | se | player-clusters | disposition |
|---|---|---|---|---|
| 2020 | +0.01414 | 0.00421 | 252 | win |
| 2021 | +0.01445 | 0.00341 | 251 | win |
| 2022 | +0.01114 | 0.00305 | 235 | win |
| 2023 | +0.01111 | 0.00306 | 233 | win |
| 2024 | +0.01129 | 0.00351 | 241 | win |
| 2025 | +0.01187 | 0.00270 | 230 | win |

The estimator on every season loaded (pooled within-player-season variance over mean), with a
player-cluster 95% interval, beside the 2022-25 figure and the constant shipped:

| phase | 2019-25 | interval | 2022-25 | shipped |
|---|---|---|---|---|
| rec | **0.971** | [0.950, 0.994] | 0.966 | 0.83 |
| rush | **0.986** | [0.949, 1.027] | 0.970 | 0.86 |
| pass | **0.826** | [0.790, 0.868] | 0.844 | 0.85 |

The 2022-25 column reproduces the issue's figures to within a few thousandths (0.962 / 0.957 /
0.842; the difference is the sample thresholds of section 1). By position (a diagnostic, the
constant is keyed by phase): rec RB 1.016, TE 0.996, WR 0.955; rush QB 0.991, RB 0.980 (TE 1.403 on
very few touchdowns); pass QB 0.826 (TE 1.273 on trick plays).

**What was and was not adopted.** Rec and rush, which the issue flagged, move to 0.97 and 0.99.
Pass, which the issue read as unchanged (0.842 against 0.85), moves 0.85 to 0.83 because the
registered estimator reads every season, and its interval [0.790, 0.868] contains the old
constant: that move is inside the noise of the figure and is adopted because the pre-registration
adopts the estimator as one triple, not because the pass phase was tested alone. (The gate is on
the three jointly; it cannot say which phase earned the gain, and section 1 chose it so.) The
adopted values are the registered estimator, rounded to two decimals: **rec 0.97, rush 0.99,
pass 0.83**.

**Diagnostic, not read by the rule.** The unfloored mean difference is +0.176 to +0.276 nats a
player-week in every season: fifteen to twenty-two times the floored one, and the reason
amendment 1 exists.
Those nats are the weeks the shipped constants' unreachable region calls impossible (a mean under
`1 - 0.83 = 0.17` is the narrowest law there is), which the higher constants make reachable down
to a mean of 0.03 (rec) and 0.01 (rush). The floored gain is the part that is not that.

**Biases, as named in section 2.** The shipped constants had seen 2022-25 (against the refit); the
leave-one-out mean carries estimation noise a log score reads as dispersion (toward the higher
constants). The gain is as large in 2020 and 2021, seasons before the 2022-25 data the shipped
constants were measured on, as in 2022-25, so the first bias is not hiding a larger effect in the
later seasons. The second bias is not separately quantified and the rule does not subtract it:
the number is reported as measured, and the pre-registration named it before the run.

## 12. After the verdict: what was done, and what the secondary table said

* `components.TD_DISPERSION` is **pass 0.83, rush 0.99, rec 0.97** (was 0.85, 0.86, 0.83), still
  declared `fitted`. `config_digest(HubConfig())` `4fad9f78` -> `839ceda9`, `fitted_digest`
  `f6b8c28a` -> `9f2d0b0c` (measured before and after; restoring the old three values reproduces
  both prior digests). The gate's incumbent is kept as `td_dispersion.INCUMBENT`, the three
  constants it was registered and run against, so the run reproduces after the constant moved.
* **The forward arm does not read the constant**: `docs/weekly-forward.md`, amended 2026-10-15,
  with the closure re-pinned (`hub.models.components` `d42ac69f7717`, closure `69a09cab8e72c77d`).
* **The secondary table (section 2a) says the opposite of what the issue hoped.** On #307's
  649 player-seasons, the sampler's expected weeks at 30 / 35 / 40 or more points go from 398 / 164
  / 64 (15% / 20% / 23% over the observed 345 / 136 / 52) to 428 / 184 / 76 (24% / 36% / 45%
  over). The refit predicts the touchdown count better and widens the tail overshoot. The gate was
  registered on the count and the table as descriptive, so the action is the pre-registered one,
  and the finding is that the overshoot at 30+ points is not caused by the touchdown dispersion;
  it is in `PER_UNIT_CV`, `COUNT_DISPERSION` or `TD_YARDS_RHO` (the last re-measured at the new
  dispersion: it still implies 0.349 / 0.369 / 0.478 against the observed 0.349 / 0.372 / 0.477).
  `docs/td-count-sampler.md` section 9 carries the restated tables.
* **Left open, deliberately**: a refit of `PER_UNIT_CV` and `COUNT_DISPERSION` at the corrected
  touchdown law, which is the candidate this result points at, and which is its own gated change.

## 13. Amended 2026-10-15 (#473): the open item in section 12 is closed as a diagnosis (prior text kept)

Section 12's last bullet leaves "a refit of `PER_UNIT_CV` and `COUNT_DISPERSION` at the corrected
touchdown law" as the candidate. `docs/td-count-sampler.md` section 10 ran the diagnosis first and
**supersedes the sentence in section 12 that puts the overshoot in those constants**: about half of
the table's overshoot is the table's own method (a planted null whose truth is the shipped sampler
reads +11% / +20% / +32% at 30 / 35 / 40), and the rest is the upper tail of the unit-count law at a
variance that is right; no setting of the two constants calibrates the tail without cutting the 20
and 25 point weeks. No refit was registered. The constant this document adopted is unchanged.
