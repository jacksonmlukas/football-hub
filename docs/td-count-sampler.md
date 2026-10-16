# The touchdown count: no ceiling, its dispersion held, and drawn with the yards (#307)

Landed 2026-10-15 under ADR-0024 (issue converted to agent work 2026-09-16 from audit IV's R8 and
M-R4). The deliverable the issue named is the gate table across the three families (section 1) and
the single restatement (section 5). **The restatement the issue anticipated -- a total re-draw that
moves the frozen board's arm B and the draft gate -- did not happen, because nothing in that path
reads this sampler (section 4).** What moved is the touchdown prop price and the shape of the
component sampler's week.

## 0. What was wrong, measured

`hub.models.components._counts` drew an under-dispersed count (`phi < 1`, every touchdown) as
`Binomial(round(mean / (1 - phi)), mean / trials)`. Two defects, both exact rather than sampled:

- **A ceiling.** The trial count *is* the support. At a WR1's receiving rate (0.36 a game,
  `phi = 0.83`) it is two, so P(3+) is exactly 0 where the Poisson gives 0.60%; the support is
  under three counts at every mean up to 0.42 (23 of the 581 means from 0.20 to 6.00 by 0.01).
- **A drift.** The rounding moves the realised variance over mean off the constant the family was
  chosen to hold: up to **9.6%** off `phi` (at a mean of 0.25, where the trials round 1.5 to 2 and
  the variance over mean is 0.870 against 0.83).

And the touchdowns were drawn after the yards and from nothing the yards said, so the weeks high on
both -- the weeks a ceiling-based decision reads -- were thinner than the measured correlation
(section 3) allows.

## 1. The gate: three families, one rule

The rule is the issue's first two acceptance criteria, stated here before the table is read from
the code: a family is adopted iff **(a)** its support has no ceiling -- P(k) > 0 for every k, so P(3+)
at a WR1 rate is non-zero -- and **(b)** its realised variance over mean is the fitted `phi` at every
mean the dispersion is reachable at (within 1%). A count of mean m < 1 cannot have variance below
m(1-m), so `phi = 0.83` is unreachable under a mean of 0.17 and the floor there is 1 - m; no family
can pass (b) below it, and the Conway-Maxwell-Poisson's answer there is the narrowest law there is
(two points either side of the mean).

Exact probabilities (no sampling), `phi = 0.83`. "Last count" is the support's end: the binomial's
is its trial count, the Katz family's is where `a + b k` turns negative, the Conway-Maxwell-Poisson
has none (tabled to mean + 10 sd + 25).

| mean | floor 1-m | binomial (shipped): last count / var-mean / P(3+) | Katz: last count / var-mean / P(3+) | Conway-Maxwell-Poisson: var-mean / P(3+) | Poisson P(3+) |
|---|---|---|---|---|---|
| 0.10 | 0.90 | 1 / 0.900 / 0 | 1 / 0.892 / 0 | 0.900 / 0.00000 | 0.00015 |
| 0.20 | 0.80 | 1 / 0.800 / 0 | 2 / 0.834 / 0 | 0.830 / 0.00001 | 0.00115 |
| 0.26 | 0.74 | 2 / **0.870** / 0 | 2 / 0.837 / 0 | 0.830 / 0.00017 | 0.00241 |
| **0.36** (WR1) | 0.64 | **2** / 0.820 / **0** | **3** / 0.830 / 0.00027 | 0.830 / **0.00133** | 0.00595 |
| 0.50 | 0.50 | 3 / 0.833 / 0.00463 | 3 / 0.830 / 0.00445 | 0.830 / 0.00591 | 0.01439 |
| 0.70 | 0.30 | 4 / 0.825 / 0.01862 | 5 / 0.830 / 0.01911 | 0.830 / 0.02036 | 0.03414 |
| 1.00 | - | 6 / 0.833 / 0.06229 | 6 / 0.830 / 0.06186 | 0.830 / 0.06137 | 0.08030 |
| 1.60 | - | 9 / 0.822 / 0.20497 | 10 / 0.830 / 0.20560 | 0.830 / 0.20053 | 0.21664 |
| 3.00 | - | 18 / 0.833 / 0.59735 | 18 / 0.830 / 0.59780 | 0.830 / 0.59378 | 0.57681 |

Worst realised-over-target gap over means 0.20-6.00 (step 0.01): binomial **9.64%**, Katz 0.83%,
Conway-Maxwell-Poisson **0.00%**.

| family | (a) no ceiling | (b) dispersion held | verdict |
|---|---|---|---|
| binomial, `round(mean/(1-phi))` trials (incumbent) | fails: P(3+) = 0 at the WR1 rate | fails: 9.64% off | REMOVE |
| Katz (`b = 1 - 1/phi < 0`) | fails: the same ceiling in a continuous parameter -- it ends at 3 at the WR1 rate, and P(3+) is a fifth of the Conway-Maxwell-Poisson's | passes at 0.83% | REMOVE |
| Conway-Maxwell-Poisson | passes: unbounded; P(3+) at the WR1 rate 0.00133, between 0 and the Poisson's 0.00595 | passes: exact where reachable | ADOPT |

Adopted: the Conway-Maxwell-Poisson (`P(k) ~ lambda^k / (k!)^nu`), `nu` solved at each mean so the
variance over mean is `TD_DISPERSION` (`components.cmp_nu`; `nu = 1` is the Poisson and a larger
`nu` is narrower). Over-dispersed counts (negative binomial) and `phi = 1` (Poisson) are untouched.
This is a distributional family replacing another under a fixed measurement, not a refit: no
`TD_DISPERSION` value moved.

## 2. Realised dispersion across the mean range, not at one point

`tests/unit/test_td_count_sampler.py::test_the_realised_dispersion_is_the_constant_at_every_mean`
holds the exact pmf's variance over mean to 1e-3 of `phi` and 600,000 draws to 2%, at seven means
from 0.2 to 6.0 and both `phi = 0.83` and `0.86`. Below the floor (mean 0.05, `phi = 0.83`)
`nu` runs to `CMP_NU_MAX` (64.0, a stop and not a fit: past about 10 the law no longer moves at any
mean a touchdown count has) and the realised value is the floor, 0.95.

## 3. Touchdowns drawn given the yards

**Measured, nflverse player-weeks, regular season 2022-25, within-player-season** (a player's
own weeks demeaned, pooled; 650 / 284 / 144 player-seasons of at least 8 games and real volume, so
the question is "in *this* player's high-yardage week, are there more touchdowns", which is what a
line fed to the sampler asks). 95% intervals are player-cluster bootstraps, 1,000 resamples.

| phase | observed corr(yards, TD) | the sampler, TD drawn apart | the sampler, structural (TD mean proportional to the week's yards) | **fitted `TD_YARDS_RHO`** (copula) | implied at the fit |
|---|---|---|---|---|---|
| rec | **0.349** [0.323, 0.375] | 0.00 | 0.433 | **0.456** [0.424, 0.491] | 0.350 |
| rush | **0.372** [0.342, 0.402] | 0.00 | 0.450 | **0.465** [0.426, 0.500] | 0.372 |
| pass | **0.477** [0.441, 0.516] | 0.00 | 0.437 | **0.506** [0.467, 0.546] | 0.478 |

The first design -- the week's expected touchdowns proportional to its yards, which is `TD_RATE`'s
own structure -- was built and **not shipped**: it overstates the receiving and rushing correlation
(0.433 and 0.450 against intervals that end at 0.375 and 0.402) and understates the passing one.
The shipped design keeps the marginal count exactly as drawn and pairs it with the yards: a
Gaussian copula at `rho` gives each week a latent score from the normal score of its yards rank and
a fresh normal, and the largest counts go to the largest scores
(`components.draw_touchdowns`). The coupling is a reordering, so the measured mean and dispersion
are untouched by construction (`test_the_coupling_reorders_the_counts_and_adds_no_new_ones`), and
`rho` is the one fitted constant it adds. It is fitted so the sampler's own within-player Pearson
correlation, on each qualifying player-season's own line, equals the observed one (bisection, 4,000
draws a line); the interval is the same inversion at the observed interval's ends.

Units and yards are drawn first and are the draws they were: **a yards prop is not re-priced.**

### Against the player-weeks

Expected player-weeks at 2, 3 and 4 or more touchdowns, summed over the same player-seasons, each
sampled on its own season-average line (3,000 draws a line), with the observed count:

| phase | weeks at >= 2 / 3 / 4 | observed | binomial (shipped) | Conway-Maxwell-Poisson, apart | **Conway-Maxwell-Poisson, given yards** | Poisson |
|---|---|---|---|---|---|---|
| rec | | 297 / 26 / 1 | 231 / 16.5 / 0.9 | 232 / 21.3 / 1.6 | 230 / 21.0 / 1.8 | 354 / 51.7 / 6.7 |
| rush | | 261 / 31 / 2 | 226 / 32.2 / 3.6 | 223 / 34.4 / 4.6 | 221 / 33.7 / 4.9 | 269 / 53.1 / 9.0 |
| pass | | 842 / 321 / 102 | 815 / 333 / 108 | 808 / 329 / 108 | 806 / 328 / 109 | 798 / 355 / 132 |

The count itself is a modest change: the binomial's expected receiving weeks at 3 or more was 16.5
against 26 observed, the Conway-Maxwell-Poisson's 21.0. Coupling does not move the count's own
tail (it is a reordering), and the table says so.

**Where the coupling matters is the total.** Observed player-weeks at or above a fantasy-points
line (PPR), 649 QB/RB/WR/TE player-seasons of at least 8 games and 8 points a game, 9,426
player-weeks, each sampled on its own season-average line (`sample_weeks`, 4,000 draws a line):

| points | >= 20 | >= 25 | >= 30 | >= 35 | >= 40 |
|---|---|---|---|---|---|
| observed | 1912 | 889 | 345 | 136 | 52 |
| shipped: binomial, apart | 1686 | 709 | 266 | 91 | 29 |
| Conway-Maxwell-Poisson, apart | 1684 | 709 | 267 | 92 | 29 |
| **Conway-Maxwell-Poisson, given yards** | **1867** | **900** | **399** | **165** | **64** |

The family change alone moves nothing at this level (the rows agree to the rounding): the ceiling
was a defect of the count and not of the total. The tail was thinned by drawing the touchdowns apart
from the yards, and that is what moves it -- the shipped sampler put the weeks at 30+ points 23%
below what was observed, 33% at 35+, 44% at 40+. The new one is on the observed at 25, and is **above
it by 16%, 21% and 23% at 30, 35 and 40**. The overshoot is reported, not tuned away: these are
in-sample season-average lines, `PER_UNIT_CV` and `COUNT_DISPERSION` are unchanged and were fitted
with the touchdowns apart, and nothing here refits them.

## 4. What reads this sampler, and so what can move

Measured by grep over `src/` and `scripts/` and by a tripwire: every function of the sampler
(`sample_weeks`, `moments`, `phase_draws`, `draw_counts`, `draw_touchdowns`, `cmp_nu`, `cmp_pmf`)
replaced by one that raises, the whole `tests/unit`, `tests/contracts` and `tests/golden` suites
run except the three files that exercise the sampler on purpose (`test_components.py`,
`test_props.py`, `test_td_count_sampler.py`). 5,174 passed; the one failure,
`test_volume.py::test_a_decomposed_line_can_be_sampled`, calls `sample_weeks` on purpose and is the
only test that reached the tripwire.

- **`hub.models.props` is the only caller in `src/`.** `components.sample_weeks` and
  `components.moments` have no caller outside tests. (`predict.moments`, which the lineup and
  championship code import, is a different function: the fitted square-root law.)
- **The draft backtest, the frozen board's arm B, the lineup optimiser and the weekly gate do not
  read it.** The draft path simulates a season from `mu`, `sd` and `skew` columns
  (`hub.draft.season.simulate_weeks`); the weekly gate and the forward arm read `td_rate`,
  `SCORING`, `EXPECTED` and `from_opportunity` from `components` and never a draw. The issue's
  premise that "every ceiling-based decision reads this distribution" and that arm B would be
  re-drawn is therefore **not so in the shipped code**, and no draft-gate figure is restated.

## 5. The single restatement

| quantity | before | after |
|---|---|---|
| a WR1's P(3+ receiving touchdowns in a week), mean 0.36 | exactly 0 | 0.00133 |
| realised touchdown variance over mean, worst case over means 0.2-6.0 | 9.64% off the constant | 0.00% |
| `sample_weeks`, a 12-point WR: sd / skew / median-over-mean / P(>= 30) | 6.447 / 0.630 / 0.943 / 0.92% | 7.369 / 0.826 / 0.902 / 2.01% |
| `sample_weeks`, a QB line (34 att, 250 yds, 1.7 TD): sd / skew / P(>= 30) | 6.625 / 0.367 / 4.21% | 7.837 / 0.496 / 7.10% |
| `sample_weeks`, an RB line (16 car, 68 yds, 0.55 TD, 3 rec): sd / skew / P(>= 30) | 6.837 / 0.535 / 3.40% | 7.910 / 0.705 / 5.72% |
| sd over sqrt(mean), the 12-point WR (`docs/weekly-spread.md` measured 2.0) | 1.86 | 2.13 |
| mean of every draw | unchanged | unchanged (the coupling reorders) |
| `player_pass_tds` and `player_anytime_td` prop prices (`hub.models.props`) | the binomial, apart | the Conway-Maxwell-Poisson, with the yards; marginal mean and dispersion unchanged |
| yards, receptions, carries and attempts props | | **unchanged for a given seed** |
| the draft gate, arm B, the weekly gate and the forward arm | | **unchanged** (section 4) |

No verdict changes and none acts on the product: no gate in `src/` reads the sampler, and the prop
log is a scorecard (`hub.models.props --report`). No prop log is committed, so there is no stored
price to restate here; a log written before this change carries the earlier `version()`.

**Prior wording, kept.** `docs/component-projection.md` and `docs/decisions.md` say touchdowns are
binomial; each carries a dated note beside that sentence.

## 6. What is not done

- **`TD_DISPERSION` is not refit.** The same player-weeks give a within-player-season variance over
  mean of 0.962 (rec), 0.957 (rush) and 0.842 (pass) against the fitted 0.83, 0.86 and 0.85. The
  rec and rush figures are nearer Poisson than the constants say, which is also why the expected
  weeks at 2 touchdowns above fall short of the observed for every under-dispersed family. How the
  fitted constants were measured is `docs/method.md`'s "dispersion fitted on observed variance"
  note, and refitting them is its own gated change; this one holds them and fixes the family.
- **The overshoot at 30+ points** (section 3) is the interaction of the new coupling with the
  other fitted spreads and is open.

## 7. The controls (rule 18)

All in `tests/unit/test_td_count_sampler.py`; each fails on the incumbent or the planted defect and
passes on the shipped sampler, and each guard was taken away in turn and a test went red:

- the shipped binomial kept as `_shipped_binomial` and shown to draw at most 2 at a WR1 rate over
  two million draws and to miss `phi` by 4 points at a mean of 0.26;
- a planted generator whose uniform is past the end of the table is capped at the last count and
  not returned one past it;
- touchdowns drawn apart (`rho = 0`) planted beside the shipped coupling: correlation 0 against
  0.25-0.5, the top yards decile carrying twice the rest's touchdowns, and the same line's P(>= 30)
  1.5 times and P(>= 35) 1.8 times higher tied than apart at one mean;
- the guards removed one at a time -- the cap, the no-mean return, the unreachable-dispersion stop,
  both refusals, the `rho == 0`, no-yards and no-touchdown returns, `rho` passed as 0, the latent
  score ignoring the yards, one search pass, a support cut to the mean, and the shipped binomial
  restored -- thirteen mutants, thirteen red, no survivors.

## 8. Version

Two names entered the digest: `components.TD_YARDS_RHO` (`fitted`) and `components.CMP_NU_MAX`
(`chosen`). `fitted_digest` `772d3bba` -> `f6b8c28a`, `config_digest(HubConfig())` `96fee74f` ->
`4fad9f78`; removing exactly those two names from the live constants reproduces both prior values.
A ledger entry written before this change is of a different key and is not compared with one after.

## 9. Amended 2026-10-15 (#472): `TD_DISPERSION` refit; sections 3, 5 and 6 restated (prior text kept)

Every table and sentence above stands as the record of #307. `docs/td-dispersion.md` is the
pre-registration and the result: the gate read **ADOPT** (six of six held-out seasons, +0.0123
nats of held-out log score per player-week, t interval [+0.011, +0.014]) and
`components.TD_DISPERSION` is now **pass 0.83, rush 0.99, rec 0.97** (was 0.85, 0.86, 0.83). The
estimator on 2019-25 reads 0.826 [0.790, 0.868], 0.986 [0.949, 1.027] and 0.971 [0.950, 0.994]; on
2022-25 alone, 0.844, 0.970 and 0.966, which is section 6's measurement to within the sample
thresholds. Section 6's first bullet ("`TD_DISPERSION` is not refit") is superseded by this.

### Section 3, the tail table, restated

The same 649 QB/RB/WR/TE player-seasons and 9,426 player-weeks (2022-25, at least 8 games and 8
points a game), `sample_weeks` on each player-season's own average line, 4,000 draws a line;
expected weeks at or above each PPR line, summed. The "registered" row is #307's sampler at the
constants it was registered against, re-measured here (it reproduces #307's "Conway-Maxwell-Poisson,
given yards" row to the sampling noise: 1,867 / 900 / 399 / 165 / 64 there).

| points | >= 20 | >= 25 | >= 30 | >= 35 | >= 40 |
|---|---|---|---|---|---|
| observed | 1912 | 889 | 345 | 136 | 52 |
| registered constants (0.85 / 0.86 / 0.83 pass / rush / rec), re-measured | 1864.5 | 898.4 | 398.3 | 163.5 | 63.9 |
| over (+) or under (-) the observed | -2.5% | +1.1% | +15.4% | +20.2% | +22.9% |
| **refit constants (0.83 / 0.99 / 0.97)** | **1873.1** | **928.3** | **427.9** | **184.3** | **75.6** |
| over (+) or under (-) the observed | -2.0% | +4.4% | **+24.0%** | **+35.5%** | **+45.4%** |

**The refit does not close the overshoot, it widens it**: 15% -> 24% at 30 points, 20% -> 36% at
35, 23% -> 45% at 40 (the same sampling noise, a point or two). Section 3's last paragraph asked
whether a corrected dispersion closes the gap; the answer is no, and the direction is the one the
mechanism gives: constants nearer Poisson put more mass on two and three touchdowns, and that mass
is the upper tail of a week's points. The gate scored the count, which the refit predicts better
in every season, and the pre-registration (`docs/td-dispersion.md`, section 2) named this table
as descriptive and not the gate; it is reported here because it is what the issue was about. The
overshoot at 30+ points is therefore **not** a dispersion error: its cause is in the other fitted
spreads (`PER_UNIT_CV`, `COUNT_DISPERSION`, `TD_YARDS_RHO`), fitted with the touchdowns apart or at
the old count law, and it is the open item it was, now with one candidate removed.

**`TD_YARDS_RHO` is not refit.** Its fit target is the sampler's own within-player Pearson
correlation of yards and touchdowns at the observed one. Re-measured on the same player-seasons
with 4,000 draws a line, at the unchanged `rho`, before -> after the dispersion moved: rec 0.349 ->
0.349, rush 0.370 -> 0.369, pass 0.478 -> 0.478 (observed 0.349 / 0.372 / 0.477), so the coupling
still implies the correlation it was fitted to.

### Section 5, the restatement, redone

The lines of section 5's table are not stored with it and its figures do not reproduce from the
lines it names, so both columns are re-measured here, at one stated line each and 1,000,000
draws (seed 7): a 12-point WR (4.65 receptions, 53.9 yards, 0.326 touchdowns: the phase's yards
per reception and `TD_RATE`), the QB line (34 attempts, 250 yards, 1.7 touchdowns) and the RB line
(16 carries, 68 yards, 0.55 touchdowns, 3 receptions).

| quantity | before (pass 0.85 / rush 0.86 / rec 0.83) | after (0.83 / 0.99 / 0.97) |
|---|---|---|
| a WR1's P(3+ receiving touchdowns in a week), mean 0.36 | 0.00133 | 0.00495 (the Poisson: 0.00595) |
| realised touchdown variance over mean, worst case over means 0.2-6.0 | 0.002% off the constant | 0.003% off the constant |
| `sample_weeks`, the 12-point WR: sd / skew / median-over-mean / P(>= 30) | 7.166 / 0.811 / 0.904 / 1.77% | 7.345 / 0.924 / 0.892 / 2.17% |
| `sample_weeks`, the QB line: sd / skew / P(>= 30) | 6.933 / 0.546 / 4.14% | 6.880 / 0.536 / 4.02% |
| `sample_weeks`, the RB line: sd / skew / median-over-mean / P(>= 30) | 6.771 / 0.823 / 0.922 / 1.84% | 7.017 / 0.946 / 0.903 / 2.34% |
| sd over sqrt(mean), the 12-point WR (`docs/weekly-spread.md` measured 2.0) | 2.069 | 2.121 |
| mean of every draw | unchanged | unchanged |
| `player_pass_tds` P(over 1.5) / P(over 0.5) / sd, the QB line | 0.5214 / 0.8478 / 1.2028 | 0.5232 / 0.8520 / 1.1883 |
| `player_anytime_td`, a WR (5 receptions, 62 yards, 0.38 touchdowns) | 0.3475 | 0.3259 |
| `player_anytime_td`, an RB (16 carries, 68 yards, 0.55 touchdowns, 3 receptions, 22 yards, 0.12 touchdowns) | 0.5156 | 0.4912 |
| yards, receptions, carries and attempts props | | **unchanged for a given seed** (checked: a QB's pass-yards and a WR's receiving-yards price are equal to the last digit) |
| the draft gate, arm B, the weekly gate and the forward arm | | **unchanged** (section 4; `docs/weekly-forward.md`, 2026-10-15 amendment) |

The anytime prices fall because the rushing and receiving constants went from 0.86 and 0.83 to
0.99 and 0.97: a narrower law puts more of its mass on exactly one touchdown, so "at least one"
was priced high. No gate in `src/` reads these, and the prop log is a scorecard; no stored price is
restated. `config_digest` `4fad9f78` -> `839ceda9`, `fitted_digest` `f6b8c28a` -> `9f2d0b0c`: one
name moved by value, `components.TD_DISPERSION` (`tests/unit/test_config.py`).

## 10. Amended 2026-10-15 (#473): the 30+/35+/40+ overshoot, diagnosed; no constant is refit (prior text kept)

Sections 3 and 9 stand as the record of what was measured. Section 9's last paragraph says the
overshoot "is in the other fitted spreads (`PER_UNIT_CV`, `COUNT_DISPERSION`, `TD_YARDS_RHO`)".
That sentence is **superseded** by this section; it is the diagnosis #473 asked for, run before any
refit was designed. **It is in-sample throughout** (the 649 QB/RB/WR/TE player-seasons, 9,426
player-weeks, 2022-25, each sampled on its own season-average line); nothing here is held out and
nothing here is a gate. All figures were measured on a copy of `data/` made into a worktree for the
purpose (the scripts are one-shot and live outside the repo; the procedure is stated so it can be
rerun).

### 10.1 The table's own method reads high when the sampler is exactly right

A control (`docs/method.md` rule 18) the table never had: plant weeks whose truth *is* the shipped
sampler, on the real lines (`phase_draws` per player-season, as many weeks as the player played),
take the average line of the planted weeks, and read the same table. Eight independent plants:

| points | >= 20 | >= 25 | >= 30 | >= 35 | >= 40 |
|---|---|---|---|---|---|
| table, real data (section 9, refit constants) | -2.0% | +4.4% | **+24.0%** | **+35.5%** | **+45.4%** |
| table, truth = the shipped sampler (mean of 8 plants; sd across plants) | +2.1% (0.7) | +6.4% (2.5) | **+11.1%** (4.5) | **+20.3%** (5.0) | **+32.3%** (13.4) |
| real minus planted-null | -4 | -2 | +13 | +15 | +13 (1 sd) |

So roughly half of the 30+ and 35+ overshoot, and all of the 40+ one to within its noise, is the
method: the line is the average of the very weeks it is then asked to explain, and the tail
probability is convex in the line. (The two candidate causes, self-inclusion and convexity in a
noisy mean, are not separated here.) What the sampler does overshoot is **about 13-15% at 30 and
35 points, with 20-25 slightly under**: a shape that is lighter-tailed than the sampler's.

### 10.2 The variance constants are not the fault

Pooled within-player-season moments, observed against the sampler at the same lines (4,000 draws a
line), per phase: the constants already reproduce the observed spreads.

| phase | units var/mean obs / sim | yards variance sim over obs | TD var/mean obs / sim | corr(units, yards) obs / sim | corr(yards, TDs) obs / sim |
|---|---|---|---|---|---|
| pass (142 player-seasons) | 2.67 / 2.45 | 0.95 | 0.84 / 0.83 | 0.71 / 0.84 | 0.47 / 0.47 |
| rush (219) | 2.11 / 2.38 | 1.05 | 0.97 / 0.99 | 0.72 / 0.72 | 0.36 / 0.38 |
| rec (453) | 1.19 / 1.20 | 1.01 | 0.96 / 0.97 | 0.76 / 0.82 | 0.35 / 0.35 |

The yards-per-unit CV implied by the observed within-player yards variance (given the observed
unit variance) is 1.01 / 1.51 / 0.77 against `PER_UNIT_CV` 1.00 / 1.47 / 0.76. The one visible miss
is rush `COUNT_DISPERSION`, 2.39 against 2.11 observed.

### 10.3 Where it sits: the unit counts' upper tail

Swapping one component of the sampler for the player's own observed weeks, in the causal order
units, then yards given units, then touchdowns given yards, on the same lines (8,000 draws a line;
a node is "units / yards / touchdowns", `e` observed weeks resampled whole, `s` simulated):

| node (U, Y, T) | >= 30 | >= 35 | >= 40 |
|---|---|---|---|
| observed | 345 | 136 | 52 |
| `sss`, the sampler | 429 | 186 | 76 |
| `ess`, units observed | 388 | 158 | 61 |
| `ees`, units and yards observed | 367 | 139 | 50 |
| `eee`, everything observed (the benchmark) | 335 | 135 | 50 |

The benchmark is itself 3% under the observed (resampling a player's ~15 weeks shrinks the variance
by (n-1)/n), so every step carries a floor, measured by running the same chain on the two planted
nulls of 10.1 (weeks at 30 / 35 / 40; two plants):

| step, weeks removed | real data | planted null (the floor) | net |
|---|---|---|---|
| units, >= 30 / 35 / 40 | 40.0 / 28.6 / 15.5 | 11.7-15.4 / 6.9-10.0 / 5.6-6.8 | **+25 / +20 / +9** |
| yards given units | 21.0 / 18.6 / 11.6 | 8.6-11.5 / 10.7-11.2 / 6.3-8.1 | +11 / +8 / +4 |
| touchdowns given yards | 32.2 / 5.7 / -0.2 | 7.4-31.7 / 10.2-16.4 / 17.2-20.6 | not resolved: the floor spans the real figure at 30 and exceeds it at 35 and 40 |

**The excess the sampler carries over the floor is in the unit counts, then in yards given units.
The touchdown step is not resolved at 30+ and is zero or negative at 35+ and 40+; neither
`TD_DISPERSION` nor `TD_YARDS_RHO` is where it is.** The component marginals say the same thing,
observed against sampler, real minus the planted-null overshoot of the same cut:

| cut | real over | net of the null |
|---|---|---|
| pass attempts >= 45 / >= 50 | +39% / +49% | +30% / +40% |
| rush carries >= 25 / >= 30 | +26% / +90% | +17% / +76% |
| rec receptions >= 10 / >= 12 | +14% / +15% | +10% / +6% |
| pass yards >= 350 / >= 400 | +45% / +119% | +33% / +99% |
| rec yards >= 125 / >= 150 | +14% / +27% | +11% / +16% |
| rush TDs >= 3 | +52% | +44% (rare: about 33 observed weeks) |
| pass TDs >= 2, 3; rush TDs >= 1, 2; rec TDs >= 1 | within +-5% | within +-5% |

The count law at a matched variance has a heavier upper tail than the games do: a negative
binomial with variance over mean 2.4 puts mass on 50 attempts and 30 carries that real games, which
a score and a clock bound, do not reach. The variance is right (10.2) and the tail is not, which is
a property of the family and not of its constant. The yards inherit it (a Gamma compounded on the
count).

### 10.4 No constant fixes it

`COUNT_DISPERSION - 1` scaled by k and `PER_UNIT_CV` by c, expected weeks at 20 / 25 / 30 / 35 / 40
(3,000 draws a line), against the observed scaled by the planted-null overshoot of 10.1 (the
target: 1952 / 946 / 383 / 164 / 69):

| k, c | >= 20 | >= 25 | >= 30 | >= 35 | >= 40 |
|---|---|---|---|---|---|
| 1.00, 1.00 (shipped) | 1874 | 928 | 427 | 186 | 77 |
| 0.75, 0.90 | 1848 | 896 | 405 | 171 | 68 |
| 0.50, 0.90 | 1825 | 879 | 390 | 162 | 64 |
| 0.50, 0.80 | 1816 | 865 | 379 | 156 | 60 |

The settings that reach the 30+ targets halve the unit overdispersion that 10.2 shows is right and
leave 20 and 25 points 6-7% **under**: a constant that cuts the tail cuts the middle with it. A
gated refit of `PER_UNIT_CV` or `COUNT_DISPERSION` on tail calibration would adopt a trade, not a
repair, and a refit on the held-out log score of weekly points would be fitting the same two
numbers to a family whose tail is the wrong shape. **No refit is registered, run or adopted;**
`PER_UNIT_CV`, `COUNT_DISPERSION`, `TD_DISPERSION` and `TD_YARDS_RHO` are unchanged, nothing in
`src/` moves, `config_digest` is `839ceda9` as before, and the forward arm is not touched.

### 10.5 The controls (rule 18)

The decomposition was run on planted weeks with the excess put in one component (truth sampler =
shipped, except the one named), chain steps at weeks removed, >= 30:

| planted | units | yards given units | touchdowns given yards |
|---|---|---|---|
| nothing (two plants) | 11.7-15.4 | 8.6-11.5 | 7.4-31.7 |
| truth units Poisson (`COUNT_DISPERSION` 1.0) | **70.9** | 10.9 | 23.3 |
| truth yards-per-unit CV halved | 10.8 | **46.0** | 21.4 |
| truth touchdowns drawn apart from yards (`TD_YARDS_RHO` 0) | 17.8 | 12.6 | **113.3** |

Each planted component takes the largest share, and the other two stay in the null band. The first
design of the decomposition scored touchdowns observed against yards simulated as a node, which
carried the sampler's own coupling into the "observed" touchdowns; the touchdown plant was then split
between yards and touchdowns (58 and 53 weeks). The control caught it, and the chain above, which
only observes touchdowns once the yards are observed, is the replacement.

### 10.6 What is open

The family of the unit counts: an overdispersed law with a bounded upper tail (as #307 replaced a
binomial of a ceiling with the Conway-Maxwell-Poisson for the touchdown count), scored on held-out
weekly units, is its own gated change and is not a constant refit. Any further claim about the
tail should be made with a leave-one-out line, not the season average the weeks were part of.
