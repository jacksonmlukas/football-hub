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
`test_props.py`, `test_td_count_sampler.py`).

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
