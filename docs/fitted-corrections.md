# The five board corrections, refitted against the column they correct

**Fitted 2026-09-07**, `hub/draft/fit_corrections.py`, issue #47. This is the first committed
fitting code for any of the five: the tables in [td-luck.md](td-luck.md) and
[durability.md](durability.md) are the output of a fit that had no harness and cannot be
re-run.

**Nothing here changes a shipped constant.** That is #48's job and this is its input. Where a
refit disagrees with the board, the disagreement is reported and the board keeps running on
what it has.

**#48 closed 2026-09-10, and this is what it did.** One of the five moved, and it moved to
zero rather than to its refit:

| coefficient | disposition | #48 |
|---|---|---|
| `TD_LUCK_BETA["QB"]` | sign-reversed, resolved | **withdrawn** — `TD_LUCK_BETA` is now `{}` |
| `TD_LUCK_BETA["WR"]` | reproduced, underpowered | **withdrawn** with it, on #186's held-out evidence rather than on its own disposition |
| `BETA["QB"]` (missed) | reproduced | unchanged, no flag |
| `BETA["WR"]` (missed) | unreproduced, resolved | unchanged, **applied behind a flag** |
| `INJURY_BETA["OUT"]` | unreproduced, underpowered | unchanged, **applied behind a flag** |

The rule the ticket set, and what each part of it cost. A **reproduced** coefficient stands.
An **unreproduced** one goes on applying, because withdrawing a correction mid-season is
itself an unmeasured change to the board and neither refit is a licence to move a shipped
number — but it no longer applies silently: `hub.draft.board.CORRECTION_FLAG` carries the
disposition and `hub.draft.report` prints it beside the ranking, so an operator sees which of
the numbers under THE PICK is disputed and how. A **sign-reversed** one is zeroed, because a
sign flip is a bug rather than a finding.

The touchdown-luck pair is the one place #48 went further than its own rule, and #186 is why:
`td_luck.WR` is *reproduced* on this page and would have stood, but the question #186 asked
was whether the correction earns its place at all, and held out neither coefficient does. The
next section is that decision.

`config_digest` moved `6bdcb663` → `9fc5b0af` and `fitted_digest` `8c248a6e` → `d9caf498`,
restated in `tests/unit/test_config.py`. A correction that stopped applying is a model change
and ADR-0006 wants the version to move for it.

## The result

`target ~ baseline + signal`, per position, over 4,595 player-seasons and eight outcome
seasons. `target` is next-season points per **team** game; `baseline` is the projection the
correction is added to. Intervals are season-clustered through
`experiment.summarise(cluster=SEASON_CLUSTER)`; the MDE beside each is
`minimum_detectable_effect` on that same bootstrap's standard error.

| coefficient | shipped | refitted | 95% CI (season-clustered) | MDE | disposition |
|---|---|---|---|---|---|
| `TD_LUCK_BETA["QB"]` | **−0.540** | **+0.504** | [−0.150, +0.939] | 0.946 | **sign-reversed**, resolved |
| `TD_LUCK_BETA["WR"]` | −0.286 | −0.117 | [−0.323, +0.095] | 0.342 | reproduced, **underpowered** |
| `BETA["QB"]` (missed) | −0.457 | −0.486 | [−0.591, −0.377] | 0.177 | **reproduced** |
| `BETA["WR"]` (missed) | −0.151 | **−0.327** | [−0.394, −0.261] | 0.110 | **unreproduced**, resolved |
| `INJURY_BETA["OUT"]` | −1.631 | −0.833 | [−1.475, −0.211] | 1.051 | unreproduced, **underpowered** |

Population, for all five: prior-season players at a drafted position who cleared the shipped
eligibility filter (`regression.MIN_GAMES` = 6 games, `durability.MIN_PPG` = 5.0 ppg), outcome
seasons **2018–2025**, eight season clusters. Per-coefficient row counts are in the table
below. Baseline `base_blend`; see the next section for what that is and is not.

**"Reproduced" is not "confirmed".** Two of the five are reproduced only because the interval
is wide enough to contain the shipped number *and* zero. The `resolved` column asks the other
question — whether the gap between shipped and refitted exceeds the MDE — and for `missed.QB`
it does not, so that row is agreement rather than evidence of agreement. At eight clusters
`t(0.975, 7)` is 2.365 against `z`'s 1.960, so a disposition taken against the pre-#45 MDE
would have called two of these five resolved that are not.

## The axis, which is the finding

Both shipped comments name their fit as `ppg_next ~ proj_ppg + signal`. **`proj_ppg` is not
what either correction touches.** `correct_projection` adds `beta * signal` to `proj_blend`,
and `predict.blend()` defines that as the mean of `proj_ppg` and `xfp_per_game`.

Those two halves disagree about touchdowns *by construction*. `xfp_per_game` is an
expectation — `hub.models.components` says in as many words that it is already regressed and
that `td_luck` is defined as actual minus expected — while `proj_ppg` carries a player's own
realised touchdowns forward. So the same residual has opposite signs against the two halves,
and the coefficient applied to their average is neither of them:

| signal, position | vs `base_xfp` (expectation) | vs `base_carry` (carries TDs) | vs `base_blend` |
|---|---|---|---|
| td_luck QB | **+0.789** | **+0.249** | +0.504 |
| td_luck WR | **+0.190** | **−0.376** | −0.117 |
| missed QB | −0.524 | −0.496 | −0.486 |
| missed WR | −0.320 | −0.351 | −0.327 |
| designation | −0.739 | −0.913 | −0.833 |

**Touchdown luck moves by more than its own shipped value when the baseline changes; missed
games barely moves at all.** That is why the two durability numbers reproduce and the two
touchdown numbers do not, and it is the same defect [pick-noise.md](pick-noise.md) found one
layer over — fitted on one axis and applied on another. There it was ranks fitted and picks
predicted; here it is one summand fitted and the sum corrected.

## What `base_carry` is, and the honest limit

`proj_ppg` for a past season **cannot be recovered**. `hub.draft.adp_history` records the
verification: ESPN does not retain a past season's draft-time view, and the projection beside
the ADP goes with it. That is the deeper reason these five had no committed harness — the fit
as its comments describe it is not reproducible by anyone, including whoever ran it.

So the harness builds three baselines from nflverse and reports every fit against all three.
`base_xfp` is exactly `board.expected_points`'s `xfp_per_game`, the recoverable half.
`base_carry` is prior-season realised points per game played — **a stand-in, not a market
projection**, and the only reconstructible column that carries a player's own touchdowns
forward the way one does. `base_blend` is their mean, which is `blend()`'s shape with the
stand-in in `proj_ppg`'s place, and is the headline because it is the shape of the column
actually corrected.

What the stand-in buys is a **bracket** rather than an answer:

- **`td_luck.QB` is resolved anyway.** It is positive at *both* ends of the bracket, so no
  ESPN projection sitting between them could produce the shipped −0.540. The sign reversal
  holds whatever the unrecoverable half would have said.
- **`td_luck.WR` is not resolvable from this repo's data.** It changes sign inside the
  bracket (+0.190 to −0.376). The shipped −0.286 is reproduced against `base_carry` and
  sign-reversed against `base_xfp`, and nothing here can say which is nearer the truth.
- The three durability rows are flat across the bracket, so the stand-in does not matter
  for them.

## The target, which was already right

`total_next / 17`, points per team game. Not a choice made here: `hub.draft.season`
draws every one of `REG_SEASON_WEEKS + PLAYOFF_ROUNDS` weeks from `mu` with no games-played
term anywhere, so `proj_blend` is *consumed* as a per-team-game rate.
[calibrate.py](../src/hub/draft/calibrate.py) fixes the same convention for `TALENT_CV` and
gives the reason — "a player who misses ten weeks really did deliver close to nothing".

`docs/durability.md` chose this target too, and its own table shows what the other choice
costs: −0.087 per game played against −0.186 per team game, a factor of 2.1 on the same data.
The target was the half that was right.

**A player with no outcome row scores zero rather than dropping out**, for `calibrate`'s other
reason: the vanished players are the busts, and dropping them reads as a more predictable
season than the season was.

## Two counts, reconciled

`docs/durability.md` publishes the designation as "1,263 player-seasons" in one place and
"n = 27" in another, and nothing says those are one fit counted two ways. Both are now on
every `Fit`:

| coefficient | `n` (regression rows) | `n_signal` (rows carrying it) |
|---|---|---|
| td_luck QB | 339 | 339 |
| td_luck WR | 1,366 | 1,356 |
| missed QB | 429 | 388 |
| missed WR | 894 | 785 |
| **designation** | **4,595** | **68** |

The designation's coefficient rests on **68 Out-or-Doubtful player-seasons inside a
4,595-row regression**. The large number is the regression's; the small one is the evidence.
That is also why its MDE (1.051) is larger than its own point estimate (0.833): the run can
say the shipped −1.631 is outside the interval, and cannot pin the level.

## Is anything sitting at a bound?

**No — not in the sense [pick-noise.md](pick-noise.md) found.** None of the five comes from a
constrained fit, so there is no analogue of a shipped intercept turning out to be exactly
`MIN_SIGMA`. `Fit.at_a_bound` checks it mechanically on every run rather than leaving it to
whoever next reads the table, and it fires on none of them.

Two things of a related shape are worth naming, and neither is an optimiser bound:

- **`INJURY_BETA` is one number serving three keys.** OUT, DOUBTFUL and INJURY_RESERVE all
  read −1.631, and `docs/durability.md` says plainly that IR *borrows* the Out coefficient
  because nobody on injured reserve appears on a practice report. A borrowed coefficient
  applied to a population it was not fitted on is the same family of defect as a wrong axis,
  and the refit cannot address it: a week-1 report has no IR rows to fit either.
- **`correct_projection` clips at zero.** A −1.631 markdown on a per-game projection floors
  every player projected under 1.631 ppg at exactly nothing, so at the bottom of the board the
  constant's size stops mattering and the clip decides. Small in effect and real in kind.

## Held out, which is the arm #48 acts on

Fitting only on strictly earlier seasons — `experiment.expanding_seasons`, asserted on the
split by a guard that `tests/contracts/test_guards_are_load_bearing.py` proves by deleting —
three arms on the same held-out rows. `shipped` is the board's own constant applied to the
no-correction fit, which is the model the board actually runs.

| coefficient | plain | fitted | shipped | shipped beats plain |
|---|---|---|---|---|
| td_luck QB | 4.976 | 4.947 | **5.124** | 2/7 seasons |
| td_luck WR | 2.749 | 2.749 | **2.763** | 1/7 |
| missed QB | 4.921 | **4.516** | **5.189** | 1/7 |
| missed WR | 3.431 | **3.238** | **3.256** | **7/7** |
| designation | **2.615** | 2.616 | 2.618 | 1/7 |

**Only `missed.WR` earns its place out of sample**, and it is the one the refit says is
*too small* rather than wrong. The two touchdown constants and the QB durability constant each
score **worse than applying no correction at all** on these seasons: `shipped` loses to
`plain` on the mean, and in five of seven years for `td_luck.QB` and six of seven for the
other two.

`missed.QB` is the interesting row: the coefficient reproduces cleanly, and applying it still
hurts held-out error while the *refitted* version helps. That is the axis again — a
coefficient of the right size added to the wrong column.

## Does the touchdown-luck correction earn its place at all (#186)

**Decided 2026-09-10: no. It is withdrawn, and #48 is where it stops applying.**

#186 was written as a refinement — estimate the shrinkage by empirical Bayes on split-half
reliability, and add a goal-line term to the expectation — and was re-scoped on 2026-09-09 to
the prior question, which is `docs/method.md` rule 8: *compute the ceiling before chasing the
gap.* The three options were remove it, keep it at a measured shrink, or keep it per position
where the sign is defensible. Held out, all three resolve to the same answer.

**The evidence, and it is the held-out table above rather than the shipped constants.**

| | plain | shipped | shipped beats plain |
|---|---|---|---|
| `td_luck.QB` | 4.976 | **5.124** | 2/7 seasons |
| `td_luck.WR` | 2.749 | **2.763** | 1/7 |

Taking the three options in turn:

- **Keep it per position where the sign is defensible.** There is no such position. QB is
  sign-reversed at *both* ends of the `base_carry`/`base_xfp` bracket (+0.789 and +0.249), so
  no projection sitting between them could produce the shipped −0.540 — that is the strongest
  form the bracket can give. WR changes sign inside the bracket, so it is not resolvable from
  this repo's data at all, and an unresolvable coefficient is not a defensible sign.
- **Keep it at a measured shrink.** `shrink_curve` in `hub/draft/fit_corrections.py` scores
  the shipped constant at each factor between nothing and whole, on the same splits and the
  same held-out rows as the table above, so this option is an arm rather than an assertion.
  Its two ends are `plain` and `shipped` by construction, and for both coefficients `plain`
  wins — so the shrink that minimises held-out error is **zero**, which is removal. A
  negative factor is deliberately not on the sweep: flipping a sign is a refit, not a
  shrinkage, and #48's rule for a sign-reversed coefficient is that it is zeroed.
- **Remove it.** What the other two reduce to.

**The ticket's shrinkage premise was also the wrong way round, and this is worth separating
from the decision.** #186 argued that regressing a noisy estimate at one hundred per cent
assumes it is measured without error. [td-luck.md](td-luck.md) records the fitted optimal
shrink on the touchdown rate as **1.0, not a partial one**. Those are two different shrinks —
one on the player's own touchdown rate inside the signal, one on the price the board puts on
the resulting residual — and conflating them is what made a refinement look available. They
happen to point the same way here, but the decision above does not rest on either: it rests
on the held-out table, where the price loses to charging nothing.

**What this does not withdraw.** `td_luck` stays on the board as a column and stays in the
report. Nothing measured about the *signal* changed: touchdown rate still has no
year-over-year persistence, and the room still prices it. What is withdrawn is the claim that
this repo knows the size of that mispricing well enough to move `proj_blend` by it. That is
the same shape as [ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md) and
[ADR-0016](adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md): shown, and never
ranked on.

**#225 is blocked on this and stays blocked.** The goal-line refinement would pay an
OOM-hazard play-by-play pull — `yardline_100` is not among the ten columns
[nflverse.py](../src/hub/fetch/nflverse.py) requests, and
[panel.py](../src/hub/models/panel.py) records a narrowed four-season `load_pbp` killed by
the OOM reaper — to sharpen a correction that loses to doing nothing. That is chasing a gap
under a ceiling that is currently below zero.

## What this does not say

It does not say the board is wrong today. Every number here is fitted against a stand-in for
half of `proj_blend`, on a population assembled from nflverse rather than from the ESPN board
the corrections run on, over eight seasons that include four the shipped fits never saw. A
refit disagreeing with a fit nobody can reproduce is not the same as the refit being right.

What it does say, and says with the bracket closed:

1. `TD_LUCK_BETA["QB"]` has the **wrong sign** against every baseline this repo can build.
2. `BETA["WR"]` is **about half** the size the same data supports, and it is the only one of
   the five that helps out of sample.
3. `INJURY_BETA["OUT"]` is about **twice** the size the data supports, on 68 observations.
4. `TD_LUCK_BETA["WR"]` is **not resolvable** without a projection that no longer exists.

## A caution about re-running

Two runs twenty minutes apart returned panels of 4,595 rows and per-position counts differing
by one, because nflverse revises its archive in place. `hub.fetch.nflverse` has `as_of`
pinning for exactly this and the harness does not currently pass one; a number quoted from
this page should be re-derived rather than assumed stable to the last digit. Left as a
follow-up rather than fixed here, and named rather than discovered later.

## Reproduce

```bash
uv run python -m hub.draft.fit_corrections --fit
```

Needs nflverse (`ff_opportunity`, `player_stats`, `injuries`) and no credential. It refuses
with a sentence and a non-zero exit when they are absent, like every other entry point here.

```bash
uv run python -m hub.draft.fit_corrections --fit --seasons 2022,2023,2024,2025
```

The four-season span the shipped constants were fitted over. Four clusters, where
`minimum_detectable_effect`'s t quantile is 3.182 against the normal's 1.960 — the 1.44x #45
corrected — so every MDE above roughly doubles and nothing is resolved.

Either run prints the shrink sweep (#186) under each coefficient's walk-forward line, as
`x0.00 <mae> (<seasons beaten>/<seasons>)` through `x1.00`, with the winning factor named at
the end. `x0.00` is the `plain` column of the held-out table and `x1.00` is its `shipped`
column, so the sweep can be read against that table directly.
