# Researched improvements

Written 2026-08-24, after the draft path was frozen-ready. Every item below is grounded in
something checked in this repo, not in general advice. Where a number appears, it was computed
here. Ordered by evidence strength, not by size.

Two rules from `CONTEXT.md` govern what "done" means for each: a **signal** is screened for
predictive power beyond consensus; a **model** is gated against the simplest thing that already
works. Every item names which it is.

---

## 1. `MARGIN_SD` is asserted, and the data disagrees

**Model. The strongest finding here, and the cheapest to act on.**

`hub/models/market.py` sets `MARGIN_SD = 13.5` under a comment reading *"Stable across decades
of NFL results"*. There is no fit, no interval and no write-up. It converts every closing
spread into a win probability, so it is the single most load-bearing number in the NFL path.

The repo already fetches the data to measure it. From `nflverse.load("schedules", …)`:

    completed games with a closing spread : 854   (2022-24)
    sd(result - spread_line)              : 12.470
    mean(result - spread_line)            : +0.700
    se(sd) ~ sd / sqrt(2n)                : 0.30

13.5 sits **3.4 standard errors high**. Not noise. The effect is to widen every game toward a
coin flip:

| favourite by | at sd 13.5 | at sd 12.47 | difference |
|---|---|---|---|
| 3 points | 58.8% | 59.5% | +0.7 pp |
| 7 points | 69.8% | 71.3% | +1.5 pp |
| 10 points | 77.1% | 78.9% | +1.8 pp |

Small per game, systematic across every game, and in the same direction every time — which is
exactly the shape a log-loss scorer punishes.

Worse, `hub.models.market` is in `config.FITTED_MODULES`, so `MARGIN_SD` is hashed into the
model version *as though it were fitted*. ADR-0006 draws the line between a measurement and a
choice; this number is currently on the wrong side of it while being registered on the right
one.

**Also measure the +0.70 mean residual** before assuming it is home-field: `spread_line`'s sign
convention and whether `result` is home-relative both need checking, and a sign error there
would look exactly like a small bias.

**Do:** fit it, with an interval, over as many seasons as nflverse retains. Write it up like
`docs/weekly-spread.md`. Gate: does the fitted value improve log-loss against the 13.5 baseline
on held-out seasons? If not, keep 13.5 and record *that* — an asserted number that survives a
fit is no longer asserted.

---

## 2. Conformal is built, tested, and connected to nothing

**Model, currently inert — the same category the lineup optimiser was in before ADR-0012.**

`hub/models/conformal.py` is 144 lines at 88% coverage. Outside its own module and tests,
nothing imports it. The only references in `src/` are a `conformal_alpha` setting in
`config.py` and a docstring in `board.py` saying `ecr_sd` is *"a crude prior on uncertainty
until conformal lands"*. It has not landed.

`docs/gaps.md` already lists the missing CLI. But a CLI is not the gap — a *caller* is. Nothing
in the weekly slate asks for an interval.

**Do:** decide what consumes it before building more of it. The honest options are the weekly
slate publishing intervals alongside probabilities, or nothing. If nothing, say so in an ADR
rather than leaving a well-tested module implying it is in the pipeline. The README used to
advertise it; that has been fixed, but the module still reads as shipped.

---

## 3. Correlation covers teammates and nothing else

**Model.** `TEAMMATE_RHO` carries three edges — QB-WR +0.232, QB-TE +0.225, QB-RB +0.054 —
measured within a team.

Nothing models correlation *between* teams in the same game. A shootout lifts both
quarterbacks, both sets of receivers and neither defence, and in a lineup holding players from
both sides of one game that is real, unpriced covariance. `docs/correlation.md` gates the
teammate work on an interval-coverage test; the same test would show whether opponent
correlation matters.

The data is there: `pbp` and `schedules` give every player-week its game, and the harness to
measure a correlation already exists.

**Do:** measure opponent-game correlation on standardised weekly points, the same way
`TEAMMATE_RHO` was measured. It either matters or it does not, and one number settles it.

---

## 4. The scheme layer is fetched and unused

**Signal, unscreened.** `participation` and `ftn_charting` landed today: personnel, formation,
box count, coverage shell, man/zone, motion, play action, screens, blitzers, pressure,
time-to-throw. 45k and 47k rows a season, 2025 included.

Nothing reads them yet, deliberately — the screen is post-draft work.

**The prior is poor and should be stated before screening**, because writing it down afterwards
is how a null becomes "promising". Team scheme is among the *least* persistent things
available: coordinators turn over, and a season-N tendency predicting season N+1 must survive
that. Five of five previous screens came back null.

**Do:** screen scheme features the way `signal-screens.md` screens everything else — partial
correlation beyond ECR, pre-registered, no `src/` written until it clears. The most promising
framing is not "does scheme predict points" but "does scheme predict **volume**", since volume
is the half of the signal that already carries forward.

---

## 5. Injury pricing is one coefficient for three states

**Model.** `INJURY_BETA` prices OUT, DOUBTFUL and INJURY_RESERVE at the same −1.631 ppg, and
`durability.py` says why: IR has no coefficient of its own, so it borrows Out's and thereby
understates it. QUESTIONABLE is carried and deliberately unpriced, because the August board
carries it at 12.6% against 2.9% at week 1 — the fitted number would come from a much sicker
group.

Both compromises are honest and both are stated. They are also the smallest model in the repo
doing the most work: injury status moves a player up to 20% of his ADP under ADR-0011.

`load_injuries` gives weekly `report_status`, `practice_status` and `report_primary_injury`.
That supports the model the grilling settled on: `P(plays week N | injury type, practice
status, weeks since onset)` as one estimand rather than two.

**Do:** fit it, gated against `INJURY_BETA`'s −1.631. The gate matters — a richer model that
does not beat one number is not an improvement, it is a liability with more parameters.

---

## 6. `sd = k·sqrt(mu)` makes half the system inert

**Model, and already documented.** [ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md)
measured the lineup optimiser at +0.00 points a game because within a position `sd` is a
deterministic function of `mu` (correlation 0.985). Two players projected at 12 points a game
are not equally volatile in reality, and the square-root law cannot say so.

This is the strongest argument for the usage and component layers, and worth restating here
because it is easy to file the null under "the optimiser does not work". It works. It is being
handed nothing to work with.

**Do:** the usage model, gated against pick-anchored volume, with `volume.py` demoted to the
cold-start path for rookies. Then re-run `hub.season.lineup_gate`.

---

## Codebase

### 7. `board.py` is still the hot spot

367 statements, **46% covered**, and `build()` still fetches, prints and degrades in one
function — the one deviation from ADR-0003 that this session only partly addressed. Its
uncovered lines are the network path and the report block.

The honest split is between fetching and assembling, and
[ADR-0003](adr/0003-make-now-dagster-in-october.md) already says the Dagster port wants that
seam anyway. Doing it twice would be waste; doing it as part of the port would not.

### 8. The board bypasses `hub.store`

`build()` writes `data/processed/draft_board.parquet` directly. `hub.store` exists, has a Hive
layout and a `write()`, and is used by the fetch layer. The board — the most-read artifact in
the repo — is the one thing that does not go through it, so it gets no partitioning and no
manifest.

Low urgency, but it is why `hub.inspect` has a special case for a bare-name dataset.

### 9. One naive datetime in production

`market.py:84` calls `datetime.now()` without a timezone. Twenty-five more are in tests and do
not matter. One does: a prediction timestamped without a zone is ambiguous in a repo whose
whole claim is that December can audit August.

`DTZ` is deliberately excluded from the ruff select for now, because fixing the test sites
changes stored fixtures. Fix the production site; leave the rest.

### 10. Coverage floors that mean something

Coverage is 80.2% against an 80% gate — passing by 0.2 points, which is a gate about to start
failing on noise. The low modules are `backtest` 30%, `board` 46%, `espn` 58%, `live` 64%,
`tune` 64%.

`backtest` at 30% is the one that matters: it is the module whose entire justification is that
a measurement nobody can re-run is worthless. Its orchestration got offline tests today; its
`main()` did not.

---

## What is deliberately not on this list

**Anything that tries to beat a market.** Objective 4 was retired on 2026-08-24 after six
measured failures. Improvements to the *props* or *spread* paths are worth making for
calibration, and their gate is log-loss against the market, not profit against it.

**Re-litigating championship equity.** [ADR-0009](adr/0009-championship-equity-does-not-pick.md)
closes it, and says explicitly that reopening means re-running the harness, not re-arguing.
Item 6 is the path back if there is one.
