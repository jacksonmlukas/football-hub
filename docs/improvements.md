# Researched improvements

Written 2026-08-24, after the draft path was frozen-ready. Every item below is grounded in
something checked in this repo, not in general advice. Where a number appears, it was computed
here. Ordered by evidence strength, not by size.

Two rules from `CONTEXT.md` govern what "done" means for each: a **signal** is screened for
predictive power beyond consensus; a **model** is gated against the simplest thing that already
works. Every item names which it is.

---

## 1. `MARGIN_SD` is asserted, and the data disagrees — DONE 2026-08-24

**Fitted and adopted: 13.5 to 12.741.** See [margin-sd.md](margin-sd.md). Two corrections to
what this item originally claimed, both found by doing the work:

* The full-sample sd is **13.214**, not the 12.470 quoted below — that was a three-season
  slice, and per-season sd runs 11.5 to 14.4. The direction held at every window (13.5 sits
  2.6 to 4.6 se high) but the magnitude did not.
* **The predictive gain is negligible**: +0.0002 held-out log-loss, about 0.057 nats a season.
  This item implied the error mattered for accuracy. It does not. It mattered for provenance —
  a number hashed into every model version as fitted, which had never been fitted.

The original text follows.

### Original text


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

## 2. Conformal is built, tested, and connected to nothing — PARTLY DIAGNOSED 2026-08-25

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

### 2026-08-25: it also could not run

Running the documented CLI for the first time against the real store:

    _duckdb.BinderException: Referenced column "margin_actual" not found

`load_scored` selected `margin_actual` from `preds`, and there is no such column. `hub.publish`
writes a prediction before kickoff and never revisits it, so `preds` records what was predicted
and never what happened. The realised margin is in nflverse's schedules as `result`. Scoring a
prediction is a **join**, not a column, and 88% test coverage did not catch it because every
test supplied its own frame.

Fixed: `load_scored` now joins on `game_id` and an unplayed game does not survive the join. Run
again, it says the true thing:

    hub.models.conformal: never reached 40 calibration points;
    the data has 0 rows across 0 weeks

**Which settles the sequencing, if not the decision.** All 48 stored predictions are for 2026
games that have not been played. There is nothing to calibrate against and cannot be until
Week 1, so "what consumes an interval" is not answerable on evidence yet — any consumer built
now would be wired to a module that has never produced a number. Revisit once a few weeks are
scored; the coverage report is the evidence that decision needs.

---

## 3. Correlation covers teammates and nothing else — MEASURED 2026-08-25

**Done, and it is real:** opposing quarterbacks correlate at **+0.148** (4.5 se), larger than
the QB-RB teammate edge the simulator already models. QB-TE +0.066, QB-WR +0.055, both past
four se; RB-RB *negative* at −0.024. See [opponent-correlation.md](opponent-correlation.md).

**Deliberately not wired.** Both consumers are inert — `lineup.optimize` per ADR-0012 and
`simulate_weeks` per ADR-0009 — so a correlation term would be a parameter plus plumbing
bought for nothing. It becomes worth wiring when the usage layer gives the optimiser real
variance to work with.

The method was validated by reproducing `TEAMMATE_RHO` to within 0.02 on all three edges.

### Original text

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

## 4. The scheme layer is fetched and unused — SCREENED 2026-08-29, null

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

### Pre-registered 2026-08-29, before the run

**The features**, per (team, season, week), from `ftn_charting`, and their **trends** — the
last three weeks against the three before, the same shape `snap_trend` uses and dark before
week 8 for the same reason:

| feature | what it is | pre-stated sign |
|---|---|---|
| `pa_rate` | share of dropbacks with play action | ? |
| `motion_rate` | share of plays with pre-snap motion | ? |
| `nohuddle_rate` | share of plays no-huddle | + — more plays, more volume |
| `screen_rate` | share of dropbacks that are screens | ? |
| `pass_rate` | dropbacks over all plays | + for receivers |

**A level is not the test; a trend is.** Team scheme is stable within a season and a player's
own recent usage already absorbs the level. What would be new information is a scheme
*changing* — the thing a coordinator change or an injury to a quarterback actually does — and
that is what a trend measures.

**Screened as an opt-in set** (`--scheme`), never merged into the default `FEATURES`, because
a collinear addition to the control set destroys real signals. That is not caution in the
abstract: adding `route_trend` to the default screen cost this repo `snap_trend` until it was
taken back out ([expected-and-routes.md](expected-and-routes.md)).

**Pre-stated expectation: null.** Five of five previous scheme-adjacent screens came back null,
team scheme is among the least persistent things available, and a trend over three weeks of a
stable quantity is mostly noise. The honest reason to run it is that it is the last unscreened
asset in the tree and leaving it unscreened is how it becomes folklore.

### Result, 2026-08-29: null, exactly as pre-stated

Against **points**, beyond season-to-date PPG and that week's consensus ECR:

| feature | partial r | t | seasons with the stated sign |
|---|---|---|---|
| play-action rate trend | +0.0229 | +1.62 | 3/4 |
| no-huddle rate trend | −0.0035 | −0.29 | 2/4 |
| pass rate trend | −0.0063 | −0.46 | 2/4 |
| motion rate trend | −0.0169 | −1.55 | 3/4 |
| screen rate trend | −0.0223 | −1.87 | 3/4 |

**Nothing clears either half of the bar**, and the default screen is untouched — the
independent signals are still snap_trend, dvp, inj_sev and td_rate_prior.

Against **volume**, which is the framing this item itself proposed as the more promising one:
**one cell of fifteen** clears — `pass_rate_trend` against pass attempts, **−0.0324 at −2.61
se**. One of fifteen tests crossing two standard errors is what the null predicts, the sign is
a *reversion* story rather than a scheme story, and it applies only to quarterbacks. Read as
the multiple-comparisons artifact `component-projection.md` already has a lesson about, not as
a finding.

**Six of six scheme-adjacent screens are now null.** The prior written down before the run was
right, which is the whole reason it was written down first.

Only FTN charting's coverage narrows the sample — it begins in **2022**, so these features get
four held-out seasons where everything else gets five. Stated rather than absorbed.

**Two defects found building it**, both now guarded:

* `trend()` silently assumed one row per (key, season, week). Called with a *team* key against
  a player-week panel it fans out by the roster size on every left join, and five chained calls
  took the process out on memory — exit 137, twice, with no output to explain it. It now raises
  with the reason, and the scheme trends are computed on the unique team-week frame and joined
  once.
* A narrowed `load_pbp` over four seasons was also killed by the OOM reaper. It was not needed:
  `participation` already carries `possession_team`, and a non-empty `route` marks a charted
  pass play — the same definition `route_share` uses, so the two agree by construction.

---

## 5. Injury pricing is one coefficient for three states — MEASURED 2026-08-25

**Done, and it is the first model in this repo to clear its gate.** See
[weekly-injury.md](weekly-injury.md).

The item below conflated two quantities, and the work separated them. `INJURY_BETA` prices a
*preseason* designation against a *season-long* projection — a draft question, unchanged. What
was missing is the *weekly* cost, and it is large and monotone in the practice report:
Questionable + did-not-practise keeps **41%** of a player's own production, Questionable + full
keeps **72%**, Out keeps **0%**.

Gated against "bench anyone ruled out", the rule a manager already follows for free. The first
attempt — an *additive* penalty table — **lost** (4.978 against 4.229), because an Out player
scores exactly zero and no additive penalty can say so. A multiplicative retention table,
declared before being run, wins at **4.059**, better in all three held-out seasons and by 0.170
MAE at 3.8 se across 3,687 player-weeks.

### Original text

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

## 6. `sd = k·sqrt(mu)` makes half the system inert — MEASURED 2026-08-25, RETIRED

**Model, and already documented.** [ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md)
measured the lineup optimiser at +0.00 points a game because within a position `sd` is a
deterministic function of `mu` (correlation 0.985). Two players projected at 12 points a game
are not equally volatile in reality, and the square-root law cannot say so.

This is the strongest argument for the usage and component layers, and worth restating here
because it is easy to file the null under "the optimiser does not work". It works. It is being
handed nothing to work with.

**Outcome: retired.** [player-spread.md](player-spread.md) measured whether per-player weekly
spread exists beyond the positional constant, and it barely does — ±9.3% in `sd`, with 85% of
any one season's estimate being noise. The shipped model is already within 0.085 MAE of the
irreducible floor set by sampling error in the outcome, so **the total prize for every future
variance model combined is 8% of the error that remains.** Nothing was adopted.

The sentence above stays as written because it was the honest reading at the time, and it was
wrong in a specific way worth keeping: two 12-ppg players really are not equally volatile, but
the gap is small and individually unmeasurable. This retires what looked like the highest-
leverage remaining item.

**Still open, on different grounds:** a usage layer may improve the *mean* (component
projection already beats points projection on skew — see
[weekly-spread.md](weekly-spread.md)). That is a separate claim with separate evidence, and it
is not what ADR-0012 was waiting for.

### Original text

**Do:** the usage model, gated against pick-anchored volume, with `volume.py` demoted to the
cold-start path for rookies. Then re-run `hub.season.lineup_gate`.

---

## Codebase

**Status, 2026-08-29.** Every item below is resolved except two, and both are held open for a
reason rather than by neglect: **#2** waits on Week 1 producing scored predictions to calibrate
against, and **#7**'s remaining thread is the fetch/assemble seam that
[ADR-0003](adr/0003-make-now-dagster-in-october.md) says the Dagster port wants anyway — doing
it twice would be waste.

### 7. `board.py` is still the hot spot — PARTLY ADDRESSED, remeasured 2026-08-29

367 statements, **46% covered**, and `build()` still fetches, prints and degrades in one
function — the one deviation from ADR-0003 that this session only partly addressed. Its
uncovered lines are the network path and the report block.

The honest split is between fetching and assembling, and
[ADR-0003](adr/0003-make-now-dagster-in-october.md) already says the Dagster port wants that
seam anyway. Doing it twice would be waste; doing it as part of the port would not.

**Two of the three jobs have since left, without touching that seam.** Rendering moved to
`hub/draft/report.py`, and the degradation policy moved into `_stage`. What remains for the
port is exactly the fetch/assemble split ADR-0003 names, which is the right thing to have left.

**Measured again 2026-08-29**, after a week in which `board.py` was still the most-touched file
in the repo:

| | when this was filed | now |
|---|---|---|
| statements | 367 | **305** |
| coverage | 46% | **95%** |
| `build()` lines | 154 | **83** |
| `try/except` inside `build()` | several | **0** |

The one open thread is the fetch/assemble seam, and it is open **on purpose**: ADR-0003 says
the Dagster port wants that seam anyway, and doing it twice is waste. This item stays PARTLY
ADDRESSED until the port, which is the honest status rather than a stale one.

### 8. The board bypasses `hub.store` — HALF DONE 2026-08-29, the half that loses data

`build()` writes `data/processed/draft_board.parquet` directly. `hub.store` exists, has a Hive
layout and a `write()`, and is used by the fetch layer. The board — the most-read artifact in
the repo — is the one thing that does not go through it, so it gets no partitioning and no
manifest.

Low urgency, but it is why `hub.inspect` has a special case for a bare-name dataset.

**Reassessed 2026-08-25: this is not tidiness, and it already cost real data.** Compare the two
paths on the same question — what happens to yesterday's numbers?

    data/processed/lines/league=nfl/season=2026/week=02/snap-20260825T045821.parquet
    data/processed/draft_board.parquet

The odds fetcher goes through `hub.store.write`, so every snapshot lands in its own timestamped
file inside a week partition and **nothing is ever overwritten**. The board does not, so it has
exactly one file, and every `make draft` destroyed the previous day's ADP — the one input that
`fit_espn_weight`, the opponent model and validating `edge` all need, and the one ESPN does not
retain. Three documented dead ends, one root cause, and the fetch layer had already solved it.

Patched around on 2026-08-25 by `hub/draft/adp_history.py`, which keeps a dated copy. That is a
second bespoke archive sitting next to a general one that already works, which is an argument
for doing this properly rather than against it.

Two further consequences found the same day: `make draft` had never worked on a fresh clone,
because `data/processed/` is created by `hub.store` and the board does not call it (fixed); and
the board is the reason `hub.inspect` needs its bare-name special case at all.

### Done 2026-08-29: every build is now kept

`board._archive` writes each build through `hub.store.write` as
`boards/league=nfl/season=2026/week=00/board-<timestamp>.parquet`. **Nothing is overwritten any
more**, which is the half of this item that was losing data rather than offending tidiness. The
catalog sees it: `store.tables()` now returns `boards` alongside `lines`, `preds`,
`ff_opportunity`, `pbp` and `adp_history`.

**Additive on purpose, and that is the whole design.** `draft_board.parquet` stays exactly
where it is — `last_good` reads it, `adherence` copies it, `hub.inspect` special-cases it and
[draft-night.md](draft-night.md) names it as the fallback for when the board will not build.
Four days before a draft is not when the artifact everything falls back to should move.

It also cannot break a build: the write is wrapped and prints `board archive skipped (...)`,
for the same reason the ADP archive is — an archival side effect that stops the board printing
is the operator-dependence `CLAUDE.md` warns about. Three tests, including that a second build
does not overwrite the first and that a failing store still lets the board through.

**What remains, after the draft:** migrate the readers to the store, retire the flat file, and
then `hub.inspect`'s bare-name special case and `hub/draft/adp_history.py` — the second bespoke
archive this item complains about — both become unnecessary.

### 9. One naive datetime in production — FIXED 2026-08-25

`market.py` stamped `predicted_at` — the provenance timestamp on every prediction row — in
naive *local* time, while `publish.py` used UTC. Now UTC, written tz-naive after conversion so
the column dtype is unchanged, matching the pattern `hub.fetch.odds` already used.

`src/` is clean of DTZ, so `DTZ` is now in the ruff select and stays clean. Test fixtures are
per-file-ignored: a timezone on a literal that exists only to be compared to another literal
is noise.

### Original text

`market.py:84` calls `datetime.now()` without a timezone. Twenty-five more are in tests and do
not matter. One does: a prediction timestamped without a zone is ambiguous in a repo whose
whole claim is that December can audit August.

`DTZ` is deliberately excluded from the ruff select for now, because fixing the test sites
changes stored fixtures. Fix the production site; leave the rest.

### 10. Coverage floors that mean something — LARGELY ADDRESSED 2026-08-27

**Superseded numbers, kept because the reasoning still holds.** Coverage is now **91%** against
the 80% gate, and every module named in the original text has moved: `backtest` 30% → **68%**,
`board` 46% → **96%**, `espn` 58% → **100%**, `live` 64% → **94%**, `tune` 64% → **86%**.

The remaining floor is `season/lineup_gate` 70%, `publish` 77%, `availability` 75% and
`backtest` 68%. None is on the draft path. `espn` was the one that mattered this week — it is
the module `--poll` talks to on the night, and its graceful-degradation path (four host/UA
combinations, then the cache) was the repo's own hard rule with no test on it at all.

**The gate itself is still the point, and it is still 80.** A gate set just under where the
repo sits is a gate that fails on noise; a gate set far under is decorative. Raising it is a
decision to make after the draft, when the number has stopped moving daily.

### Original text

Coverage is 80.2% against an 80% gate — passing by 0.2 points, which is a gate about to start
failing on noise. The low modules are `backtest` 30%, `board` 46%, `espn` 58%, `live` 64%,
`tune` 64%.

`backtest` at 30% is the one that matters: it is the module whose entire justification is that
a measurement nobody can re-run is worthless. Its orchestration got offline tests today; its
`main()` did not.

---

### 11. The roster shape is half-parameterised — FIXED 2026-08-27

**Explored 2026-08-27** as part of the architecture review.

`FLEX_CAPACITY` and `FLEX_FROM` are read: `optimize._need_score`, `evaluate.starter_points`,
`season.lineup_points` and `season/lineup.py` all consult them. `FLEX_SLOTS` — the count
itself — is defined at `season.py:31` and read by **nobody**.

Which means the flex arity is hardcoded in four places at once:

| where | how |
|---|---|
| `season.lineup_points:86` | `np.max(np.stack(leftovers))` |
| `season/lineup.py:102` | `yield (*base, f)` for a single `f` |
| `season/lineup_gate.py:84` | `flex.append(flex[0])` |
| `draft/evaluate.py:47` | `max(leftovers)` |

The consequence is precise, and worse than four copies of one rule. Set `RosterConfig.flex = 2`
and the halves disagree:

    FLEX_CAPACITY   7 -> 8      (the draft-side need logic adapts)
    the four scorers            (still field exactly one flex)

So the draft would correctly stop calling for flex-eligible players at eight, while every
lineup score in the repo kept fielding seven. `config.py:32-38` says the roster shape used to
live in five places and "nothing made them agree"; the unification reached `STARTERS`,
`FLEX_FROM` and `FLEX_CAPACITY` and stopped one constant short.

**Fixed.** All four read `FLEX_SLOTS`, and `lineup_gate` reads `FLEX_FROM` instead of inlining
`("RB", "WR", "TE")`. Two needed more than a constant swap: taking one leftover per position
and then the best of those is correct only for a single flex, since with two slots the best
pair can come from one position. The vectorised and greedy scorers now take the top
`FLEX_SLOTS` across all leftovers, and the exhaustive enumerator yields combinations. At
`FLEX_SLOTS = 1` every one reduces to what it did before, which is why the existing suite
passed untouched. Seven tests in `tests/unit/test_flex_arity.py`.

---

### 12. Five fitted constants describe code nothing can reach — FIXED 2026-08-27

**Explored 2026-08-27.** Two modules are reachable only from their own tests, and both are
registered in `FITTED_MODULES`, so their constants move the digest that identifies a
prediction.

**`draft/projection.py`.** `adjust_consensus` and `build` have no production caller —
`grep` for importers finds `config.py` naming the module and `tune.py:217` taking
`weighted_signal`, and nothing else. The formula `ecr * exp(-lam * z)` is written three
times: here, in `tune.apply`, and inline in `evaluate.simulate_draft:61`. The copy that is
*not* called is the one the other two are copies of. And `DEFAULT_LAMBDA = 0.0`, matching
`DraftConfig.projection_lambda = 0.0`, so even if something called it the result is
`adj_ecr == ecr` identically.

**`models/components.py`.** `SCORING`, `td_rate`, `points` and `scoring_mismatch` are
reachable. `sample_weeks`, `moments`, `project`, `regress_touchdowns`, `points_expr` and
`_counts` are not. `predict.py:17-21` records why: component-derived spread was measured
worse than the fitted square-root law, 1.365 against 1.140, P(better) 0.0%.

The digest, counted rather than asserted:

    35 fitted constants total
     4 unreachable  PER_UNIT_CV, YARDS_PER_UNIT, COUNT_DISPERSION, TD_DISPERSION
                    (all used only inside sample_weeks)
     1 unreachable  projection.DEFAULT_LAMBDA

So **5 of 35** describe code no prediction can reach. `SCORING`, `TD_RATE` and
`FALLBACK_TD_RATE` are genuinely live and stay.

**Fixed, and the two halves were separable exactly as expected.**

`projection.adjusted(df, lam)` is now the only implementation; `tune.score` and
`evaluate.simulate_draft` call it, `adjust_consensus` and `build` are gone along with the
`ranks_moved` column nothing read, and the clip is a named `Z_CLIP` rather than a bare -3/3
written twice.

The digest is 35 → 30. `hub.draft.projection` moved to `NOT_FITTED`;
`hub.models.components` is no longer registered wholesale, its three live constants are named
in `FITTED_EXTRA`, and the four that are not live are named in a new `NOT_IN_DIGEST` with
their reason. **The code is kept** — ADR-0007 says a measurement that steered a decision stays
in the tree — it simply stops identifying predictions that cannot reach it. Naming the
exclusions one at a time keeps that a decision on the record rather than a module quietly
falling off a list, and the registry guard was taught about both mechanisms so it still
catches a genuinely new unregistered constant.

Also fixed on the way: `test_underperformer_moves_up_the_board` called `build()` at the
shipped lambda of 0.0, where both players get identical `adj_ecr` — it was asserting stable
sort order and nothing about the adjustment.

Minor, same family: `predict.components()` is a two-line delegate to `volume.decompose` whose
only callers are its own tests, and `components.moments` and `predict.moments` are two
different functions of the same name one import apart.

---

### 13. ESPN's league year is hardcoded twice, and the failure is absorbed — FIXED 2026-08-27

**Explored 2026-08-27.** `fetch/espn.py:210` builds `League(..., year=2026)` as a literal.
`player_market(limit, season=2026)` takes a season parameter, passes it to `_parse_market`,
and that filters `st.get("seasonId") == season`. The two numbers are independent.

Confirmed offline rather than reasoned about:

    _parse_market(payload, 2026) -> proj_ppg 14.2
    _parse_market(payload, 2027) -> proj_ppg None      # full frame, no error

**The interesting part is what happens next.** The board's substance guard checks `adp`, not
`proj_ppg`. And `proj_blend` is a coalesce:

    pl.coalesce((proj_ppg + xfp_per_game) / 2.0, proj_ppg, xfp_per_game)

so an all-null `proj_ppg` silently falls through to `xfp_per_game`. The board builds, THE
PICK works, nothing warns, and ESPN's projection has simply stopped contributing. The new
dtype contract check does not catch it either: the column is correctly typed `Float64`, it
just has no values.

This fires exactly once, on the 2027 rollover, for whoever updates `SEASON_AHEAD` and not
`league_settings`.

It is also why that module has **three** different injection seams for one dependency —
`transaction_counts(fetch=...)`, the `_parse_market` parse/fetch split, and
`state.sync_from_espn(_league_factory=...)` — and why the tests monkeypatch
`league_settings` rather than pass an adapter. Every call site also discards half its tuple
return (`lg, _` at three sites, `_, slots` at one).

**Fixed — and the season turned out to be hardcoded five times, not two:**
`board.SEASON_AHEAD`, `league_settings`'s `League(year=2026)`, `state.sync_from_espn`'s
`year or 2026`, `playoff_sos`'s default and `publish --season`. Parameterising the year alone
would have put the same defect in a new place.

`hub.config.SEASON_AHEAD` is the owner and all five read it. `league_settings(year,
league_id)` takes both and returns a `LeagueView` rather than a tuple every call site was
discarding half of. `player_market` passes its season through as the year, which is the fix.

Three regression tests, one asserting both that the League is built for the season asked for
*and* that the projection survives the filter — so it cannot pass by the filter being
removed. Verified against live ESPN: 165 drafted players priced, roster and scoring checks
both running.

---

### 14. The poll loop catches only KeyboardInterrupt — FIXED 2026-08-27

**Fixed the evening of 2026-08-27, seven days out.** The loop now catches `Exception`, prints
`poll error (serving stale, the board is unaffected): ...`, and keeps polling.

Two things beyond the three lines this item asked for:

* **Repeated identical errors print once**, with a `recovered after N failed polls` line when
  the sync comes back. The reason is the one already in `poll`'s own docstring — the sync was
  made quiet because a thousand lines over three hours scrolls the board out of view, and a
  traceback repeating every ten seconds does exactly that.
* **`KeyboardInterrupt` is no longer named in the loop.** It is a `BaseException`, so
  `except Exception` cannot swallow it; the clause was dead code. There is a test that says so,
  since the loop no longer says it itself.

Four tests, each checked to fail against the old loop. The related `fetch/espn` coverage gap
went with it: 63% → 100%, including the `_get` degradation ladder, which was the repo's own
hard rule with nothing exercising it.

### Original text

**Found 2026-08-27 while covering the poller; deliberately not fixed before the freeze.**

`live.poll`'s loop is wrapped in `except KeyboardInterrupt`. Anything else — a render error on
an unexpected board shape, a disk error inside `save` — propagates, kills the poller, and skips
the line that tells you where the fallback is:

    stopped. site/data/draft_board.json is still correct.

`hub.fetch.espn.poll` already has the right shape for this and prints
`poll error (serving stale)` rather than dying, so there is precedent in the repo.

**Severity is low and that is why it waits.** `sync_from_espn` swallows its own failures by
design and returns last-good, so the common case — ESPN unreachable — is already survived, and
there is a test for it. The runbook already says "If it dies, the board and `--pick` still
work". What is lost is the reminder, not the fallback.

**Do:** after the draft, catch `Exception` in the loop, print it, and keep polling. Three lines.
It is draft-path code and the freeze holds until 2026-08-31.

*(The freeze was lifted on 2026-08-27, which retired the only reason this was waiting.)*

---

### 15. `durability` and `regression` are one module written twice — DONE 2026-08-29

**Explored 2026-08-27.** Both files are the same four functions in the same order:
`prior_season(season, cache)` → `<signal>(season)` → `attach(board, season)` →
`correct_projection(board, column="proj_blend")`, each with a per-position `BETA` dict.
`attach` differs by three tokens; `correct_projection` is the same
`pos.replace_strict(BETA) * col` then `.clip(0.0)`. The `_norm` join dance is written twice.

**One claim from the review does not survive checking, and it was the severe one.** The
concern was that either signal could silently fail to attach while `correct_projection` still
ran and contributed zero. It cannot: `correct_projection` opens with
`if column not in board.columns or "td_luck" not in board.columns: return board`, and
`BuildReport` reports the missing stage as `built without: td_luck`. Guarded, and surfaced.

**What is real** is the duplication, and one measurable cost. `nflverse._cache_path` keys on
the sorted column set — deliberately, so a caller asking for six columns is never served an
earlier caller's five — so callers asking for different slices of `player_stats` download it
separately. There were four slices: durability 7 columns, regression 13, backtest 6,
lineup_gate 6. **Two of those four were merged on 2026-08-27** when `hub.models.experiment`
took ownership of the column list; durability and regression remain distinct.

**Do, post-draft:** one prior-season-correction shape parameterised by signal and coefficient.
Low urgency — the duplication costs a reader's time and one extra download, not correctness.

### Done 2026-08-29, and less of it than the item asked for

`hub/draft/prior_signal.py` holds the two pieces that genuinely were one thing written twice:

* **`join_by_player(board, signal, column)`** — the `_norm` join dance, twelve identical lines
  in both files, including the rule that a player with no prior season keeps a **null and not a
  zero**. Zero is "played every game" or "scored exactly as expected"; null is "we do not
  know"; and filling one with the other calls every rookie durable and every rookie lucky.
* **`priced(column, beta)`** — `beta[position] * column`, with an unlisted position at zero
  rather than a pooled default, because a position absent from a `BETA` is one the fit found
  nothing for.

**What was deliberately not merged**, against the item's own suggestion: the two
`correct_projection` bodies are *not* the same function. Durability prices a trait **and**
today's designation, over two different position sets; touchdown luck prices one term. One
shape with flags to cover both would cost more than the duplication does.

Each module keeps its own `BETA` — which is also what keeps `config_digest` at `281b7b7a`,
since the digest is over the constants those modules declare and both are in `FITTED_MODULES`.
Verified: digest unchanged, and the rebuilt board has the same 449 rows, 241 non-null `missed`,
320 non-null `td_luck` and a `proj_blend` mean of 7.1265.

The extra `player_stats` download the item mentions is untouched — that is
`nflverse._cache_path` keying on the column set, and merging the two column lists is a separate
decision about what each module needs.

---

### 16. `store.tables()` is one commit old and already used inconsistently — DONE 2026-08-29

**Explored 2026-08-27.** Two things, both small, both about a seam that exists and is bypassed.

**The predicate is written twice.** `store.connect()` (store.py:58) and `store.tables()`
(store.py:82) each carry their own copy of "a directory that is an identifier and holds
parquet". `tables()`'s own docstring claims it is "discovered the same way `connect` discovers
them, **so the two cannot disagree**" — which nothing enforces. The invariant is asserted in
prose, not in code.

**And `publish.py` ignores the seam.** `store.tables` was added earlier the same day to fix
CLIs that raised `CatalogException` against a fresh clone. `publish.py` has the identical
exposure and uses it **zero** times, handling the case with four bare `except Exception:`
blocks instead (publish.py:70, 91, 102, 239). So a genuine schema break, a DuckDB lock and a
typo in the SQL all reach the manifest as `"stale": true, "reason": "no predictions"`.

**Adjacent, and the reason it is worth doing together:** `store.LAYOUT` owns the
`week={week:02d}` zero-padding, and `publish` re-derives it three times
(`f"preds_wk{week:02d}"` at :65 and :184, `f"{week:02d}"` as a query param at :69) then reads
it back as `int(got["w"][0])` at :242. Four encodings of one partition key in a 333-line file,
because the format is not part of any query helper's interface.

**Do, post-draft:** one predicate, shared. `publish` asks `store.tables` like its two
neighbours do. The partition format becomes part of a query helper rather than a convention
spelled four ways.

### Done 2026-08-29, ahead of the flip rather than the draft

Brought forward because `publish` is the module that writes the **public** artifacts and the
repo goes public on 2026-09-04 with a scheduled watchdog. It touches nothing the draft reads.

**One predicate.** `store.is_table(dir)` — an identifier, holding parquet — and both `connect`
and `tables` call it. The docstring claim that the two "cannot disagree" is now enforced by
there being one of them, with a test that compares `connect`'s actual view list against
`tables()` on the same store.

**`publish` asks instead of catching.** Three of the four bare `except Exception` blocks were
guarding "this clone has no `preds` yet", which is exactly the question `store.tables` answers.
They are now `if "preds" not in store.tables(base)`, and **a genuine failure surfaces**: there
is a test that a broken query raises rather than arriving on the page as
`stale: true, reason: no predictions`. The fourth wraps a network call on an unattended
schedule and stays broad — but it now prints which failure it was.

**One spelling for the partition.** `store.week_key(week)` owns the zero-padding, and a test
pins it against `LAYOUT`'s own `week={week:02d}`. It was spelled four ways in one 333-line
file — two filenames, a query parameter, and the read-back — because the format belonged to
nobody.

`store` coverage 97%, `publish` 76%.

---

### 17. `ROOT` is declared eight times, and it does not matter much — DONE 2026-08-29

**Investigated 2026-08-27; recorded rather than fixed.** The review flagged three
declarations of the repo root; there are eight — `store`, `publish`, `board`, `tune`, `state`,
`nflverse`, `cfbd`, `odds` — at two different `parents[]` depths.

**Checked at runtime rather than argued: all eight resolve to the same path**, and each depth
is correct for its own nesting (2 for top-level modules, 3 for package modules). Each has one
or two uses. There is no drift and no latent bug, so the case for a shared declaration is
tidiness, and tidiness alone does not earn a change to eight files.

**The part that is real** is narrower. `hub/draft/adp_history.py` and `hub/draft/adherence.py`
take `ROOT` and `BOARD_PARQUET` from `board.py` — a 780-line module that pulls in nflreadpy,
numpy, polars, contracts, durability, regression, report, state, availability, picks and
playoff_sos — purely to learn a `Path`. The cost shows up immediately: `board.main` has to
import `adp_history` inside the function body, with a comment saying why.

**Do, if anything:** a `hub/paths.py` leaf holding `ROOT` and the two or three derived paths,
so a module wanting a filename does not import a board builder. That removes two
function-local imports. It is the smallest item on this list and is here so the next review
does not rediscover the eight declarations and overrate them.

### Done 2026-08-29, the half that is not tidiness

`hub/paths.py` exists — `ROOT`, `DATA`, `PROCESSED`, `BOARD_PARQUET` — importing nothing but
`pathlib`, with a test pinning that it stays a leaf. `hub.draft.adp_history` takes `ROOT` from
it and **no longer imports a board builder at all**, so that cycle is gone.

**And then the rest of it**, once the same session was already re-verifying the draft path for
#18. `board.py` takes `ROOT` and `BOARD_PARQUET` from the leaf instead of declaring them;
`adherence.py` takes the path from the leaf and keeps `board_age_hours` from the module, which
is a function and belongs there.

**The point of the item, delivered:** `board.main`'s function-local `import adp_history` is
gone, and with it the comment reading *"imported here rather than at module scope because it
reads ROOT from this module"*. Eight function-local `hub` imports remain in `board.py` and all
eight are deliberate — lazy loads of heavy or optional paths, not cycle workarounds.

The five other `ROOT` declarations are untouched. They are one or two uses each in modules that
import nothing from the leaf, and #17's own judgement stands: tidiness alone does not earn a
change to five more files.

---

### 18. `board_as_of` is not reproducible — FIXED 2026-08-29

**Found 2026-08-28 while building the waiver gate.** Two calls to `board_as_of(2024)` in one
process return the same 1,103 players in a **different row order**, and `wk15_17_sos` differs
below 1e-6.

**Diagnosed 2026-08-29, and it is not what the first guess said.** Not a threaded sum: it
reproduces with `POLARS_MAX_THREADS=1`. It is in `playoff_sos._dvp_from_stats`, and it is a
**two-stage aggregation where each stage is deterministic and the pair is not**:

```
per_week = stats.group_by([defence, position, week]).agg(points.sum())   # stage 1
dvp      = per_week.group_by([defence, position]).agg(allowed.mean())    # stage 2
```

Stage 1 emits its rows in a hash-dependent order that varies between calls. Stage 2 then takes
a **mean over those rows**, and floating-point addition is not associative — so a different row
order inside each group gives a different answer at **7.1e-15**. Either stage run twice on a
fixed frame is identical; the composition is not, four times out of four.

**The fix is one `sort` between the stages, and it is verified**: with
`.sort([defence, position, week])` inserted, four consecutive runs are bit-identical; without
it, they are not.

**Applied 2026-08-29**, on the reasoning that it makes the board *more* deterministic four
days before the draft rather than less. The alternative was leaving row order to vary randomly
on the night; this changes it once, now, under verification.

**Two causes, not one.** The sort between the aggregations fixed every *value* — after it,
two boards agree column for column — and the row order still varied, because `build` ended with
`sort("ecr")` and ties ordered arbitrarily. It now sorts on **`["ecr", "player"]`**. The
tiebreaker cannot move a ranking: it only orders players who share an ECR, which was arbitrary
before.

**Verified**: three consecutive `board_as_of(2024)` builds are now identical frames;
`config_digest` is unchanged at `281b7b7a`, so no prediction moves; THE PICK at slot 3 is
unchanged. The workaround in `weekly_gate_data` is removed. There is a test on the tiebreaker
that needs no network.

**It moved a published number, which is the point of having fixed it.** The weekly gate drew
its rosters from `board_as_of`, so a board that was not reproducible meant a roster sample that
was not either: the frozen gate reported **+0.711 [+0.313, +1.129]** and now reports
**+0.215 [−0.249, +0.659]**, every time. The verdict is unchanged (SHOW) and so is everything
the write-up concluded, but the interval now contains zero where it excluded it — the earlier
figure was one draw presented as a result. Restated in
[weekly-blend-gate.md](weekly-blend-gate.md) and
[ADR-0017](adr/0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md) rather than quietly
left. The class of bug is worth a moment's
thought beyond this instance — **any `group_by` feeding a float aggregation to another
`group_by` has it**.

**Audited 2026-08-29**, by AST, for functions containing two or more aggregations. Three:

| | |
|---|---|
| `playoff_sos._dvp_from_stats` | the original. Draft path, **not fixed**, fixed after the draft |
| `weekly_screen.route_share_from_plays` | **safe** — the aggregations are `len()` counts and a `max()`, and both are exact |
| `weekly_screen.build_panel` / `prior_means` | **the same defect, in code written the night before**. Fixed |

The third is the useful part of the audit: sum-then-mean without a sort between them, written
while diagnosing the identical bug elsewhere. `prior_means` now sorts on the way out, which
fixes every consumer at once rather than each separately, and three consecutive panel builds
are bit-identical on `ppg_before`, `dvp`, `td_rate_prior` and `targets_prior` where `dvp` was
not before. The published screen is unchanged.

It matters because the draft indexes the board by **row**: an unstable order moves picks, which
moves rosters, which moves every downstream measurement. `hub.season.weekly_gate` wobbled by
~0.04 points a team-week between identical runs, and the first number published from it
(−0.684) was one draw from an unstable process rather than a result.

**This is not confined to the weekly gate.** `hub.draft.backtest` and `hub.season.lineup_gate`
both draft from `board_as_of`, so their published numbers carry the same wobble. ADR-0009's
−17.30 and the lineup gate's structural zero are both large enough that ±0.04 changes nothing,
but "large enough not to matter" is a judgement that should be on the record rather than
assumed.

**Worked around, not fixed:** the weekly gate sorts the board by player before drafting, which
made it deterministic across three consecutive runs. Sorted in the gate rather than in
`board.build`, deliberately — that is draft-path code six days from a live draft and its
tie-breaking must not move tonight.

A measurement that cannot be reproduced to the last bit is one ADR-0007 says should not be
steering anything — and `backtest` and `lineup_gate` both draft from this function, so ADR-0009's
−17.30 and the lineup gate's structural zero carry the same wobble. Both are large enough that
±0.04 changes nothing, but "large enough not to matter" should be on the record rather than
assumed.

### 19. The screen owned the Panel, and two other modules reached in for it — DONE 2026-08-30

`weekly_screen.py` was 867 lines doing two jobs. One was the screen proper: the partial
correlation, the cell structure, the pre-registered verdict. The other was the **Panel** — one
row per player-week, every feature measured before its outcome — assembled from eleven network
sources. The second is 65% of the file and the screen is only its first caller.

The tell was already written down as item 17, *what a caller does when an import feels wrong*.
`models/weekly.py` and `season/weekly_gate_data.py` both needed a Panel, and both fetched one
with a **function-local** `from hub.models.weekly_screen import build_panel` buried inside a
function body. Neither of them screens anything. A module named for one question was the only
way to ask a different one.

**Done 2026-08-30.** `src/hub/models/panel.py` owns assembly; `weekly_screen.py` keeps the
statistic and the verdict and is now 329 lines. Both function-local imports are ordinary
module-level ones. `build_panel`'s five boolean flags became one `PanelSpec`, so a call site
reads back as the shape it asked for instead of as a flag combination the reader has to decode.
The screen's 49 tests split 26/23 by what they exercise, and `Panel` went into `CONTEXT.md` —
it is a domain concept, not an implementation detail.

Two more restatements of the same literal fell out and are collapsed rather than moved:

* `TREND_MIN_WEEK = 8` was declared in `weekly.py` **and** in the screen. The Panel owns it now,
  since the trend it gates is computed there, and the better half of each comment was merged.
* `tuple(range(1, 15))` was written out in `weekly_screen`, `weekly_gate` **and**
  `draft/season.py` — a bare `15` in three files for a league length with one owner. It moved to
  `hub.config` rather than to `draft.season`, because `test_models_does_not_reach_into_draft`
  forbids the direction the screen would have needed. A new test pins the *literal*, not the
  name: re-declaring `range(1, 15)` anywhere under `src/hub/` now fails.

That guard test earned its place immediately — it was the thing that caught the first attempt
importing `FANTASY_WEEKS` from `draft/`.

**Verified unchanged.** The frozen weekly gate re-runs at **+0.215** points per team-week,
2,000 roster-weeks over 160 rosters, 16.0% unranked, 0.1% join failure, per-season
+0.983 / +0.027 / +0.355 / −0.504 — identical to the pre-split figures in
[weekly-blend-gate.md](weekly-blend-gate.md), through the network path and not only under the
unit tests. 1,361 tests pass; `config_digest` stays `281b7b7a` and `fitted_digest` `d5598b96`.

---

### 20. The Gate's statistic was written twice, and one copy was not reproducible — DONE 2026-09-04

Three modules run a **Gate**: `draft.backtest`, `season.lineup_gate`, `season.weekly_gate`.
All three build a paired frame with a `diff` column, bootstrap it, and read a verdict off the
interval. The bootstrap existed twice. `experiment.summarise` resampled *rows*;
`weekly_gate.cluster_bootstrap` resampled *rosters*. Same returned dict, one difference: what
counts as an independent observation.

Neither was wrong. `backtest` pairs one row per (season, draft) and `lineup_gate` one row per
roster — independent draws, so rows are the right unit there. `weekly_gate` pairs one row per
roster-*week*, fourteen readings sharing a roster's players, bye and draft. Each was right; the
rule keeping them right lived in two places, and which one you got depended on which function
you imported.

**Done 2026-09-04.** One statistic: `summarise(paired, *, cluster=None)`, where `cluster` names
the columns identifying one independent observation. `cluster_bootstrap` is deleted;
`weekly_gate` passes `CLUSTER = ("season", "roster")` and its hand-rolled print block is now
`paired_report`, which gained `places` and `show_n` so the weekly gate's tenth-the-size effect
still reads as +0.215 rather than rounding to +0.22.

**And the move found a live defect.** The clusters arrived in `.unique()` order, and since the
bootstrap indexes into that order a permutation of the same cluster means moved the percentiles
while leaving their average alone. That is why this repo's own docs recorded
**+0.215 [−0.249, +0.659]** for a run that re-ran as **+0.215 [−0.251, +0.663]** — the mean was
bit-stable and the interval never was. It is the same class of defect as #18, one layer down:
a hash-ordered frame feeding a seeded computation. Clusters are sorted now, the interval is
**[−0.242, +0.684]** every run, and a test asserts a shuffled input gives an identical summary.

Nothing about the decision moves: same mean, same 3/4 seasons, same monotone decay, and an
interval that contained zero still contains it.

### 21. The Artifact was a concept with no module — DONE 2026-09-04

`CLAUDE.md`'s degradation rule — *a panel whose data is missing says so and keeps rendering* —
was written three ways inside one function. `publish_all` used a `record()` closure for four
artifacts, a hand-rolled dict literal for `draft_board`, and `survivor` returned **its own
manifest entry** rather than a payload, so it was appended raw and never recorded at all.

The two that bypassed `record` were also the two that always reported `generated_at: null`, so
the page could not age them. A stale board and a fresh one looked identical.

**Done 2026-09-04.** `Artifact(name, produce, reason)` owns the contract and `publish_all` is a
list comprehension over six declarations. `survivor` returns a payload or None like every other
producer, printing its own failure the way `live` does. A parametrised test now asserts all six
go stale with a reason and carry last-good's timestamp — for `draft_board` and `survivor` that
test could not previously be written. `_generated_at` also guards the branch where a
list-shaped artifact is read for a key it cannot have, which was unreachable before and is not
now.

---

### 22. Four defects from a full-repo review — FIXED 2026-09-04

A read of all 13,665 lines of `src/hub`, not a diff. Four findings, each reproduced before
being believed. Three are the same shape: a local accounting or ordering assumption that the
code *next door* already knows to guard.

**`fetch/cfbd.py` — a failed call spent quota nothing counted.** Both counters were
incremented after `_http_get` returned, so every 429, 500 and timeout -- all of which
`raise_for_status` turns into an exception -- spent a real call against the 1,000/month free
tier that neither `quota_used()` nor `_CALLS_THIS_RUN` ever saw. Stubbing the call to raise:
**20 real requests, 0 recorded, and the MAX_CALLS_PER_RUN=12 "this is a loop" ceiling never
fired.** Counted in a `finally` now. A connection error that never reached CFBD over-counts by
one, which is the safe direction. `fetch/odds.py` was immune only because it reads the balance
from the server's own `x-requests-remaining` header instead of keeping a local counter.

**`draft/tune.py` — the Spearman that fits `projection_lambda` mis-ranked ties.**
`actual.argsort().argsort()` gives tied values distinct sequential ranks; Spearman requires the
mean rank. `actual_points` is `fill_null(0.0)`, so every player with no production is an exact
tie. On a 450-row board with 180 zeros: true rho **+0.650**, computed **+0.563**, and **+0.623**
for the same data in a different row order -- so `score()` was not a function of its data.
`sweep` re-sorts by `adj_ecr` per lambda, so each lambda carried its own arbitrary
tie-breaking into a comparison against the others. Fixed with `average_ranks`, written in
numpy rather than `scipy.stats.rankdata` because scipy is not a declared dependency and a
fitted constant should not rest on a transitive one. **The sweep was re-run: `lam = 0.00`
still wins on both metrics**, and lambda-sweep.md records why it was robust -- the tables
report deltas between lambdas, and a shared bias cancels in the difference.

**`store.write` — the docstring promised immutability and nothing enforced it.** The path is a
pure function of its key and `write_parquet` truncates, so any caller reusing a name destroyed
the previous partition. Three of four call sites dodged this with a timestamp or digest, each
with a comment about the hazard; `fetch/nflverse` used the default `name="part"` and overwrote
every week on every refresh, losing the record of any upstream stat correction. The promise is
now enforced: `replace=False` refuses to overwrite differing data, identical bytes are a no-op
so `make slate` stays idempotent, and the two callers that genuinely mean replacement say so.
`ratings` needed it too -- `predicted_at` is `datetime.now()`, so an identical-config re-run
lands on its own filename with different content, and the guard would have broken the weekly
ritual. Caught by checking rather than by shipping.

**`models/panel.py` — the reported scrape lead came from the wrong scrape.**
`weekly_consensus` paired `ecr.min()` with `lead_days.first()` over an unsorted `group_by`.
Measured on live rankings: **1,216 of 37,151 player-weeks carry two scrapes and 1,211 of those
have two different leads**, so `LEAD_DAYS` -- what weekly-screen.md cites for the confound the
whole screen is read against -- was partly computed from rows that did not supply the ECR.
`injury_severity` sorts before its own `first()` for exactly this reason, forty lines above.
The aggregation is now `best_per_week`, split out so it can be tested without the network.

**Verified unchanged.** The frozen weekly gate re-runs at **+0.215, CI [-0.242, +0.684]**,
3/4 seasons -- `lead_days` feeds no statistic, only the report. 1,423 tests pass (+12),
`config_digest` 281b7b7a and `fitted_digest` d5598b96 unmoved.

**Looked at and cleared**, so the sweep is legible as more than four hits: every `zip()` in
the tree already passes `strict=`; there are no mutable default arguments; the other four
`.first()` aggregations are each sorted first or take a value constant within the group;
`_counts` guards all three dispersion regimes; `conformal.interval` cannot be reached with an
empty window; `teammate_rho` handles both key orientations. Two near-misses are worth naming
because the reasoning is the useful part: `correlate.pair_correlations` canonicalises its key
without swapping the values, which is harmless **only** because `z` is standardised per
player-season so both marginals are ~(0,1); and `survivor.solve` compares a solver float with
`== 1`, which is genuinely fragile but CBC rounds cleanly and no failure could be produced, so
it was left alone rather than padding the list.

---

---

## What is deliberately not on this list

**Anything that tries to beat a market.** Objective 4 was retired on 2026-08-24 after six
measured failures. Improvements to the *props* or *spread* paths are worth making for
calibration, and their gate is log-loss against the market, not profit against it.

**Re-litigating championship equity.** [ADR-0009](adr/0009-championship-equity-does-not-pick.md)
closes it, and says explicitly that reopening means re-running the harness, not re-arguing.
Item 6 is the path back if there is one.
