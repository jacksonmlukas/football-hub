# Decision log

Everything settled during the design conversation, so no future session has to re-derive it.
If something here conflicts with the code, the code is wrong or this file is stale — say so.

## Facts about Jackson's setup

Given, not assumptions to re-litigate.

| Fact | Value |
|---|---|
| Fantasy platform | ESPN, private league (cookies in `.env`) |
| League | 12-team, full PPR, redraft, snake |
| Roster starters | QB1 / RB2 / **WR3** / TE1 / FLEX1 (not the ESPN default) |
| Draft slot | **3** — picks 3, 22, 27, 46, 51, 70, 75 (19/5 alternating waits) |
| Draft date | **Sep 3, 2026** — hard deadline |
| Room composition | **Mixed** — some draft off ESPN's board, some use outside rankings |
| Data budget | Free tier only |
| Modeling philosophy | Hybrid: market prior + model on the residual |
| Betting | Pools plus occasional small bets. No real bankroll. |
| CFB scope | All 136 FBS. CFB *fantasy* is permanently cut. |
| Machine | Apple M-series, 16GB |
| Claude plan | Max 5x |
| Repo | Private until Sep 4, then public as a portfolio artifact |
| Testing | Strict TDD, test-first |
| Track record | Public, warts and all |
| Pool sizes | **Unknown.** Last blocker for Track D. |

### Objective order (governs every tradeoff)

1. Win the fantasy league and pools
2. Actually understand the modeling techniques
3. Portfolio artifact that lands interviews

**Demonstrable market edge (positive CLV) was objective 4 until 2026-08-24. It is now an
explicit non-goal.** The standing decision below -- backtest to audit ourselves, not to find
edges -- had already retired it in practice, and the list had not caught up. Six measured
attempts have now failed to beat a market: four screens, VOR ordering at -5.06 points per
team-game, and championship equity at -19.66. Leaving it stated invites the move this document
records going wrong twice in one day: when a pre-registered rule fires against the complicated
thing, an ambition to find edge is what makes "the test was unfair" feel reasonable.

The system measures itself against markets. It does not try to beat them.

Edge ranked last. That is why gates route rather than kill: a model failing its gate becomes a
documented study, not a deletion. It is also why we derive before importing (hand-roll split
conformal before MAPIE, the survivor DP before pulp) — the slower path serves objective 2.

Shipping priority: fantasy > pickem/spreads > survivor > CFB predictions > props > awards.
Props and awards are roadmap-only: free-tier odds data cannot support props, since the credit
multiplier makes one props pull cost more than a month's quota.

## Corrections issued during design

Six things I got wrong then fixed. Do not revert to the earlier version.

1. **MCP is not categorically expensive.** Claude Code ships MCP Tool Search, which defers tool
   definitions and loads them semantically. Revised rule: MCP *definitions* are cheap, MCP
   *outputs* are not. Fine for interactive work (GitHub, PM); wrong for data fetching, where a
   CLI prints a 20-line summary instead of returning 60KB of JSON.
2. **ffverse does not require R.** `nflreadpy` ships `load_ff_opportunity` and
   `load_ff_rankings` natively. No R anywhere in this project.
3. **All 136 FBS fits on CFBD's free tier.** The quota only dies if you loop over teams.
4. **The poller does not need concurrency.** One scoreboard request covers every game in a
   league. Only `summary` fans out, and it is tiered and capped.

5. **espn_api exposes no ADP; ESPN's raw API does.** The board stood in `Player.posRank`,
   which is a *positional* rank (WR5 -> 5) parsed from a `positionalRanking` key that
   free-agent payloads omit -- so it was always `[]`, and subtracting a positional rank from
   an overall consensus rank would have been meaningless even had it carried values.
   `espn_api` reads `ownership.percentOwned` and discards the rest of that block, where
   **`ownership.averageDraftPosition`** lives. Real ADP comes from the `kona_player_info`
   view via `lg.espn_request.league_get`, one bulk request for the whole pool
   (`hub/fetch/espn.py:player_adp`). Verified Aug 21 2026: 463 players, Gibbs at 1.49.
   Do not go back to `posRank`, and do not add an ADP dependency -- ESPN has it.

6. **`load_ff_rankings("draft")` is 31 ranking pages stacked in one frame, not one board.**
   Redraft, dynasty, best-ball, superflex and IDP pages each carry their *own* `ecr` scale.
   `consensus()` selected `page_type` away before `.unique(keep="first")`, so every player's
   ECR came from whichever page happened to sort first -- 27 distinct pages, the largest
   contributors `best-overall` (419 players, best-ball) and `redraft-lb` (216, IDP
   linebackers). The symptom was kickers topping the edge list on *positional* ECRs of
   11-30, which is the same error as the `posRank` one: a scoped rank read as an overall.
   Correct slice for this league is `page_type == "redraft-overall"`, which resolves to
   `/nfl/rankings/ppr-cheatsheets.php` -- redraft, full PPR, no superflex, no IDP. Filtered
   to QB/RB/WR/TE, since K and DST are rostered but drafted off ESPN's own list.
   Board went 1474 -> 452 players and 957 -> 104 missing xFP. A missing page now raises
   `ContractViolation` rather than silently rebuilding the mongrel.

   General lesson, now twice: **before differencing two ranks, ask what population each was
   computed over.** Every bug in the draft board so far has been a version of that.

## Alternatives rejected

| Instead of | Rejected | Because |
|---|---|---|
| DuckDB | SQLite | Row-oriented, single-threaded per query, no ASOF JOIN |
| DuckDB | Postgres | Single user, no concurrent writers, no network |
| Pyrefly | mypy | Slower, and it let real unsoundness through (polars union bug) |
| Pyrefly | ty | Beta; weakest on generics and Protocols, and `Forecaster` is a Protocol |
| Make → Dagster | Prefect | Pipeline is asset-shaped; partitioned backfills are the reason |
| Vanilla JS + uPlot | Observable Framework, Evidence | Node build step for four charts and a table |
| Hand-rolled contracts | patito, pandera | Would lose the domain-specific range checks that are the point |
| GitHub Projects | Linear, Todoist, Notion | Free, lives with the code, public = portfolio evidence |
| Sync tiered poller | asyncio, thread pool | 13 req / 3 min; concurrency raises odds of ESPN blocking |
| Own tokenizer | Porting from the World Cup sim | Repos stay independent; re-deriving serves objective 2 |

## The draft edge

Naive signal was `edge` = ECR minus ESPN ADP, assuming the room drafts off ESPN's board. The room
is **mixed**, which breaks it specifically: sharps take the ECR-favorable players first, so the
largest edges are gone before pick 22. What falls to you is the high-edge player the sharps also
passed on, which usually means consensus has not priced something real.

Fix is `hub/draft/availability.py`: blend both boards by room composition, model pick position as
a distribution, rank by `cost_of_waiting` = VOR × P(gone by next pick). Run `fit_espn_weight()`
before the draft to estimate the blend from league history rather than the 0.5 prior.

**Slot 3 rule** (`draft_mode()`): rounds 1/3/5/7 (picks 3, 27, 51, 75) have a 19-pick wait after,
so take who will not survive. Rounds 2/4/6 (picks 22, 46, 70) have a 5-pick wait, so take the
highest VOR and ignore availability.

**Three-WR effect:** ~43 startable WRs makes WR replacement sit *below* RB
(QB 19.8 / RB 12.4 / WR 9.8 / TE 9.6). With the WR pool consumed by starters, the flex tilts back
toward RB, so flex allocation is ~even (0.45 RB / 0.50 WR), not WR-dominant.

## Known gotchas

| Gotcha | Fix |
|---|---|
| Worktree sessions lose gitignored `.env`, silently degrading to ECR-only mode | `.worktreeinclude` lists `.env` |
| Desktop app does not inherit exported shell vars | Local environment editor: env dropdown → Local → gear |
| Pyrefly's default preset is `basic`, silencing most errors | `preset = "default"` in `[tool.pyrefly]` |
| Hatchling cannot infer the package (`football-hub` vs `src/hub`) | `[tool.hatch.build.targets.wheel] packages = ["src/hub"]` |
| PEP 735 `[dependency-groups]` are not extras, so `.[dev]` fails | dev lives in `[project.optional-dependencies]` |
| Polars types `min()`/`max()` as the union of every dtype | Guard None, then `cast(float, ...)` — do not silence |
| `site.api.espn.com` began 403ing scripted traffic Aug 2026 | Retry `site.web.api.espn.com` and non-browser User-Agents |
| conda auto-activating shadows the venv Python | `conda config --set auto_activate_base false` |
| CFBD quota dies if you loop over teams | Bulk week endpoints only |
| ESPN does not retain historical ADP (past seasons return the sentinel) | `fit_espn_weight` is not identifiable; snapshot ADP pre-draft to fix next year |
| FantasyPros and ffopportunity disagree on suffixes, so an exact join drops players | `_join_expected_points` matches on `state._norm` |
| Replacement level set by 1-game samples put WR *above* RB | `MIN_GAMES = 10`; restores the documented three-WR effect |
| Rookies have no prior-season xFP, so mu was 0 for top-168 picks | `_impute_xfp` interpolates from consensus rank within position |
| Scoring the season on the same projection the greedy ranks on | Absolute P(win) is inflated; read `lift`, not the level |
| `espn_api` has no ADP field at all | Raw `kona_player_info` -> `ownership.averageDraftPosition` |
| ESPN parks undrafted players on a shared tail ADP (~170), not null | `_adp_saturation_cutoff()`: lowest value shared by >= `teams` players |
| Subtracting ranks drawn from different populations | Rank consensus *within* the ADP pool before differencing |

## Weeks 15-17 SoS (pre-draft gate: met)

`hub/draft/playoff_sos.py`, column `wk15_17_sos` on the board, `--sos` to view. 1.00 is a
league-average defence *for that position*; the ratio is never collapsed across positions
because a defence soft against WRs need not be soft against RBs. Built from two bulk pulls:
2025 PPR points allowed per defence per position, and the published 2026 weeks 15-17
schedule. 435 of 452 players placed; the 17 without a team get null, not a default 1.0.

Caveats, since this is a tiebreaker and not a ranking:

- last season's defence is a noisy guide to this one, and rosters move
- the spread is widest at TE and QB, where the pool is small, so the extremes are
  partly small-sample rather than real schedule strength
- it moves players *inside* a tier. Taking a worse player for a softer week 16 is not
  what this column is for.

Most actionable form is the same-tier swap: Nabers (1.10) over Garrett Wilson (0.90) at an
identical ADP of 36; Tyler Warren (1.18) over Colston Loveland (0.85) at 48 vs 45.

## Conflict: championship-leverage.md vs the live league

`docs/championship-leverage.md` states **8 of 12 make the playoffs, 3 weeks, no byes**, and
builds its central strategic argument on it. The live league says otherwise. From raw
`mSettings.scheduleSettings`, verified Aug 21 2026:

| Field | Value |
|---|---|
| `playoffTeamCount` | **6**, not 8 |
| `matchupPeriodCount` | 14 |
| `playoffMatchupPeriodLength` | 1 |
| `playoffReseed` | False |
| `playoffSeedingRule` | TOTAL_POINTS_SCORED |

Six of twelve with three one-week rounds means **seeds 1-2 get byes**. Per the rule at the top
of this file, the API wins and the doc is stale. The consequences invert its argument:

- "Regular season is nearly a formality" -- false. Half the league misses.
- "P(make playoffs) comfortably north of 85%" -- far lower at 6/12.
- "dP(champ)/d(regular-season win) is close to zero" -- false; wins are the scarce good.
- "Seeding buys marginally easier matchups and nothing else" -- false. A bye skips an entire
  single-elimination round, which is the largest single seeding prize available.
- The "uncomfortable corollary" (marginal value of wins possibly negative) does not follow.

`hub/draft/season.py` implements the API's structure: 14 weeks, top 6, two byes, no reseed.
The doc's downstream reasoning about ceiling-over-floor needs redoing against 6-of-12 before
any of it drives a draft decision.

**Resolved for the lineup layer (2026-08-23), in `hub/season/lineup.py`.** The redo did not
produce a new blanket rule. Inverting "never sacrifice ceiling" into "always take the floor"
would be the same error pointed the other way, and the doc's own point 3 already says the
right thing: variance preference is state-dependent and sign-flips. So the weekly objective
is P(win this matchup), and ceiling-or-floor is an output rather than a setting. On a test
roster a 35-point underdog quadruples his win probability (0.9% -> 3.6%) by starting the
volatile player, and a 35-point favourite gives up three projected points to avoid him.

**Resolved for the draft layer (2026-08-23), measured in `hub/draft/leverage.py` and written
up in `docs/six-of-twelve.md`.** 20,000 simulated seasons against the real structure:

- League-average roster makes the playoffs **50%**, not "north of 85%".
- A marginal regular-season win is worth **4-6 pp of title equity** (9+ for a strong
  roster), not "close to zero". The gradient steepens with strength.
- P(title | seed) is 37.9% / 26.9% for the bye seeds vs 12.5-6.0% for seeds 3-6. Seed 2 is
  worth 2.2x seed 3 across one place, so "seeding buys marginally easier matchups and
  nothing else" is false.

The ceiling advice split in two rather than inverting. At a fixed team mean, **season-long
outcome spread pays at every roster strength** (10x for a weak roster, +39% for a strong one,
even while costing it 15 pp of playoff probability — berths trade well for byes because the
seeding payoff is convex), while **weekly boom-bust does not** (flat weak, negative strong,
because head-to-head wastes surplus). The doc treated these as one quantity.

Worth recording as a near-miss: the first sweep found variance helping everywhere, because
a starting lineup is a max over the roster, so scaling player spread by 1.8x raised the
team's *mean* by 15% before any variance effect. `calibrate` removes it and a test pins it.
The correction reversed the sign of the answer.

**`TALENT_CV` fitted (2026-08-23), 0.35 -> 0.41.** `six-of-twelve.md` named it as the one
number that would overturn its conclusions, since the season-long sweep is a sweep in
exactly that quantity. Fitted against this league's own past drafts -- a pick is market
opinion recorded before week 1, so it cannot be revised after the fact the way a stored
projection can. 460 drafted skill players over 2023-25: **0.411, 95% CI [0.384, 0.437]**,
putting the old guess 4.6 se low. Rerunning at the fitted value moved every number slightly
and strengthened every conclusion; nothing reversed. `hub.draft.calibrate`,
`docs/talent-cv.md`.

Two things that would have biased it and are handled: weekly sampling noise is subtracted
per player (it is 0.154 of the 0.439 raw dispersion), and 2022 is excluded because a quarter
of its drafted players have retired out of ESPN's universe and those are the busts. Only the
second moment was fitted; the quantiles and near-zero skew came out right unprompted, which
is the model's normal-multiplicative talent assumption validating itself.

**Made per-position the same day.** `TALENT_CV_BY_POS = {QB 0.41, RB 0.49, WR 0.41, TE 0.32}`,
shrunk toward the pool in proportion to each estimate's noise -- 51 tight ends do not support
an independent number. Only RB (+2.6 se) and TE (-3.8 se) really differ; QB and WR sit inside
one standard error and shrink back onto the pooled value, which is the honest answer rather
than a tidier one. Effect on valuation is real but small: two rosters with identical
projections, one RB-tilted and one TE-tilted, move about 0.4 pp of title equity apart beyond
what slot eligibility already explains.

**It also surfaced a model bug the constant had been absorbing.** Weekly points were drawn as
N(realised talent, 0.55 * *projection*) and clipped at zero, so a player projected at 15 whose
talent collapsed still averaged 3.3 points a game -- 22% of his projection, manufactured
entirely by the clip. The simulation could not produce a bust, and the calibration kept asking
for a larger constant (0.457) to make the aggregate dispersion come out right. `simulate_weeks`
now scales weekly spread by realised talent; an average player is unchanged, and the gap
between the fit and what the model needs collapses from 10% to under 2%.

**Weekly spread fitted (2026-08-23), and it was the wrong shape rather than the wrong value.**
`sd = 0.55*mu` assumed spread proportional to the mean. Across 1,174 player-seasons of
nflverse weekly scoring the exponent in `sd = k*mu^b` is **b = 0.498 +/- 0.012** -- the
Poisson value, ~42 se from the assumed 1. Shipped as `sd = k*sqrt(mu)` with
k = {QB 1.88, RB 2.07, WR 2.13, TE 1.99}; RMSE predicting a player's weekly sd falls 31-36%.
`docs/weekly-spread.md`.

That exponent is derived, not fitted-and-hoped: fantasy points aggregate count-driven
components, and a sum of counts has spread growing with the square root of its mean.
Touchdowns are 54% of a QB's weekly variance off 32% of his points.

Three consequences. Relative volatility is now a property of the projection rather than a
setting -- a 5 ppg flier is nearly twice the per-point lottery a 20 ppg starter is, where
the old constant called them identical. Most of the apparent *position* difference in weekly
CV (0.47 to 0.73) was positions differing in mean points; once the law is right, k spans only
1.88-2.13. And the sd floor of 2.0 is gone -- it gave an unprojected player real spread,
which the best-lineup max turned into free points off the bench.

`TALENT_CV` was refitted against the new law since the two are coupled: pooled 0.41 -> 0.42,
per-position by ~0.01, all inside their intervals, nothing downstream reversed. The QB
anomaly flagged earlier resolved as suspected -- under the sqrt law QB sits at -0.0 se from
the pool, so 0.55 had been over-subtracting for the steadiest position.

**Component projection layer built (2026-08-24), `hub/models/components.py`.** Screened
first: aggregating components and regressing touchdowns beats carrying points forward,
RMSE 3.30 vs 3.38 over 1,049 player-season pairs, 99.7% on a paired bootstrap. Optimal shrink
on a player's own TD rate is **1.0** -- it carries no information beyond his yardage. But the
gain on the *mean* is only ~2%, concentrated in RB and WR, and QB is marginally worse. The
payoff is the distribution, not the mean, and the doc says so.

Distribution families are measured rather than assumed: volume counts are overdispersed
(carries 2.39, attempts 2.45, receptions 1.20 -- usage moves with game script) so negative
binomial; touchdowns are *under*dispersed (0.83-0.86, bounded scoring chances) so binomial.
Yardage is compound -- a week's yards are a sum over catches -- and modelling it as constant-CV
Gamma reimposes the proportional spread law this layer replaces. The first version did exactly
that and produced spread growing as mu instead of sqrt(mu).

Validated against 760 real player-seasons: mean unbiased to +0.05 ppg, sd 6.05 vs 6.76
observed -- but 5.3% of the observed figure is within-season role drift, so detrended the gap
is 2.7%. Skew 0.73 vs 0.60, overstated except at RB where it lands at 0.68 vs 0.67; likely
cause is volume and efficiency co-moving within a game, recorded not fixed.
`docs/component-projection.md`.

**`simulate_weeks` now draws a shifted gamma** matched to (mean, spread, skew) rather than a
normal. The median week is ~0.90 of the mean in reality and 1.00 under a normal, so the model
had been flattering every floor-based decision.

**Found: `hub.draft.leverage` was running a superseded model.** It re-implemented
`simulate_weeks`' internals instead of calling it, and so kept proportional spread and normal
draws long after the simulator moved on -- meaning every number in `six-of-twelve.md` was
computed against a model the repo had stopped using. Now routed through the simulator with a
behavioural test pinning them together. Conclusions all survived; one refined -- extra weekly
spread mildly *helps* a weak roster (2.5% -> 3.1%) where it read as flat, because a skewed
distribution gives an underdog upside a normal hid.

**Volume model screened (2026-08-24) and came back null.** Year-over-year persistence
confirms the premise -- volume persists better than points (targets 0.805, carries 0.791 vs
ppg 0.775) and touchdown rate per yard has *zero* persistence (-0.004 receiving, -0.030
rushing), independently confirming the full-shrink result by a different route. But shrinking
volume and efficiency by their measured persistence does not beat regressing touchdowns
alone: RMSE 3.311 vs 3.303, paired bootstrap +0.008 [-0.057, +0.070], P(better) 40.2%.
QB alone shows +9.7% at P=97.5%, and is *not* taken -- one of four positions tested, interval
touching zero, and shipping the slice that worked is the failure mode the screens exist to
avoid. Likely cause: shrinking toward one positional mean over-shrinks the studs; an
ADP-implied prior is a different and untested thing. `project()` unchanged.

**L1 teammate correlation gated and passed, narrowly (2026-08-24).** Measured within-game
correlation of standardised weekly points: QB-WR1 +0.323, QB-WR2 +0.293, QB-TE1 +0.252,
everything else within +/-0.06 of zero -- including WR1-WR2 at +0.025, against the doc's
+0.16. At position level: QB-WR +0.232, QB-TE +0.225, QB-RB +0.054, rest ~0.

The gate (fit 2022-24, evaluate held-out 2025): for a lineup holding a QB and his own pass
catchers, independence gives 72.9% coverage on a nominal 80% interval -- materially
overconfident -- and correlation puts it at 80.4% with a better log score. For a lineup with
no QB, independence is already calibrated (79.3% vs 79.1%) and correlation changes nothing.

So L1 is three numbers, not a copula over six positions conditioned on game total and spread.
`hub/season/lineup.py` now prices lineup spread with teammate covariance; stacking helps an
underdog and hurts a favourite, worked out per matchup rather than as a view on stacking.
`docs/correlation.md`.

Still open: `simulate_weeks` has no NFL-team input, so the draft optimizer and leverage
harness still treat a stacked roster as independent -- understating stack variance at draft
time, in the weeks a stack is for. Opponent-side correlation is also unpriced.

**Volume model rebuilt on an ADP-implied prior (2026-08-24) -- the diagnosis held, the
verdict is mixed.** The earlier null shrank volume toward a positional mean; swapping that
for a prior implied by the market's own draft pick turns it into a 99.9% result against the
same baseline (RMSE 3.474 vs 3.720, paired bootstrap -0.247 [-0.385, -0.104]). A WR1 and a
WR5 really do not regress to the same place, and the market prices where each one does.

But it does **not** beat simply reading the pick: market alone 3.578, model 3.474, 87% on a
paired bootstrap -- below anything here counts as a result. It also ties a trivial blend of
the two point predictions (3.482), which is the tell that the structure is not buying
anything the arithmetic did not.

So `hub/models/volume.py` ships as a **decomposition**: `decompose(pick, position,
target_ppg)` reproduces the market's mean exactly and supplies only the shape. That shape is
what `components.sample_weeks` needs and what a points projection cannot give -- the bridge
between the board and the sampler. `docs/volume-model.md`.

Fitted on 526 pairs from the league's own drafts joined to nflverse via the `ff_playerids`
crosswalk (a real ID join, 90% match, not name matching), held out by season. Volume is
trusted less than efficiency (keep 0.5 vs 0.7): volume is what a changed situation moves, and
the pick is the only input that knows it changed. Extrapolation clamped to each position's
observed pick range -- unclamped the TE curve claims 10.8 targets a game at pick 3, past
every data point it has.

**Touchdown luck shipped to the board (2026-08-24), `hub/draft/regression.py`.** The chain:
TD rate per yard has zero year-over-year persistence (+0.004 rec, -0.030 rush), full
regression is optimal, and the *market does not fully regress it*. Regressing draft position
and next-season points on the same standardised prior-season yardage and touchdown points:
for QB the room weights touchdowns at 1.02 against volume where their true predictive weight
is -0.05 -- gap +1.07 [+0.21, +2.11], 98.9%. RB +0.25 (91%), WR +0.10 (83%), TE -0.14 (27%).

Do **not** pool these: pooled it reverses sign, because QBs earn far more TD points and go
much later. Simpson's paradox, and the first version of the test reported the pooled row.

Honest strength: QB is a result, RB/WR are directional, TE is nothing. Shipped where the
earlier QB-only volume win was not, because the mechanism was measured first and independently
and predicts this sign, three of four positions show it, and the QB gap is twentyfold rather
than marginal.

Distinct from the board's existing `fp_over_expected` (realised vs *opportunity*-expected
points): they correlate at +0.16 on the live board and are signed opposite, so a real overlap
would show strongly negative. `docs/td-luck.md`.

**Historic ADP: not used, and not worth using.** Checked while considering refitting the
volume curves on ADP rather than realised pick. ESPN retains 169 as an "undrafted" sentinel
for 78%/73%/69% of players in 2022-24 and **100% in 2025** -- the whole season is a constant
170.0. Refitting on it would mean less data and a signal truncated at 169. More basically,
historic ADP has no role in *predicting*: the curve is fitted once on history, where realised
picks are the cleaner record, and what predicts 2026 is 2026 ADP, which is live and complete.

**Decomposing a projection into components does not beat the sqrt law** for weekly spread:
mean |error| 1.365 vs 1.140 predicting a player's observed weekly sd, P(components better)
0.0%. It underestimates (6.22 vs 7.19 observed) because a decomposed line is the average
player at that pick with no player-specific shape. `weekly_moments` unchanged.

**The board now leads with the objective (2026-08-24).** `--pick N` runs championship
equity by default and prints the pick; `--no-win-prob` falls back to the VOR shortlist. The
VOR list is labelled as the shortlist that feeds the simulation, not as the answer -- every
selection should maximise P(win the league), and the board was leading with a proxy for it.

**Two things had to change for that to be honest.** The defaults were too thin: at 12 draft
rollouts x 150 seasons the ordering swung 3.5 places between runs, which is unusable. Raised
to 24 x 300 (~15s end to end, fine on a draft clock), where the swing is 0.9 places.

And even at 48 x 600 (50s/run) the top pick still alternated between Nacua and McCaffrey,
because they are genuinely tied -- +2.72 vs +2.58 with standard errors near 0.4. Printing a
strict order there asserts a distinction the simulation cannot make. `rank_tiers` now marks
everything within two pooled standard errors of the leader as tied for the lead, and the
board says so and suggests breaking the tie on something the simulation does not model. The
TAKE tier contains both in 5 of 5 seeds; the strict top pick does not.

**Touchdown luck now reaches the objective, at QB and WR.** ESPN's *projection* carries the
same touchdown bias the draft room does, but only for quarterbacks: `ppg_next ~ proj_ppg +
td_luck` gives -0.540 [-1.057, -0.125], 99.5%. WR is -0.286 [-0.797,
+0.170], 89% -- applied at Jackson's direction, and recorded as a judgment call rather than a
95% result. RB stays out on sign, not threshold: +0.253 means ESPN is if anything conservative
there, so correcting it would move the projection the wrong way. Effect: Josh Allen
22.0 -> 19.5, Davante Adams -1.13, CeeDee Lamb +0.47. A signal that is
only printed is decoration; `hub.draft.optimize` scores seasons against `proj_blend`, so this
is where it had to go.

**Durability priced per player (2026-08-24), `hub/draft/durability.py`.** `TALENT_CV`
already carried availability, but only as a positional average, so every RB had identical
injury risk. Games missed persists year over year at r = +0.407 -- stronger than the folklore
-- and the practical form is a 35-point spread: a player who missed 6+ games last season
misses 3+ this season 76% of the time against a 55% base rate, one who missed none 41%.

The projection does not price it: `ppg_next ~ proj_ppg + missed` leaves -0.186 per game on
points per *team* game, P(<0) = 100%. One season is enough -- two years scores R2 0.4597 vs
0.4614 for one, second-year beta -0.040 vs -0.159.

**The surprise is that running backs are already priced**: -0.065 at 71%, nothing. The market
discounts RB durability because everyone knows RBs break, so there is no residual. The
inefficiency is QB (-0.457, 100%) and WR (-0.151, 99.6%), which are applied. TE (89%) is not.

**Current injury status is now priced too, for Out/Doubtful/IR only.** The first pass claimed
nothing could be fitted; that was asserted without checking and was wrong -- nflverse
publishes weekly injury reports back to 2019, and a week-1 designation is the closest
analogue to an August one. Fitted: Out/Doubtful -1.631 [-2.554, -0.736], 100%; Questionable
-0.949 [-2.495, +0.459], 90.2%.

Questionable stays unpriced, and mainly not for the p-value: an August QUESTIONABLE is a
different population from a week-1 one -- 12.6% of the August board against 2.9% at week 1,
**4.4x more common**. Out/Doubtful/IR transfer far better (4.1% vs 2.1%) and are applied at
every position, since being ruled out is news rather than a trait the market has discounted.
IR borrows the Out coefficient -- nobody on IR is on a practice report, and starting there
means missing 4+ games by rule, so it understates rather than overstates.

On the pick-3 tie that prompted this: Nacua missed 1 game in 2025, McCaffrey 0. The trait
flags neither, and RB is unpriced regardless, so the model says the concern is smaller than
intuition suggests for both. Verdict unchanged. `docs/durability.md`.

**The market beats our decision layer on realised outcomes (2026-08-24).** Backtest: draft a
past season off that season's market, score on what actually happened. Three seasons, 40
drafts each, slot 3. Following the market (`adp_need`) lands at exactly 1/12 best-roster rate
and beats value-over-replacement by **5.06 points per team game, [-7.90, -2.18], P(better) =
0.0%**. Raw projected points is catastrophic (-18.6).

**The in-simulation answer said the opposite and was an artifact.** Scored on `proj_blend`,
vor_need reads +17.25 pp; scored on realised points it is negative. `optimize.py` always
carried the caveat that the season is scored on the same projection the greedy ranks on --
this quantifies it at ~17 pp, larger than any real effect measured here. An ADP-scored version
is circular the other way, so neither in-sim column is evidence.

Two harness bugs found and fixed en route, both of which produced a wrong answer first: a
strawman opponent that drafted position-blind (five QBs, no TE) and handed every strategy
+24 pp; and an additive need weight on incomparable scales that had *my* VOR strategy drafting
4.9 QBs in a one-QB league. `hub.draft.optimize`'s lexicographic rule (need gates, value
breaks ties) is correct and was adopted. The result survived the fixes.

Consequence: do not try to out-project or out-order the market. ADP already contains the
scarcity argument VOR exists to make. The remaining edges are the corrections the projection
demonstrably omits (TD luck, durability, current OUT/IR) and the things that are not about the
player at all (slot, survival to next pick, weeks 15-17 SoS). `docs/market-value.md`.

**Scoring weights are the league's, not ours.** `components.SCORING` was hardcoded full PPR
and assumed. Verified against `mSettings`: matches on all nine items, no per-position
overrides. `make draft` now reads the league's weights and prints a loud mismatch if they
diverge -- half-PPR would otherwise silently mis-score everything.

**Direction set: project components, aggregate to points.** Weekly *spread* now comes from
the component structure; the weekly *mean* still arrives as a points projection. Doing the
same to the mean is the next build -- volume persists where touchdowns do not, and it is the
only route to the component-level correlation `championship-leverage.md` calls L1.
`hub.fetch.nflverse` now carries `player_stats` (weekly, per player, all components), and
full-PPR reconstructs from them to within 0.01 for 99.4% of player-weeks.

## Why we backtest: to audit ourselves, not to find edges

**Decided 2026-08-24 (Jackson).** Standing direction, so a future session does not relitigate
it.

**Do not backtest looking for an edge over the market.** Sportsbooks and the drafting room
have already done that work, and their price is the aggregated result of everyone who has
tried. Four independent attempts in this repo have now failed the same way:

| attempt | result |
|---|---|
| Component projections vs carrying points forward | +2% RMSE, real but tiny |
| Volume model shrunk to a positional mean | null (3.311 vs 3.303) |
| Volume model shrunk to an ADP-implied prior | beats history 99.9%, **does not beat the pick** (87%) |
| Value-over-replacement draft ordering | **loses** to drafting the market, -5.06 pts/team-game, P(better) 0.0% |

Jackson's framing, and it is right: *the market prices outcomes well.* Trying to out-project
or out-order it is arbitrage against people with more money and better data.

**Do keep backtesting to audit our own machinery**, which is a different job and the one that
has repeatedly paid. Sportsbooks price player and game outcomes; nobody prices "which pick
maximises P(win) from slot 3 in a 12-team, 6-of-12 league", so that question can only be
answered against realised outcomes. In one session the backtest:

- Overturned a **+17.25 pp** in-simulation result that was pure circularity — the season was
  scored on the same projection the greedy ranked on, and we would otherwise still be acting
  on it ([market-value.md](market-value.md)).
- Caught a strawman opponent model that drafted position-blind, finishing with five
  quarterbacks and no tight end, and handed every strategy +24 pp for beating it.
- Caught an additive roster-need weight on incomparable scales that had *our own* VOR
  strategy drafting **4.9 quarterbacks** in a one-quarterback league.

All three produced confident, wrong answers first. No market price would have caught any of
them.

**So the rule:** a backtest whose hypothesis is "we beat the market" is not worth running.
A backtest whose hypothesis is "our code does what we think" is worth running every time.

**Where the remaining edge is**, given all of the above — narrow, and all of it either a
correction the market's projection demonstrably omits or something the market never priced at
all:

- Corrections measured *against* the market rather than replacing it: touchdown luck at QB and
  WR ([td-luck.md](td-luck.md)), durability at QB and WR, being ruled out today
  ([durability.md](durability.md)).
- Things that are not about the player: your slot, who survives to your next pick, weeks 15-17
  schedule under a 6-of-12 bracket ([six-of-twelve.md](six-of-twelve.md)).

**Practical note — odds API, checked 2026-08-24 after Jackson set the key.**

Key works: 500 credits on the plan, 496 remaining after probing. `/sports` and `/events` are
free and do not count against quota. Useful operational fact: **an event with no bookmakers
returns 200 and costs nothing**, so probing for coverage is free.

| question | answer |
|---|---|
| Season-long player totals? | **No.** Nine naming variants tested, all rejected as invalid markets |
| Per-game player props in August? | **Yes, but only for the marquee opener** |
| Coverage of the week-1 slate | **1 of 16 games** — 3 books, 24 players |
| Two ordinary Sunday games | 0 books, 0 players |
| Cost for a full week's props | ~4 credits/event, **64/week** |

**Not usable for the Sep 3 draft.** 24 players priced against ~170 draftable is not a board.
The earlier caution here was right that season-long props do not exist, and wrong that nothing
would be available in August — the opener is priced early because it is a marquee game, and
the rest of the slate arrives closer to kickoff.

**Season totals do not exist here, and that is settled.** Batch-probing
`player_pass_yds_season`, `player_reception_yds_season`, `player_receptions_season`,
`season_player_*`, `player_season_*` and four more all come back `INVALID_MARKET`. The only
futures market is `outrights`, which is team-level (Super Bowl winner), not player-level.
Some books do post season-long player lines; The Odds API does not carry them. Probing for
this is free — invalid markets return 422 at zero cost.

**Genuinely useful in-season, with a quota caveat.** The valid player markets are
`player_pass_yds`, `player_rush_yds`, `player_reception_yds`, `player_anytime_td`,
`player_receptions`, `player_rush_attempts`, `player_pass_tds`, `player_tds_over` and
`player_pass_yds_alternate` — which is the **full component set, volume included**. That
matters more than the yardage markets alone: volume is the persistent part (targets 0.805,
carries 0.791 year over year) and touchdowns are the noise, so a market that prices
`player_receptions` and `player_rush_attempts` is pricing exactly the half worth having.
These are the components `hub/models/components.py` aggregates. That is the convergence of both
of Jackson's directions: money-backed market prices, expressed as component stats rather than
as a fantasy aggregate. But 64 credits/week against a 500/month plan covers about two weeks a
month before the spread snapshots are counted, so weekly full-slate prop pulls do not fit the
free tier. Pulling only the players on your roster and your opponent's would.

## Draft-night pick feed: what ESPN actually publishes

**Tested 2026-08-24 against a live practice draft**, because the whole live path had never
run against a draft in progress.

### What is readable, and what is not

| | readable via REST? |
|---|---|
| Real league, **completed** draft | **Yes** — 2025 replay pulled all 192 picks |
| Mock/practice room, **mid-draft** | **No** — 52 picks made, every view returned zero |
| Real league, **mid-draft** | **Untested**, and untestable without drafting |

The mock finding is solid: with the draft at round 5, pick 53 on screen and the drafter
holding 5 of 16 roster spots, `mDraftDetail`, `mRoster`, `mTransactions2` and `players_wl`
all reported no picks and no rostered players. `draftapi.fantasy.espn.com` does not resolve,
and `communication/` 404s for the room. ESPN serves a mock draft over a websocket to the
client and writes nothing to the read API.

**A retraction worth recording.** This was first concluded from a league id that was stale --
the practice room had been recreated and the live one had a different id -- and from a draft
that had not started. Both facts were consistent with "the API is blind" and neither was
evidence for it. The conclusion happened to be right, and the reasoning was not. Re-tested
against the live room with 52 picks down before it was recorded.

### What this means for 2026-09-03

The real draft is a persistent league, not an ephemeral lobby room, and its completed draft is
demonstrably readable. Whether it publishes *during* the draft is genuinely unknown. So:

1. Try `--sync` first on the night. If picks appear, draft night is automatic.
2. If it returns zero, fall back to reading the room directly.

### Reading the room

Not screenshots. OCR over ~170 player names trades a solved problem for an unsolved one. The
Claude-in-Chrome extension reads the DOM and returns exact strings, and can be polled.

The efficient thing to read is **not the pick feed** but the player list itself: with ESPN's
`Show Drafted` toggle off, the visible list *is* the available pool, so there is nothing to
accumulate and no way to fall behind by missing a poll. Tracking picks incrementally is the
version that breaks when a sync is missed.

`hub.draft.state.sync_from_espn` now takes a `league_id`, so any room can be targeted; the
configured league stays the default so draft night needs no extra argument.

### How FantasyPros solves this, and what we would build

Their Draft Assistant syncs with ESPN mock drafts, which the REST API cannot support -- so
whatever they do is client-side. Their own setup instructions give it away: the ESPN draft
room must be open **in a Chrome tab**, the assistant in another, and the documented fix for a
broken sync is to **reload the draft room tab**. That is a content script attaching at page
load, not an API integration.

Three ways such a script can get picks, in increasing robustness: scraping the DOM with a
`MutationObserver` (breaks on any reskin, and the player list is virtualised so only rendered
rows exist); reading the app's own JS state; or wrapping `window.WebSocket` before ESPN's
bundle runs and reading the frames the client itself renders from. The last cannot drift from
what is on screen, and is what we would build.

**Decided (Jackson, 2026-08-24): nice-to-have, not draft-night infrastructure.** It is more
moving parts than anything else in the repo with ten days to the draft, and two unknowns are
unresolved -- whether ESPN uses a websocket or long-polling, and whether picks carry player
ids or names. A half-built socket interceptor at 9:01 PM on Sep 3 is worse than typing names.

### So typing picks is the primary path, and it had a sharp edge

`--taken` recorded whatever it was given and said only "N picks recorded". A mistyped name
matched nothing, so the misspelt player stayed on the board as available and the next
recommendation could hand back someone already drafted -- silently, which is the worst
version of it.

It now checks each name against the board and says so, with a suggestion when one is close:

    NOT ON THE BOARD: 'Bijan Robinsen' -- did you mean 'Bijan Robinson'?
      until that is fixed, 'Bijan Robinson' is still shown as available

A name matching nothing gets no guess, because kickers and defences are drafted every round or
two and the board excludes them by design -- guessing at those would train the reader to
ignore the warning.

## Open questions

1. **Pool configuration** — entries, payout, rebuys. Under ~20 entries play near max win
   probability; above ~100, survival stops being sufficient and the objective becomes
   P(finish first), which is materially more contrarian. Only Jackson can get this.
2. Which sportsbooks are accessible in NY (determines whether line shopping is possible).
3. Custom domain vs `username.github.io`.
4. Whether this survives into 2027 or is a one-season artifact.
