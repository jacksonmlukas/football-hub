# Pro Football Reference: what we already read, and eight angles on the rest

**Written 2026-09-05.** A research note, not a plan. Nothing here is committed to, and every
angle carries the gate it would have to clear. No effect size below has been measured — the
point of the note is to say which questions are worth spending a screen cell on, and which
are not.

## We already read it, and it is behind the best signal we have

PFR arrives through nflverse rather than directly. `hub.fetch.nflverse` serves snap counts,
`hub.models.panel.snap_share` and `hub.models.spread` read `offense_pct`, and that feeds
`snap_trend` — the feature that cleared the weekly screen at +0.043 and the subject of
[ADR-0013](adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md). `hub.contracts` already
characterises the source's quirks: the percentage ceiling sits at 1.05 rather than 1.0 because
`st_pct` reaches 1.01 from PFR's own rounding.

So the question is not whether to use PFR. It is what else nflverse redistributes, and whether
any of it buys something.

## What else is reachable

Four loaders, none of them wired into `hub.fetch`:

| loader | shape (2025 weekly) | carries |
|---|---|---|
| `load_pfr_advstats(stat_type="rush")` | 2,355 rows | yards before/after contact and their averages, broken tackles, carries |
| `load_pfr_advstats(stat_type="rec")` | 4,533 rows | broken tackles, drops and drop rate, targeted passer rating, interceptions |
| `load_pfr_advstats(stat_type="pass")` | 684 rows | pressures, hurries, hits, blitzes, sacks taken, bad-throw rate, drops by receivers |
| `load_pfr_advstats(stat_type="def")` | 7,926 rows | per-defender coverage: targets, completions and yards allowed, ADOT, missed-tackle rate |
| `load_snap_counts` | 26,612 rows | `defense_pct` and `st_pct`, both fetched and validated, both unread |
| `load_draft_picks` | 773 (3 drafts) | round, pick, age, college — and career outcomes, see the trap below |
| `load_combine` | 650 (2 classes) | forty, bench, vertical, broad jump, cone, shuttle, height, weight |

There is no air yards, no depth of target and no yards-after-catch on the *receiving* side.
Whatever the receiving equivalents of the rushing contact splits would be, PFR does not
publish them here.

## Two constraints that govern every angle below

**A screen cell is not free.** `docs/weekly-screen.md` and the 2026-09-05 audit put the family
at roughly 2,000 partial correlations with no multiplicity correction, and the power analysis
is worse than the count: the screen detects `|r|` around 0.027, while the features that have
cleared run 0.023 to 0.043 — incremental R² near 0.16%. A screen tuned to find effects too
small to survive a decision gate will keep clearing them. Adding features widens that family.
The angles worth taking are the ones carrying information the board **structurally lacks**,
not the ones offering another small usage correlate.

**Everything in advstats is a week-*w* outcome.** The panel's rule is that a predictor must be
published *for* week `w` before kickoff — the rule the audit caught `wind` violating. So none
of this is a pre-kickoff fact for the week it describes. It is usable lagged, over weeks
`1..w-1`, exactly as `snap_trend` already is; or aggregated over a prior season as a preseason
board input. The second use is the more interesting one, because that is where the board's
gaps are.

## The angles, ranked, each with the gate it must clear

### P1. Yards before contact as the board's first team-context signal

The board carries no representation of offensive-line quality, coordinator change, or scheme —
audit finding C6, and the reason a back who changed teams is priced on last season's
production as though nothing moved. Yards *before* contact is substantially a measurement of
the blocking in front of him, attached to a player row.

This is the strongest angle in the note, and not because the correlation is expected to be
large. It is the only candidate carrying a category the board has none of. Every other usage
signal competes with `snap_share` and `ppg_before` for the same variance; this one does not.

**The gate:** prior-season yards before contact per carry, as a preseason board input, screened
against the existing controls with the sign pre-stated. It has to clear the same bar
`snap_trend` did — the pre-stated sign holding in every held-out season, and roughly 2 se —
and then a decision gate, which is where the audit expects most cleared signals to die.
**What would kill it:** if it is collinear with the back's own prior production, it adds
nothing that `ppg_before` does not already carry.

### P2. Draft capital as the prior for players with no NFL history

Every other angle here is an NFL weekly statistic, and a rookie has none. The audit is explicit
that rookies are half an early board, and that the board's imputation makes their projection a
deterministic transform of consensus rank — so the board cannot disagree with consensus about
them by construction. Draft round and pick number are the classic prior for exactly this
population, are pre-draft by construction, and cost two integer columns with no crosswalk pain.

**The gate:** does draft capital, added to the imputation, let the board's rookie projections
diverge from consensus rank *and* score better? The second half matters — divergence alone is
easy and worthless. **What would kill it:** consensus already prices draft capital. Analysts
know where players went. The honest prior here is that most of the signal is already in ECR,
and the residual is small.

### P3. Yards after contact as a returning-player role signal

The complement to P1: what the back did once the blocking ran out. Two backs at identical
yards per carry can be entirely different assets, and nothing on the board distinguishes them.

**The gate:** as P1, and it should be screened *jointly* with it — they will correlate, and the
repo has already been bitten by a collinear twin destroying a real signal, which is why
`route_trend` sits outside the default screen despite clearing on its own.
**What would kill it:** joint screening annihilates both, the `snap_trend`/`route_trend`
outcome repeating.

### P4. Passing drops as the QB analogue of touchdown luck

`hub.draft.regression` already ships a correction of exactly this shape: a prior-season
quantity that the market misprices because it looks like performance and behaves like luck.
Drops by a quarterback's receivers are that shape — they cost him completions and yards he
did not fail to earn, and receiver hands are not stable across a roster year.

The attraction is that the *machinery* already exists. This would be a new coefficient in an
existing correction framework rather than a new kind of thing.

**The gate:** the same one `td_luck` cleared — a walk-forward `ppg_next ~ proj + drops`
regression with a season-clustered interval, fitted against `proj_blend`. Under the corrected
fitting discipline, not the uncommitted kind the audit found.
**What would kill it:** the QB sample is thin. 684 rows for a full season is roughly one
starter per team-game, so a season-clustered fit has very little to work with, and the audit's
MDE argument applies with force.

### P5. Pressure faced as a QB stability signal

Pressure rate is more a property of the line and the scheme than of the quarterback, and is
generally more stable year over year than the sacks it produces. A quarterback whose sack rate
spiked while his pressure rate did not is a candidate for regression.

**The gate:** does pressure-rate-adjusted prior production predict next-season points better
than raw prior production? **What would kill it:** the same thin QB sample as P4, and the
market may already price line quality through the team's win total.

### P6. Special-teams snap share, already fetched and unread

The cheapest thing in the note. `st_pct` is already pulled, already contract-validated, and
`hub.contracts` says outright that it rides along unread. The hypothesis is inverse and almost
mechanical: a genuine offensive contributor does not play special teams, so a skill player with
high `st_pct` has a smaller role than his rank implies.

**The gate:** one screen cell, sign pre-stated negative, against the usual controls.
**What would kill it:** `offense_pct` already says this more directly, making `st_pct` a noisier
restatement of a column the screen holds.

### P7. Per-defender coverage stats as a better defence-versus-position

`def` advstats are per **defender**, not per defence: targets, completion rate allowed, yards
per target, ADOT, missed-tackle rate. Audit finding C9 says the current `playoff_sos` DvP is
raw, unadjusted for the offences each defence faced, unregressed, and blind to personnel
turnover — a real weakness. Per-defender data could support a genuinely better version.

**The gate:** does a personnel-aware DvP beat the raw one at the thing DvP is used for?
**What would kill it, and why this is deferred rather than ranked higher:** using per-defender
coverage properly needs matchup assignment — which corner travels, who covers the slot — and
that is a modelling project, not a feature. The effort is an order of magnitude above
everything else here, and `dvp` already cleared the screen in its raw form.

### P8. Not recommended: combine athleticism

Forty, vertical, cone, shuttle, broad jump, bench, height, weight. The appeal is that they
apply to rookies, where P2 also lives. The problem is that draft capital is the better version
of the same information — teams observe the testing and price it — so combine numbers are
largely subsumed by where a player was drafted. Coverage is also partial: not every prospect
tests, and non-participation is not random.

**If pursued anyway,** it must be screened *after* draft capital, as a residual. Screened alone
it will look predictive for a reason that is not its own.

### P9. Not recommended: receiving advanced stats as a family

Broken tackles, drops, drop rate, interceptions on targets, targeted passer rating. Counts are
low, several are outcome-heavy, and `receiving_rat` in particular is mostly a measurement of
the quarterback throwing to him. Screening five thin columns as a family is exactly the
multiplicity cost the note opens with, for the least promising material in the set.

**If any single one is taken,** drop rate is the defensible pick, and it should be entered as a
**pre-stated null** — the literature's position is that drops do not persist, and a null that
comes back significant is a finding against the pre-registration rather than a feature.

## The trap in `load_draft_picks`

That table mixes two kinds of column and only one is safe. `round`, `pick`, `age`, `college`
and `position` are pre-draft facts, fixed at the moment of the draft. But `w_av`, `car_av`,
`dr_av`, `allpro`, `probowls`, `seasons_started`, `games`, `to` and every career stat total are
**updated to the present** — they describe what the player went on to do.

Joining those onto a 2022 historical board imports 2023-2025 outcomes into a 2022 prediction.
That is `docs/method.md` rule 2, and this repo already has a 7.4-sigma result in its record
from precisely this shape of mistake. If a fetcher for this table is ever written, the contract
should **drop the outcome columns at ingest** rather than trusting a caller to avoid them —
the same reasoning that put the `scrape_date` lower bound in `tune.holdout`.

## What is deliberately not concluded

- **No effect size is claimed.** Nothing here has been screened. The ranking is by whether the
  information is structurally absent from the board, not by expected correlation.
- **Nothing is recommended for the weekly screen as a family.** Given the multiplicity position,
  the defensible move is one or two pre-registered candidates, not a sweep.
- **Direct scraping is out of scope and not considered.** Sports Reference's terms prohibit bulk
  collection and this repository is public and publishes artifacts. Everything above is reachable
  through nflverse, which is also the path `hub.fetch` is currently being consolidated onto —
  a direct fetcher would reopen the side door that work is closing.
- **The join cost is real and unpriced here.** These tables key on `pfr_player_id` and
  `pfr_player_name`, not nflverse's `player_id`. `hub.models.spread` already pairs snap counts
  with `load_ff_playerids`, so the pattern exists, but no angle above has costed the crosswalk.

## Reproduce

```
uv run python -c "import nflreadpy as nfl; \
  print(nfl.load_pfr_advstats(seasons=[2025], stat_type='rush', summary_level='week').columns)"
```

`stat_type` takes `pass`, `rush`, `rec`, `def`; `summary_level` takes `week` or `season`;
weekly data begins in 2018. `load_snap_counts`, `load_draft_picks` and `load_combine` take
seasons directly.
