#!/usr/bin/env zsh
# zsh, not bash: macOS ships bash 3.2, which has no `declare -A` and cannot parse the
# `"$(cat <<'BODY' ... BODY)"` bodies below. Homebrew bash is not assumed.
# Seed the tracker from docs/audits/2026-09-20-method-audit.json (S series).
#
# Not Audit V. #325 pre-commits that Audit V runs after week 4 is played and scored; this pass
# ran 2026-09-20 on the code alone, so it is filed as a separate method audit and #325 is
# unchanged. Decided by the maintainer 2026-09-20.
#
# Conventions this follows, from docs/agents/issue-tracker.md and triage-labels.md:
#   * tickets are GitHub issues, driven by `gh`; heredoc bodies
#   * ordering is a native `blocked_by` edge, never a prose obligation
#   * `ready-for-agent` when ADR-0024's one-axis test passes; `ready-for-human` when the
#     alternatives are different objects, and then the body carries a PROPOSED draft
#   * nothing here closes a ticket, so no --reason is needed anywhere in this file
#
# S1 is the root blocker. Every ticket that re-runs or restates a gate is blocked_by it,
# for the same reason nine gate tickets were blocked_by #311: re-running before the rule is
# fixed produces a number that will have to be restated a second time.
#
# Usage:
#   scripts/seed_method_audit_tickets.sh       # dry run: print every gh call, change nothing
#   scripts/seed_method_audit_tickets.sh --apply  # create the milestone, the tickets and the edges
#
# Re-running with --apply creates DUPLICATES. It is not idempotent. Run the dry run first.

set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
MILESTONE="Method audit 2026-09-20"
AUDIT="docs/audits/2026-09-20-method-audit.json"
ROOT_TICKET=325   # Audit V proper. This audit is not it; #325 gets a pointer, not a delivery.
FREEZE_TICKET=326 # S1 is `blocking`, and per #325 a blocking finding holds the freeze open.

[[ -f "$AUDIT" ]] || { echo "missing $AUDIT" >&2; exit 1; }

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# create <label> <title> <body> -> echoes the new issue number (or a placeholder on a dry run)
declare -A NUM
create() {
  local key="$1" label="$2" title="$3" body="$4"
  if (( APPLY )); then
    local url
    url="$(gh issue create --title "$title" --body "$body" \
             --label "$label" --milestone "$MILESTONE")"
    NUM[$key]="${url##*/}"
    echo "  created ${NUM[$key]}  $title"
  else
    NUM[$key]="<$key>"
    echo "  DRY  gh issue create --label $label --milestone '$MILESTONE' --title '$title'"
  fi
}

# blocks <blocker-key> <blocked-key>
blocks() {
  local b="${NUM[$1]}" c="${NUM[$2]}"
  if (( APPLY )); then
    local id; id="$(gh api "repos/$REPO/issues/$b" --jq .id)"
    gh api --method POST "repos/$REPO/issues/$c/dependencies/blocked_by" -F issue_id="$id" >/dev/null
    echo "  #$c blocked_by #$b"
  else
    echo "  DRY  #$c blocked_by #$b"
  fi
}

say "Repo: $REPO   apply=$APPLY"

say "1. Milestone"
if (( APPLY )); then
  gh api "repos/$REPO/milestones" --method POST -f title="$MILESTONE" \
    -f description="Findings from $AUDIT (S series, 2026-09-20; a code-only pass, not Audit V / #325). S1 is the root blocker." \
    >/dev/null 2>&1 || echo "  milestone already exists"
else
  echo "  DRY  create milestone '$MILESTONE'"
fi

say "2. Tickets"

create S1 ready-for-agent \
"S1: the gate's two halves are one half -- \`won == total\` implies \`lo > 0\`" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S1** (blocking), measurement **M-S1**.

`experiment.gate` adopts on `summary["lo"] > 0 and won == total`. Every `run_gate` call site in
`src/` passes `cluster=SEASON_CLUSTER`, so `summarise`'s bootstrap units are
`paired.group_by(["season"]).agg(mean("diff"))` -- the identical vector `per_season` returns as
`seasons["gain"]`. A nonparametric bootstrap resamples only observed units, so every resample mean
is a convex combination of those k numbers. If all k are positive, every draw is positive and
`lo > 0` follows by construction.

**`won == total` implies `lo > 0`. The ADOPT conjunction reduces to `won == total`.** The converse
does not hold, so the interval half is strictly implied by the sign half and never binds against it.

The gate is therefore a one-sided sign test of size `2^-k`: 12.5% at k=3, 6.25% at k=4, 3.1% at k=5.
Power is `Phi(delta/s)^k`. Simulated at 20,000 trials per cell (M-S1), the ADOPT rate matches both
to three decimals and (all-positive) never once disagreed with (lo>0) in 20,000 draws.

At the repo's own published standard errors:

| gate | k | s | delta at 80% power | power at a realistic delta |
|---|---|---|---|---|
| draft | 4 | 7.34 | 11.78 pts/team-game | 0.136 at delta = 2.0 |
| weekly blend | 4 | 0.382 | 0.613 pts/team-week | 0.378 at delta = 0.3 |

The printed MDE, the t-interval `small_sample_report` renders, and `p_better` are all decorative
with respect to the decision.

This is upstream of #R9 and #R10. Those are about the wrong standard error and the wrong bar; this
is about a conjunction that reduces to one term whatever standard error you hand it.

### Acceptance

- [ ] The interval half is computed on something the sign half does not determine. Options, ascending
      cost: (a) read the t-interval `small_sample_report` already renders off `t_quantile` --
      implemented, printed, unread; (b) a wild cluster bootstrap, the textbook k=4 remedy;
      (c) a hierarchical posterior (phase 2, separate ticket).
- [ ] A `power` field is computed **before** the verdict and `SHOW` prints it. A SHOW that does not
      say what it could have detected is not a result.
- [ ] A unit test simulates the rule under the null and asserts the realised size is the intended
      one. This is the check that would have caught the defect and no existing safeguard performs it.
- [ ] `verdict()` docstrings that claim two independent halves are corrected.

### Reproduce

```sh
uv run python -c "
import numpy as np, polars as pl
from hub.models.experiment import summarise, per_season, SEASON_CLUSTER as C
r = np.random.default_rng(0); bad = 0
for _ in range(2000):
    m = r.normal(0, 1, 4)
    df = pl.DataFrame({'season': np.repeat(np.arange(4), 50), 'diff': np.repeat(m, 50)})
    s = summarise(df, cluster=C); p = per_season(df)
    bad += int(((p['gain'] > 0).sum() == 4) and not (s['lo'] > 0))
print('disagreements', bad)"
```

M-S1's numbers come from a faithful reimplementation, not from the repo's own functions -- the
reviewer could not rebuild the venv. **Run against the real `summarise`/`per_season` on 2026-09-20
at `bb334e0`: 0 disagreements in 2,000 draws, null ADOPT rate 0.061 against 2^-4 = 0.0625.** The
claim holds on the repo's own code.

**Holds #326 open.** This finding is tagged `blocking`; per #325, a blocking finding holds the freeze
until it lands, and only the maintainer lifts it.
BODY
)"

create S2 ready-for-human \
"S2: thirteen of fifteen recorded measurements are failures to detect, published as closures" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S2**. Blocked by S1: the restatement should
be written once, against the fixed rule.

Classifying `docs/method.md`'s fifteen rows by whether ADOPT was reachable at the realized n:

- #6, #7, #10, #15 -- all-season losses. ADOPT unreachable by construction.
- #1-5, #9, #11 -- screens whose 80%-power detectable effect sits above any plausible signal. The age
  screen's detectable r is about 0.075 against a measured +0.0076.
- #13 -- died on the unanimity half at k=3, where the null takes that half one time in eight.
- #8 -- unmeasurable by its own account.

Two designs of fifteen had ADOPT reachable, and one of those (the injury table) reached it through
the missing cluster (#R9) and a `verdict()` with no bar at all (S3).

The closures are written as equivalence claims with no equivalence bound: *"the programme is closed"*
and *"age adds nothing to ECR"* (`docs/signal-screens.md:41, 93`); *"+0.00 -- a structural zero"*
for the lineup optimiser (`docs/method.md`), a point null asserted from a design whose MDE was never
computed. `gate()`'s own SHOW text keeps the distinction correctly; the prose downstream does not.

### PROPOSED (needs an `ADOPTED:` line -- ADR-0024 different objects)

Three options for what "the published number moves" means here, per rule 13:

1. **Text-only restatement, now.** Each of the thirteen rows gains the effect its design could have
    detected, and the verdict word changes from a closure to a bounded null. No re-run needed.
    *Recommended*: it is the cheapest honest action and rule 13 does not permit waiting.
2. **Restate only after the phase-2 re-run.** Defensible, and leaves thirteen closures standing in
    `README.md` for months while their author knows the instrument was underpowered -- which is the
    exact shape rule 13 was written against.
3. **Restate and withdraw.** Move all thirteen to *superseded and unestablished*, as #167's three
    claims were when no re-run was possible. Stronger than 1, and arguably overstates the case for the
    four all-season losses, where the verdict does survive.

This is a decision about the record and it is the maintainer's. The decision-free half -- computing
each row's detectable effect -- is split out as its own ticket.

### Acceptance

- [ ] An `ADOPTED:` line naming the option
- [ ] `README.md`'s conclusions table and `docs/method.md`'s fifteen-row record carry the restatement,
      dated, beside the original text (never over it)
- [ ] `docs/track-record.md` reflects it
BODY
)"

create S2M ready-for-agent \
"S2a: compute the detectable effect for each of the fifteen recorded measurements" \
"$(cat <<'BODY'
The decision-free half of S2, split under ADR-0024 step 2.

For each row of `docs/method.md`'s fifteen-measurement table, compute what that design could have
detected at 80% power under the **fixed** rule from S1: the between-season s, k, and the resulting
delta. One table, one axis, no decision -- the deliverable is the table.

Where the inputs to a row no longer exist (measurement #8, and any row whose board cannot be
rebuilt), say so in the row rather than estimating.

S2 reads this table. Blocked by S1 because the rule determines the formula.
BODY
)"

create S3 ready-for-agent \
"S3: injury.verdict is an argmin -- the one adopted model faced the lowest bar in the repo" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S3**.

`injury.verdict` (`src/hub/models/injury.py:349-372`) computes `m = {c: mean(mae_c)}`, takes
`best = min(m, key=...)`, and returns ADOPT if `best` is `table` or `retention`. No interval, no
every-season half, no significance test of any kind. Four candidates, argmin, adopt.

So the **0.170 MAE at 3.8 se, n=3687** in `README.md` and `docs/weekly-injury.md:82` is not produced
by the pre-registered `verdict()` that decided the adoption. The module's other bar, `type_verdict`,
does read `paired_gain` -- which #R9 found unclustered over player-weeks.

This is rule 1's incident in reverse. Rule 1's case was a `verdict()` documenting two halves and
implementing one. Here the `verdict()` documents and implements none, while the write-up quotes a t
computed elsewhere. Consequence: the repo's single ADOPT, cited in `README.md` as *"the one that
cleared its gate"*, cleared a weaker gate than everything that was removed.

### Acceptance

- [ ] `verdict()` faces the same two halves as every other gate, computed under S1's fixed rule
- [ ] Re-run; `README.md` and `docs/weekly-injury.md` restated with whatever comes back
- [ ] If it no longer clears, that is the finding and it is published as one

Blocked by S1 (the rule) and S4 (the baseline it is measured on).
BODY
)"

create S4 ready-for-agent \
"S4: the injury retention baseline is a within-season lookahead" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S4**. The audit file says `verified: read`;
**confirmed by direct read on 2026-09-20 at `bb334e0`** before filing: `observations()` groups healthy
weeks by `(gsis_id, season)` over the whole season, so a week-5 designation's baseline includes weeks
6-18 of the same season.

`injury.observations` (~`src/hub/models/injury.py:121-124`) computes `baseline` as the player's mean
over the healthy weeks of **that season**, including weeks after the designated week being predicted.
`walk_forward` fits the retention table on strictly earlier seasons -- correct -- and then predicts
`now["baseline"] * retention` (~line 342), handing the held-out arm the player's realised in-season
healthy level.

All four arms share the column, so the **contrast** survives. What does not survive is the magnitude
as a statement about live error, and the deployability: on a Sunday there is no such column.
`roster.availability` (`src/hub/season/roster.py:57-85`) uses ESPN's `projected_total/projected_avg`
ratio, which is season-level and explicitly says nothing about which games are missed.

Secondary: the healthy-week population is the player's healthier-and-better subset, so retention is
measured against an inflated denominator.

Not among the three leakage surfaces #R11 names. Believed new.

### Acceptance

- [ ] `baseline` is a strictly-prior-weeks expanding mean
- [ ] Re-run; `docs/weekly-injury.md` and `README.md` restated. The gain will shrink and may vanish;
      either answer is publishable and the current one is not
- [ ] A note on what production would use instead, since `grep` finds no caller of `retention_table`
      or `predict_retention` outside `injury.py` -- the one adopted model is not in the lineup path
BODY
)"

create S5 ready-for-agent \
"S5: state/gate-width.json holds two intervals published nowhere, and the ledger overwrites" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S5**. Read at `e0071e2`:

```json
"draft":  {"clusters": 4.0, "lo": -18.761, "hi":  -7.665, "requires_review": true},
"weekly": {"clusters": 4.0, "lo":  -1.152, "hi":  -0.541, "requires_review": true}
```

Neither matches any published figure. `docs/gate-power.md` carries draft widths of -19.66, -15.65,
-12.48, -11.59, -12.19 and -11.88; `docs/weekly-blend-gate.md` carries [-1.347, -0.640].

So runs exist whose numbers reached no page, and both entries have carried `requires_review: true`
unaddressed. `review_width` writes `state[name] = {...}` -- one record per gate name, overwritten on
the next run. The repo's entire cross-run memory of how many times a gate has been run is a dict with
three keys.

This is the mechanism behind #B2. The forking paths are not only across anchors and bases; they are
across re-runs, and nothing could count them even if someone wanted to.

### Acceptance

- [ ] The ledger is append-only, keyed by `(gate, config_digest, data_digest, timestamp)`, and records
      the **verdict** alongside the width
- [ ] The two `requires_review` flags are discharged: which run produced each, and whether its number
      was ever published
- [ ] `state/README.md` describes the new shape

Prerequisite for the anytime-valid inference ticket.
BODY
)"

create S6 ready-for-human \
"S6: NOT-RUNNABLE is conditioned on a ceiling only the ADOPT direction has an incentive to measure" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S6** (severity: decision).

`gate()` (`src/hub/models/experiment.py:902-911`) fires NOT-RUNNABLE only when
`reading(summary,"mde") is Field.VALUE and reading(summary,"ceiling") is Field.VALUE and mde > ceiling`.
A ceiling appears only when a caller hands one in, and by `docs/gate-power.md`'s own account two of
three gates have never measured one -- for them the branch *"does not fire and cannot"*.

The docstring's reasoning is sound in isolation: *"A gate that measured no ceiling has not shown that
it cannot run; it has shown nothing."* The asymmetry is in **who bears that**. An underpowered design
is protected from publishing a null (SHOW) and from adopting, and is not protected from REMOVE. And
REMOVE is not a label: per `gate-power.md`'s own table the module moves to `hub.exhibits`.
`championship_equity` and `leverage` are there now, removed on evidence S1 shows to be a 14%-power
sign test.

This is the mirror of #R6. R6: a branch the gate cannot *leave* granted a standing licence to ship.
S6: a branch the gate cannot *reach* withholds a protection from one direction only.

### PROPOSED (needs an `ADOPTED:` line -- ADR-0024 different objects)

1. **Require a ceiling before any verdict.** A gate that measured none returns NOT-RUNNABLE in both
    directions. *Recommended*: it makes the guard symmetric and forces the ceiling measurement that
    two gates have avoided.
2. **Keep the guard and name the exemption.** State in the pre-registration that REMOVE is reachable
    without a ceiling, the way #300 named NOT-RUNNABLE an exemption. Cheaper, and leaves the asymmetry
    in place with a label on it.

Either way: **nothing further is deleted from `hub.exhibits` until this is decided and S1 has landed.**

### Acceptance

- [ ] An `ADOPTED:` line naming the option
- [ ] ADR-0019 amended, dated, with the superseded text kept visible
- [ ] A test pinning the chosen branch structure
BODY
)"

create S7 ready-for-human \
"S7: decide what bar a game-level betting model faces, before one exists" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S7** (severity: decision).

`src/hub/holdout.py` is entirely fantasy-draft -- eleven keys, all `predict.*` and `availability.*`.
`MARGIN_SD` is not among them. There is no leave-one-season-out machinery for any game-level constant,
and effective n at game level is **seasons, not games**.

For an ATS gate at k=10 seasons of ~285 games: per-season SE ~= `sqrt(0.25/285)` = 2.96pp, SE of the
mean ~= 0.94pp, MDE ~= `(2.262 + 0.842) * 0.94` = **2.9pp**. Break-even at -110 is 52.38%, so the
interval half is blind to every profitable-but-realistic edge. The binding half is worse: a true 53%
model has `P(season hit rate > 50%) = Phi(2.62/2.96) ~= 0.81`, so `P(positive in all 10) ~= 0.12`.

The house rule is calibrated for **constant refits**, where rejecting churn is right and a false
negative costs nothing. Applied to a betting edge a false negative costs the entire enterprise. Same
rule, opposite loss function.

Not currently binding, because no ATS or cover model exists -- which makes this the right time to
decide, while no number the decision could favour exists yet (rule 1's sibling rule).

### PROPOSED (needs an `ADOPTED:` line -- ADR-0024 different objects)

1. **The house rule unchanged**, with the power stated in the pre-registration so a REMOVE is read as
    what it is.
2. **The house rule at higher k**, using more seasons of line history where it exists.
3. **A decision-theoretic bar**: expected value at a stated stake, with a pre-registered utility.
    *Recommended*, and it is the same move as the phase-2 posterior ticket, so they should be decided
    together.

Blocked by S1: option 3 is only coherent against a rule that reports an effect size.
BODY
)"

create S8 ready-for-agent \
"S8: build a state-space team rating -- ratings.py is a passthrough and six consumers rest on it" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S8**.

`ratings.forecaster()` (`src/hub/models/ratings.py:107`) returns `MarketBaseline()`; `rated_games()`
is `schedule.priced_games` with nothing applied since #299 pulled the QB layer. No Elo, no state-space
rating, no margin regression exists anywhere in `src/`.

`survivor.grid_from_schedule` builds `win_prob` as `normal_cdf(close_spread / MARGIN_SD)`. The
survivor IP, `pool.Field`, `weekly`, `leverage`, `buyback` and `sensitivity` all read it. Unpriced
games are dropped, so a coverage hole shrinks the slate rather than degrading it -- there is nothing
to fill it with.

`market.py:1-13` states the falsifiability argument correctly and never builds the challenger.
`props.py:1-10` says the game-level spread was skipped precisely **because** it is both the input and
the thing being scored: the asymmetry was identified and then used as a reason not to do the work.

### Acceptance

- [ ] Glickman-Stern state-space rating: Normal margin, team strengths as a random walk, home
      advantage a parameter, fitted on `schedules` back to 1999
- [ ] `ratings.forecaster()` returns it behind a flag; `MarketBaseline` stays the default until gated
- [ ] Walk-forward log-loss against the close, under the **fixed** gate from S1, with the power
      stated before the run
- [ ] Both outcomes are publishable: it loses and you have a calibrated second opinion and a residual
      to study, or it does not and the record contains something new

Blocked by S1. CFB extension is a separate ticket.
BODY
)"

create S8C ready-for-agent \
"S8a: extend the team rating to CFB -- 136 FBS teams in scope, zero modelled" \
"$(cat <<'BODY'
`schedule.SLATES` is `{"nfl": _nfl_slate}` and `priced_games(league="cfb")` raises
`LeagueUnavailable`. `/ratings/sp` and `/player/returning` are in the CFBD quota budget and are read
by no model.

This is the largest unexploited surface in the repo **precisely because** G5 and non-conference games
are not priced by a sharp market -- the one place an independent rating plausibly has edge, rather
than a second opinion on a price that is already good.

Quota is not the constraint: ~56-90 calls a month against 1,000, with ~900 explicitly earmarked for
backfill. Worth noting in passing that the Big Ten availability regime spends 7 of a ~14-call weekly
budget on one conference, for a study that has not run.

### Acceptance

- [ ] The S8 rating fitted on CFB results, with SP+ and returning production as priors
- [ ] `LeagueUnavailable` degrades to the rating instead of refusing
- [ ] Scored against `/lines` where a price exists, and reported as unpriced coverage where none does

Blocked by S8.
BODY
)"

create S9 ready-for-agent \
"S9: the favourite-longshot location bias is measured, published, and consumed by nothing" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S9**.

The repo's own residuals by bucket (`docs/margin-sd.md`): pick'ems **-1.56 pts** (se 0.53, n=569),
6-9 point favourites +0.78, 9-14 +0.91. In probability: 7+ favourites priced 0.771 and win **0.803**,
+3.2pp at 2.4 se on n=859; 10-14 +0.036; 14+ +0.063.

`docs/weekly-coverage.md`: *"Nothing is refit on this."*

7+ favourites are exactly the population survivor picks from. `margin.survival_beside` puts the priced
18-pick survival product at 0.0099 against a realised 0.0206 -- low by roughly a factor of two. For
survival that is conservative. For the pool's dollar figures it is not: those are P(survive)-weighted
shares of a pot, so every `expected_dollars` the module prints is corrupted by it.

Distinct from audit IV rank 9, which is about `MARGIN_SD` and non-normality. This is the **location**
half, separately actionable without touching the shape. Do not revisit the key-number bump model:
`margin.py:170-186` diagnoses correctly why it loses.

### Acceptance

- [ ] A shrunk, spread-dependent location correction fitted and applied in `grid_from_schedule`
- [ ] Adopted as a **provisional rule under ADR-0014** -- it cannot be gated at this n (see S7) -- with
      a stated horizon and every application logged. The ADR counts its members; this would be the
      second, and the count growing is itself the signal the ADR asks for
- [ ] Every published survivor and pool dollar figure re-derived and restated
BODY
)"

create S10 ready-for-agent \
"S10: resolve the leverage term -- it is unresolved for want of CPU, not evidence" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S10**.

`pool.leverage` (`src/hub/season/pool.py:1975-2075`): 2 of 25 comparisons clear 2 sigma where chance
gives ~1.1, 24 of 25 positive in sign, magnitude growing with concentration. Reported honestly and
pooled into nothing.

The module's own arithmetic says resolving a +$3 term at concentration 16 needs **~4,000 trials per
arm**. `WEEKLY_TRIALS` is **400**, over 6 candidates.

So "unresolved" is a statement about a constant in the config, not about the world. Everything else
this repo closes for want of a resource is closed for want of **seasons**, which is honest because
seasons cannot be bought. Trials can. And the leverage term is the entire economic case for
contrarian survivor play, in a pool that is live now.

### Acceptance

- [ ] The leverage study re-run at 4,000 trials per arm; whatever it says is published
- [ ] `WEEKLY_TRIALS` and the study's trial count are separated, so the weekly path stays fast
- [ ] `docs/championship-leverage.md` (or the pool's own page) carries the result beside the
      superseded `LEVERAGE` caveat string

Separate but adjacent, and worth its own ticket if this one grows: `field_concentration` is the
module's central knob, is unfitted, and is swept 1 to 16 (chalk share 5.7% to 27.0%). Public weekly
pick percentages are published free and would replace the sweep with a measurement. No fetcher exists
anywhere in `fetch/`.
BODY
)"

create S11 ready-for-agent \
"S11: one correlation constant, two incompatible semantics, and lineup.py reads the weaker one" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S11**.

`TEAMMATE_RHO` is estimated in `correlate.pair_correlations` as Pearson r on within-player-season
z-scores of realised PPR -- measured on **points**.

- `correlated_normal` (`src/hub/models/predict.py:583`) applies it via Cholesky to the **latent
  normal**, before `skewed()`. Right layer for a copula, and it means the realised points correlation
  is not 0.232. #R12 already names this attenuation.
- `predict.group_sd` (`:621-636`) applies the **same constant** directly to the **points** variance as
  `2*rho*sd_i*sd_j`, and `src/hub/season/lineup.py:135` builds the entire lineup objective on it.

The two consumers disagree about what the number is. They cannot both be right, and R12's finding is
about only one of them.

Also: `teammate_rho` returns 0 for unmeasured pairs, so WR-WR is exactly zero in the shipped matrix,
while `docs/correlation.md:78-84` describes an implied +0.05 between catchers from a star topology the
matrix does not build. Doc and code disagree.

### Acceptance

- [ ] A decision on which layer the constant belongs to, with the conversion applied for the other
      consumer
- [ ] A test that fails if the two drift apart again
- [ ] `docs/correlation.md` reconciled with the shipped matrix in the same commit
BODY
)"

create S12 ready-for-agent \
"S12: free nflverse data unpulled, and four sources with no as-of archive" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S12**. Four separable pieces; split if it grows.

**1. Unpulled free data.** `load_nextgen_stats` and `load_depth_charts` exist in `nflreadpy` and are
called nowhere in `src/hub/fetch/nflverse.py`. `participation` and `ftn_charting` are on disk for
**2024 only** -- one season -- so aDOT, air-yards share and route participation are historically
incomplete by accident rather than by decision. These are the inputs #C5, #G4 and audit IV ranks 2 and
3 need. There is also no weather fetch of any kind: `wind` was correctly withdrawn as RECORDED (#170)
and nothing replaced it with a pre-kickoff **forecast**, which is a legitimate PRE_KICKOFF feature and
is free.

**2. No as-of archive on four sources.** `injuries`, `snap_counts`, `participation` and
`ftn_charting` are pulled as bulk multi-season files with no snapshotting, so a later nflverse revision
silently overwrites the historical record. `store.py`'s own docstring records the earlier version of
this bug. Any measurement resting on those four is not reproducible at the vintage it was taken. ADP,
odds and `ff_rankings` **are** properly snapshotted.

**3. Odds cadence.** By the repo's own 2026-09-11 count, 260 of 272 games never moved across the
polling window. Two polls a week (Wed 11:00, Sat 14:00 UTC) cannot see a market move, and there is no
distinct opening capture -- the earliest poll stands in. Budget is not the constraint: ~56 credits a
month against 500. 4-6 polls a week with a Thursday and a Sunday-morning capture costs ~50.

**4. Unmeasured join-failure rate.** `names.player_key` is exact-match after normalization plus a
hardcoded `ALIASES` dict of **six** manually-discovered pairs. `relaxed_key` exists only to *classify*
a probable join failure for bias auditing, not to repair one. There is no measured failure rate for
the ESPN or CFBD crosswalks at all.

### Acceptance

- [ ] nextgen and depth charts pulled; `participation` and `ftn_charting` backfilled for every season
      nflverse retains
- [ ] As-of snapshotting for the four unarchived sources
- [ ] Odds cadence raised, with the credit budget restated
- [ ] Unmatched rows counted per source per season, and the rate on the board's status line
BODY
)"

create S13 ready-for-agent \
"S13: conformal.interval's finite-sample correction is undone by a polars default" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, finding **S13**.

`conformal.interval` (`src/hub/models/conformal.py:49-58`) computes `k = min(1, (1-alpha)*(n+1)/n)`
and then `residuals.abs().quantile(k)`. polars defaults to `interpolation="nearest"`, which rounds the
order statistic **down** as often as up. Split conformal requires the `ceil((1-alpha)(n+1))`-th order
statistic, i.e. `interpolation="higher"`.

So the carefully-argued `(n+1)/n` correction on one line is negated about half the time by the line
below it.

Scope note: `conformal.py` is wired to **game margin** only -- `load_scored` joins `margin_mean` to
`schedules.result` and `base.Conformalized` rewrites `margin_lo`/`margin_hi`. No player projection is
conformalised anywhere. That larger gap is audit IV rank 4 and already has a ticket.

### Acceptance

- [ ] `interpolation="higher"`
- [ ] A test planting a residual set where `nearest` and `higher` differ, so the default cannot come
      back silently
BODY
)"

create RULE ready-for-human \
"S-method: add rule 17 -- check a decision rule for degeneracy before pre-registering it" \
"$(cat <<'BODY'
From `docs/audits/2026-09-20-method-audit.json`, `method_rule_to_add`.

> **A decision rule must be checked for degeneracy against its own inputs before it is
> pre-registered.** A conjunction whose second term is implied by its first is one term, and writing
> it down as two is how a rule comes to be trusted for a strictness it does not have.

**Rationale.** ADR-0019 was argued, documented, unit-tested and enforced by a contract test, and its
second half has been inert since it was written. Every safeguard in this repo was pointed at whether
the rule was **followed**. None was pointed at whether the rule, as written, said anything. The check
is cheap -- simulate the rule under the null and confirm the realised size is the intended one -- and
it is the one check that would have caught S1.

Proposed home: `docs/method.md` as rule 17, with M-S1 as the incident, in the same form as rules 1-16.

Blocked by S1: the rule is written from a discharged incident, not a live one.
BODY
)"

create README ready-for-agent \
"S-docs: index the 2026-09-20 method audit in docs/audits/README.md" \
"$(cat <<'BODY'
`docs/audits/README.md` indexes I-IV and says *"Audit V is #325."* That line stays true: #325
pre-commits that Audit V runs after week 4 is played and scored, and this audit ran 2026-09-20 on the
code alone with `weeks_played [1, 2]`. So `docs/audits/2026-09-20-method-audit.json` is indexed as a
**separate method audit**, not as V, and #325 still owes the artifact read and the Phase 2/3 re-plan.
The file's `schema.relation_to_audit_v` says the same.

Per that README's own convention, the file is **never edited** and re-ordering lives in the tracker.
So this ticket only adds the index entry and any known divergence between `plan` as written and what
the tracker actually did.

Known at filing time:

- S1 was run against the real `summarise`/`per_season` before filing (0 disagreements in 2,000; null
  ADOPT 0.061 vs 0.0625) and S4 was confirmed by direct read. The JSON keeps the reviewer's own
  `verified` values; the confirmations live on the tickets.

- `plan` phase 0 is in-season and dated; steps may be re-ordered as weeks pass. Record re-orderings
  here, not in the JSON.
- The audit carries an `overlap_with_audit_iv` section listing 15 items found independently this round
  that were already on the tracker from audits I-IV. No S ticket was filed for any of them. If one of
  those R/B/C/G tickets is later closed as a duplicate of an S ticket, that is a divergence worth
  recording.
BODY
)"

say "3. Dependency edges (S1 is the root blocker)"
for k in S2 S2M S3 S7 S8 RULE; do blocks S1 "$k"; done
blocks S4 S3
blocks S2M S2
blocks S8 S8C

say "4. Point #$ROOT_TICKET at the audit (not a delivery) and record the hold on #$FREEZE_TICKET"
if (( APPLY )); then
  gh issue comment "$ROOT_TICKET" --body "Not this ticket, but adjacent: a code-only method audit ran 2026-09-20 (\`$AUDIT\`, S series, 13 new findings, 15 overlaps with audit IV listed and not re-filed) with \`weeks_played [1, 2]\`, so the precondition here was not met and it is filed under milestone '$MILESTONE' rather than as Audit V. This ticket is unchanged: it still owes the week-4 artifact read and the Phase 2/3 re-plan. Its blocking finding, **S1** (#${NUM[S1]}), holds #$FREEZE_TICKET open per the rule in this body: \`experiment.gate\`'s ADOPT conjunction reduces to \`won == total\`, a sign test of size 2^-k."
  gh issue comment "$FREEZE_TICKET" --body "Held open by #${NUM[S1]} (S1, \`blocking\`, from \`$AUDIT\`): the gate's ADOPT conjunction reduces to \`won == total\`. Per #325 a blocking finding holds this freeze until it lands, and only the maintainer lifts it. Nothing further leaves \`hub.exhibits\` until S1 and S6 (#${NUM[S6]}) are settled."
else
  echo "  DRY  gh issue comment $ROOT_TICKET --body '<pointer: method audit landed, not Audit V; S1 holds #$FREEZE_TICKET>'"
  echo "  DRY  gh issue comment $FREEZE_TICKET --body '<held open by S1>'"
fi

say "Done. apply=$APPLY"
(( APPLY )) || echo "This was a dry run. Re-run with --apply to create anything."
