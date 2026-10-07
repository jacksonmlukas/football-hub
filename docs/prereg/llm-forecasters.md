# LLM forecasters vs. the market: pre-registration, 2026 NFL

Oct 6, 2026 · @Jackson

## Purpose and status

This fixes every rule for a live test of LLM win-probability forecasts on the remaining 2026 NFL regular season, scored against the betting market. It lives at `docs/prereg/llm-forecasters.md` and is tagged twice: `prereg-llm-v0` before the burn-in, holding every analysis rule including δ, and `prereg-llm-v1` after the burn-in, before the first scored week. Every change after v0 goes in the deviations log with its date and reason.

Only forward forecasts count. Nothing is backtested, because no one can rule out that a model has seen past results.

**Burn-in.** The first full NFL week (Thursday through Monday) after v0 runs the full pipeline but is not scored. Using only each model's deviations from the market, never outcomes, it reports the power δ gives for H2 and whether each model deviates enough for H1 to be estimable. Nothing is tuned from it.

Any change to prompts, settings, or models after the burn-in either reruns the burn-in or goes in the deviations log. v1 adds only the burn-in report and any logged changes; scoring starts with the next full week.

**Before v0 (the burn-in can't start without these):**

- [ ] The four models (at least three providers): exact version strings and each one's stated training cutoff
- [ ] Each model's settings: temperature where the API accepts it, plus reasoning effort or token budget
- [ ] The odds source (the same feed as the hub's market prior) and the source for probable starting QBs, with each source's terms checked for what may be republished
- [ ] The burn-in week, the first scored week, and the expected game count through Week 18
- [ ] δ, for a substantive reason: the market's own average log-loss improvement from the forecast-time snapshot to the close in past seasons, if the feed has historical snapshots; otherwise a round number with a one-sentence justification. Never set from burn-in power
- [ ] The H1 estimability threshold: the minimum standard deviation of a model's log-odds deviations from the market
- [ ] The Parity version or commit that runs H2
- [ ] A spending cap with each provider that offers one, and the season cap the code enforces

**Before v1 (after the burn-in):**

- [ ] The burn-in report: power for H2 at δ, and each model's deviation standard deviation against the threshold
- [ ] Every change since v0 entered in the deviations log

## Questions and predictions

One question is the confirmatory test and one is secondary; everything else is estimated with intervals and no test. Predictions are written now so the result can't be reframed later.

|  | Question | Prediction |
| --- | --- | --- |
| H1 (confirmatory) | Line-aware: do a model's deviations from the market predict outcomes? (encompassing test) | No detectable added information. Reported as the estimate of c with an upper bound, never as a confirmed absence: c = 0 means the deviations are noise, c = 1 means they deserve full weight |
| H2 (secondary) | Line-aware: is every LLM no better than the market by more than δ? | Yes, for all four |
| Descriptive | How far each model's forecasts sit from the market (mean absolute gap) | Line-aware models stay close to the line |
| Descriptive | Blind: how much worse than the market is each LLM? | Every model clearly worse |
| Exploratory | Hub model vs. the market | Within noise |
| Exploratory | Calibration, favorite/underdog bias, which model is best, how much the line helps each model | No predictions |

## Contestants

Eleven forecasters are scored against one benchmark: four LLMs in two conditions each, the hub model, and two simple baselines.

| Contestant | What it is | Fixed at v0 |
| --- | --- | --- |
| LLM × 4, blind | Sees the info packet without the line | Version string, settings, stated training cutoff, prompt hash |
| LLM × 4, line-aware | Same packet plus the market's de-vigged probability and spread | Version string, settings, stated training cutoff, prompt hash |
| Hub model (exploratory) | football-hub's pick'em model, forecasting in the same run from the same market snapshot | Nothing. Each forecast logs the commit that produced it, so weekly changes are allowed |
| Home constant | P(home win) = a fixed historical home win rate | The rate and the seasons it comes from |
| Coin flip | P = 0.5 for every game | Nothing |
| Market (benchmark) | De-vigged moneyline at forecast time | Source and method (see Market benchmark) |

Models with later training cutoffs know more about the 2026 season before the packet adds anything. Cutoffs are recorded and reported next to every blind result, so differences between models aren't read as pure skill.

No model gets web search, tools, or retrieval. A model deprecated mid-season stops; no substitute counts toward the confirmatory tests.

## Information packet

Every LLM sees the same frozen packet for a game, and no field may carry information timestamped after the forecast run starts. The packet is stored verbatim and hashed.

| Field | Blind | Line-aware |
| --- | --- | --- |
| Teams, home/away/neutral (neutral flagged), kickoff time | Yes | Yes |
| Probable starting QB for each team, from the latest source update before the run | Yes | Yes |
| Rest days for each team | Yes | Yes |
| Season record and point differential through the last completed game | Yes | Yes |
| Last three results with scores | Yes | Yes |
| Official injury report statuses (Out, Doubtful, Questionable) from the latest report published before the run | Yes | Yes |
| Market's de-vigged home win probability and the spread, from this run's snapshot | No | Yes |

**As-of rule.** Every field is fetched during the run and records its source and fetch time. Fetch time alone proves nothing, so each field type also gets a content check:

- Records, point differentials and recent results use only games completed before the run start.
- The injury report's own publication time precedes the run start.
- The starting-QB source's update time precedes the run start.
- Market fields come from this run's snapshot.

A packet that fails any check is not forecast. A leak found later voids that game for every contestant and is logged.

The packet schema is a data contract in the repo. Adding or removing a field after v0 is a deviation.

## Forecast procedure

The forecast job runs once a day in a 09:30–11:30 ET window, and each game belongs to exactly one day's run: its kickoff date if kickoff is at or after 13:00 ET, otherwise the day before. That leaves at least 90 minutes between the end of the window and any same-day kickoff, so the 60-minute rule below only cuts a game if a run takes more than 30 minutes. International morning games and any game before 13:00 ET (such as an early Thanksgiving game) are forecast the day before.

GitHub schedules run in UTC, so the job is triggered every 15 minutes from 13:30 to 16:30 UTC, which covers the window under both daylight and standard time. Each trigger converts to ET in code and exits unless it is inside the window and today's run hasn't finished. A workflow concurrency group plus a lock check before any model call keep two triggers from running at once. The extra triggers are the backup when GitHub delays or drops a scheduled run. No step needs a person.

Runs are idempotent: each forecast for a given game and contestant is written once, and a run skips any already written. A retry after a crash only fills the gaps and never pays twice.

1. Snapshot the market for every game due in this run.
2. Build and validate each game's packet (as-of rule).
3. For each LLM and condition, draw 5 samples with the fixed prompt and that model's pinned settings (temperature where the API accepts it, otherwise the provider default; reasoning effort or token budget for reasoning models). The forecast is the mean of the valid samples.
4. Clip every probability to \[0.01, 0.99\], for all contestants, before scoring.
5. Write one JSONL line per contestant and game, and have the workflow commit it.

**Output format.** Each sample is JSON: `p_home_win` (a number) and `rationale` (at most 60 words, logged, never scored or cited as evidence of how the model reasoned). A sample that fails to parse is retried up to twice. With fewer than 3 valid samples, that contestant has no forecast for the game.

**What each line records.** Game id, contestant, model version, condition, prompt hash, packet hash, market snapshot id, every sample, the mean, run start and finish times (UTC), the workflow run id, and cost.

**Proof of timing.** The workflow, not a laptop, commits the forecasts. From the first scored slate, each run also timestamps the hash of its forecast file with OpenTimestamps and commits the receipt beside it. That record is permanent, unlike GitHub's run logs (kept at most 90 days) or commit times (set by whoever commits). A forecast that finishes less than 60 minutes before kickoff is excluded and logged.

**Missed runs.** A failed run sends an email through GitHub's workflow-failure notifications. The Tuesday job also lists every completed game that has no forecast, so a lost slate shows up in the weekly report.

**Spending guard.** Before calling any model, the job sums the cost log. If the remaining budget is below one run's estimated maximum cost, it stops without calling models, since not every provider offers a hard cap.

**Saving output.** Forecasts are committed to a dedicated `forecasts` branch that only the workflow writes to, so code pushes never conflict with it; the leaderboard page reads from that branch. The run also saves its output as a workflow artifact before committing.

## Market benchmark

The benchmark is the de-vigged moneyline snapshotted in the same run as the forecasts, so the market and the models see the same moment. The closing line is reported only as a secondary comparison, because it includes later news the models never saw.

**Source.** The same odds feed as football-hub's market prior, named at v0. When the feed lists several books, de-vig each book separately, then take the median probability. Taking each side's median first can pair odds no single book offered.

**De-vig.** Proportional: convert both sides to decimal odds and normalize the implied probabilities so they sum to 1.

```latex
p_{\text{home}} = \frac{1/o_{\text{home}}}{1/o_{\text{home}} + 1/o_{\text{away}}}
```

**Known bias.** Proportional de-vig leaves some favorite-longshot bias in the benchmark, which slightly flatters forecasters that lean toward longshots. It is named in the limitations, not corrected.

**Missing line.** A game with no moneyline in the snapshot is scored on its own terms but left out of every comparison against the market.

**Closing line.** A separate job, triggered every 15 minutes during game windows, snapshots each game within 30 minutes of its kickoff. It is best-effort: a game without a closing snapshot drops out of the secondary comparison only, and this job's failures never touch the forecast job.

## Scoring and the confirmatory test

The confirmatory test asks whether a line-aware model's deviations from the market carry information about outcomes. Log-loss differences from the market are secondary. Every interval treats the NFL week as the unit.

**Encompassing test (H1).** For each line-aware LLM, a logistic regression of the result on the market's log-odds and the model's deviation from them:

```latex
\Pr(y_g = 1) = \sigma\!\left( a + b\,\mathrm{logit}(m_g) + c\,\left[ \mathrm{logit}(p_g) - \mathrm{logit}(m_g) \right] \right)
```

Here y\_g is 1 for a home win, m\_g is the market probability, and p\_g is the model's. A positive c means the model's deviations add information the market lacked. The test is one-sided, c > 0 at α = 0.05, using a leave-one-week-out jackknife for the standard error and t with W − 1 degrees of freedom, with a Holm correction across the four models. The jackknife is conservative with this few weeks, and it's simple to check by hand.

The reported result is the estimate of c with its one-sided 95% upper confidence bound, per model and not adjusted for multiplicity. A null reads as "the deviations deserve at most X of full weight," not as a confirmed absence. A model whose log-odds deviations have a standard deviation below the threshold set at v0 is reported as "not estimable (copies the market)" and leaves H1. That outcome is a finding in its own right.

**Log-loss difference (H2).** Computed within each week, then averaged across weeks, so the estimand is exactly what the interval estimates:

```latex
\Delta_c = \frac{1}{W} \sum_{w=1}^{W} \frac{1}{|G_{c,w}|} \sum_{g \in G_{c,w}} \left[ \ell(p_{c,g}, y_g) - \ell(m_g, y_g) \right], \qquad \ell(p, y) = -\left[ y \ln p + (1-y) \ln (1-p) \right]
```

G\_{c,w} is the set of games in week w where contestant c and the market both have a forecast; W is the number of scored weeks; a week with no eligible games for a contestant is dropped for that contestant. Positive Δ means worse than the market. Brier score is reported the same way as a secondary measure.

**Intervals.** For H2, a 95% t-interval on the weekly values of Δ with W − 1 degrees of freedom (about 12 weeks after the burn-in). A week-block bootstrap is reported only as a robustness check, since with this few clusters it gives intervals that are too narrow. Weeks are the unit because games in the same week share news shocks.

**H2, α = 0.05.** For each line-aware LLM, a one-sided non-inferiority test for the market: reject Δ ≤ −δ, with δ fixed before the burn-in. The claim "no LLM beats the market by more than δ" is made only if all four reject. Because every model must pass on its own, no multiplicity correction is needed (an intersection-union test). Parity runs this test at the version pinned at v0, and a repo test checks its output against a hand computation on a fixed dataset.

- Equivalence within ±δ (two one-sided tests) is reported alongside H2.
- Each model's mean absolute gap from the market is reported next to H1 and H2, so a pass that comes from copying the line is visible.
- Blind models and the hub model get estimates and intervals only, no tests.

**Window.** Every eligible game from the first scored week (the week after the burn-in) through Week 18. A game postponed into another week counts in the week it is played. The confirmatory analysis runs once, after Week 18's last game. Everything else, including calibration (5 equal-count bins) and per-model comparisons, is labeled exploratory.

## Exclusions and edge cases

Every exclusion is logged with its reason and counted in the final report.

| Case | Rule |
| --- | --- |
| Tie | Game excluded from scoring |
| Kickoff moved more than 24 hours | Original forecast void; the game is forecast again under the normal timing rule |
| Forecast finishes less than 60 minutes before kickoff | That contestant's forecast excluded |
| Packet fails as-of validation | Game not forecast by anyone |
| Leak found after the fact | Game voided for every contestant |
| A contestant has no forecast (API failure, parse failures) | Game kept for the others; paired comparisons use only games both sides have. Missing rate reported per contestant |
| No moneyline in the snapshot | Left out of comparisons against the market |
| Model deprecated | Contestant ends when deprecation is announced. Its games are reported as exploratory and it leaves H1 and H2, which then cover the models still running |
| Spending cap hit | The spending guard stops the runs; resuming is a logged deviation |

## Reporting and deviations

Weekly numbers are descriptive only; the confirmatory (H1) and secondary (H2) analyses run once, after Week 18, and are published whatever they show.

- **Weekly, Tuesday.** A scheduled job scores the finished week and updates the leaderboard page, labeled "Interim: descriptive, not a test."
- **After Week 18.** The confirmatory analysis exactly as written here, plus the exploratory results, clearly separated. Every number carries its n and a 95% interval.
- **Always public.** Forecasts, prompts, derived market probabilities, the cost log, and this document live in the repo. Raw odds and other source data, including full packets, are published only where each source's terms allow; otherwise their hashes are.

**Deviations log.** Any change after v0 gets a row before it takes effect.

| Date | Change | Reason | Affects confirmatory tests? |
| --- | --- | --- | --- |
|  |  |  |  |
