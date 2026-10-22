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
- [ ] The retrieval arm (added 2026-10-22, #511): which LLM it pairs with, and Tavily's per-credit price and the promo's expiry

**Before v1 (after the burn-in):**

- [ ] The burn-in report: power for H2 at δ, and each model's deviation standard deviation against the threshold
- [ ] Every change since v0 entered in the deviations log

> **Restated 2026-10-22 (#448, method decisions adopted 2026-10-22, items 1–9; before `prereg-llm-v0`, so not deviations). Under method rule 13 every passage below that these replace stands as first written; the prior values are quoted here.**
>
> 1. **H2's implementation.** Prior: "Parity runs this test at the version pinned at v0" and the checklist's "The Parity version or commit that runs H2". Now H2 runs on this repository's `hub.forecasters.score` (#452), pinned at a commit as `[analysis].h2_implementation = "hub.forecasters.score@<commit>"` in `conf/forecasters.toml`; scoring refuses any other form, a Parity version included. The repo test that checks H2 against a hand computation on a fixed dataset is `tests/unit/test_forecasters_score.py`. The value is still TBD: this restatement's lane changes `score.py`, so the pin is the merge commit of that lane on `main`, filled in once it has landed and before v0.
> 2. **The clip.** Prior: "Clip every probability to \[0.01, 0.99\], for all contestants, before scoring." Now the clip applies to the contestants only. The betting market's de-vigged probability is the benchmark, not a contestant, and is scored as de-vigged, never clipped.
> 3. **The H1 family.** Prior: a model below the threshold "leaves H1", with no word on which games measure it. Now estimability (the SD of each line-aware model's log-odds deviations from the market, against the threshold) is measured once, on the burn-in week, from forecasts only. The Holm family is fixed then, before any scored game, and is never re-measured on scored games. The burn-in report (#451) writes the measurement to `docs/prereg/llm-forecasters.burn-in.json` (the season, the burn-in week, the threshold and each model's SD), committed with `prereg-llm-v1`. Scoring refuses while that record is missing, was measured against a threshold other than the pinned one, or leaves out a line-aware model. A family member whose fit fails on the scored games stays in the family with p = 1, so Holm's m is the burn-in's.
> 4. **H2's level.** One-sided non-inferiority at α = 0.05, as written. The matching 90% two-sided t interval on the weekly values (the one whose lower end is the test's bound) is reported beside the 95% interval for every Δ.
> 5. **δ = 0.02 nats of log loss per game.** It is the round-number fallback the checklist allows, because no historical forecast-time snapshots were bought. The justification: δ = 0.02 nats a game is roughly a quarter (24%) of the 0.0846 nats by which the betting market's log loss on NFL games (0.6085: de-vigged closing moneylines, 2,628 regular-season games 2016–2025 from nflverse schedules, ties excluded, measured 2026-10-22) beats a coin flip's (ln 2 ≈ 0.6931), so an LLM that beat the market by more than δ would have closed a substantial share of the gap between knowing nothing and knowing the line. *Context, not the reason:* #452's simulation put H2's power at about 125 scored games at 0.69–1.00 for near-market models at this δ. The rule above forbids setting δ from power, and it was not set from power.
> 6. **The H1 estimability threshold = 0.25**, the SD of a model's log-odds deviations from the market.
> 7. **The injury report's timestamp** (#447). No source gives a report publication time, so the report's time is the newest item `date` strictly before the run start. An item dated at or after the run start is dropped, never shown. This replaces "The injury report's own publication time precedes the run start" as the content check.
> 8. **The run refuses while δ or the H1 threshold is TBD.** `runner.require_pinned` will not call a model under such a config, so no forecast can be made before the analysis that judges it is fixed.
> 9. **Retrieval pairing** (#511). The retrieval arm pairs with the cheapest of the four pinned models, by per-token price as pinned in `conf/forecasters.toml`. The pairing value itself is set with the models.
>
> Item 10 of the same decision (forecast records carry the forecast-time market probability, and scoring joins results from nflverse's schedule) is implementation, not a rule change. Each record now carries `market_home_win_prob` from its run's snapshot. Where a crash-and-retry leaves one game on two snapshots, the earliest run's is the benchmark, and the count is reported.

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

> **Restated 2026-10-22 (#511; adopted 2026-10-20 as its option 1, before `prereg-llm-v0`). Under method rule 13 the text above stands as first written: "eleven forecasters" and "no model gets ... retrieval" are the prior values.** There are **twelve** forecasters. The twelfth is **LLM + retrieval**, a contestant *paired* with one bare arm, so the paired difference isolates retrieval:
>
> | Contestant | What it is | Fixed at v0 |
> | --- | --- | --- |
> | LLM + retrieval (paired, exploratory) | One of the four LLMs, same version string and settings, shown its own line-aware packet unchanged plus news search results retrieved for the game before the run started | Which LLM it pairs with, the queries, the search cap, the lookback, the snippet budget, the prompt hash, the credit cap |
>
> - **Pairing.** It pairs with `<model>:line_aware` and is recorded as `<model>+retrieval:line_aware`. Each record carries `paired_with` (the twin's contestant name) and `base_packet_hash` (the twin's packet hash, which must equal the twin's own `packet_hash`). The quantity reported is retrieval minus bare, game by game, on games both have, averaged within week and then across weeks as for H2, with a 95% t-interval on the weekly values. It is exploratory: no test, and H1 and H2 stay four models.
> - **Searches.** Four a game, fixed: `"<team> injury report"` and `"<team> inactives"` for the home team, then the away team, through Tavily's news search (basic depth, one credit each, last 7 days, 5 results each). A longer query list is refused before any search.
> - **As-of rule for retrieved items.** Each item is judged by its own published date. It is kept only if that date is strictly before the run start and no more than 7 days earlier. An item with no usable date is dropped; fetch time never stands in. Every dropped item is logged with its URL, title, raw date and reason, and counted. The rule is written into the packet's retrieval block.
> - **Budget and hash.** At most 10 items, newest first, each snippet cut to 600 characters. The packet hash covers the bare packet and the whole retrieval block (queries, items, drops, counts). The block, snippets included, is committed in the record.
> - **Spend.** Tavily credits are counted separately from dollars, in the study's own state file, against a season cap of 1,500 credits checked before each game's searches. They are also priced into the cost log at a per-credit price pinned at v0 (0 while the promo's credits are free), so the season dollar cap and the spending guard include retrieval.
> - **Failures.** A failed search means that game has no retrieval forecast, logged with the reason and the credits spent (the "contestant has no forecast" row under Exclusions). When the credit cap is reached the arm stops and the other eleven continue. Resuming it is a logged deviation.

> **Models pinned 2026-10-22 (#448; the maintainer's 2026-10-21 rule: the newest flagship from Anthropic, OpenAI and Google plus one cheaper fourth). Recorded under method rule 13; nothing above is changed.** Every value is read from the provider's own docs on 2026-10-22 and is what `conf/forecasters.toml` carries.
>
> | Name | Provider | API model id | Stated cutoff | Temperature | Reasoning effort | Max output | USD / MTok in / out | Source (accessed 2026-10-22) |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | llm1 | Anthropic | `claude-opus-5-5` | Jun 2026 (reliable knowledge and training data) | none (sampling parameters return 400) | `low` (`output_config.effort`; thinking always on) | 1,000 | 4.00 / 20.00 | platform.claude.com/docs/en/about-claude/models/overview; …/models/opus-5-5/overview; …/models/opus-5-5/migration-guide; …/about-claude/pricing |
> | llm2 | OpenAI | `gpt-6.1-sol` | Apr 30, 2026 | none (removed unless effort is `none`, which this model rejects) | `low` (`reasoning.effort`) | 1,000 | 2.00 / 10.00 | developers.openai.com/api/docs/models/gpt-6.1-sol; …/guides/latest-model; …/guides/reasoning; …/changelog |
> | llm3 | Google | `gemini-4-argon` | not published | none (ignored on Gemini 4; Google says remove it) | `low` (`thinkingConfig.thinkingLevel`; thinking can't be disabled) | 1,000 | 4.00 / 20.00 (2.00 / 10.00 through 2026-11-30) | ai.google.dev/gemini-api/docs/models/gemini-4-argon; …/generate-content/latest-model; …/generate-content/thinking; …/pricing |
> | llm4 (cheaper) | OpenAI | `gpt-6-luna` | May 18, 2026 | none (as llm2) | `low` (`reasoning.effort`) | 1,000 | 0.10 / 0.50 | developers.openai.com/api/docs/models/gpt-6-luna; …/guides/reasoning; …/changelog |
>
> - **Which model is the flagship.** Each provider also sells a premium tier above these: Anthropic's Claude Fable 5.1 and OpenAI's GPT-6 Astra, both $10 / $50. They are excluded. Anthropic's docs label Opus 5.5 "Latest" (released 2026-09-22) and name it the model to start with. GPT-6.1 Sol (released 2026-09-29) is the newest model in OpenAI's "Flagship models" list. Google calls Gemini 4 Argon its "flagship reasoning model". Two premium models at their own ceiling would cost about $196 over the season, more than the $150 cap on their own.
> - **Ids.** Anthropic says every Claude id from the 4.6 generation on, dateless ones included, is a pinned snapshot. OpenAI lists one snapshot for each model, the dateless id, and publishes no dated one. Google lists `gemini-4-argon` as Stable ("Stable models usually don't change") and publishes no dated version. None of these is a moving alias.
> - **Settings.** All four reason. Opus 5.5 and Gemini 4 Argon can't turn thinking off, GPT-6.1 Sol rejects `none`, and Luna is held to the same rule. So effort is `low`, the lowest level all four accept, and temperature is none on all four. The config spells "none" as "send no parameter", so it cannot ask Luna for effort `none`.
> - **Why 1,000 output tokens, not 400.** In all three APIs thinking counts toward the cap. A cap hit while the model is still thinking returns no answer but is billed (OpenAI: status `incomplete`; Google: `MAX_TOKENS`). The answer itself is about 150 tokens (JSON plus a rationale of at most 60 words). A cap of 400 leaves about 250 tokens for thinking; 1,000 leaves about 850. The live dry run measures real output tokens. A change to the cap after v0 goes in the deviations log.
> - **Gemini's price.** Google's pricing page shows $2 / $10 through 2026-11-30 and $4 / $20 from 2026-12-01. The study window (weeks 9–18) covers both. The pin uses the higher price, so the cost log and the spending guard overstate November and never understate December.
> - **The fourth model and the retrieval pairing.** It is the cheapest model by cost per call at 2,000 input + 400 output tokens:
>
>   | Model | 2,000 × in + 400 × out | Per call |
>   | --- | --- | --- |
>   | llm1 `claude-opus-5-5` | 2,000 × $4 + 400 × $20 = $0.0080 + $0.0080 | $0.0160 |
>   | llm2 `gpt-6.1-sol` | 2,000 × $2 + 400 × $10 = $0.0040 + $0.0040 | $0.0080 |
>   | llm3 `gemini-4-argon` | 2,000 × $4 + 400 × $20 = $0.0080 + $0.0080 | $0.0160 |
>   | llm4 `gpt-6-luna` | 2,000 × $0.10 + 400 × $0.50 = $0.0002 + $0.0002 | **$0.0004** |
>
>   The order is the same at 1,000 output tokens. `[retrieval].paired_with = "llm4"`. Claude Haiku 5.5 has the same list price as Luna ($0.10 / $0.50 for prompts up to 100k tokens). Luna gets the tie: Anthropic's docs say its current tokenizer fits about 555k words in 1M tokens against about 750k before, so the same packet is likely billed as more Claude tokens. Either choice leaves three providers.
> - **Tavily.** `[retrieval].usd_per_credit = 0.008`, the pay-as-you-go price on tavily.com/pricing (read 2026-10-22). It counts the arm's searches against the dollar cap even while they fall inside the 1,000 free monthly credits.
> - **Season cost.** 14 games a week × weeks 9–18 (10 weeks) = 140 games. Each game has twelve records, and nine of them call a model: four models × two conditions × 5 samples, plus the paired arm's 5 samples on a prompt up to 2,500 tokens longer, plus 4 Tavily credits. The hub, home and coin records cost nothing. With every call at the 1,000-token ceiling and no retries, a game costs $0.28 (llm1) + $0.14 (llm2) + $0.28 (llm3) + $0.007 (llm4) + $0.004 (the arm's calls) + $0.032 (searches) = $0.744. The season is **$104**, under the $150 cap. At 400 output tokens a call it is $61.50. Parse retries add up to twice a sample's calls. The spending guard covers them: before each run it refuses when the remaining budget is below that run's worst case, which is about $44 for a 13-game Sunday at these settings.

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
