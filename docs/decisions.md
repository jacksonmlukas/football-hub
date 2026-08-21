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
4. Demonstrable market edge (positive CLV)

Edge ranks last. That is why gates route rather than kill: a model failing its gate becomes a
documented study, not a deletion. It is also why we derive before importing (hand-roll split
conformal before MAPIE, the survivor DP before pulp) — the slower path serves objective 2.

Shipping priority: fantasy > pickem/spreads > survivor > CFB predictions > props > awards.
Props and awards are roadmap-only: free-tier odds data cannot support props, since the credit
multiplier makes one props pull cost more than a month's quota.

## Corrections issued during design

Five things I got wrong then fixed. Do not revert to the earlier version.

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
| `espn_api` has no ADP field at all | Raw `kona_player_info` -> `ownership.averageDraftPosition` |
| ESPN parks undrafted players on a shared tail ADP (~170), not null | `_adp_saturation_cutoff()`: lowest value shared by >= `teams` players |
| Subtracting ranks drawn from different populations | Rank consensus *within* the ADP pool before differencing |

## Open questions

1. **Pool configuration** — entries, payout, rebuys. Under ~20 entries play near max win
   probability; above ~100, survival stops being sufficient and the objective becomes
   P(finish first), which is materially more contrarian. Only Jackson can get this.
2. Which sportsbooks are accessible in NY (determines whether line shopping is possible).
3. Custom domain vs `username.github.io`.
4. Whether this survives into 2027 or is a one-season artifact.
