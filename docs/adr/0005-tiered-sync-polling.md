# Tiered sync polling, no async

**Status:** accepted.

**Decision.** Plain synchronous loop with two tiers. No asyncio, no thread pool.

**The concurrency problem was overstated, and the fix is endpoint selection.** One scoreboard
request returns score, status, possession, and down/distance for *every* game in the league --
13 NFL games or 60+ CFB games in a single response. That is everything the dashboard renders.
The per-game cost only appears on the `summary` endpoint, which adds win probability and box
score.

| Tier | Endpoint | Cadence | Requests |
|---|---|---|---|
| 1 | `scoreboard` | every 45s | 1 per league |
| 2 | `summary` | every 4th tick | 1 per game of interest, capped at 12 |

Worst case is about 13 requests per three minutes. Sync is comfortably sufficient.

**Why not add the thread pool anyway.** These endpoints are undocumented and ESPN publishes no
rate limits, with the standing community guidance being to keep volume low and cache
aggressively. Concurrency here buys nothing and raises the odds of getting blocked. If tier 2
ever needs to grow, `ThreadPoolExecutor(max_workers=8)` is a four-line change -- but the right
first move is to shrink the watch list instead.

**Games of interest** are your fantasy starters' teams plus your survivor pick, not the full
slate. If that list exceeds 12 you are watching games you have no stake in.
