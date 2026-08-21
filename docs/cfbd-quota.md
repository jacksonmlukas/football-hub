# CFBD quota architecture — all 136 FBS on the free tier

Free tier is 1,000 calls/month. All-FBS coverage fits comfortably. The quota only dies if you loop.

## The rule

**Never loop over teams or games.** Every CFBD endpoint that matters accepts a `year` and `week`
and returns the entire slate in one response. Pull the bulk payload, cache it, filter in Polars.

| Pattern | Calls for a full week |
|---|---|
| `for team in fbs_teams: get(/games?team=...)` | 136 |
| `get(/games?year=2026&week=N)` | 1 |

## Weekly budget (in-season)

| Endpoint | Calls |
|---|---|
| `/games?year&week` | 1 |
| `/lines?year&week` | 1 |
| `/games/teams?year&week` (box scores) | 1 |
| `/ratings/sp?year` | 1 |
| `/stats/season?year` | 1 |
| `/player/returning`, `/talent` (preseason only) | 0 in-season |
| **Total** | **~5–8/week → ~35/month** |

That leaves 900+/month of headroom. Spend it on the one-time historical backfill.

## Historical backfill — do this in the first week of a billing month

2015–2025 across games, lines, box scores, and SP+ runs roughly 250–350 calls. Doing it early in
the cycle means a bug that double-fetches doesn't cost you the season. Cache to
`data/raw/cfbd/{endpoint}/{year}.parquet` and never re-fetch a completed season.

## Terms

Multiple keys, alias emails, and rate-limit circumvention are explicit violations and get access
revoked. One key, cache aggressively. Redistributing CFBD data is also prohibited, which matters if
you make this repo public: commit code and model outputs, never raw CFBD payloads. `data/raw/` is
already gitignored for this reason.

## Academic tier

CFBD raises free limits for verified students and researchers registering with a `.edu` address.
Worth checking whether your Berkeley address still resolves.

## Live layer

ESPN covers all 136 FBS live for free (`groups=80` on the scoreboard endpoint, `limit=200` so a
full Saturday doesn't truncate). Live scoring never touches CFBD quota.
