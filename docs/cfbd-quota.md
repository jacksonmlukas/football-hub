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

## Which week gets fetched, and what a run that fetched none says

The budget above assumes a scheduled run fetches *a* week. Until issue #56 none ever did: the
Makefile passed `--week` only when a human had set `WEEK` in the environment, and the scheduled
job set none, so `make slate` ran the fetcher with no week every Wednesday and Saturday and the
leading `-` swallowed the refusal. Nothing spent quota and nothing said so.

**The week is counted from `CFB_WEEK_ONE`** — the date of the college season's first game, set
once in `.env` locally and as a repository variable in Actions. `hub.fetch.cfbd.configured_week`
snaps that date back to the Tuesday its week opens (which is where CFBD's own week numbers change
over) and counts seven-day blocks from there. The argument for a *date* rather than a week number
is in that function's docstring and is worth one line here: a pinned `CFB_WEEK=3` is right for
seven days and silently wrong for the rest of the season — in November it would fetch week 3 from
cache and record a successful refresh. A start date stated once in August stays true through
January.

Unset, unparseable, before the first game, or past week 15, no week is fetched and the run says
so. `--week N` overrides everything, and is how a backfill or a rerun asks for one week.

**Every run leaves a record in `site/data/cfbd.json`**, in the three states `hub.publish` uses for
every other producer:

| `fetched` | `stale` | means |
|---|---|---|
| `true` | `false` | the week was fetched and had rows |
| `true` | `true` | the week was fetched and came back empty — the source answered |
| `false` | `true` | nothing was fetched, and `reason` says why |

Counts per endpoint, never rows: `site/data` is committed and redistributing CFBD payloads is a
terms violation (see below). The slate workflow reads the record back and raises a warning
annotation when nothing was fetched — visible without scrolling a log, and never fatal, because an
unconfigured optional source must not fail a Sunday.

**A failed run says what kind of failure, not what the failure said.** The claim above has to
survive the failure path, and at first it did not: that path wrote the exception's text, and a
contract violation quotes the values that broke it. Measured 2026-09-05, a planted bad frame put
`week range [99, 99] outside [1, 20]; homePoints range [131, 131] outside [0, 120]` into the
record — four payload values and a column name, committed. Truncating at 400 characters bounded
how much escaped and left it rows all the same. What is recorded now is the exception's *type*,
plus the HTTP status where there is one, because a status code is three digits from the protocol
rather than a field of anybody's payload. The message goes to stderr, which is a log and not a
commit.

**What "empty" means to the contract.** A week that answers `[]` is the source saying "nothing
here", not a shape change, so the row minimum both CFBD contracts declare is relaxed for it — but
the contract still runs. `[]` parses to a frame with no rows *and no columns*, and there is no
shape in it to check. A response with no rows and columns of its own — `{"data": []}` from a
reshaped endpoint — is a rename wearing an empty week's clothes, and everything the contract
declares except the row count still applies to it. Skipping validation whenever a frame had no
rows, which is what this did for a while, took the check off the only two contracts the
provenance work constrains.

**What this costs.** `WEEKLY` is three endpoints — games, lines, box scores — so a fetched week is
three calls. The two scheduled runs can land in different college weeks, so the worst case is two
weeks a week: **6 calls, ~26 a month**, comfortably inside the 5–8/week the table above budgets.
Two runs inside one college week cost three, because the second reads the cache.

## No test spends a call

`hub.fetch.cfbd._http_get` is the only function in the module that reaches the network, and it
refuses outright when `PYTEST_CURRENT_TEST` is set, unless the node running is under
`tests/golden/`.

It sits there rather than in a fixture because of what happened without it. A contract test drove
the CLI while patching neither the environment reader nor the response cache, which was harmless
for exactly as long as the CLI had no week to resolve — and became three live calls on every run
of the suite once it began counting one from `CFB_WEEK_ONE`. Nothing was actually spent: measured
2026-09-05, the `.env` on that machine carried neither the anchor nor a key. But `.env.example`
now instructs a developer to set the anchor, and the key is already a repository secret, so what
stood between the suite and the account was which machine happened to run it. Patching the
transport per test is the right habit and every unit test here does it; it is not a guarantee,
because the guarantee has to hold for the test nobody remembered to patch.

`tests/golden/` is the exception, because it exists to diff a live response against the frozen
fixture — it is the only thing in the repo that can say whether the two CFBD contracts, both
written from documentation, resemble reality. It is marked `golden` and deselected by default, so
running it is a deliberate act: `uv run pytest -m golden` with a key on the machine spends one
call for the week it checks, and none once that response is cached.

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
