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
| `/lines?year&week`, refreshed, per availability-report capture (#215) | 7/week in season |
| **Total** | **~12–15/week in season → ~60/month; ~5–8/week outside the report regime** |

That leaves roughly 940/month of headroom in season and more outside it. Spend it on the
one-time historical backfill.

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

## The Big Ten availability archive: one call per deadline, stated before it ran

Issue #215's disposition: the availability reports the Big Ten began publishing on 19 September
2026 are captured by a scheduled workflow (`bigten.yml`, `hub.fetch.bigten`), and each capture
pairs the reports with an odds snapshot taken in the same run. The reports themselves cost
nothing here — they come from `bigten.org`, which is public and unmetered, and **CFBD has no
availability endpoint at all** (checked against its OpenAPI spec on 2026-09-11, not from
memory). The snapshot is the metered half, and it is exactly one call:

| What | Endpoint | Calls |
|---|---|---|
| The odds snapshot beside each capture | `/lines?year&week`, `refresh=True` | 1 per capture |
| Anything per team, per game, per conference | — | 0, by construction (`hub.fetch.cfbd`) |

`refresh=True` because the point is the price *at this deadline*; a cached week is the price at
some earlier one. `bulk` still refuses to spend past the run ceiling or the month, and serves the
cached week when it holds one — the capture then records the snapshot as an earlier price rather
than as this deadline's, read off the call counter rather than a clock.

**Per week: 7 calls.** Seven deadlines (`hub.fetch.bigten.DEADLINES`): the evening deadline Tuesday
through Saturday night ET (01:30 UTC Wednesday through Sunday), and two gameday captures on
Saturday (15:00 and 21:00 UTC). Each is one `/lines` call. **Per month: about 30**, on top of the
slate's ~26 above, so roughly **56 a month** against 1,000 — and against the 488 that remained
when this was written, comfortably. A missed run spends nothing; a run with no
`BIGTEN_ARCHIVE_KEY` spends nothing (see below); the job stops when `CFB_WEEK_ONE` says the
regular season is over, which is week 15 and the conference championship.

**No run spends what it cannot keep.** The snapshot is a CFBD payload, `data/raw/` is gitignored
for that reason, and a runner is ephemeral — so a snapshot taken on a runner and not committed is
a call spent on nothing. The workflow encrypts each run's parquet with the `BIGTEN_ARCHIVE_KEY`
repository secret (`openssl enc -aes-256-cbc -pbkdf2`) and commits the ciphertext under
`archive/bigten/lines/`, which is retrievable by the one person holding the key and redistributed
to nobody. Without that secret the capture step passes `--skip-lines` and the CFBD call is not
made; the reports are archived either way. To set it: `openssl rand -hex 32` into the repository
secret. To read one back:

    openssl enc -d -aes-256-cbc -pbkdf2 -pass env:BIGTEN_ARCHIVE_KEY \
        -in archive/bigten/lines/2026/w03/2026-09-17T0130Z.parquet.enc -out /tmp/w03.parquet

**The reports are committed in the clear**, under `archive/bigten/availability/` — the one tracked
archive in this repo. They are the conference's own publication, made for exactly this
dissemination, and losing one is the thing the ticket is about. That is a decision the maintainer
can reverse by moving `hub.fetch.bigten.ARCHIVE` under `data/`; nothing else depends on it.

**What the first live run has to show, and nothing here can show before it.** Nothing in this
repo executes GitHub Actions and the 2026 page held no report on the day this was written. So
the first evening deadline after 16 September is the test, and these are the things to check on it:

1. `site/data/bigten.json` says `fetched: true`. If `stale: true` with `PageShapeChanged` in the
   reason, the conference's page moved and the raw page is under `archive/bigten/availability/
   2026/page/` — the bytes are kept; the parser is what needs fixing.
2. `documents.new` is at least 2 on a day the conference published: the article record and one
   linked file. If it is 1, the reports are in the body rather than linked — still archived, in
   the article record — and `Article.links()` needs to learn the shape.
3. `lines.rows` is a number and `lines.why` is null, and `state/cfbd-quota.json` moved by exactly
   one. If `lines.why` says *earlier capture*, `bulk` refused; the reason is on the run's stderr.
4. `missed.count` is 0. A nonzero count on the first run means a cron fired and the run did not,
   which is the GitHub delay `live.yml` records — the next run attributes itself correctly.
5. A commit landed on `main` from the job, with `archive/` in it. The one on the Wednesday
   deadline before the first deadline may be a stamp only; that is the empty page, not a failure.

**What is deliberately not built.** Nothing parses a report into player rows. The 2024 PDFs have
a clean text layer (one page per team-game, designation headings over jersey-and-name lines,
measured with `pypdf` on 2026-09-11), so a parser is realistic — but the 2026 reports carry five
midweek designations and three gameday ones in a shape nobody has seen, and a parser written
against a guess is the fixture-shaped incident this repo already has. The archive keeps the
designations as given because it keeps the bytes; the parser is the ticket after the first
capture exists.

## The cache can be corrected, and says how old it is

Until #175 the cache was permanent by file existence: if `data/raw/cfbd/{endpoint}/{stem}.parquet`
was there it was returned, with no maximum age, no refresh parameter and nothing recording when
the bytes arrived. On a metered source that is the expensive half rather than the cheap one — it
is both the reason to cache and the reason a stale price could never be put right.

Three things changed, and the default did not:

- **A written entry carries its capture time**, in a `.capture.json` beside it — the same
  sidecar-next-to-the-entry shape `hub.fetch.nflverse` writes its `Pin` in. `cfbd.captured_at(...)`
  reads it back without fetching anything.
- **`refresh=True` re-fetches**, and `max_age=timedelta(...)` re-fetches an entry captured longer
  ago than that. On the CLI: `--refresh` and `--max-age-hours`. Three endpoints in a week, so a
  refreshed week is three calls, which is why neither has a default.
- **A refusal serves rather than raises.** A refresh that would pass the run ceiling or the monthly
  budget, or one with no key to make it with, prints what it did and hands back the cached payload
  with its capture time. Nothing to serve is still a raise.

An entry written before this change has nothing beside it and reads back as an **unknown** capture
time — never the file's mtime, which a clone, a copy or a restore rewrites, and which therefore
dates the file rather than the fetch. Unknown means "ask again"; a plausible wrong number means
"no need to". An unknown age is treated as past any stated `max_age` for the same reason.

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
