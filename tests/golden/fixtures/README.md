# Contract fixtures

Frozen payloads for `tests/contracts/`. Contract tests run against these rather than the
live API, because live-API tests in CI are flaky and prove nothing about our parsing. The
nightly job in `tests/golden/` is what actually catches an upstream rename.

## Provenance

Two kinds of fixture, and the filename says which. There is a third answer to the provenance
question and it has no file here: a contract that no test validates against any frozen
payload. Five of the fourteen in `hub.contracts` are in that position, so what this directory
records about them is nothing, and `verified_against_live=None` is how they say so — the
absence of a row below is not evidence of a capture. It was six until 2026-09-05: a row here
is only evidence a validation can reach, and `espn_scoreboard_cfb.json` was recorded below as a
capture while the one test reading it asserted fields by hand, so the resolver saw nothing
and the contract said `None` beside its own evidence. See
`tests/contracts/test_every_contract_is_applied.py`, which resolves which payload each
contract is actually checked against and requires the flag to match.

**A row here is also not evidence about the part of a payload that was trimmed away.** A
capture cut down past the path the production reader takes still reads as a capture from
that table, and cannot fail when the source changes shape on the path the code walks — which
is what the ESPN capture was until #73. So the rule for trimming is: remove anything you
like except a path a reader takes.

**And a row here is not evidence for a sentence the code writes about the payload.** #73
left the trim cut the other way: `_overlay_row` said an event-level `status` also exists and
is deliberately not read, and the trimmed capture carried it on none of its four events, so
the one file that could have settled the claim contradicted it instead. A capture that keeps
only what a reader reads cannot evidence a deliberate *not*-reading. So there is a second
rule, narrower than the first and with exactly one instance below: a field kept past the
reader's paths must say in this file why, and a test must fail when the file and the
sentence stop agreeing.

| Fixture | Source |
|---|---|
| `nflverse_pbp.json` | **Captured** 2026-08-23, real 2025 play-by-play, first 8 rows of `PBP_COLS` |
| `nflverse_ff_opportunity.json` | **Captured** 2026-08-23, real 2025 weekly ff_opportunity |
| `nflverse_schedules.json` | **Captured** 2026-08-23, real 2025 games with a published spread |
| `nflverse_ff_rankings.json` | **Captured** 2026-09-05 from the `all` archive, 8 rows scraped 2025-08-29 — see below |
| `nflverse_injuries.json` | **Captured** 2026-09-05, real 2024 injury report, 8 rows — see below |
| `nflverse_snap_counts.json` | **Captured** 2026-09-05, real 2024 snap counts, 8 rows — see below |
| `espn_scoreboard_cfb.json` | **Captured** 2026-09-05, the college board, all three states, trimmed — see below |
| `espn_scoreboard_nfl.json` | **Captured** 2026-09-05, the NFL board, `pre` only — see below |
| `cfbd_games.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `cfbd_lines.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `odds_spreads.synthetic.json` | **Hand-built** from The Odds API's documented response shape |
| `bigten_captures.synthetic.json` | **Hand-built** index of `hub.fetch.bigten`'s own archive: the 2026 page held no report on 2026-09-11, so nothing existed to capture — see below |

**The Big Ten index is hand-built for a different reason.** `bigten_captures.synthetic.json`
describes a shape this repo writes rather than one a third party returns, so a "real capture"
of it would be a run of `hub.fetch.bigten --capture` on a day the page carries a report. The 2026
page existed and held nothing on 2026-09-11, five days before the first report was due. The
first real capture replaces it: copy `archive/bigten/availability/captures.json` here under a
name without `.synthetic` and flip the flag.

The `.synthetic` suffix is not decoration. Those three were written by hand because no
`CFBD_API_KEY` or `ODDS_API_KEY` exists on this machine, so **they prove our parser handles
the shape we believe the API returns, not the shape it actually returns.** A synthetic
fixture cannot catch a rename, which is the failure contracts exist for. Replace them with
real captures the day a key is added; until then treat those two sources as unverified
against reality and read the `tests/golden/` job as the thing that would close the gap.

## The two ESPN scoreboard captures

Both captured on 2026-09-05, one unauthenticated GET each:

    https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=200
    https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard

— the URLs `hub.fetch.espn.scoreboard("cfb")` and `scoreboard("nfl")` build, with the
User-Agent `_get` sends first. The scoreboard endpoint needs no key, which is why the live
overlay may refresh every five minutes with no secret and why this is the one source here that
can be re-captured on demand. Two requests, not a loop.

**Why there are two, and why they are not equivalent.** `live_state` reads both leagues
through the same code, so one board evidences most of what the other would — but only one of
them could be caught with a game in progress. Until #75 only the college board was frozen and
the NFL one, which is the league the poller *defaults* to, had no frozen shape at
all: a rename would have waited for the nightly canary. Both are frozen now, and what each
can prove is set out below rather than left to the filenames.

### `espn_scoreboard_cfb.json` — the live path

The college board that afternoon held games in all three states at once. **Four of the 99
events**, chosen to hold every state and both situation cases:

| Event | State | Why it is in the file |
|---|---|---|
| Marshall at Penn State | `in` | `situation` carries `possession` and `downDistanceText` |
| Texas State at Texas | `in` | live, but `situation` carried neither — ESPN sent it between plays, after a kick |
| Missouri State at Texas A&M | `pre` | scores are `"0"`, not absent |
| ECU at Alabama | `post` | a final |

Each event keeps `id`, `date`, `name` and one competition holding `status.type`
(`state`, `shortDetail`), its competitors (`homeAway`, `score`, `team.abbreviation`) and a
`situation` reduced to the two fields the reader reads. Everything else ESPN sent — venue,
odds, broadcasts, links, and a `situation.lastPlay` block naming players and linking their
headshots — is gone. It is third-party data a fixture does not need, and dropping it is the
same instinct as the CFBD note below.

**The one exception, and the reason for it.** The Marshall at Penn State event also keeps its
**event-level `status`**, reduced to the same two fields, and nothing reads it. That is the
second trim rule at the top of this file, and this is its only instance. `_overlay_row` says
in so many words that an event-level `status` exists in the response and is deliberately not
read; the capture #73 left carried it on none of its four events, so the file a reader would
check the sentence against contradicted it — the mirror of the defect #73 itself removed,
which was a capture that kept *only* the event-level copy and so could not exercise the
competition-level path production walks. Keeping one event with both is what makes
"deliberately not read" a choice a reader can watch being made.
`test_the_event_level_status_is_in_the_capture_and_still_unread` in
`tests/unit/test_fetch_espn.py` holds the three ends together: it fails if the comment stops
claiming it, if the capture stops carrying it, or if a row's state starts coming off the
event. If that sentence ever leaves `_overlay_row`, this field leaves the capture with it.

That is what makes this file able to fail. `tests/contracts/test_source_contracts.py` runs it
through `espn.live_state`, and three shape changes tried against it on 2026-09-05 — renaming
`team.abbreviation`, renaming `homeAway`, and dropping `competitions[].status` — each stop
every event resolving and raise; two events sharing an id fails the contract. An `id` arriving
as a number is *not* caught, because `SCOREBOARD_TYPES` coerces it back to `Utf8` first, in
production as well as here.

### `espn_scoreboard_nfl.json` — the NFL board, before kickoff

**All sixteen of the sixteen events on that board, and every one of them `pre`.** The 2026
NFL season had not started on 2026-09-05 — the earliest event in the file kicks off on 9/9 —
so this is what the NFL board actually looked like that day. The trim is the one
above minus its exception: `id`, `date`, `name`, and the competition's `status.type`,
competitors and (absent, for a game not yet under way) `situation`. Nothing was kept past the
reader's paths here; the event-level-status evidence lives in the college file.

**What it evidences.** Sixteen of sixteen events resolve through `live_state`, so the field
names the reader walks are evidenced on the NFL board rather than assumed from the
college one. That is the whole of what it earns, and it is enough: the poller *defaults* to
this league, and until now its field names were inherited from a college capture rather than
observed.

It was first justified by `possession` and `down_distance` arriving all-null on a quiet
board. That justification is false — neither column is in `SCOREBOARD_TYPES` or the
contract's `required`, so nothing validates their dtype and dropping both outright still
passes. The claim had sat in `espn.py` for months; this fixture is what made it checkable.

**What it cannot evidence, and why no fixture here does.** Nothing about a NFL game
in progress: `situation.possession` and `downDistanceText` are absent from all sixteen
events, and so is a `post` state. An in-progress NFL game did not exist on
2026-09-05 to capture. The three ways to have one anyway are to hand-write a payload, to edit
a `pre` event into an `in` one, or to wait; the first two produce a fixture asserting a shape
nobody observed, which is worse than the gap, because it would go green on a board ESPN never
sends and read as coverage. So the gap is left open and named: the NFL in-progress
path is watched live by `tests/golden/test_golden.py`, which runs both leagues through
`live_state` nightly, and it is not frozen. Re-capture this file during a Sunday once the
season is under way and the gap closes on real data — that is the only thing that should
close it.

## The three #33 added

Captured on 2026-09-05 by one `nflreadpy` call each — `load_ff_rankings("all")`,
`load_injuries([2024])`, `load_snap_counts([2024])` — with no key, since these are public
nflverse releases. No loop over teams or games; one bulk pull per source.

**What was trimmed, and why the trim keeps every reader's path.** Each file keeps exactly the
columns its contract declares and drops the rest. That is the whole of what a reader takes:
`FF_RANKINGS` covers the nine columns `hub.draft.board._select_consensus` selects plus the
`scrape_date` the as-of filter bounds on; `INJURIES` covers the identity columns plus the
`report_status`/`practice_status` pair `hub.models.injury` fits its retention table on;
`SNAP_COUNTS` covers the identity columns plus `offense_pct`, which is the one number
`hub.models.panel.snap_share` reads. The rule from the ESPN re-capture holds — remove
anything except a path a reader takes — and here the contract *is* that list.

**Which rows, and why those.** Not the first eight of each. The rankings file carries two
`page_type` values because 47 ranking pages are stacked in the real frame on their own ECR
scales and telling them apart is that column's job. The injuries file carries five rows with
a game designation and three without, because `report_status` is null on 21,490 of 40,204
rows over 2019-25 and an all-null column would arrive typed `Null` and fail the contract for
the wrong reason. The snap-counts file carries six starters and two players who took no
offensive snap, so the bottom of the `offense_pct` range is exercised as well as the top.

CFBD prohibits redistributing its data. A hand-built two-row shape sample is not a dataset,
but a real capture would be -- if you replace these, keep them minimal.

## `panel_archive/` — the seven #108 added

Everything above this line is one payload per source, a handful of rows, read by a contract
test. This directory is different in kind: it is an **archive** rather than a payload, and it
exists to be *run through*, not parsed. `hub.models.panel.build_panel` reads eleven sources and
every one is marked `pragma: no cover - network`, so the entire assembly — eleven joins, five
per-source narrowings, the expanding aggregates that are the one statement of the
before-its-outcome rule — was covered by a single `inspect.getsource` call that grepped the
function's own text. `tests/panelarchive.py` serves these files to the assembly at the last
call before the wire, so that everything above that line is the production path.

Captured on **2026-09-05** by one bulk `nflreadpy` call per source, plus one run of
`hub.draft.board.board_as_of(2024)`. No key, no loop over teams or games.

| Fixture | Source |
|---|---|
| `panel_archive/player_stats.json` | **Captured**, `load_player_stats([2023, 2024], summary_level="week")`, 410 rows |
| `panel_archive/schedules.json` | **Captured**, `load_schedules()`, all 416 REG games of weeks 1-14 |
| `panel_archive/snap_counts.json` | **Captured**, `load_snap_counts([2023, 2024])`, 410 rows |
| `panel_archive/injuries.json` | **Captured**, `load_injuries([2023, 2024])`, 101 rows |
| `panel_archive/ff_rankings.json` | **Captured**, `load_ff_rankings("all")`, the `weekly-op` page, 456 rows |
| `panel_archive/ff_opportunity.json` | **Captured**, `load_ff_opportunity(stat_type="weekly")`, 1,000 rows — see below |
| `panel_archive/draft_board.json` | **Captured** output of `board_as_of(2024)`, its top 200 rows |

**Each file carries its own dtypes.** These are `{"dtypes": {...}, "rows": [...]}` rather than
a bare list, and that is not a second dialect for its own sake. JSON has three scalar types and
nflverse has a dozen: inferred, a float column whose captured values are all whole numbers
comes back `Int64`, and a column null on every captured row comes back `Null`. Both change what
the assembly does with it, so the dtype map is part of the capture — it records the frame
nflverse returned rather than what JSON could carry.

**What was trimmed, and why the trim keeps every path the assembly takes.** Sixteen players,
four at each position, over weeks 1-14 of 2023 and 2024. Sixteen because the trend features
reach back six calendar weeks and the priors expand over every earlier week, so the binding
constraint is *consecutive weeks*, not players: a capture of four hundred players over three
weeks would leave every trend null and every leakage assertion true over nothing.
`test_the_trend_features_have_something_to_compute_on` is the floor that says so, in numbers.
Columns are cut to what the reader takes — the twenty-three `weekly_stats` names, the eleven
`game_context` and `week_windows` read off a schedule, `offense_pct` and the REG marker for
snaps, the status/practice pair plus identity for injuries, and `FF_RANKINGS.required` exactly,
since that is the column list `load_rankings` asks the loader for.

**Which sixteen, and why those.** Six change team between the two seasons or miss weeks, so the
calendar-grid reindexing in `trend` and `recent_mean` is exercised on real gaps rather than on
a frame that never has one. All sixteen are on the frozen board, so the gate-side assembly
joins rather than resolving to nothing. And two — **Michael Pittman in 2024 week 6 and Gus
Edwards in 2024 week 10** — were really absent from that week's cross-position consensus page
and really scored (12.5 and 5.5). Those two cells are the only reason
`hub.season.weekly_gate.VOID_FLOOR` guards something a test can reach: three roster-weeks of
3,080 over the twenty-draft **Cohort**, 0.097% against a pre-registered floor of 2%.

**The board is a capture of an output, not of inputs.** `board_as_of(2024)` returned 1,103
players and the file keeps its top 200 in the board's own `(ecr, player)` order, every column.
Rows only, because the draft simulation reads columns this file has no business guessing at,
and 200 because twelve teams over fourteen rounds take 168 of them. Freezing the board's
*inputs* instead would mean a whole season of `ff_opportunity` and the redraft archive, to
exercise a seam that belongs to `hub.draft.board` and has six test files of its own. What
`assemble_universe` owns is everything from the board onward, and that runs here for real.

**The 650 rows of `ff_opportunity` that join to nothing.** `FF_OPPORTUNITY` declares
`min_rows=1000` and the production loader validates before the panel sees the frame, so a file
holding only the sixteen players' 350 rows would fail the contract rather than the test. The
remainder is every other row in the same (season, week) cells, in `(season, week, player_id)`
order, taken until the file reaches exactly 1,000. They join to no Panel row and are there for
the floor and nothing else.

**What this archive cannot drive, and why no file here does.** `PanelSpec(routes=...)` and
`PanelSpec(scheme=...)` read `load_participation` and `load_ftn_charting`, which are
play-level. A capture deep enough to make a six-week team trend real is thousands of rows of
`offense_players` strings — larger than everything else here together — to exercise two opt-in
features the repo has already measured as null, whose arithmetic is covered by
`route_share_from_plays` and `scheme_rates_from_plays` on their own. So the gap is left open
and named, the way the NFL in-progress board above is: `tests/panelarchive.py` installs both as
refusals that raise, so the day someone turns a flag on they get that sentence instead of a
silent live fetch. What is *not* covered by this archive is the call: whether the scheme trend
is taken on the team-week frame and joined once, which is the hazard `trend`'s own guard names.

**One number here is an artefact of the trim and must not be read as production.** The board
carries 200 players and the frozen consensus page carries 16 of them, so 88% of roster-week
cells come back unranked against roughly 64% in a real run. `coverage` reports `unranked` and
`join_failure` as two numbers for exactly this reason — only the second is what `VOID_FLOOR`
reads, and only the second is asserted.
