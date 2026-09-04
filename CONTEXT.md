# Football Hub

A fantasy-football and college-football modelling repo built around one draft: a 12-team
full-PPR ESPN snake league, slot 3 of 12. Everything here exists to answer one question at a
time — *which player, at this pick, gives the highest chance of winning the league?*

## Language

### Market signals

There are three markets in this repo and they are not interchangeable. **Never write "the
market" unqualified** — it has meant all three of these in the same paragraph, and the reader
cannot tell which.

**Draft market**:
Where your league actually takes a player, as ESPN's average draft position (ADP).
_Avoid_: the market, ADP alone when the contrast with consensus matters.

**Consensus**:
FantasyPros' expert consensus ranking (ECR), a rank rather than a pick number. The only market
signal available for past seasons, since ADP is published for the current season only.
_Avoid_: the market, rankings.

**Betting market**:
Sportsbook prices — closing spreads and player props. Treated as already-efficient: the repo
backtests against it to audit itself, never to beat it.
_Avoid_: the market, the book, Vegas.

**Snapshot**:
One dated, immutable capture of the betting market's spread on a game, written by
`hub.fetch.odds` and never rewritten. Many per game across a week; the as-of join picks the
one that was live at a given moment. Contrast the schedule's own `spread_line`, which is a
single field that *moves* — the same quantity, but not a record of it.
_Avoid_: the line, the closing line (a snapshot is rarely the close).

**Price source**:
Which input priced a prediction — a dated snapshot, or the moving field it falls back to.
Carried on the row and in the version string, because a prediction priced from a snapshot
can be shown afterwards and one priced from a moving field can only be asserted. Two
predictions that agree on the number and differ here are different artifacts.
_Avoid_: line source, provenance alone.

**Edge**:
Consensus rank minus draft-market pick, on a common scale. Positive means your leaguemates,
drafting off ESPN's board, will let this player fall past his consensus value. Displayed and
never sorted on — it needs historical ADP to validate and none exists
([ADR-0010](docs/adr/0010-edge-is-displayed-but-never-ranked-on.md)).

**Corrected ADP**:
The draft market's ordering, moved by the corrections this repo has measured and the market
has not priced, bounded at 20% of a player's own ADP. What THE PICK ranks on.
_Avoid_: adjusted ADP, our ADP.

**Correction**:
A fitted markdown or markup to a projection, in points per game, where a measurement says the
market is wrong — touchdown luck, durability, current designation. Each carries a coefficient
with an interval. An unfitted quantity is not a correction, however useful.

**Player key**:
The comparable form of a player's name, so that one player named three ways by three sources
is one player. Lower-cased, accents folded, punctuation and generational suffixes dropped:
`A.J. Brown` and `AJ Brown` share a key, `Marvin Harrison Jr.` and `Marvin Harrison` share a
key. `hub.names.player_key`, and every cross-source join in the repo goes through it —
`docs/decisions.md` records an exact join silently dropping players as the bug it fixes.
_Avoid_: normalised name, `_norm` — it was private and lived in `hub.draft.state` until
2026-08-27, which is why ten modules across three packages imported a private helper.

### The board

**Board**:
One row per drafted-position player, carrying every signal joined onto a consensus ranking. It
is the frame every draft-night decision reads.

**xFP**:
Expected fantasy points — what a player's opportunity was worth, separating volume from
finishing. Sourced from nflverse's ff_opportunity, per game.
_Avoid_: expected points, projected points.

**Replacement level**:
The points of the last startable player at a position, given league size and flex allocation.
Only players with a real games sample define it; a rookie is priced against it without setting
it.

**VOR**:
Value over replacement — a player's points per game minus his position's replacement level.
The shortlist quantity, not the recommendation.

**Talent**:
How wrong a preseason projection turns out to be about a player's season, as a fraction of his
projected points. Drawn once per simulated season, before any weekly scoring.

### Draft decisions

**League**:
How this one is shaped — the starting **Slots** (QB1/RB2/WR3/TE1/FLEX1), which positions may
fill the flex, how many weeks the regular season runs — and the one rule for filling it:
best available to each required Slot, then the flex. Read by both halves of the repo, the
draft and the in-season path, which is why it is a leaf and not part of either.
_Avoid_: roster (which is *your* players, not the league's shape), settings.

**Scarcity** / **Value**:
The two modes a pick can be in, set by the wait until your next turn. Scarcity means a long
wait, so rank by who will not survive it; value means a short wait, so rank by VOR.

**Cost of waiting**:
VOR multiplied by the probability a player is gone before your next pick. The scarcity-mode
ranking.

**Room**:
The eleven opponents, as a model: each follows consensus with fitted pick noise and fills its
own starting slots before taking depth. Not the eleven actual people — a simulated room is
what makes a draft replayable.

**Slot**:
A draft position, 1 to `teams`. Yours is 3. `roster_for(state, slot)` reconstructs any seat's
roster from the snake order rather than tracking it.
_Avoid_: seat, pick position.

**Run**:
Three or more picks at one position inside the last five — the moment replacement level at
that position moves and a static board goes stale.

**THE PICK**:
The single recommendation the draft-night output leads with: best available in the draft
market that fills an unfilled starting slot.

### How a thing earns its place

**Signal**:
A claim that some quantity predicts outcomes *beyond what consensus already knows*. Tested by
a **screen**: partial correlation against ECR. Six have been screened; five were null and one —
the in-season snap-share trend — survived. Note what separates it: the five nulls all asked
consensus about information it had had all summer, and the one that survived asked about
something published on a Monday.

**Model**:
A component that produces a projection or a decision. Tested by a **gate**: does it beat the
simplest thing that already works? Championship equity was gated against following the market
and lost by 19.66 points a team-game.

**Provisional rule**:
A decision rule adopted *without* a gate, because no gate can run at available n — never
because one ran and was failed. It requires a signal that passed a screen, a rule written down
before use, a log of every application, and a stated horizon. It is reported as judgment and
never as validated. Exactly one thing qualifies today: the snap-share trend as a waiver
tiebreaker. See [ADR-0014](docs/adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md).
_Avoid_: heuristic, hunch — both hide that it is pre-registered and logged.

**Screen** / **Gate**:
The two tests, and they are not interchangeable. A screen asks "is this real?"; a gate asks
"is this better than what it replaces?" A signal can pass one and fail the other: the snap-share
trend screened positive and then lost its gate to consensus rank, because a partial correlation
says a quantity *adds* to the board and never that it should *be* the board
([ADR-0013](docs/adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md)). Confusing them is how a well-built, well-tested model
ships while being confidently worse than a one-line rule.

A third case exists and is not a third test: a signal passes a screen and its decision cannot
be gated at any n this project will see. That is a **provisional rule**, above.
_Avoid_: validate, backtest — both hide which question is being asked.

### In-season

**Usage**:
The count vector a player's week produces — targets, carries, pass attempts, receptions. The
quantity a weekly model predicts, because volume persists (targets r = 0.805) and efficiency
does not (yards per carry r = 0.108, touchdowns per yard r ≈ 0).
_Avoid_: opportunity (which is what **xFP** already prices), volume (which names only half of
it — a carry and a target are not the same unit).

**Panel**:
The frame every week-level question is asked on: one row per (player, week), with every feature
measured *strictly before* that week's kickoff. Pre-kickoff facts published *for* the week — the
line, the injury report, the opponent — are week-*w* information; anything derived from play uses
weeks before *w*. It is the unit of measurement, not a table: a screen correlates within its
(season, week) cells, a projection is fitted on it, and a gate is assembled from it.
_Avoid_: dataset, training data — both hide that the before-its-outcome rule is what makes it a
Panel rather than a join.

**Weekly projection**:
A player's expected points for one *named* week, against a named opponent. Distinct from
**xFP**, which is per-game and season-long. It is **shown and never ranked on**: measurably more
accurate than the flat projection (+0.074 MAE at 5.9 se) and it still lost the lineup decision
to consensus rank at −0.304 points a team-week, see
[ADR-0016](docs/adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md).
_Avoid_: projection unqualified — say weekly projection or xFP, since the whole open question
is whether the week adds anything.

### Season simulation

**Championship equity**:
A candidate's probability of winning the league, from simulating the season to a champion.
Measured worse than consensus at actually picking — −19.66 points per team game across
2022–25 — so it does not choose the pick; see [ADR-0009](docs/adr/0009-championship-equity-does-not-pick.md).
_Avoid_: P(win) as a standalone term — say championship equity, and reserve `p_win` for the
column.

**Lift**:
The *difference* in championship equity between taking one candidate and taking another. The
decision quantity. Absolute equity is inflated, because the season is scored on the same
projection the ranking used; the difference is paired and survives that.
_Avoid_: edge (which means something else entirely here).

**Co-leader**:
A candidate the simulation cannot separate from the best one — within two standard errors of
the difference. A tie is reported as a tie rather than broken by noise.
