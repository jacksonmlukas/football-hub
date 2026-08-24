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

**Edge**:
Consensus rank minus draft-market pick, on a common scale. Positive means your leaguemates,
drafting off ESPN's board, will let this player fall past his consensus value.

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

### Season simulation

**Championship equity**:
A candidate's probability of winning the league, from simulating the season to a champion.
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
