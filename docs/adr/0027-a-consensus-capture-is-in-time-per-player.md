# A consensus capture is in time per player, not per week

**Status:** accepted 2026-10-20 (option 2 of #515, adopted by the maintainer).

**Decision.** A **Consensus capture** counts as point-in-time for a player if it was taken before
*that player's* kickoff. A player whose game had started when the capture was taken is left out of
that week's comparison against consensus. This replaces the per-week rule: "taken after the
previous week's last game and before the target week's first game day".

## Why

The per-week rule could not be met. The nflverse rankings archive publishes the weekly consensus
page only on Fridays (all six 2026 scrapes: 09-11 through 10-16), so the scrape always lands after
Thursday's game. No 2026 capture ever qualified, and nothing scored against consensus this season
had a ranking to score against (#515).

The per-player rule keeps the guarantee where it matters: no ranking is credited with a game it
could have seen. The cost is that Thursday's players drop out of the comparison, roughly 2 of
about 14 games a week.

## Alternatives declined

- **Scrape FantasyPros directly on Tuesday and Wednesday.** This keeps the per-week rule, but it
  adds a direct source whose terms have not been checked.
- **Retire the comparison against consensus.** That would leave #492's provisional ECR rule with
  no forward record of the kind its reopening condition names.

## Consequences

- The strict and lineup-only captures stop being different kinds of capture. A Friday capture is
  in time for Sunday's players, both for evaluation and for setting the lineup.
- Readers that score against consensus filter by the player's own kickoff, not the week's first.
- The restatement of what earlier documents meant by "pre-registered against consensus" is dated
  (method rule 13); the prior text is kept.
