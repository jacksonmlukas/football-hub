# The draft simulator indexes the board

**Status:** accepted 2026-08-24.

**Decision.** `simulate_remaining_draft` takes the whole board, carries a `gone` mask, and
returns row indices **into the board**. Every seat's roster is seeded from the players it
already holds. `draft_pool` — "players still available, priced" — is deleted.

## Why

It used to index a derived frame of only-available players, and carried this warning:

> Roster indices returned by `simulate_remaining_draft()` index THIS frame, not the full
> board. Everything that consumes those rosters — mu, sd, pos — must be taken from here too,
> or the simulation silently scores the wrong players.

That warning was written by someone who had been bitten. It also concealed a worse
consequence: because the pool excluded players already drafted, the simulated rosters
contained only picks made *during* the simulation. Championship equity was therefore blind to
what you already held. Holding a quarterback, `win_probability` ranked a second quarterback
above a startable running back; driven as a lead strategy it finished with four of them; and
on the live 2026 board it named a third and fourth running back at three of the first six
turns while QB, WR and TE sat empty.

Fixing that needs held players inside the rosters, which needs them inside the frame the
indices point at. The alternative was a *third* frame — available, held, board — with indices
into one of them and the same warning, now harder to obey. Indexing the board removes the
class of error instead of documenting it.

## Consequences

- Rosters are seeded for **all twelve seats**, not just yours. Seeding only yours would leave
  eleven opponents fielding future picks alone, making them weaker than they are and
  inflating your `p_win` — a number the output already has to apologise for.
- A recorded pick that is not on the board — a kicker, a defence, a misspelling — is skipped.
  `DRAFTED_POSITIONS` excludes K and DST deliberately, so this is the normal case from about
  round 13 of every real draft, and `suggest_unmatched` already flags a misspelling at the
  point a human can still fix it.
- `draft_pool` fails the deletion test: after this change its one production caller is
  `market_pick`, which ranks on `adp` or `ecr` and never reads the `mu_pick` blend that
  `draft_pool` existed to attach. That caller now takes `remaining(board, state)`.
- Do not reintroduce "a pool of available players, priced". It is the obvious-looking
  refactor, it reads better than a mask, and it is how the blindness got in.
