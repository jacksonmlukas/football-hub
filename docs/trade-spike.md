# Does this league trade?

**Measured 2026-08-23. Answer: no. Do not build the trade evaluator.**

`docs/championship-leverage.md` puts the trade evaluator last in its sequencing and gates it
explicitly: *"Trades are last because the league's trade activity is unknown. Do not guess:
`espn-api` exposes transaction history. Pull two or three prior seasons and count actual
trades."* This is that spike. The league has existed since 2022, so all four completed
seasons were counted rather than two or three.

## The numbers

| Season | Trades | Teams that traded | Waiver adds | Drops |
|---|---|---|---|---|
| 2022 | 6 | 7 of 12 | 319 | 313 |
| 2023 | 4 | 5 of 12 | 264 | 274 |
| 2024 | 2 | 2 of 12 | 254 | 257 |
| 2025 | 3 | 6 of 12 | 247 | 240 |
| **Total** | **15** | — | **1,084** | **1,084** |

3.75 trades per season across twelve teams. Trades are **1.4%** of all player movement, and
the trend is down: six in the league's first year, two or three since.

## What that means for the roadmap

**A trade evaluator would run on well under one decision per season.** Each trade involves
two of twelve teams, so the expected number of trades *you* are party to is 2 x 3.75 / 12 =
**0.6 per season**. Even a perfect evaluator, delivering its answer instantly, would be
consulted less than once a year. L4 is not worth building.

**The volume is on the wire, by a factor of thirty.** 271 acquisitions a season is about 23
per team, roughly 1.6 adds per team per week. That makes `championship-leverage.md`'s
priority 3 — waiver availability under known priority — the layer that actually earns its
build cost, and it should absorb the effort L4 would have taken.

One caveat on that, from the same doc: waivers run on reverse standings with no FAAB, so a
good team picks last every week and contested adds are effectively unavailable. High volume
is not the same as high *accessible* volume. The waiver layer needs to model priority, not
just availability, or it will recommend players who will be gone.

## How it was measured, and why not the obvious way

`espn_api`'s `recent_activity` is the documented route and **it does not work here**: it
returns HTTP 400 for every past season of this league. The cause is not the wrapper. It
reads `.../leagues/{id}/communication/?view=kona_league_communication`, and ESPN 404s that
endpoint for every season but the current one — confirmed directly against both the
`seasons/{yr}` and `leagueHistory` paths, at every page size. For 2026 it returns 200 with
zero topics, since nothing has happened yet.

What does survive is the per-team `transactionCounter` on the `mTeam` view, which
`leagueHistory` serves for all four seasons. `hub.fetch.espn.transaction_counts` reads it.

**A trade increments the counter on both teams**, so the league total is half the sum;
acquisitions and drops have one side each and are not halved. Getting that wrong would
double every number above. `trade_summary` reports any season whose trade sum is odd, which
would mean the two-sided assumption broke — a three-way deal, or a team dropped from league
history. No season here was odd, so the halving is sound for all four.

## What this does not say

It does not say trades are a bad idea, or that this league would reject one. It says the
observed frequency is too low to justify building and maintaining an evaluator for it. If a
trade is ever offered, evaluate it by hand with the tools that already exist — the roster
simulator in `hub/draft/season.py` will price a roster swap directly.
