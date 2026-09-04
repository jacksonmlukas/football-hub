# Live scores are not part of the record

**Status:** accepted 2026-09-04.

**Decision.** `site/data/live.json` is **not committed**. It is generated in the deploy job
and shipped with the same Pages upload as everything else. Every other artifact stays
committed, and the commit stays the timestamp.

## The question

`pages.yml` deploys what is committed, deliberately, and says why:

> This deploys what is committed; it does not run `hub.publish`. […] it would break the
> record. `docs/track-record.md` rule 1 says a prediction counts only if its commit predates
> kickoff […] Publishing straight to Pages would leave nothing to pin. **The commit is the
> timestamp.**

A live overlay wants to refresh every ten minutes during a game window. Read literally, the
rule above says commit it — roughly seventy-two commits a Sunday, thirteen hundred a season,
into a repository whose history *is* the record. Which makes the record harder to read in
order to protect it.

## Why the two compose

Rule 1 pins **claims this repo makes**. A prediction is unfalsifiable unless you can show
when it was made, so it must be frozen and dated by something outside this repo's control.

A live score is **someone else's fact**. It is ESPN's, it is checkable against ESPN at the
time, and this repo asserts nothing by relaying it. Pinning it proves nothing and protects
nothing — there is no claim to protect.

The codebase had already drawn this line, in `publish.live`'s own docstring, before the
deploy question came up:

> Predictions are frozen at lock and must never be regenerated mid-slate: a win probability
> that drifts while games are in progress is indistinguishable from one that was always going
> to look right, which destroys the pre-registration the whole record depends on. **So: scores
> move, model numbers do not, and the page says which is which.**

Committing the thing that moves puts both categories through one mechanism and loses the
distinction the page is built to display.

## What it fixes today

`live.json` is committed right now, and the deployed page therefore shows whatever was last
committed. **The current design guarantees the staleness the watchdog is reporting** — the
heartbeat cannot be fresh between commits, however often the poller runs. Issue #4 is that
guarantee, observed.

## The cost, accepted

A fresh clone has no `live.json`, so a local `make serve` renders "Live scores unavailable —
model numbers below are still correct" until something generates one. That is the designed
degradation, it is the honest state of a clone that has not fetched anything, and the manifest
already reports the panel absent rather than pretending otherwise.

## What would reopen this

A live artifact that started carrying a claim of ours rather than a relay of someone else's —
an in-play win probability computed here, say, rather than read from the feed. That is a
prediction wearing a scoreboard's clothes, and it would belong back inside rule 1.
