# Public since Sep 4

The draft was Sep 3 and the leaguemates are a mixed room. Publishing the board before then
would have handed them the exact artifact built to beat them, so the repo stayed private
until the draft was over. It flipped on 2026-09-04. This is the record of what private cost,
what it did not, and what is now checked rather than remembered.

## What private cost, while it lasted

**GitHub Pages did not work.** Pages is available in public repositories on GitHub Free;
private-repo Pages requires Pro. And even on Pro the site is still publicly reachable — truly
private Pages needs an organization on Enterprise Cloud. So there was no configuration where
a private repo got you a hidden dashboard, and the answer until the flip was to run it
locally:

```
make serve      # http://localhost:8000
```

Pages now deploys from `main`. `make serve` is still how you look at it without pushing.

**Actions minutes are metered.** Free gives 2,000 minutes/month on private repos; public
repos are unmetered. That is why CI ran on push only and the watchdog and golden crons were
commented out — and why `.github/workflows/watchdog.yml`, which records what it cost when the
watchdog cron was left on by mistake anyway, is the one place that incident is written down.
Both crons went back on at the flip.

**Whether they are still on is not recorded here.** `scripts/preflight_public.sh` reads the
workflow files and reports it, because for part of 2026-09-04 this document and two workflow
comments all said the crons were off while they were running.

## What private did not cost you

**Pre-registration still works.** The track-record guarantee is that predictions were
committed before kickoff, and git timestamps do not care about repo visibility. The full
history came across with the flip, and every pre-draft commit is dated and verifiable.
Nothing was lost by waiting.

## What the flip did, and what the gate does now

Done on 2026-09-04: the repo is public, Pages deploys from `main`, both crons are live, and
`CFBD_API_KEY` exists as a repository secret and is passed to the `golden` job. Run
`scripts/bootstrap_project.sh` if the project board is still empty.

`scripts/preflight_public.sh` is no longer a pre-flip gate — there is nothing left to gate.
It runs before a push, which is the only moment anything it finds is still preventable:

1. It scans **every commit**, not just HEAD. A cookie committed on day one and deleted on day
   two is still in the repo forever, and now that the repo is public, reachable by SHA is
   indistinguishable from published.
2. So a hit is no longer "do not flip". It is **rotate the credential first**, then rewrite
   history with `git-filter-repo`. Removing it is not un-publishing it; assume anything ever
   committed is compromised.
3. It checks the four unattended workflows still have live `schedule:` blocks, structurally
   rather than by grepping for the words — a commented-out trigger is a workflow that simply
   never runs, with nothing anywhere failing to say so.

### The golden job, which had never once run

Worth keeping because it was wrong twice, in two different ways, and looked fine both times.

It gates on `event_name == 'schedule'` while the `schedule:` trigger was commented out, so it
could never fire — and its own comment, and the README, described it as nightly. That was
found 2026-08-24. It had a *second* reason to do nothing, found 2026-08-25: the step ran
`pytest tests/golden`, which the `addopts = "-m 'not golden'"` in `pyproject.toml` deselects
entirely, so it would have gone green having run zero tests. It now runs `pytest -m golden`,
and accepts `workflow_dispatch` so it can be exercised on demand rather than debugged after
the fact. Verified locally: 6 passed, 1 skipped (CFBD, no key).

That skip was the third way to look fine while doing nothing, and it is why the job is now
passed `CFBD_API_KEY` — without a key the CFBD shape check skips in Actions exactly as it
does locally, which is the test's designed behaviour and, from the outside, a green run.

## Verify on a fresh clone first — this is not optional

Added 2026-08-25, after doing it once and finding three bugs that only exist on a machine
without local state:

```bash
git clone <this repo> /tmp/freshclone && cd /tmp/freshclone
uv sync --all-extras
uv run pytest tests/unit tests/contracts -q --cov=hub --cov-fail-under=80
make draft && uv run python -m hub.draft.board --pick 3
make slate
```

What that found, all of it invisible here:

* **`make draft` had never worked on a fresh clone.** `data/processed/` is created by
  `hub.store`, the board does not go through `hub.store`, and every developer machine already
  had the directory. The first command in the README.
* **The ECR-only fallback crashed before THE PICK.** Two report sections read `adp` while
  guarding on a different flag, so a board built without an ESPN key — which is every
  stranger, and this repo too if ESPN is down on draft night — exited 1 on
  `ColumnNotFoundError`.
* **Two tests passed only because this machine has data.** They monkeypatched `store.sql`
  while a `store.tables()` guard returned before reaching it. They would have failed in CI,
  which also checks out fresh.

Note the fresh clone resolves to **Python 3.13**, while this working copy runs 3.11 —
`requires-python = ">=3.11"` and there is no `.python-version`. Both pass, but they are not the
same interpreter, and the public gets the one nobody develops on. Either pin it or keep running
this check.

The good news from the same run: with no keys at all, `make slate` completes and publishes all
five artifacts, CFBD and the odds API degrade with a sentence each, and the board builds 452
players on consensus alone.

## Scope: cut

**CFB fantasy is permanently out.** Not on the roadmap, not "later." Removed so it stops
consuming design attention.

Still on the roadmap, sequenced late: awards and season futures, the staking module, player
props. Note that props are roadmap-only in practice, since free-tier odds data cannot support
them — the credit multiplier makes a props pull cost more than a month's quota.

## Relationship to the World Cup sim

Fully independent. No shared library, no monorepo, no imported tokenizer.

Track B re-derives the football tokenizer from scratch rather than porting `event-tokenizer`.
That is slower and it is the point: understanding the methods ranks second among this project's
objectives, and football's structure differs enough from soccer's that a port would smuggle in
the wrong assumptions anyway. Football has discrete downs, so down/distance/yardline/clock/score
is a near-Markov state and you need far less history than a continuous-possession sport.


## Author identity: raised 2026-08-29, resolved at the flip

`scripts/preflight_public.sh` passes on secrets — no credential patterns in any commit, `.env`
never committed, no raw payloads or data files anywhere in history. `gitleaks` is still the
one unrun check and needs `brew install gitleaks`.

**What the preflight did not check until 2026-08-29, and now does.** Flipping public exposes
the *author* of every commit, not only its contents. At the time it found 141 commits from
the GitHub noreply address and **34 from a personal one**, with `git config user.email`
already set to noreply — so the exposure was historical and not growing.

It was raised as a decision rather than a defect: publishing under a real address is a choice
many people make deliberately, and the only reason to surface it then was that changing it
was far cheaper before the flip than after.

**It was changed.** History now carries 242 commits from
`103243720+jacksonmlukas@users.noreply.github.com` and five from the Actions bot, and none
from a personal address; `git config user.name` is a real name rather than the `Your Name`
placeholder it also flagged. The `git filter-repo --mailmap` run rewrote every SHA after the
first affected commit, and the bounded cost that made it cheap held: the three short SHAs
referenced across `docs/next.md` and
[ADR-0007](adr/0007-measurements-that-steer-the-product-are-committed-code.md) all still
resolve.

The preflight still runs the check, and it is still a warning rather than a failure — the
next commit from a misconfigured machine puts a personal address back in a public history,
and that is much cheaper to catch on the push than after.
