# Private now, public Sep 4

The draft is Sep 3 and your leaguemates are a mixed room. Publishing the board before then
hands them the exact artifact built to beat them. So the repo stays private until the draft
is over, then flips.

## What private costs you

**GitHub Pages does not work.** Pages is available in public repositories on GitHub Free;
private-repo Pages requires Pro. And even on Pro the site is still publicly reachable — truly
private Pages needs an organization on Enterprise Cloud. So there is no configuration where a
private repo gets you a hidden dashboard.

Run it locally instead until Sep 4:

```
make serve      # http://localhost:8000
```

**Actions minutes are metered.** Free gives 2,000 minutes/month on private repos; public repos
are unmetered. Keep CI on push only and leave the watchdog cron disabled until the flip. It has
nothing to watch before Week 1 anyway.

## What private does not cost you

**Pre-registration still works.** The track-record guarantee is that predictions were committed
before kickoff, and git timestamps do not care about repo visibility. When you flip, the full
history comes with it, and every pre-draft commit is dated and verifiable. Nothing is lost by
waiting.

## Flip checklist

1. Draft ends Sep 3.
2. `./scripts/preflight_public.sh` — must pass. It scans **every commit**, not just HEAD,
   because flipping exposes the entire history. A cookie you committed on day one and deleted
   on day two is still in the repo forever.
3. If it fails on a credential: rewrite history with `git-filter-repo`, then rotate the
   credential anyway. Assume anything ever committed is compromised.
4. Flip to public.
5. Settings > Pages > deploy from `main`.
6. Re-enable the watchdog cron in `.github/workflows/watchdog.yml`, **and** the
   `schedule:` trigger in `.github/workflows/ci.yml` — the `golden` job gates on
   `event_name == 'schedule'` and has never once run, because the trigger was absent.

   It had a *second* reason to do nothing, found and fixed 2026-08-25: the step ran
   `pytest tests/golden`, which the `addopts = "-m 'not golden'"` in `pyproject.toml`
   deselects entirely, so it would have gone green having run zero tests. It now runs
   `pytest -m golden`, and the job also accepts `workflow_dispatch` so it can be exercised
   before the flip rather than debugged after it. Verified locally: 6 passed, 1 skipped
   (CFBD, no key). **If you want the CFBD shape check to actually run in Actions, add
   `CFBD_API_KEY` as a repository secret and pass it to that job** — otherwise it skips
   there too, which is the test's designed behaviour, not a failure.
7. Run `scripts/bootstrap_project.sh` to seed the project board.

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
