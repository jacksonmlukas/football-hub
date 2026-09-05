.PHONY: setup draft slate check serve preflight

setup:           ## Create the venv and install, dev extras included
	uv sync --all-extras

draft:          ## Build the draft board (safe to re-run; cached)
	uv run python -m hub.draft.board --league-size 12 --scoring ppr

slate:          ## Weekly pregame refresh -> site/data/*.json
# nflverse and ratings are the NFL spine: if either fails the slate is wrong, so they are
# fatal. CFB and odds are optional sources -- both need a key this repo does not require,
# and neither feeds the fantasy path. They report their own status and the slate continues,
# because a weekly refresh that halts on an unconfigured extra is a system that needs an
# operator, and those die in October.
# WEEK is the NFL week and is optional. Unset, it expands to nothing and `hub.publish` picks
# its own default -- `--week $(WEEK)` with WEEK unset would pass a bare `--week` and argparse
# would exit 2, which is how `make slate` came to fail on a clean checkout. The same `$(if
# $(strip ...))` guard is why CFB_WEEK below is written the way it is.
	uv run python -m hub.fetch.nflverse --refresh
# CFB_WEEK, not WEEK, and it is normally unset. WEEK is the *NFL* week -- a different
# calendar with a different week-1 date and three more weeks in it -- so passing it here
# would have fetched a confidently wrong college week. Left unset, `hub.fetch.cfbd` counts
# the week from $CFB_WEEK_ONE, the date of the college season's first game, which is a fact
# stated once in `.env` (or as an Actions variable) rather than a number somebody has to
# bump every Wednesday. `make slate CFB_WEEK=3` is the one-off override, for a backfill.
#
# The leading `-` stays, and so does the reason for it: an optional source that is
# unconfigured must not take a Sunday down. What changed with issue #56 is that the run now
# leaves a record in `site/data/cfbd.json` saying whether it fetched, so a swallowed exit
# code is no longer the only thing that knew.
	-uv run python -m hub.fetch.cfbd $(if $(strip $(CFB_WEEK)),--week $(CFB_WEEK))
	-uv run python -m hub.fetch.odds --snapshot
	uv run python -m hub.models.ratings --fit
# The roster changes every week the waiver wire does, so it is refreshed here rather than
# written once after the draft. Optional like the other ESPN-cookie sources: if the league is
# unreachable the CLI serves last-good and the panel says so, and the rest of the slate is not
# about your team.
	-uv run python -m hub.season.roster --write
	uv run python -m hub.publish --all $(if $(strip $(WEEK)),--week $(WEEK))

check:          ## Quota + cache health, prints a summary only
	uv run python -m hub.fetch.cfbd --quota

serve:          ## Local dashboard while the repo is private (Pages needs a public repo)
	cd site && uv run python -m http.server 8000

preflight:      ## Secret scan of full history before flipping public
	./scripts/preflight_public.sh
