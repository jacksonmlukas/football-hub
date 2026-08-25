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
# WEEK is optional. Unset, it expands to nothing and the CLIs pick their own default --
# `--week $(WEEK)` with WEEK unset passes a bare `--week` and argparse exits 2, which is how
# `make slate` came to fail on a clean checkout.
	uv run python -m hub.fetch.nflverse --refresh
	-uv run python -m hub.fetch.cfbd $(if $(strip $(WEEK)),--week $(WEEK))
	-uv run python -m hub.fetch.odds --snapshot
	uv run python -m hub.models.ratings --fit
	uv run python -m hub.publish --all $(if $(strip $(WEEK)),--week $(WEEK))

check:          ## Quota + cache health, prints a summary only
	uv run python -m hub.fetch.cfbd --quota

serve:          ## Local dashboard while the repo is private (Pages needs a public repo)
	cd site && uv run python -m http.server 8000

preflight:      ## Secret scan of full history before flipping public
	./scripts/preflight_public.sh
