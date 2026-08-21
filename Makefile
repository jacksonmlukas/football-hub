.PHONY: setup draft slate live check serve preflight

setup:
	uv venv && uv pip install -e .

draft:          ## Build the draft board (safe to re-run; cached)
	uv run python -m hub.draft.board --league-size 12 --scoring ppr

slate:          ## Weekly pregame refresh -> site/data/*.json
	uv run python -m hub.fetch.nflverse --refresh
	uv run python -m hub.fetch.cfbd --week $(WEEK)
	uv run python -m hub.models.ratings --fit

live:           ## Sunday poller (local, not Actions)
	uv run python -m hub.fetch.espn --poll --interval 45

check:          ## Quota + cache health, prints a summary only
	uv run python -m hub.fetch.cfbd --quota

serve:          ## Local dashboard while the repo is private (Pages needs a public repo)
	cd site && uv run python -m http.server 8000

preflight:      ## Secret scan of full history before flipping public
	./scripts/preflight_public.sh
