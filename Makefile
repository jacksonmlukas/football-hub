.PHONY: setup draft slate live check serve preflight

setup:
	uv venv && uv pip install -e .

draft:          ## Build the draft board (safe to re-run; cached)
	python -m hub.draft.board --league-size 12 --scoring ppr

slate:          ## Weekly pregame refresh -> site/data/*.json
	python -m hub.fetch.nflverse --refresh
	python -m hub.fetch.cfbd --week $(WEEK)
	python -m hub.models.ratings --fit

live:           ## Sunday poller (local, not Actions)
	python -m hub.fetch.espn --poll --interval 45

check:          ## Quota + cache health, prints a summary only
	python -m hub.fetch.cfbd --quota

serve:          ## Local dashboard while the repo is private (Pages needs a public repo)
	cd site && python -m http.server 8000

preflight:      ## Secret scan of full history before flipping public
	./scripts/preflight_public.sh
