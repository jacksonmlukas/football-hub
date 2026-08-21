#!/usr/bin/env bash
# Quality gate on every edit. Ordered cheapest-first so Claude sees the fastest signal first.
# Pyrefly runs in well under a second on a repo this size, which is the whole reason it
# belongs in a per-edit hook and mypy did not.
set -uo pipefail

types=$(uv run pyrefly check 2>&1)
if [ $? -ne 0 ]; then
  echo "Type check failed. Fix before continuing:" >&2
  echo "$types" | tail -20 >&2
  exit 2
fi

out=$(uv run pytest tests/unit tests/contracts -q --no-header -x 2>&1)
if [ $? -ne 0 ]; then
  echo "TDD gate failed. Fix before continuing:" >&2
  echo "$out" | tail -30 >&2
  exit 2
fi
echo "$out" | tail -1
