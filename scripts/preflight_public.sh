#!/usr/bin/env bash
# Run before flipping the repo public. Flipping exposes the ENTIRE history, not just HEAD --
# a secret committed and later removed is still sitting in an old commit, permanently.
set -uo pipefail
fail=0

echo "==> Scanning full history for ESPN cookies and API keys"
# espn_s2 is 250+ chars of base64-ish; swid is a braced UUID.
PATTERNS='ESPN_S2=.\{40,\}|espn_s2["'"'"']*\s*[:=]\s*["'"'"'][A-Za-z0-9%+/=]\{60,\}|SWID=\{\?[0-9A-F]\{8\}-|CFBD_API_KEY=.\{20,\}|ODDS_API_KEY=.\{20,\}'
if git rev-list --all >/dev/null 2>&1; then
  HITS=$(git rev-list --all | while read -r c; do
    git grep -I -n -E "$PATTERNS" "$c" 2>/dev/null | head -5
  done | sort -u)
  if [ -n "$HITS" ]; then
    echo "  FAIL: credential-shaped strings found in history:" >&2
    echo "$HITS" | head -20 >&2
    echo "  Do not flip public. Rewrite history (git-filter-repo) and rotate the credentials." >&2
    fail=1
  else
    echo "  ok: no credential patterns in any commit"
  fi
fi

echo "==> Checking .env was never committed"
if git log --all --name-only --pretty=format: 2>/dev/null | sort -u | grep -qx '.env'; then
  echo "  FAIL: .env appears in history" >&2; fail=1
else
  echo "  ok"
fi

echo "==> Checking no raw third-party payloads are tracked"
# CFBD and nflverse both prohibit redistribution. These are gitignored, but verify
# nothing slipped in. processed/ is included: it was NOT checked here until 2026-08-23,
# and by then 45 parquet files (3.5MB of nflverse play-by-play) were tracked while this
# gate reported PASS. Derived data is still redistributed data.
TRACKED=$(git ls-files 'data/raw/*' 'data/interim/*' 'data/processed/*' 2>/dev/null)
if [ -n "$TRACKED" ]; then
  echo "  FAIL: raw data tracked in git:" >&2; echo "$TRACKED" | head >&2; fail=1
else
  echo "  ok"
fi

echo "==> Checking no data file survives anywhere in history"
# The index check above only sees the current tree. Flipping public exposes every commit,
# and a file removed in commit N is still sitting in commit N-1 -- which is exactly what
# happened on 2026-08-23: 45 parquet files were committed, untracked the next commit, and
# this script reported PASS with 3.5MB of play-by-play still reachable by SHA.
HIST=$(git rev-list --objects --all 2>/dev/null \
       | grep -E ' data/(raw|interim|processed)/|\.parquet$' | head -20)
if [ -n "$HIST" ]; then
  echo "  FAIL: data files reachable in history:" >&2
  echo "$HIST" | head -10 >&2
  echo "  Rewrite with git-filter-repo before flipping public. Untracking is not enough." >&2
  fail=1
else
  echo "  ok"
fi

echo "==> Checking gitleaks if available"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --no-banner --redact || fail=1
else
  echo "  skipped (brew install gitleaks for a real scan)"
fi

echo
if [ $fail -eq 0 ]; then
  echo "PASS. Safe to flip public."
  echo "Then: enable Pages (Settings > Pages > main branch), and the watchdog cron starts working."
else
  echo "BLOCKED. Fix the above before making this repo public." >&2
fi
exit $fail
