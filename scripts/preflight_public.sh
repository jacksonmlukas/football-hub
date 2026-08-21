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
# CFBD prohibits redistribution. data/raw/ is gitignored, but verify nothing slipped in.
TRACKED=$(git ls-files 'data/raw/*' 'data/interim/*' 2>/dev/null)
if [ -n "$TRACKED" ]; then
  echo "  FAIL: raw data tracked in git:" >&2; echo "$TRACKED" | head >&2; fail=1
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
