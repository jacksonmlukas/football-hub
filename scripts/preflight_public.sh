#!/usr/bin/env bash
# Run before flipping the repo public. Flipping exposes the ENTIRE history, not just HEAD --
# a secret committed and later removed is still sitting in an old commit, permanently.
set -uo pipefail
fail=0

# espn_s2 is 250+ chars of base64-ish; swid is a UUID, braced in the cookie and routinely
# pasted without the braces -- which is how it was set in a repository secret on 2026-09-04,
# and the form the old pattern could not see.
#
# **ERE, not BRE.** These are passed to `grep -E`, where `\{40,\}` is a literal brace, a 4, a
# 0, a comma and a brace -- not an interval. Every pattern here was written in BRE and was
# therefore inert: measured 2026-09-04 by planting three synthetic credentials in a throwaway
# repo, against which this script printed "ok: no credential patterns in any commit" and
# exited PASS. The self-check below exists so that can never again be silent.
#
# `[[:space:]]` rather than `\s`: this runs on BSD grep locally and GNU grep in CI, and only
# one of them knows `\s`.
#
# The value charsets are named rather than left as `.`, which is not tightening for its own
# sake: `ODDS_API_KEY=.{20,}` matches the sentence "ODDS_API_KEY= in SETUP.md for where to
# get one", and a gate that fires on the documentation telling you to set a key is a gate you
# learn to ignore -- which is how the 2026-08-23 miss became possible. A real cookie or key
# carries no spaces and no prose punctuation.
PATTERNS='ESPN_S2=[A-Za-z0-9%+/=_-]{40,}|espn_s2["'"'"']*[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9%+/=]{60,}|SWID=[{]?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-|CFBD_API_KEY=[A-Za-z0-9+/=_-]{20,}|ODDS_API_KEY=[A-Za-z0-9+/=_-]{20,}'

# A scan reporting "no credential patterns in any commit" makes two claims: that it looked,
# and that it found nothing. Only the second was ever true. This proves the first on every
# run, against a synthetic value that is not a credential and never was.
echo "==> Verifying the patterns match anything at all"
CANARY="ESPN_S2=$(printf 'a%.0s' $(seq 1 50))"
if printf '%s\n' "$CANARY" | grep -qE "$PATTERNS"; then
  echo "  ok: self-check planted a synthetic cookie and the patterns matched it"
else
  echo "  FAIL: the patterns match nothing, so the scan below cannot fail" >&2
  echo "  This is not a clean repo; it is a broken gate. Fix PATTERNS." >&2
  fail=1
fi

echo "==> Scanning full history for ESPN cookies and API keys"
if git rev-list --all >/dev/null 2>&1; then
  # `grep -v PATTERNS=` so this script's own declaration is not a hit. Narrower than
  # excluding the file, which would hide a credential that landed inside it.
  HITS=$(git rev-list --all | while read -r c; do
    git grep -I -n -E "$PATTERNS" "$c" 2>/dev/null | grep -v 'PATTERNS=' | head -5
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

# Flipping public exposes the AUTHOR of every commit, not only its contents, and an address
# in a commit object cannot be edited without rewriting every SHA after it. This is a warning
# rather than a failure: publishing under a real address is a choice many people make on
# purpose, and it is only a problem if it is a surprise.
echo "==> Checking commit author addresses"
NOREPLY=$(git config user.email 2>/dev/null | grep -c 'users\.noreply\.github\.com' || true)
OTHER=$(git log --all --format='%ae%n%ce' 2>/dev/null | sort -u \
        | grep -v 'users\.noreply\.github\.com' | grep -v '^$' || true)
if [ -n "$OTHER" ]; then
  echo "  WARNING: history carries addresses that are not GitHub noreply:"
  echo "$OTHER" | sed 's/^/    /'
  git log --all --format='%ae' 2>/dev/null | sort | uniq -c | sort -rn | sed 's/^/    /'
  echo "    Flipping public publishes these. To change them you must rewrite history"
  echo "    (git filter-repo --mailmap), which changes every SHA -- and docs/ references"
  echo "    $(grep -rho '\`[0-9a-f]\{7\}\`' docs/ 2>/dev/null | sort -u | wc -l | tr -d ' ') of them."
  echo "    Not a failure. A decision, and one that is far cheaper before the flip."
else
  echo "  ok: every commit is authored from a noreply address"
fi
if [ "$NOREPLY" -eq 0 ]; then
  echo "  WARNING: git config user.email is not a noreply address, so new commits will add more"
fi
NAME=$(git config user.name 2>/dev/null || echo "")
case "$NAME" in
  "Your Name"|""|"user"|"root")
    echo "  WARNING: git config user.name is '$NAME' -- a placeholder, on a public repo" ;;
esac

echo "==> Checking gitleaks if available"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --no-banner --redact || fail=1
else
  echo "  skipped (brew install gitleaks for a real scan)"
fi

echo
if [ $fail -eq 0 ]; then
  echo "PASS. Safe to flip public."
  echo
  echo "Then, in this order -- none of these happen on their own:"
  echo "  1. Flip the repo public."
  echo "  2. Check the 'schedule:' block in .github/workflows/watchdog.yml is live."
  echo "     It is, as of 2026-09-04 -- the comment above it saying otherwise is stale."
  echo "  3. Uncomment the 'schedule:' block in .github/workflows/ci.yml, which is what"
  echo "     runs the golden tests against the live APIs."
  echo "  4. Enable Pages (Settings > Pages > main branch)."
else
  echo "BLOCKED. Fix the above before making this repo public." >&2
fi
exit $fail
