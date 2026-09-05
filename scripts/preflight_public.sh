#!/usr/bin/env bash
# The repo is public. Every commit in it is readable by anyone, and that is already true of
# everything below -- so a hit here is not "do not flip", it is "this is disclosed, rotate it
# now". The scan covers the ENTIRE history rather than HEAD because a secret committed and
# later removed is still sitting in an old commit, permanently, and being reachable by SHA is
# indistinguishable from being published.
#
# It still runs before a push, which is the only moment any of it is preventable.
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
#
# One named variable per credential shape rather than a single blob. The self-check below
# reports the name, so a pattern that goes dead is identified instead of being folded into
# "the patterns match nothing" -- which is what the 2026-09-04 canary said when it failed,
# on evidence about the first alternation only (issue #54).
P_ESPN_S2='ESPN_S2=[A-Za-z0-9%+/=_-]{40,}'
P_ESPN_S2_QUOTED='espn_s2["'"'"']*[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9%+/=]{60,}'
P_SWID='SWID=[{]?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-'
P_CFBD='CFBD_API_KEY=[A-Za-z0-9+/=_-]{20,}'
P_ODDS='ODDS_API_KEY=[A-Za-z0-9+/=_-]{20,}'
PATTERNS="$P_ESPN_S2|$P_ESPN_S2_QUOTED|$P_SWID|$P_CFBD|$P_ODDS"

# label:pattern-variable, one per synthetic sample the self-check plants. Two labels share
# P_SWID: the braces are optional in that pattern, and the brace-less form is the one that
# was actually pasted into a repository secret on 2026-09-04. The coverage check compares
# distinct *variables* against the alternations in PATTERNS, not labels.
CANARY_CASES='espn-cookie:P_ESPN_S2 espn-cookie-quoted:P_ESPN_S2_QUOTED braced-swid:P_SWID brace-less-swid:P_SWID cfbd-key:P_CFBD odds-key:P_ODDS'

# Every sample is synthetic and assembled at runtime from parts. Not style: the scan below
# reads every commit, this script is in every commit, so a credential-shaped literal here
# would make the gate fail on itself permanently. Held by
# tests/unit/test_preflight.py::test_the_gate_and_its_own_tests_are_not_credential_hits,
# which commits this file into a throwaway repo and runs the real scan over it.
canary_sample() {
  B64=$(printf 'a%.0s' $(seq 1 70))           # 70 > the 60 the quoted branch demands
  UUID='1A2B3C4D-5E6F-7081-9203-A4B5C6D7E8F9' # a bare UUID; no SWID= in front of it here
  KEY=$(printf '0f1e2d3c%.0s' $(seq 1 4))     # 32 > the 20 both key branches demand
  case "$1" in
    espn-cookie)        printf 'ESPN_S2=%s\n' "$B64" ;;
    espn-cookie-quoted) printf 'espn_s2: "%s"\n' "$B64" ;;
    braced-swid)        printf 'SWID={%s}\n' "$UUID" ;;
    brace-less-swid)    printf 'SWID=%s\n' "$UUID" ;;
    cfbd-key)           printf 'CFBD_API_KEY=%s\n' "$KEY" ;;
    odds-key)           printf 'ODDS_API_KEY=%s\n' "$KEY" ;;
    *)                  return 1 ;;
  esac
}

# A scan reporting "no credential patterns in any commit" makes two claims: that it looked,
# and that it found nothing. Only the second was ever true. This proves the first on every
# run, against synthetic values that are not credentials and never were.
#
# One sample per alternation, not one sample. Until 2026-09-05 this planted a single ESPN
# cookie and, on failure, printed "the patterns match nothing, so the scan below cannot
# fail" -- a claim about all five alternations from evidence about one. The other four were
# exactly as unproven here as they had been while every pattern was inert BRE.
# GUARD pattern-canary [test_the_scan_proves_it_can_match_before_reporting_clean or test_a_dead_pattern_is_named_and_blocks_the_flip or test_a_pattern_with_no_synthetic_sample_is_a_failure]: refuses to report a clean history through a scanner that has gone even partly blind.
echo "==> Verifying every credential pattern can still match"
canary_fail=0
COVERED=0
COVERED_VARS=''
for CASE in $CANARY_CASES; do
  LABEL=${CASE%%:*}
  VAR=${CASE#*:}
  PAT=${!VAR:-}
  SAMPLE=$(canary_sample "$LABEL") || SAMPLE=''
  if [ -z "$PAT" ] || [ -z "$SAMPLE" ]; then
    echo "  FAIL: $VAR is named by the self-check as $LABEL but has no pattern or no sample" >&2
    canary_fail=1
    continue
  fi
  if ! printf '%s\n' "$SAMPLE" | grep -qE "$PAT"; then
    # Named, and scoped: the other patterns are untouched by this one dying, which is the
    # distinction the old single-cookie message could not draw.
    echo "  FAIL: $VAR no longer matches a synthetic $LABEL, so that one credential shape" >&2
    echo "    is invisible to the scan below. The other patterns are unaffected." >&2
    canary_fail=1
  elif ! printf '%s\n' "$SAMPLE" | grep -qE "$PATTERNS"; then
    echo "  FAIL: $VAR matches a synthetic $LABEL on its own but not once joined into" >&2
    echo "    PATTERNS, so the join dropped it." >&2
    canary_fail=1
  fi
  case " $COVERED_VARS " in
    *" $VAR "*) ;;
    *) COVERED_VARS="$COVERED_VARS $VAR"; COVERED=$((COVERED + 1)) ;;
  esac
done

# The other half of "every alternation": a sixth pattern joined into PATTERNS with no sample
# beside it is a sixth unproven pattern, and the loop above cannot see what it was never
# given. Counting `|` is exact only while no pattern carries one inside a bracket expression;
# if that ever changes this over-counts and fails loudly, which is the safe direction.
ALTS=$(( $(printf '%s' "$PATTERNS" | tr -cd '|' | wc -c) + 1 ))
if [ "$ALTS" -ne "$COVERED" ]; then
  echo "  FAIL: PATTERNS joins $ALTS alternations and the self-check exercises $COVERED of" >&2
  echo "    them ($COVERED_VARS). An alternation with no synthetic sample is one nothing" >&2
  echo "    proves can match. Add a case to CANARY_CASES and canary_sample." >&2
  canary_fail=1
fi

if [ "$canary_fail" -eq 0 ]; then
  echo "  ok: self-check planted a synthetic sample for each of the $ALTS patterns and every"
  echo "    one of them matched"
else
  echo "  This is not a clean repo; it is a broken gate. Fix the pattern named above." >&2
  fail=1
fi
# /GUARD

# GUARD credential-history-scan [test_a_planted_credential_blocks_the_flip or test_a_credential_committed_and_then_removed_still_blocks or test_the_brace_less_swid_is_caught_too]: greps every commit for the five credential shapes; a hit is already-disclosed, not preventable.
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
# /GUARD

# GUARD env-file-in-history [test_a_committed_env_file_blocks_the_flip]: refuses a history containing .env at all, whatever is in it -- that is where every credential in this project lives.
echo "==> Checking .env was never committed"
if git log --all --name-only --pretty=format: 2>/dev/null | sort -u | grep -qx '.env'; then
  echo "  FAIL: .env appears in history" >&2; fail=1
else
  echo "  ok"
fi
# /GUARD

# GUARD tracked-raw-data [test_a_staged_but_uncommitted_data_file_blocks_the_flip]: the index half -- CFBD and nflverse both prohibit redistribution, and this sees a file staged before it is ever committed.
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
# /GUARD

# GUARD data-in-history [test_a_parquet_committed_then_untracked_still_fails]: the history half -- the 2026-08-23 miss was 45 parquet files untracked one commit later and still reachable by SHA.
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
# /GUARD

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

# `schedule:` blocks go stale silently: a commented-out trigger is a workflow that simply
# never runs, with no failure anywhere to notice. watchdog.yml and ci.yml spent the private
# period with theirs off to stay inside the 2,000 free Actions minutes private repos are
# metered at (watchdog.yml records what that cost when it was left on by mistake); c984746
# turned both back on and left every comment and checklist step still saying they were off.
# Reading the file is the only version of that claim that cannot drift.
#
# Structural, not two greps. `schedule:` also matches a `with:` input and `- cron:` also
# matches a line inside a heredoc, so testing for the two independently passes a file whose
# only real schedule block is commented out -- the same shape as the BRE patterns above that
# matched nothing, in the check added to stop a claim going stale.
#
# Comments are stripped first, and only where a `#` starts a line or follows whitespace, so a
# `#` inside a quoted scalar survives. Stripping can only ever hide a real cron and produce a
# spurious WARNING; it cannot invent a trigger, so it fails in the safe direction.
# GUARD schedule-liveness [test_a_commented_out_schedule_is_reported or test_a_cron_outside_the_schedule_block_is_not_a_live_schedule]: reads the on:/schedule:/- cron: nesting rather than grepping for the two independently, which passed a workflow whose only real schedule block was commented out. A WARNING, not a fail=1: a cron that is off may be off on purpose.
has_live_schedule() {
  sed -e 's/^[[:space:]]*#.*//' -e 's/[[:space:]]#.*//' "$1" | awk '
    function ind(s) { match(s, /^ */); return RLENGTH }
    { if ($0 ~ /^[[:space:]]*$/) next
      i = ind($0)
      if (insched && i <= sind) insched = 0
      if (i == 0) { inon = ($0 ~ /^on:[[:space:]]*$/); insched = 0 }
      else if (inon && !insched && $0 ~ /^ +schedule:[[:space:]]*$/) { insched = 1; sind = i }
      else if (insched && $0 ~ /^ *- *cron:/) found = 1 }
    END { exit found ? 0 : 1 }'
}

# All four workflows that are meant to run unattended. watchdog.yml and ci.yml are the two
# that were deliberately switched off; pages.yml and slate.yml never were, and are here
# because a schedule silently commented out is exactly as invisible in them.
echo "==> Checking the unattended workflows still run on a schedule"
for WF in watchdog ci pages slate; do
  F=".github/workflows/$WF.yml"
  if [ ! -f "$F" ]; then
    echo "  skipped: no $F here"
  elif has_live_schedule "$F"; then
    echo "  ok: $F has a live schedule: block"
  else
    # A warning, not a failure: a cron that is off is a thing to know, not a credential in
    # the history, and it may well be off on purpose.
    echo "  WARNING: $F has no live 'schedule:' trigger, so it runs only when something"
    echo "    else triggers it. Nothing will fail to tell you that."
  fi
done
# /GUARD

echo "==> Checking gitleaks if available"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --no-banner --redact || fail=1
else
  echo "  skipped (brew install gitleaks for a real scan)"
fi

echo
if [ $fail -eq 0 ]; then
  # "PASS" is asserted by tests/unit/test_preflight.py::test_a_clean_repo_passes.
  echo "PASS. Nothing in this history is exposed that should not be."
else
  # Not "do not flip" any more. The history is already published, so everything above is
  # already disclosed; removing it from the repo is the second step, after rotating it.
  echo "BLOCKED. Treat everything above as already disclosed: rotate the credential first," >&2
  echo "then rewrite the history with git-filter-repo. Removing it is not un-publishing it." >&2
fi
exit $fail
