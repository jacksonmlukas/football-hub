#!/usr/bin/env bash
# Every open ticket a commit said it only *advanced*, that no later commit said it closed.
#
# "Closes #N" on the default branch is acted on by GitHub: the ticket closes itself.
# "Advances #N" is the honest trailer for work that landed a coherent half, and it is inert
# -- which is the point, and also the defect. It leaves an obligation on some *future*
# commit, and that commit has no way to know it inherited one.
#
# Measured on 2026-09-05, four tickets had ever carried "Advances". Two of them (#34, #53)
# were finished and sat open, and both were finished by commits about something else --
# #34's last criterion landed under a message calling it "the catch on the way through",
# #53's inside two commits about review findings. Neither had reason to think about a ticket
# it was silently completing. The other two (#62, #71) were genuinely partial and correctly
# open, so the trailer is working; what was missing is this sweep.
#
# So it prints a queue to read, not a refusal. Everything on it is either still partial --
# fine, leave it -- or finished and open, which is the case no trailer can catch. It stays
# short by construction: a ticket leaves the moment something closes it.
#
# Usage:
#   scripts/partial_work.sh            # open candidates with their titles, needs gh
#   scripts/partial_work.sh --numbers  # every candidate number, no network
set -euo pipefail

numbers_only=false
[[ "${1:-}" == "--numbers" ]] && numbers_only=true

log=$(git log --format='%B')

# A trailer, not a mention: "Advances #34" must start its line, so a sentence saying "this
# advances #34" mid-paragraph is not read as one.
advanced=$(printf '%s\n' "$log" | grep -oE '^Advances #[0-9]+' | grep -oE '[0-9]+' | sort -un || true)
resolved=$(printf '%s\n' "$log" | grep -oE '^(Closes|Fixes|Resolves) #[0-9]+' | grep -oE '[0-9]+' | sort -un || true)

# Advanced and never resolved by a trailer. comm needs both sides sorted, and both are.
candidates=$(comm -23 <(printf '%s\n' "$advanced") <(printf '%s\n' "$resolved") | grep -E '^[0-9]+$' || true)

if $numbers_only; then
  printf '%s\n' "$candidates"
  exit 0
fi

rows=""
skipped=0
while read -r n; do
  [[ -z "$n" ]] && continue
  # An issue that was deleted, or a checkout with no gh auth, must not take the sweep down.
  # A row whose state could not be read is still a row worth reading.
  state=$(gh issue view "$n" --json state,title --jq '"\(.state)\t\(.title)"' 2>/dev/null || printf 'UNKNOWN\t(could not read; is gh authenticated?)')
  case "$state" in
    CLOSED*) skipped=$((skipped + 1)) ;;
    *) rows+=$(printf '  #%-5s %s\n' "$n" "$(printf '%s' "$state" | tr '\t' ' ')")$'\n' ;;
  esac
done <<< "$candidates"

if [[ -z "$rows" ]]; then
  echo "Nothing advanced is still open."
else
  echo "Advanced and still open -- each is partial work, or finished and never closed:"
  echo
  printf '%s' "$rows"
  echo
  echo "For each: check whether every acceptance criterion is now met. If it is, close it"
  echo "with a reason -- see the close conventions in docs/agents/issue-tracker.md."
fi
[[ $skipped -gt 0 ]] && echo "($skipped already closed, not listed.)"
exit 0
