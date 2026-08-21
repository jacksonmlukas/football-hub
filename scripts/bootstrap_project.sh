#!/usr/bin/env bash
# Seeds GitHub Projects with the track structure and gated milestones.
# Requires: gh CLI, authenticated, with `project` scope:
#   gh auth refresh -s project,read:project
set -euo pipefail

REPO="${1:?usage: bootstrap_project.sh <owner/repo>}"
OWNER="${REPO%%/*}"

echo "==> Creating project board"
PROJECT_URL=$(gh project create --owner "$OWNER" --title "Football Hub 2026" --format json | jq -r .url)
echo "    $PROJECT_URL"

echo "==> Creating milestones"
create_ms () {
  gh api "repos/$REPO/milestones" -f title="$1" -f due_on="$2" -f description="$3" >/dev/null \
    && echo "    $1"
}
create_ms "M0 Draft"        "2026-09-03T00:00:00Z" "Draft board live. Hard deadline, does not move."
create_ms "M1 Week 1"       "2026-09-09T00:00:00Z" "Survivor IP solver + lineup optimizer + conformal coverage gate."
create_ms "M2 Week 4 gate"  "2026-10-01T00:00:00Z" "Track A go/no-go: does Bayesian rating beat the closing line?"
create_ms "M3 Week 8 gate"  "2026-10-29T00:00:00Z" "Track B go/no-go: does the sequence model beat plain xFP?"

echo "==> Creating issues"
mk () { gh issue create --repo "$REPO" --title "$1" --body "$2" --milestone "$3" --label "$4" >/dev/null && echo "    $1"; }

gh label create track-a --repo "$REPO" --color 0E8A16 --description "Bayesian ratings" 2>/dev/null || true
gh label create track-b --repo "$REPO" --color 5319E7 --description "Sequence model" 2>/dev/null || true
gh label create track-c --repo "$REPO" --color 1D76DB --description "Conformal / calibration" 2>/dev/null || true
gh label create track-d --repo "$REPO" --color D93F0B --description "Pool optimization" 2>/dev/null || true
gh label create infra   --repo "$REPO" --color BFD4F2 --description "Pipeline / tooling" 2>/dev/null || true

mk "2026 projection model (PPR)" \
   "Board currently ranks on 2025 xFP. Needs a forward projection: target share + route participation backbone, regressed TD rate. Rookie prior handled separately (xfp is null by construction)." \
   "M0 Draft" "infra"
mk "Live draft poller + run detection" \
   "Poll ESPN draft endpoint every 10s. Remove drafted players, recompute VOR and replacement level live, flag positional runs (3+ picks at one position in last 5)." \
   "M0 Draft" "infra"
mk "Survivor IP solver" \
   "18-week assignment under use-each-team-once. Maximize P(survive all remaining), not greedy weekly max. pulp or OR-Tools." \
   "M1 Week 1" "track-d"
mk "Pool-aware objective" \
   "Switch objective from P(survive) to P(finish first). Needs scraped pick percentages + pool config: entries, payout structure, rebuys." \
   "M1 Week 1" "track-d"
mk "Split conformal on fantasy projections" \
   "MAPIE, rolling calibration window. Gate: empirical coverage within 3pts of nominal by Week 2." \
   "M1 Week 1" "track-c"
mk "State-space team ratings (NFL)" \
   "NumPyro random walk on latent strength. Student-t likelihood on margin. Off/def split as a follow-up." \
   "M2 Week 4 gate" "track-a"
mk "Conference hyperpriors for CFB" \
   "136 FBS teams, unbalanced schedules. Group-level partial pooling on top of team-level. Preseason prior from returning production." \
   "M2 Week 4 gate" "track-a"
mk "Backtest harness with CLV accounting" \
   "Closing lines only. Temporal splits. Log-loss + Brier + bootstrap interval on the delta vs market-only baseline." \
   "M2 Week 4 gate" "track-c"
mk "Football event tokenizer" \
   "Re-derive from scratch (WC sim stays independent -- no shared lib). Vocabulary: play_type x outcome x field_zone, plus explicit down/distance/score/time state tokens. Round-trip invariants first." \
   "M3 Week 8 gate" "track-b"
mk "Adaptive conformal + Mondrian by position" \
   "Exchangeability breaks mid-season (injuries, drift). Adaptive CI updates target coverage online. Mondrian partitions by position group." \
   "M3 Week 8 gate" "track-c"

echo
echo "Done. Add issues to the board:  gh project item-add <number> --owner $OWNER --url <issue-url>"
echo "Board: $PROJECT_URL"
