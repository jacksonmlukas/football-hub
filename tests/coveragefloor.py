"""How many statements each module is allowed to leave untested, and no more.

CI gates on `--cov-fail-under=88`, which is a floor on the **total**. The repo sits near 94%,
so there are roughly six points of slack and any single module can spend all of them alone.

That is not hypothetical. PR #104 took `src/hub/models/weekly.py` from 98% to 22% -- 143
statements of the projection model stopped being exercised, because a feature branch wrote its
tests into a file that belonged to a different module -- and the total only fell to 89.62%.
The gate passed. The loss was invisible in CI and turned up because a human read the diff.

A total answers "is the repo broadly tested", which is not the question a gate is for. The
question is "did this change stop testing something that was tested", and a total cannot see
it: a module going to zero looks the same as a few uncovered lines spread thin.

**Missing statements rather than percentage**, which the ticket left to be decided here. A
percentage moves when a module grows, so well-tested new code and newly untested code both
shift it and the number has to be re-read to tell which happened. A count of unexercised
statements only rises when something stopped being exercised, which is the event being
gated. It also survives a refactor that moves code between modules better: the count follows
the code, where two percentages both move and neither says why.

The same counted-ratchet shape as `OUTSTANDING` in `tests/contracts/test_avoided_terms.py`:
committed numbers that may fall but not rise, updated by `scripts/coverage_ratchet.py --update`
so an improvement is recorded rather than silently banked. A module absent from this list is
a failure, not an exemption -- `tests/contracts/test_coverage_floor.py` is what says so.

**CI is the authority for these numbers.** They were first generated from a full local run,
and coverage can differ between machines wherever a branch is reached on one and not the
other. If CI reports a module as worse than its floor on a run that changed nothing about
that module, the floor is what is wrong -- regenerate it from CI's report rather than adding
slack, because slack is the property this file exists to remove.
"""
from __future__ import annotations

# module path (as coverage reports it) -> statements it may leave untested
FLOOR: dict[str, int] = {
    "src/hub/__init__.py": 0,
    "src/hub/cli.py": 0,
    "src/hub/config.py": 0,
    "src/hub/contracts.py": 1,
    "src/hub/draft/__init__.py": 0,
    "src/hub/draft/adherence.py": 1,
    "src/hub/draft/adp_history.py": 0,
    "src/hub/draft/availability.py": 17,
    "src/hub/draft/backtest.py": 79,
    "src/hub/draft/board.py": 15,
    "src/hub/draft/calibrate.py": 2,
    "src/hub/draft/cohort.py": 0,
    "src/hub/draft/durability.py": 5,
    "src/hub/draft/evaluate.py": 14,
    "src/hub/draft/fit_corrections.py": 13,
    "src/hub/draft/live.py": 10,
    "src/hub/draft/optimize.py": 3,
    "src/hub/draft/picks.py": 1,
    "src/hub/draft/playoff_sos.py": 0,
    "src/hub/draft/prior_signal.py": 0,
    "src/hub/draft/projection.py": 0,
    "src/hub/draft/regression.py": 4,
    "src/hub/draft/report.py": 1,
    "src/hub/draft/season.py": 0,
    "src/hub/draft/state.py": 6,
    "src/hub/draft/tune.py": 13,
    "src/hub/exhibits/__init__.py": 0,
    "src/hub/exhibits/championship_equity.py": 0,
    "src/hub/exhibits/leverage.py": 1,
    "src/hub/fetch/__init__.py": 0,
    "src/hub/fetch/bigten.py": 7,
    "src/hub/fetch/cfbd.py": 2,
    "src/hub/fetch/espn.py": 0,
    "src/hub/fetch/nflverse.py": 14,
    "src/hub/fetch/odds.py": 10,
    "src/hub/fetch/pool.py": 2,
    "src/hub/inspect.py": 6,
    "src/hub/jsonio.py": 0,
    "src/hub/league.py": 0,
    "src/hub/models/__init__.py": 0,
    "src/hub/models/base.py": 2,
    "src/hub/models/component_error.py": 8,
    "src/hub/models/components.py": 4,
    "src/hub/models/conformal.py": 3,
    "src/hub/models/correlate.py": 4,
    "src/hub/models/coverage.py": 3,
    "src/hub/models/eval.py": 3,
    "src/hub/models/experiment.py": 0,
    "src/hub/models/injury.py": 5,
    "src/hub/models/margin.py": 1,
    "src/hub/models/market.py": 1,
    "src/hub/models/panel.py": 1,
    "src/hub/models/predict.py": 2,
    "src/hub/models/props.py": 1,
    "src/hub/models/ratings.py": 4,
    "src/hub/models/scoring_rules.py": 1,
    "src/hub/models/spread.py": 9,
    "src/hub/models/volume.py": 1,
    "src/hub/models/weekly.py": 3,
    "src/hub/models/weekly_screen.py": 0,
    "src/hub/names.py": 0,
    "src/hub/paths.py": 0,
    "src/hub/publish.py": 32,
    "src/hub/schedule.py": 0,
    "src/hub/season/__init__.py": 0,
    "src/hub/season/journal.py": 0,
    "src/hub/season/lineup.py": 13,
    "src/hub/season/lineup_gate.py": 32,
    "src/hub/season/pool.py": 1,
    "src/hub/season/roster.py": 0,
    "src/hub/season/survivor.py": 10,
    "src/hub/season/weekly_gate.py": 1,
    "src/hub/season/weekly_gate_data.py": 0,
    "src/hub/store.py": 5,
}
