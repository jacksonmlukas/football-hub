"""Method rule 18: every check gets a positive control -- plant the condition it exists to
detect and confirm it fires -- before the check is trusted. This file is the rule's own hook.

Rule 18 is prose in `docs/method.md`, and prose does not fire. All four of its incidents shared
one property: nothing forced a look. A rule that can only be obeyed by someone remembering it
is rule 1's incident -- documented, not implemented, and quoted in the write-up. So the rule
has a contract, in the shape `test_each_gate_declares_its_ceiling_arm` and `test_avoided_terms`
already use: a registry that every decision function in `src/` must appear in, naming the
test that plants its condition, and a check that each named test exists.

**What counts as a check.** A top-level function in `src/hub` named `verdict`, `*_verdict`,
`gate`, `run_gate`, `screen` or `*_screen` -- the names this repo gives the functions that
turn a measurement into a disposition -- plus any function or method marked
`@hub.declare.decision`, the attribute a guard whose name is its job carries (`Ledger.record`).
Name and mark are the discovery rule; a decision function with neither escapes this file, and
the fix is to name it what it is or mark it. There is no hand-kept list of exceptions (#387).

**What counts as a control.** A test that constructs the case the decision exists to detect
-- a challenger that clears the bar, a gain inside the noise, a ceiling below the MDE, a
degenerate rule -- and asserts the decision fires. Both directions where the decision has
two. A test that only exercises the default branch is not a control: it is the shape rule 15
names, a fixture that sets the condition under which the check is trivially right.

**The registry is the decision, not an implementation detail.** A new `verdict()` fails here
until it names its control or its ticket; a control test renamed or deleted fails here; an
entry for a function that no longer exists fails here. `OWED` is for a decision that has no
honest control *yet* -- it names the ticket that gives it one, and nothing else.

**This file's own positive control** is the last test: a planted `verdict` in a temporary
tree is discovered and reported unregistered. A contract that cannot be seen to fail has not
been seen to work.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"
TESTS = ROOT / "tests"

_DECISION_NAME = re.compile(r"^(verdict|[a-z_]+_verdict|gate|run_gate|screen|[a-z_]+_screen)$")



def _marked(node: ast.FunctionDef) -> bool:
    """Carries `@decision` (`hub.declare.decision`), spelled bare or dotted."""
    return any((isinstance(d, ast.Name) and d.id == "decision")
               or (isinstance(d, ast.Attribute) and d.attr == "decision")
               for d in node.decorator_list)


# module.function -> (test file, the control tests). Each named test plants the condition
# its decision exists to detect. Where a decision has two directions, both are named.
CONTROLLED: dict[str, tuple[str, tuple[str, ...]]] = {
    # #386: `gate(paired, *, cluster, within, ceiling, actions)` is the frame-in interface --
    # a paired frame in, a `GateRun` out -- and every control named below drives it with a
    # real paired frame rather than a hand-built summary dict, which is the substantive part
    # of the move: a test that hand-builds `{"lo": ..., "mean": ...}` cannot catch a defect in
    # how `summarise` produces that dict, and none of these do any more. `_verdict` (the
    # renamed pre-#386 `gate`) is behind the seam and needs no entry of its own -- its own
    # name does not match this file's discovery regex, deliberately, so it cannot be
    # rediscovered as an unregistered decision.
    "hub.models.experiment.gate": ("tests/unit/test_experiment.py", (
        "test_the_size_check_flags_a_planted_degenerate_rule",
        "test_the_fixed_rule_s_null_size_is_not_degenerate",
        # #381: the first two of these were #335's `test_a_tie_blocks_adopt_...` and
        # `..._remove_...` under (A); (C) renamed and inverted them, and the other two are the
        # cases (C) is new in (no resolved season; the interval half's veto).
        "test_an_abstention_does_not_block_adopt_when_every_resolved_season_won",
        "test_an_abstention_does_not_block_remove_when_every_resolved_season_lost",
        "test_a_gate_with_no_resolved_season_can_neither_adopt_nor_remove",
        "test_the_interval_half_still_vetoes_a_resolved_unanimous_frame_whose_pooled_interval_crosses_zero",
        "test_no_ceiling_measured_is_not_runnable_in_both_directions",
        "test_the_gate_sweep_verdicts_are_unmoved",
        "test_not_runnable_preempts_every_branch_but_void",
        "test_void_still_preempts_not_runnable",
    )),
    # #385: `review_width`'s guard moved to `Ledger.record`. Required controls per the
    # ticket: an unreadable ledger is not replaced, different digests do not compare, the
    # same key does, `seasons` round-trips, and -- new here -- #384's own case, two entries
    # at one digest pair with different recipes.
    "hub.ledger.Ledger.record": ("tests/unit/test_ledger.py", (
        "test_an_unreadable_history_costs_a_line_and_not_the_run_and_is_not_replaced",
        "test_two_runs_at_different_digests_are_not_compared",
        "test_the_same_digest_still_compares_and_a_data_digest_alone_is_enough_to_block",
        "test_a_ledger_entry_round_trips_its_seasons_field",
        "test_same_digests_different_recipes_do_not_compare",
    )),
    "hub.models.experiment.run_gate": ("tests/unit/test_gate_run.py", (
        "test_a_void_condition_preempts_every_branch_and_is_the_caller_s_sentence",
        "test_a_ceiling_below_the_effect_says_so_loudly_in_the_gate_s_own_places",
    )),
    # #432: the capture's refusal (rule 18's own control for it) and the read that excludes a
    # late capture the write never saw -- both planted on either side of the deadline.
    "hub.fetch.consensus.write_capture": ("tests/unit/test_fetch_consensus.py", (
        "test_a_capture_written_after_kickoff_is_refused_and_nothing_lands",
        "test_a_capture_is_append_only_atomic_and_idempotent",
    )),
    # #437: the reading's own refusals -- before the horizon (dates, then the data), a different
    # arm, too few weeks -- each planted and flipped.
    "hub.season.weekly_forward.read_forward": ("tests/unit/test_weekly_forward.py", (
        "test_a_run_before_the_horizon_reads_no_outcome",
        "test_dates_past_the_horizon_with_week_14_absent_from_the_data_is_not_yet",
        "test_a_different_arm_refuses_the_reading_and_loads_nothing",
        # #456: the pin is the arm's import closure; an edit in `hub.models.panel` refuses, an
        # exempt module's edit does not.
        "test_an_edit_to_the_models_panel_refuses_the_reading_and_an_unchanged_closure_reads",
        "test_an_edit_to_an_exempt_module_does_not_refuse_and_the_exemption_holds",
        "test_fewer_than_the_pre_registered_weeks_is_not_runnable_and_loads_nothing",
        "test_a_planted_effect_is_detected_in_both_directions",
    )),
    "hub.season.weekly_forward.admit": ("tests/unit/test_weekly_forward.py", (
        "test_a_capture_after_kickoff_is_excluded_at_the_read_and_named",
        "test_a_stale_page_is_refused_and_the_latest_valid_capture_is_the_one_used",
        "test_a_capture_first_committed_after_the_deadline_is_not_admitted",
    )),
    "hub.models.margin.verdict": ("tests/unit/test_margin.py", (
        "test_a_better_challenger_is_adopted",
        "test_a_challenger_better_on_average_but_not_every_season_is_not_adopted",
        "test_a_binding_ceiling_makes_the_same_challenger_not_runnable",
    )),
    "hub.models.margin.shape_verdict": ("tests/unit/test_margin.py", (
        "test_a_lump_symmetric_about_the_spread_is_adopted",
        "test_a_lump_on_the_favourite_s_side_keeps_the_gaussian",
        "test_a_binding_ceiling_makes_the_shape_gate_not_runnable",
    )),
    "hub.models.starter_change.verdict": ("tests/unit/test_starter_change.py", (
        "test_verdict_adopts_when_the_lower_bound_clears_delta",
        "test_verdict_removes_when_the_interval_excludes_zero_negatively",
        "test_verdict_is_not_runnable_when_the_mde_exceeds_delta_and_names_the_events_needed",
    )),
    # #343: these three read the shared Gate now (`Harness.decide`/`Harness.run`), so each
    # control plants a case through that seam -- an adoptable candidate, a candidate that does
    # not win a season, a design whose ceiling cannot be resolved (stage 2), and a frame that
    # never measured a ceiling (S6) -- rather than a hand-built every-season-and-2-se number.
    "hub.models.component_error.verdict": ("tests/unit/test_component_error.py", (
        "test_a_real_mis_scaling_is_taken_by_the_gate_in_every_held_out_season",
        "test_a_calibration_that_worsens_the_linear_loss_is_not_taken",
        "test_one_season_against_it_is_enough_to_withhold_adoption",
        "test_a_ceiling_the_design_cannot_resolve_is_not_runnable_not_a_null",
    )),
    "hub.models.spread.verdict": ("tests/unit/test_spread.py", (
        "test_a_candidate_that_wins_every_season_and_clears_the_interval_is_adopted",
        "test_winning_on_average_but_losing_a_season_is_not_enough",
        "test_a_gain_too_small_to_distinguish_from_noise_is_not_adopted",
        "test_the_pooled_interval_is_the_season_clustered_t_interval",
        "test_a_gate_with_no_headroom_declared_is_not_runnable_even_when_the_candidate_is_clean",
    )),
    "hub.models.injury.type_verdict": ("tests/unit/test_injury.py", (
        "test_a_type_effect_that_wins_everywhere_and_clears_the_interval_is_adopted",
        "test_a_gain_the_seasons_cannot_agree_on_is_not_adopted",
        "test_no_ceiling_measured_is_not_runnable_even_when_the_effect_is_clean",
        "test_a_ceiling_below_the_design_s_resolution_is_not_runnable",
    )),
    # #360: `injury.verdict` is the Gate over retention and `out_zero`, no longer an argmin, so
    # both directions are planted: a clean win adopts, and each way of not clearing it does not.
    "hub.models.injury.verdict": ("tests/unit/test_injury.py", (
        "test_a_table_that_beats_both_simple_rules_is_adopted",
        "test_winning_on_average_but_losing_a_season_is_not_adopted",
        "test_a_gain_too_small_to_distinguish_from_noise_is_not_adopted",
        "test_no_ceiling_measured_is_not_runnable_for_retention",
        "test_a_ceiling_below_the_design_s_resolution_is_not_runnable_for_retention",
    )),
    "hub.models.coverage.verdict": ("tests/unit/test_coverage.py", (
        "test_the_verdict_reads_the_pre_registered_band",
        "test_the_gate_refuses_when_the_interval_leaves_the_band",
    )),
    # #310: the audit-time verdict and its per-position reading. The plants are a marginal
    # outside 80 +/- 2 (fails), a position below N = 8,377 (no verdict) and one at it (rules).
    "hub.models.coverage.audit_verdict": ("tests/unit/test_coverage_audit.py", (
        "test_a_marginal_outside_eighty_plus_or_minus_two_fails_the_audit",
        "test_a_position_below_the_minimum_gets_no_verdict_only_its_deviation_and_sigma",
    )),
    "hub.models.coverage.group_verdict": ("tests/unit/test_coverage_audit.py", (
        "test_a_position_below_the_minimum_gets_no_verdict_only_its_deviation_and_sigma",
        "test_a_position_at_the_minimum_does_return_a_verdict_at_root_two_binomial_se",
    )),
    # #312: the verdict reads the two-sided p at the run's degrees of freedom. Each control
    # plants a t between the old flat 2.0 and the t quantile -- the only band where the two
    # rules disagree -- and the headline pre-stated null from the ticket's own figures.
    "hub.models.weekly_screen.verdict": ("tests/unit/test_weekly_screen.py", (
        "test_a_t_past_two_and_short_of_the_quantile_no_longer_clears",
        "test_the_bar_is_the_t_quantile_at_the_runs_own_degrees_of_freedom",
        "test_the_headline_pre_stated_null_reads_as_null_under_the_p_bar",
        "test_one_season_has_no_p_and_so_clears_nothing",
    )),
    "hub.models.weekly_screen.screen": ("tests/unit/test_weekly_screen.py", (
        "test_screen_reports_the_status_the_p_bar_gives",
    )),
    "hub.draft.adherence.verdict": ("tests/unit/test_adherence.py", (
        "test_twelve_of_sixteen_meets_it",
        "test_eleven_of_sixteen_misses_it",
    )),
}

# module.function -> the ticket that gives it a control. Nothing else belongs here.
OWED: dict[str, str] = {}


def _decisions(src: Path) -> dict[str, Path]:
    """`module.function` -> file, for every top-level function whose name says it decides,
    and every function or method marked `@decision` -- `Ledger.record` (#385) is a guard
    behind a class, found by its mark rather than a list naming it (#387)."""
    found: dict[str, Path] = {}
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = ".".join(("hub", *path.relative_to(src).with_suffix("").parts))
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                qualified = f"{module}.{node.name}"
                if _DECISION_NAME.match(node.name) or _marked(node):
                    found[qualified] = path
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if not isinstance(sub, ast.FunctionDef):
                        continue
                    qualified = f"{module}.{node.name}.{sub.name}"
                    if _marked(sub):
                        found[qualified] = path
    return found


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name.startswith("test_")}


def test_every_decision_function_is_registered():
    found = _decisions(SRC)
    registered = set(CONTROLLED) | set(OWED)
    missing = sorted(set(found) - registered)
    assert not missing, (
        f"decision functions with no positive control registered: {missing}. Rule 18: name "
        f"the test that plants the condition each exists to detect (CONTROLLED), or the "
        f"ticket that will (OWED). A verdict nobody has seen fire has not been seen to work.")


def test_no_registry_entry_names_a_function_that_is_gone():
    found = _decisions(SRC)
    stale = sorted((set(CONTROLLED) | set(OWED)) - set(found))
    assert not stale, f"registry entries for decision functions that no longer exist: {stale}"


def test_no_decision_is_both_controlled_and_owed():
    both = sorted(set(CONTROLLED) & set(OWED))
    assert not both, f"in both CONTROLLED and OWED: {both}"


def test_every_named_control_test_exists():
    gone: list[str] = []
    for decision, (file, tests) in CONTROLLED.items():
        names = _test_names(ROOT / file)
        gone += [f"{decision}: {file}::{t}" for t in tests if t not in names]
    assert not gone, (
        f"controls named here that no longer exist: {gone}. A renamed or deleted control "
        f"leaves its decision uncontrolled; update the registry with the new name.")


def test_every_owed_entry_names_its_ticket():
    bad = [d for d, why in OWED.items() if not re.search(r"#\d+", why)]
    assert not bad, f"OWED entries that name no ticket: {bad}"


def test_a_planted_decision_function_is_discovered_and_reported(tmp_path):
    """Rule 18 applied to this file. A tree with a `verdict` nobody registered must be
    found by `_decisions` and, held against the registry, must come back as missing --
    otherwise the first test above is a check that cannot fail."""
    pkg = tmp_path / "hub" / "models"
    pkg.mkdir(parents=True)
    (pkg / "planted.py").write_text(
        "from hub.declare import decision\n\n\n"
        "def verdict(x):\n    return 'ADOPT'\n\n\ndef helper():\n    pass\n\n\n"
        "class Keeper:\n    @decision\n    def flag(self):\n        return True\n\n"
        "    def unmarked(self):\n        return True\n", encoding="utf-8")
    found = _decisions(tmp_path / "hub")
    planted = {"hub.models.planted.verdict", "hub.models.planted.Keeper.flag"}
    assert found == dict.fromkeys(planted, pkg / "planted.py"), found
    missing = set(found) - (set(CONTROLLED) | set(OWED))
    assert missing == planted, "a marked method is found by its mark, with no list naming it"
