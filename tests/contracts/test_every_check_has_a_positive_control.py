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
turn a measurement into a disposition. That naming is the discovery rule; a decision function
named otherwise escapes this file, and the fix is to name it what it is -- or, for a guard
whose name is its job (`review_width`), to list it in `EXPLICIT_GUARDS`, which is a deliberate
act with a control beside it.

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

# Guards that decide something without carrying a decision name. Named here so the naming
# rule stays the discovery rule for everything else, and so a guard added to this list is a
# deliberate act with a control beside it. `review_width` flags a narrowing interval; on
# 2026-09-21 it compared a --holdout run against a non-holdout one and flagged nothing real.
EXPLICIT_GUARDS: tuple[str, ...] = ("hub.models.experiment.review_width",)

# module.function -> (test file, the control tests). Each named test plants the condition
# its decision exists to detect. Where a decision has two directions, both are named.
CONTROLLED: dict[str, tuple[str, tuple[str, ...]]] = {
    "hub.models.experiment.gate": ("tests/unit/test_experiment.py", (
        "test_the_size_check_flags_a_planted_degenerate_rule",
        "test_the_fixed_rule_s_null_size_is_not_degenerate",
        "test_a_tie_blocks_adopt_even_when_every_other_season_won",
        "test_a_tie_blocks_remove_even_when_every_other_season_lost",
        "test_no_ceiling_measured_is_not_runnable_in_both_directions",
    )),
    "hub.models.experiment.review_width": ("tests/unit/test_experiment.py", (
        "test_the_width_is_recorded_for_the_next_run_and_carries_the_review_flag",
        "test_two_runs_at_different_digests_are_not_compared",
        "test_the_same_digest_still_compares_and_a_data_digest_alone_is_enough_to_block",
    )),
    "hub.models.experiment.run_gate": ("tests/unit/test_gate_run.py", (
        "test_a_void_condition_preempts_every_branch_and_is_the_caller_s_sentence",
        "test_a_ceiling_below_the_effect_says_so_loudly_in_the_gate_s_own_places",
    )),
    "hub.models.margin.verdict": ("tests/unit/test_margin.py", (
        "test_a_better_challenger_is_adopted",
        "test_a_challenger_better_on_average_but_not_every_season_is_not_adopted",
        "test_a_binding_ceiling_makes_the_same_challenger_not_runnable",
    )),
    "hub.models.margin.shape_verdict": ("tests/unit/test_margin.py", (
        "test_a_lump_symmetric_about_the_spread_is_adopted",
        "test_a_lump_on_the_favourite_s_side_keeps_the_gaussian",
    )),
    "hub.models.starter_change.verdict": ("tests/unit/test_starter_change.py", (
        "test_verdict_adopts_when_the_lower_bound_clears_delta",
        "test_verdict_removes_when_the_interval_excludes_zero_negatively",
        "test_verdict_is_not_runnable_when_the_mde_exceeds_delta_and_names_the_events_needed",
    )),
    "hub.models.component_error.verdict": ("tests/unit/test_component_error.py", (
        "test_the_verdict_refuses_a_calibration_that_worsens_the_linear_loss",
        "test_the_verdict_adopts_only_when_every_held_out_season_improves",
    )),
    "hub.models.spread.verdict": ("tests/unit/test_spread.py", (
        "test_a_candidate_that_wins_every_season_and_clears_two_se_is_adopted",
        "test_a_gain_too_small_to_distinguish_from_noise_is_not_adopted",
    )),
    "hub.models.injury.type_verdict": ("tests/unit/test_injury.py", (
        "test_a_type_effect_that_wins_everywhere_and_clears_two_se_is_adopted",
        "test_a_gain_too_small_to_distinguish_from_noise_is_not_adopted",
    )),
    # One direction only: `injury.verdict` is an argmin with no bar (S3, #360, frozen), so
    # there is no "too small to adopt" case to plant until it has one. The positive
    # direction is controlled; the negative is #360's to add.
    "hub.models.injury.verdict": ("tests/unit/test_injury.py", (
        "test_a_table_that_beats_both_simple_rules_is_adopted",
    )),
    "hub.models.coverage.verdict": ("tests/unit/test_coverage.py", (
        "test_the_verdict_reads_the_pre_registered_band",
        "test_the_gate_refuses_when_the_interval_leaves_the_band",
    )),
    "hub.draft.adherence.verdict": ("tests/unit/test_adherence.py", (
        "test_twelve_of_sixteen_meets_it",
        "test_eleven_of_sixteen_misses_it",
    )),
}

# module.function -> the ticket that gives it a control. Nothing else belongs here.
OWED: dict[str, str] = {
    "hub.models.weekly_screen.verdict": (
        "#312: the screen's verdict does not read the p it computes, so a planted effect "
        "cannot be asserted to fire on the number the screen reports until it does"),
    "hub.models.weekly_screen.screen": (
        "#312: same instrument; its per-feature disposition is `verdict`'s"),
}


def _decisions(src: Path) -> dict[str, Path]:
    """`module.function` -> file, for every top-level function whose name says it decides."""
    found: dict[str, Path] = {}
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = ".".join(("hub", *path.relative_to(src).with_suffix("").parts))
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            qualified = f"{module}.{node.name}"
            if _DECISION_NAME.match(node.name) or qualified in EXPLICIT_GUARDS:
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
        "def verdict(x):\n    return 'ADOPT'\n\n\ndef helper():\n    pass\n", encoding="utf-8")
    found = _decisions(tmp_path / "hub")
    assert found == {"hub.models.planted.verdict": pkg / "planted.py"}, found
    missing = set(found) - (set(CONTROLLED) | set(OWED))
    assert missing == {"hub.models.planted.verdict"}
