"""The config exists to make model versions honest. These tests pin that."""
from dataclasses import dataclass, field

import pytest
from omegaconf.errors import ValidationError

from hub import config, declare
from hub.config import (
    UNPINNED,
    DraftConfig,
    HubConfig,
    PollConfig,
    PoolConfig,
    RosterConfig,
    UnpinnedRead,
    config_digest,
    data_digest,
    digests,
    fitted_constants,
    fitted_digest,
    flex_capacity,
    flex_positions,
    pool_digest,
    required_starters,
    roster_mismatch,
    starters,
)


def test_digest_is_stable_across_identical_configs():
    assert config_digest(HubConfig()) == config_digest(HubConfig())


def test_changing_a_hyperparameter_changes_the_digest():
    """The whole point: two lambdas must never share a model version."""
    a = HubConfig()
    b = HubConfig(draft=DraftConfig(projection_lambda=0.16))
    assert config_digest(a) != config_digest(b)


def test_operational_settings_do_not_change_the_digest():
    """Changing the poll interval must not invalidate a model version."""
    a = HubConfig()
    b = HubConfig(poll=PollConfig(scoreboard_interval=90))
    assert config_digest(a) == config_digest(b)


# --- the pool's rules are settings, and stay out of the model's digest -----
#
# The rules of a survivor pool are decisions someone made, not measurements, so they belong
# here where a correction is a re-run. But `config_digest` is stamped on every prediction row
# and folded into `FitSpec`, so covering them would mean the commissioner confirming a buyback
# cap invalidates every cached NFL fit -- a new model version for a number no model reads.


def test_the_pool_config_is_reachable_and_carries_every_rule():
    """What the solver and the money layer read. A rule that is not here is a rule somebody
    hardcoded, which is the thing this dataclass exists to prevent."""
    pool = HubConfig().pool
    for rule in ("entry_fee", "buyback_fee", "buyback_cap", "buyback_cutoff_week",
                 "buyback_restores_ledger", "double_pick_weeks", "field_size",
                 "co_survivor_rule", "co_elimination_rule", "field_concentration"):
        assert hasattr(pool, rule), rule
    # The other half, since #160: a rule that *is* here has to be read by the simulation, or
    # it is a knob a reader takes for a lever. `tie_eliminates`, `max_entries_per_person` and
    # `playoff_continuation` were here and read by nothing; each left with its reason written
    # where it was. `test_every_pool_field_is_read_by_the_simulation` in test_publish holds it.
    for gone in ("tie_eliminates", "max_entries_per_person", "playoff_continuation"):
        assert not hasattr(pool, gone), f"{gone} is back, and nothing reads it"


def test_a_pool_rule_does_not_move_the_model_digest():
    """The reason `pool` is excluded. Confirming the buyback cap must not invalidate a cached
    fit or issue a new model version for the weekly projection, which cannot read a pool rule."""
    a = HubConfig()
    b = HubConfig(pool=PoolConfig(buyback_cap=0))
    assert config_digest(a) == config_digest(b)


def test_the_pool_digest_moves_when_a_pool_rule_moves():
    """And the other half: two survivor runs under different rules are different runs, and
    their outputs have to say so."""
    base = PoolConfig()
    assert pool_digest(base) != pool_digest(PoolConfig(buyback_cutoff_week=5))
    assert pool_digest(base) != pool_digest(PoolConfig(double_pick_weeks=(13, 14)))


def test_the_pool_digest_is_stable_across_identical_rules():
    assert pool_digest(PoolConfig()) == pool_digest(PoolConfig())


def test_double_pick_weeks_default_to_thirteen_through_eighteen():
    assert HubConfig().pool.double_pick_weeks == (13, 14, 15, 16, 17, 18)


def test_double_pick_weeks_are_overridable_to_empty():
    """An empty tuple reduces the solver to one pick a week, which is today's behaviour and
    the fallback if the rule is ever misread."""
    assert HubConfig(pool=PoolConfig(double_pick_weeks=())).pool.double_pick_weeks == ()


def test_double_pick_weeks_is_a_tuple_not_a_set():
    """`config_digest` builds a structured config and OmegaConf rejects a `set` annotation
    outright -- so the wrong type here is a startup error for the whole repo, not a survivor
    one. Pinned because a set is the obvious thing to reach for."""
    assert isinstance(HubConfig().pool.double_pick_weeks, tuple)

    @dataclass
    class Bad:
        weeks: set = field(default_factory=set)

    with pytest.raises(ValidationError, match="Unexpected type annotation"):
        config_digest(Bad())


def test_a_buyback_cap_of_zero_means_no_buybacks():
    """Zero is representable and distinct from absent. A cap that read as unlimited when set
    to zero would simulate a field that refills for free."""
    assert HubConfig(pool=PoolConfig(buyback_cap=0)).pool.buyback_cap == 0


def test_the_confirmed_rules_carry_the_commissioner_s_answers():
    """Buybacks run *through* week 6 and a re-entry keeps its used teams. Both were confirmed
    rather than assumed, and both change what a buyback is worth -- re-entering in week 6 with
    six teams spent is a weaker entry than the same $20 in week 2."""
    pool = HubConfig().pool
    assert pool.buyback_cutoff_week == 6
    assert pool.buyback_restores_ledger is True


# --- the digest covers the fitted constants, not just the settings ---------
#
# The docstring promised "everything that can change a prediction" and delivered `HubConfig`.
# Refitting TALENT_CV from 0.35 to 0.42 changed every prediction in the repo and left the
# model version identical, so the track record claimed one model had made both sets of rows.

def test_refitting_a_constant_changes_the_digest(monkeypatch):
    """The failure this pair of functions exists to prevent, stated directly."""
    from hub.models import predict
    before = config_digest(HubConfig())
    monkeypatch.setattr(predict, "TALENT_CV", 0.35)
    assert config_digest(HubConfig()) != before


def test_the_digest_is_stable_when_nothing_is_refitted():
    """A digest that moved on its own would make every week look like a new model."""
    assert fitted_digest() == fitted_digest()
    assert config_digest(HubConfig()) == config_digest(HubConfig())


def test_the_constants_that_moved_predictions_are_all_covered():
    """Named individually because these are the ones that were outside the digest, and a
    module dropped from FITTED_MODULES would otherwise fail silently -- the digest would
    still be a plausible-looking hash."""
    got = fitted_constants()
    for name in ("predict.TALENT_CV", "predict.TALENT_CV_BY_POS", "predict.WEEKLY_K",
                 "predict.WEEKLY_SKEW", "predict.TEAMMATE_RHO", "components.SCORING",
                 "components.TD_RATE", "volume.VOLUME_CURVE", "regression.TD_LUCK_BETA",
                 "durability.BETA", "durability.INJURY_BETA",
                 "availability.PICK_NOISE_SLOPE"):
        assert name in got, name


def test_a_constant_in_a_module_full_of_paths_is_covered_by_its_own_declaration():
    """`MIN_GAMES` decides who is eligible to set replacement level, so it moves every VOR
    on the board -- and it lives in a CLI module full of filesystem paths that must not be
    hashed. Before #253 that took a second list keyed by hand; now the constant says
    `chosen(10)` where it is written and nothing else in the module is looked at."""
    assert "board.MIN_GAMES" in fitted_constants()
    assert not [k for k in fitted_constants() if k.startswith("board.") and k != "board.MIN_GAMES"
                and k != "board.FLEX_SHARES"]


def test_a_constant_moves_the_version_without_touching_any_module():
    """#253: "this constant moves the version" is a question the digest answers from a
    substituted value, not from a monkeypatched attribute -- so a test of one constant
    cannot leak into the next through module state."""
    have = fitted_constants()
    assert fitted_digest(have) == fitted_digest()
    assert fitted_digest({**have, "predict.TALENT_CV": 0.35}) != fitted_digest()
    assert fitted_digest({**have, "predict.TALENT_CV": have["predict.TALENT_CV"]}) == fitted_digest()


def test_the_walk_finds_the_declarations_and_only_declarations():
    """What the digest is derived from. Every covered key is a `fitted` or `chosen`
    declaration in the tree; every declaration keys one constant; and a declaration is a
    module-level assignment through one of the three spellings and nothing else."""
    decls = declare.declarations()
    keys = [d.key for d in decls]
    assert len(keys) == len(set(keys)), "two declarations share a key"
    assert set(fitted_constants()) == {d.key for d in decls if d.covered}
    assert set(declare.excluded()) == {d.key for d in decls if not d.covered}
    assert {d.kind for d in decls} == declare.SPELLINGS
    # The spelling has to be at module level and assign one name: a call inside a function
    # or a tuple unpacking is not a declaration.
    src = (
        "from hub.declare import chosen, fitted, not_an_input\n"
        "A = fitted(1.0)\n"
        "B: int = chosen(2)\n"
        "C = not_an_input(3, why='eight words are the least an argument can be made in')\n"
        "D, E = fitted(4), fitted(5)\n"
        "def f():\n    F = fitted(6)\n"
        "G = 7\n"
    )
    got = declare.declared_in(src, "hub.x.y")
    assert [(d.name, d.kind, d.covered) for d in got] == [
        ("A", "fitted", True), ("B", "chosen", True), ("C", "not_an_input", False)]
    assert got[2].why is not None and got[2].why.startswith("eight words")
    assert got[0].key == "y.A"


def test_an_exclusion_without_its_argument_is_refused_not_recorded():
    with pytest.raises(ValueError, match=r"hub\.x\.y\.C .*without an argument"):
        declare.declared_in("C = not_an_input(3, 'too short')\n", "hub.x.y")
    with pytest.raises(ValueError, match=r"hub\.x\.y\.C"):
        declare.declared_in("C = not_an_input(3)\n", "hub.x.y")


def test_two_declaring_modules_with_one_stem_are_refused(monkeypatch, tmp_path):
    """The stem-keyed scan this replaces would have let `hub.draft.season` and a second
    `season` module collide in silence; the walk refuses the collision by name."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "season.py").write_text("from hub.declare import fitted\nX = fitted(1.0)\n")
    (tmp_path / "b" / "season.py").write_text("from hub.declare import fitted\nY = fitted(2.0)\n")
    monkeypatch.setattr(declare, "SRC", tmp_path)
    declare.declarations.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="share the stem 'season'"):
            declare.declarations()
    finally:
        declare.declarations.cache_clear()


# --- coverage is a recorded decision, not a property of how a name is typed (#201) ---
#
# Three unrelated spellings each decided, at some point, whether a value identified a model
# version: whether it was capitalised (#187's `_FACTOR_CACHE_MAX`), whether it was a float
# (#201's shape constants), and whether a re-export happened to be a number (#199's
# `TYPE_CHECKING`, since `bool` subclasses `int`). None of the three is a fact about the
# constant. What follows pins the replacement: covered by default, out only by name.

SHAPE_CONSTANTS = [
    ("hub.config", "REG_SEASON_WEEKS", 13),
    ("hub.league", "PLAYOFF_TEAMS", 8),
    ("hub.league", "PLAYOFF_ROUNDS", 2),
    ("hub.draft.optimize", "DEFAULT_ROUNDS", 15),
    ("hub.draft.cohort", "ROUNDS", 15),
    ("hub.draft.cohort", "DRAFTS", 40),
]


@pytest.mark.parametrize(("module", "name", "moved_to"), SHAPE_CONSTANTS)
def test_moving_a_simulation_extent_moves_the_model_version(monkeypatch, module, name,
                                                            moved_to):
    """#201's acceptance criterion, one constant at a time.

    Each of these sets the extent of a random draw -- weeks played, playoff rounds simulated,
    field size, draft rounds, drafts per cohort. By #196 the extent of a draw fixes the
    seed-to-outcome map, so a run on either side of one of these prices every Gate interval
    differently while computing from identical inputs. Before this ticket all six moved
    nothing: the sweep that keeps modules honest looked for a `float` and every one of them
    is an `int`, which ADR-0006 recorded as a stray-threshold limitation and which is nothing
    of the kind.
    """
    import importlib
    before = config_digest(HubConfig())
    monkeypatch.setattr(importlib.import_module(module), name, moved_to)
    assert config_digest(HubConfig()) != before, (
        f"{module}.{name} sets the extent of a random draw and two runs across it are not "
        f"the same model, but the digest cannot tell them apart")


def test_a_cache_bound_does_not_identify_a_model_version(monkeypatch):
    """#187's finding, closed. `_FACTOR_CACHE_MAX` bounds how many Cholesky factorisations
    `hub.models.predict` keeps; every prediction is identical on either side of it.

    The assertion that matters is the second one. It was already out of the digest -- but only
    because someone spelled it with a leading underscore, and it would have been *in* the
    digest spelled without one. Now it is out because `NOT_IN_DIGEST` says so and says why.
    """
    from hub.models import predict
    before = config_digest(HubConfig())
    monkeypatch.setattr(predict, "_FACTOR_CACHE_MAX", 8192)
    assert config_digest(HubConfig()) == before
    assert "predict._FACTOR_CACHE_MAX" in declare.excluded()


@pytest.mark.parametrize("key", ["predict._FACTORS", "predict._FACTOR_CACHE_MAX",
                                 "predict._EIG_FLOOR", "predict._INDEPENDENT_FLOOR"])
def test_each_exclusion_argues_for_itself(key):
    """A skip-list entry with no argument is the same silence as no entry at all. Each of
    these is a claim about what the constant does, and the claim is what a reader checks --
    at the constant now, where `not_an_input` carries it (#253)."""
    assert len(declare.excluded()[key].split()) >= 15, declare.excluded()[key]


def test_every_exclusion_in_the_tree_says_enough_to_be_checked_later():
    thin = {k: w for k, w in declare.excluded().items()
            if len(w.split()) < declare.MIN_REASON_WORDS}
    assert not thin, thin


def test_a_private_name_nobody_excluded_is_covered_anyway():
    """The other half of dropping the spelling rule, and the one that would regress quietly.

    `regression._PHASES` carries the four- and six-point scoring weights `td_luck` is computed
    in -- as much a model input as `components.SCORING` beside it -- and `volume._UNITS` routes
    which stat feeds which phase. Both were invisible to the sweep for no reason but an
    underscore. Under the new rule the default is *covered*, so an underscore buys nothing and
    a constant leaves the digest only by being argued out of it.
    """
    got = fitted_constants()
    assert "regression._PHASES" in got
    assert "volume._UNITS" in got


def test_a_value_the_digest_cannot_hash_is_refused_rather_than_dropped(monkeypatch):
    """The type test survives only as an assertion, and this is why it has to.

    As a *filter* it silently dropped whatever it did not list, which is how a cache dict and
    a `TYPE_CHECKING` re-export each got the answer wrong in opposite directions. A value the
    digest cannot turn into stable text is now a loud failure naming the constant, so the
    choice between hashing it and excluding it is made by a person on the record.
    """
    from hub.models import predict
    monkeypatch.setattr(predict, "MIN_SKEW", object())
    with pytest.raises(RuntimeError, match=r"predict\.MIN_SKEW"):
        fitted_constants()


def test_roster_reflects_three_wr_league():
    assert starters(RosterConfig())["WR"] == 3


# `test_flex_shares_sum_to_one` moved to `test_replacement_level.py` with the constant it
# checks. What stays here is the config-level half of #184: the claim is not about the three
# values, it is about where they may live and what may reach them.

def test_the_flex_shares_are_not_a_config_field():
    """ADR-0006, on a quantity nobody derived.

    `flex_rb`, `flex_wr` and `flex_te` were `RosterConfig` fields, so `roster.flex_rb=0.9` on
    a Hydra command line would have moved the replacement index at RB, WR and TE and every
    VOR on the board with it. Losing the knob is #184's point, and the assertion is written
    against the *resolved* config rather than the dataclass so a `conf/` file cannot put one
    back either.
    """
    resolved = config.resolved_config()
    for gone in ("flex_rb", "flex_wr", "flex_te"):
        assert not hasattr(RosterConfig(), gone), (
            f"{gone} is a Hydra-overridable field again. The flex shares set replacement "
            f"level at three positions; they live in `hub.draft.board.FLEX_SHARES`.")
        assert not hasattr(resolved.roster, gone)
    # Eligibility is the other half and genuinely is a league rule, so it stays.
    assert resolved.roster.flex_from == ["RB", "WR", "TE"]


def test_the_flex_shares_are_covered_by_the_digest():
    """Out of the config is not enough -- out of the config and out of the digest would be a
    number that changes every VOR on the board while the model version says nothing, which is
    the exact failure ADR-0006 was written after. `hub.draft.board` is a CLI full of paths,
    so the shares declare themselves `chosen` where they are written (#253) -- a stated
    choice, not a measurement, and covered because coverage is owed by anything that
    changes a prediction."""
    decl = {d.key: d for d in declare.declarations()}["board.FLEX_SHARES"]
    assert decl.kind == "chosen" and decl.module == "hub.draft.board"
    assert fitted_constants()["board.FLEX_SHARES"] == {"RB": 0.45, "WR": 0.50, "TE": 0.05}


def test_slot_is_three():
    assert RosterConfig().slot == 3


# --- the league shape is declared once -------------------------------------
#
# It was declared five times: this config, `board.SLOTS`, `season.STARTERS`,
# `evaluate.STARTERS`, and a bare `< 7` in `optimize._need_score` that was rb+wr+te+flex
# worked out by hand. All five agreed, and nothing made them agree.

def test_the_flex_is_not_a_position():
    """`starters` includes FLEX so a board can count roster spots; `required_starters` does
    not, so a lineup optimiser cannot start a quarterback in the flex."""
    cfg = RosterConfig()
    assert "FLEX" in starters(cfg)
    assert "FLEX" not in required_starters(cfg)


def test_flex_capacity_is_derived_not_typed():
    """The magic 7. Deriving it is the whole point: a second flex has to move it."""
    assert flex_capacity(RosterConfig()) == 7
    assert flex_capacity(RosterConfig(flex=2)) == 8
    assert flex_capacity(RosterConfig(rb=3)) == 8


def test_every_module_reads_the_same_shape():
    """The five declarations, now one. If a commissioner change moves `RosterConfig` and
    any of these stays put, this is what notices."""
    from hub.draft import board, evaluate, optimize, season
    cfg = RosterConfig()
    assert season.STARTERS == required_starters(cfg)
    assert evaluate.STARTERS == required_starters(cfg)
    assert board.SLOTS == starters(cfg)
    assert season.FLEX_FROM == flex_positions(cfg)
    assert evaluate.FLEX_FROM == flex_positions(cfg)
    assert board.FLEX_ELIGIBLE == flex_positions(cfg)
    assert optimize.FLEX_CAPACITY == flex_capacity(cfg)


def test_the_draft_values_depth_against_the_real_flex_capacity():
    """`_need_score` ranks flex-eligible depth above surplus, up to capacity. The literal 7
    meant a league rule change would have left the draft valuing depth against the old
    roster while every other module used the new one."""
    from hub.draft.optimize import _need_score
    full = {"RB": 3, "WR": 3, "TE": 1}          # seven flex-eligible players held
    assert sum(full.values()) == flex_capacity(RosterConfig())
    assert _need_score(full, "RB") == 0          # surplus
    assert _need_score({"RB": 2, "WR": 3, "TE": 1}, "RB") == 1   # still room
    assert _need_score({"RB": 1, "WR": 3, "TE": 1}, "RB") == 2   # unfilled required slot


# --- the roster shape belongs to the league, so check it ------------------

def test_a_league_running_our_lineup_reports_no_mismatch():
    assert roster_mismatch({"QB": 1, "RB": 2, "WR": 3, "TE": 1, "RB/WR/TE": 1}) == {}


def test_a_second_flex_is_reported():
    """The change that would silently move replacement level at every position."""
    got = roster_mismatch({"QB": 1, "RB": 2, "WR": 3, "TE": 1, "RB/WR/TE": 2})
    assert got == {"FLEX": (2, 1)}


def test_slots_we_do_not_draft_are_not_a_disagreement():
    """ESPN reports K, D/ST, bench and IR. Nothing here drafts them."""
    assert roster_mismatch({"QB": 1, "RB": 2, "WR": 3, "TE": 1, "RB/WR/TE": 1,
                            "K": 1, "D/ST": 1, "BE": 7, "IR": 1}) == {}


def test_a_league_that_reports_nothing_is_not_a_disagreement():
    """Degradation: no cookies means no slots, which must not print a false alarm."""
    assert roster_mismatch({}) == {}


# --- one owner for the roster shape, one level down ------------------------
#
# The drafted-position tuple was written out twelve times under four names --
# `DRAFTED_POSITIONS`, `SKILL`, `POSITIONS`, `SCORING_POSITIONS`. That is the defect this
# module was written to fix for `RosterConfig` itself ("it used to be five"), repeated one
# level down: a superflex or a K/DST league would move the roster and none of the twelve.


def test_drafted_positions_are_the_positions_with_a_starting_slot():
    from hub.config import DRAFTED_POSITIONS, drafted_positions
    assert DRAFTED_POSITIONS == tuple(required_starters(RosterConfig()))
    assert drafted_positions() == DRAFTED_POSITIONS


def test_drafted_positions_track_the_roster_they_are_derived_from():
    """The whole point: change the league and the tuple follows, in one edit."""
    from dataclasses import replace

    from hub.config import drafted_positions
    superflex = replace(RosterConfig(), qb=2)
    assert drafted_positions(superflex) == drafted_positions()
    assert required_starters(superflex)["QB"] == 2


def test_the_drafted_positions_are_written_once():
    """The AST guard that stops the four names coming back."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.Tuple, ast.List)):
                continue
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            # Exactly four, each once. `leverage.POS` is a 14-slot archetype roster whose
            # *values* are these positions with multiplicity -- a fixture league's shape, not
            # a second declaration of which positions the league drafts.
            if len(node.elts) == 4 and sorted(vals) == ["QB", "RB", "TE", "WR"]:
                offenders.append(f"{path.relative_to(src)}:{node.lineno}")
    assert not offenders, (
        "the drafted positions written outside hub.config -- they are "
        f"required_starters(RosterConfig()).keys() and belong to the roster: {offenders}")


# --- one owner for the season ----------------------------------------------
#
# `SEASON_AHEAD` got one owner because it was declared five times. Only half of that landed:
# `playoff_sos` read it from here and then carried `dvp_season: int = 2025` beside it, so a
# rollover had to move two numbers in lockstep or the defence-adjusted ratios would come from
# a two-year-stale season with no warning. Eleven more sites said the year by hand.


def test_the_completed_season_is_the_one_before_the_drafted_one():
    from hub.config import SEASON_AHEAD, SEASON_COMPLETED
    assert SEASON_COMPLETED == SEASON_AHEAD - 1


def test_playoff_sos_reads_both_seasons_from_config():
    """The half of the SEASON_AHEAD fix that did not land the first time."""
    import inspect

    from hub.config import SEASON_AHEAD, SEASON_COMPLETED
    from hub.draft import playoff_sos
    sig = inspect.signature(playoff_sos.playoff_sos)
    assert sig.parameters["season_ahead"].default == SEASON_AHEAD
    assert sig.parameters["dvp_season"].default == SEASON_COMPLETED


def test_no_default_hardcodes_the_current_season():
    """A literal that happens to equal this season is the rollover bug waiting to happen.

    Deliberately scoped to the *current* two years rather than to any four-digit year:
    `evaluate`'s 2023/2024 defaults pin a sweep that was actually run and recorded, and a
    fixed past year is a historical pin, not a season that has to move in September.
    """
    import ast
    import pathlib

    from hub.config import SEASON_AHEAD, SEASON_COMPLETED
    live = {SEASON_AHEAD, SEASON_COMPLETED}
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            found = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found = list(node.args.defaults) + [d for d in node.args.kw_defaults if d]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "add_argument"):
                found = [kw.value for kw in node.keywords if kw.arg == "default"]
            for d in found:
                if isinstance(d, ast.Constant) and d.value in live and d.value is not True:
                    offenders.append(f"{path.relative_to(src)}:{d.lineno} = {d.value}")
    assert not offenders, (
        "a season default written by hand -- use SEASON_AHEAD or SEASON_COMPLETED so the "
        f"rollover is one edit: {offenders}")


def test_the_fantasy_weeks_are_derived_from_the_season_length():
    """It was `tuple(range(1, 15))` in three modules -- a literal 15 for a league length that
    already had an owner. It lives in `config` and not in `draft.season` because `models/` may
    not reach into `draft/`, and the screen and the Weekly projection both need it."""
    from hub.config import FANTASY_WEEKS, REG_SEASON_WEEKS
    assert FANTASY_WEEKS == tuple(range(1, REG_SEASON_WEEKS + 1))
    assert len(FANTASY_WEEKS) == REG_SEASON_WEEKS


def test_no_module_restates_the_season_length():
    """The literal, not the name: a re-declared `range(1, 15)` would pass the test above."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    bad = []
    for path in root.rglob("*.py"):
        if path.name == "config.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "range" and len(node.args) == 2
                    and [a.value if isinstance(a, ast.Constant) else None
                         for a in node.args] == [1, 15]):
                bad.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not bad, f"the league length is restated in: {bad}"


# --- the data digest sits beside the config digest, never inside it --------
#
# R1: every gate output carries a content-derived data digest alongside its existing config
# digest, so a moved archive shows up as a changed digest rather than a silently different
# number. `docs/next.md` records the input moving under a harness and nothing saying so: two
# `--diagnose` runs at one seed disagreed on pick 3 because `build()` refetched live ESPN ADP
# each time.
#
# Beside and not inside is the load-bearing half, and it is an ADR-0006 question. The config
# digest folds into the model version to answer "which model made this row"; nflverse revises
# `player_stats` and `pbp` in place, so a data digest folded in would issue a new model version
# every time the archive was refetched, with no line of code and no fitted constant having
# moved. That is the mirror of the TALENT_CV failure ADR-0006 was written for -- there the
# version stood still while the model changed, here it would move while the model stood still --
# and either way the version stops identifying a model.


@dataclass(frozen=True)
class _StubPin:
    """The three fields `data_digest` reads. A stub rather than `hub.fetch.nflverse.Pin`, so
    these tests describe the contract and not one caller's dataclass; the real `Pin` is fed
    through the same function below."""

    source: str
    as_of: str | None
    digest: str
    pinned_at: str | None = None


def test_a_moved_archive_moves_the_data_digest():
    """The defect in one line: same source, same as-of, different bytes."""
    before = data_digest([_StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa")])
    after = data_digest([_StubPin("ff_opportunity", "2026-09-04", "bbbbbbbb")])
    assert before != after


def test_the_data_digest_names_the_source_and_the_as_of_too():
    """A pin is a claim about a date and a source as well as about rows, the same reasoning
    `pin_digest` folds all three."""
    rows = "aaaaaaaa"
    base = data_digest([_StubPin("ff_opportunity", "2026-09-04", rows)])
    assert data_digest([_StubPin("ff_opportunity", "2026-08-01", rows)]) != base
    assert data_digest([_StubPin("player_stats", "2026-09-04", rows)]) != base


def test_the_data_digest_does_not_depend_on_the_order_pins_are_listed_in():
    """Two gates reading the same two sources must agree, whichever they loaded first."""
    a = _StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa")
    b = _StubPin("player_stats", None, "bbbbbbbb")
    assert data_digest([a, b]) == data_digest([b, a])


def test_listing_one_pin_twice_is_the_same_run():
    """A gate that loads one source through two call sites pinned one archive, not two."""
    a = _StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa")
    assert data_digest([a, a]) == data_digest([a])


def test_when_the_rows_were_taken_does_not_move_the_digest():
    """`pinned_at` is a wall-clock stamp on a source that revises in place. Folding it in
    would move the digest on a refetch that returned byte-identical rows -- the digest would
    then report drift that had not happened, which is as useless as missing drift that had."""
    rows = "aaaaaaaa"
    early = _StubPin("player_stats", "2026-09-04", rows, pinned_at="2026-09-04T01:00:00+00:00")
    late = _StubPin("player_stats", "2026-09-04", rows, pinned_at="2026-09-05T09:00:00+00:00")
    assert data_digest([early]) == data_digest([late])


def test_a_read_that_names_no_bytes_unpins_the_whole_digest():
    """Issue #165, in one line. A run that read two sources and pinned one has to say
    `unpinned` -- publishing a hash of the one it managed names the wrong set of bytes."""
    pinned = _StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa")
    assert data_digest([pinned, UnpinnedRead("player_stats")]) == UNPINNED


def test_a_partial_run_is_not_the_run_that_read_only_the_part_it_pinned():
    """The reason the partial case is worse than the total one: it is invisible. Without
    this, a run over three sources with two pinned compared *equal* to a run that read only
    those two, and there was nothing in either digest to tell them apart."""
    two = [_StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa"),
           _StubPin("player_stats", "2026-09-04", "bbbbbbbb")]
    assert data_digest([*two, UnpinnedRead("pbp")]) != data_digest(two)


def test_one_unpinned_read_is_enough_however_many_pinned_ones_surround_it():
    """All of the bytes or none of them. A majority of pins is not a digest."""
    many = [_StubPin(f"s{i}", "2026-09-04", f"{i:08x}") for i in range(6)]
    assert data_digest([*many[:3], UnpinnedRead("pbp"), *many[3:]]) == UNPINNED


def test_an_unpinned_read_is_told_apart_without_reading_the_sources_back():
    """The sentinel is what a reader sees, so the answer to "which bytes" is legible from
    the printed line alone rather than by going back to the cache tree."""
    mixed = data_digest([_StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa"),
                         UnpinnedRead("player_stats", "2026-09-04")])
    assert mixed == UNPINNED
    with pytest.raises(ValueError):
        int(mixed, 16)


def test_an_unpinned_read_satisfies_the_shape_data_digest_reads():
    """It stands where a pin stands, which is what lets a caller record one without the
    callers downstream growing a branch for it."""
    read = UnpinnedRead("player_stats", "2026-09-04")
    assert (read.source, read.as_of, read.digest) == ("player_stats", "2026-09-04", UNPINNED)
    assert UnpinnedRead("pbp").as_of is None


def test_a_run_that_pinned_nothing_says_so_rather_than_hashing_air():
    """An unpinned run must not publish a plausible-looking hash. Most of the tree still
    reaches nflverse directly rather than through the fetch layer until U2, so this is the
    common case today and it has to read as what it is."""
    assert data_digest([]) == "unpinned"


def test_the_data_digest_is_the_length_the_others_are():
    """A gate prints the three side by side."""
    assert len(data_digest([_StubPin("ff_opportunity", None, "aaaaaaaa")])) == 8


def test_the_sentinel_shares_a_width_with_a_digest_and_nothing_else():
    """The sentinel and an eight-character digest occupy one column and one return type, so
    the only thing standing between them is that one of them is not a number.

    Width is on purpose -- three digests line up where they are printed. Hex would not be:
    an unpinned run would be reporting provenance it does not have, which is the
    `cfg_digest = "default"` defect one module over. That leaves the difference resting on a
    word, and a word survives a rename only if something is checking, so this checks. A
    sentinel that came back `deadbeef` would pass every other test in this file.
    """
    assert len(UNPINNED) == len(data_digest([_StubPin("ff_opportunity", None, "aaaaaaaa")]))
    with pytest.raises(ValueError):
        int(UNPINNED, 16)


# --- one statement of the pin fold -------------------------------------------

def test_one_pin_folds_the_way_a_set_of_them_does():
    """`pin_fold` is the whole of the form, and both levels reach it.

    `nflverse.pin_digest` hashes one pin's content with its labels; `data_digest` hashes a
    set of already-digested pins with theirs. They were two spellings of one form with a
    comment on each saying the other matched -- which is a claim, not a mechanism. Hashing
    the fold by hand here would put a third spelling in the tree, so what is asserted is what
    the fold has to *do*: all three parts load-bearing, and a null as-of not silently equal
    to some other as-of.
    """
    from hub.config import pin_fold
    base = pin_fold("ff_opportunity", "2026-09-04", "aaaaaaaa")
    assert base != pin_fold("player_stats", "2026-09-04", "aaaaaaaa"), "source is not folded"
    assert base != pin_fold("ff_opportunity", "2026-08-01", "aaaaaaaa"), "as-of is not folded"
    assert base != pin_fold("ff_opportunity", "2026-09-04", "bbbbbbbb"), "content is not folded"
    assert base != pin_fold("ff_opportunity", None, "aaaaaaaa")


def test_the_fetch_layer_folds_a_pin_through_the_same_call():
    """The reason the fold moved into this module: `hub.config` may not import a fetch layer,
    so the shared form has to live at the bottom and be reached upward."""
    import inspect

    from hub.fetch import nflverse
    assert "pin_fold(" in inspect.getsource(nflverse.pin_digest), (
        "nflverse.pin_digest has gone back to spelling the fold out for itself")


# --- the config a run resolves, as against the one the dataclasses default to -


def test_the_resolved_config_reads_the_override_tree(tmp_path, monkeypatch):
    """`resolved_config` composes `conf/`, so an override there reaches the digest.

    The claim that makes it worth having: a run's provenance names the configuration the run
    had. Asserted against a tree written here rather than against the repo's own `conf/`,
    which overrides nothing that diverges from a default -- so pointing at the real one would
    be a test that passes equally well against `return HubConfig()`.
    """
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  - hub_config\n  - _self_\n\nroster:\n  wr: 4\n")
    monkeypatch.setattr(config, "CONF_DIR", tmp_path)
    got = config.resolved_config()
    assert isinstance(got, HubConfig), "a caller gets the same type either way"
    assert got.roster.wr == 4
    assert config_digest(got) != config_digest(HubConfig()), (
        "an override that changes the roster shape did not move the model version")


def test_an_absent_conf_tree_degrades_to_the_defaults(tmp_path, monkeypatch):
    """Graceful degradation: an installed wheel has no `conf/`, and a fetch that cannot
    print provenance is worse than one that prints the defaults it ran under."""
    monkeypatch.setattr(config, "CONF_DIR", tmp_path / "absent")
    assert config.resolved_config() == HubConfig()


def test_a_conf_tree_that_does_not_compose_is_not_degraded_past(tmp_path, monkeypatch):
    """The other half, and the one worth being deliberate about. ADR-0004 chose structured
    configs so a typo in `conf/` is a startup error; swallowing one here would put it back to
    surfacing mid-Sunday, and would stamp rows with a digest for a configuration no run had.
    """
    from hydra.errors import ConfigCompositionException
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  - hub_config\n  - _self_\n\nroster:\n  wrr: 4\n")
    monkeypatch.setattr(config, "CONF_DIR", tmp_path)
    # The type and the message, not a bare `Exception`: "something went wrong" is satisfied
    # by a broken import or a stale global, and would still pass against a function that had
    # stopped composing anything at all.
    with pytest.raises(ConfigCompositionException, match="wrr"):
        config.resolved_config()


def test_the_repos_own_conf_still_agrees_with_the_dataclass_defaults():
    """Not a property -- a measurement, and the one several other checks lean on.

    `draft/backtest.py` and `models/ratings.py` stamp rows through `resolved_config`, and
    `test_fetch_nflverse.py` checks the provenance line against `config_digest(HubConfig())`
    on the strength of these two agreeing. They agree because `conf/config.yaml` sets
    `teams`, `slot` and `wr` to the values the dataclasses already default to. The day an
    override diverges, this is the test that says which of the two moved and that the answer
    is `conf/` rather than a bug.

    **Moved 2026-09-07 (#150): `281b7b7a` -> `ab32cf62`.** The superseded digest is kept
    here rather than edited away, because the artifacts under `site/data/` still stamp it
    and that is correct -- each records the model version that produced it. What moved is
    the code: `PICK_NOISE_INTERCEPT` and `PICK_NOISE_SLOPE` were refitted and
    `PICK_NOISE_SLOPE_CI` published beside them, so `fitted_digest` went `d5598b96` ->
    `3d6fc111`. ADR-0006 is the reason that is a feature: a refit that did *not* move the
    model version would be the bug.

    **Moved again 2026-09-07 (#172): `ab32cf62` -> `105b2b64`**, `fitted_digest` `3d6fc111`
    -> `df2c1948`. Same reason, one layer further back. `hub.draft.calibrate`'s bootstrap was
    corrected -- the curve is refitted inside each resample, the unit is the player rather
    than the player-season, and the noise term takes its own draw -- which widened every
    per-position standard error. `_shrink` reads those errors to decide how much of the
    spread between positions is real, so `TALENT_CV_BY_POS` moved RB 0.50 -> 0.48 and
    TE 0.32 -> 0.33 without any raw estimate changing. The pooled `TALENT_CV` did not move at
    all; only its interval did, [0.380, 0.434] -> [0.370, 0.453]. Restated in
    docs/talent-cv.md.

    **Moved again 2026-09-07 (#187): `105b2b64` -> `0c6fab17`.** `hub.models.predict` is in
    `FITTED_MODULES`, so every upper-case module-level name in it is swept into the digest,
    and #187 added four while making non-factorable correlation blocks repair rather than
    silently fall back to independence.

    One of the four is a real model change and earns the move: `NOT_A_TEAM` excludes `FA`
    from being correlated as though it were a roster. It had been a 43-player block with a
    minimum eigenvalue of -0.914, harmless only because it failed to factor and fell back to
    independence -- which is the right answer for a free agent reached by accident, and which
    repair would have turned into a confidently wrong correlated draw. Excluding it moved the
    gate's own output, -14.36 to -15.23 on the two-draft run.

    **The other three do not earn it, and that is a finding rather than a fix.**
    `_FACTOR_CACHE_MAX` is a cache bound, `_FACTORS` is the cache itself, and `_EIG_FLOOR` is
    the tolerance that decides what counts as positive semi-definite -- only the last is
    arguably a modelling choice. A cache size should not identify a model version. It does
    here because the rule is "an upper-case module-level name in a registered module", which
    is a naming convention standing in for the question of whether something is a
    measurement. That is #201's finding one step further on: there the coverage was decided
    by whether a constant was written as a float, here by whether it was written in capitals.

    The digest was checked to be stable at runtime despite `_FACTORS` being mutable and
    populated during a draw -- it is not hashed by content, so the model version does not
    move as a process runs.

    **Moved again 2026-09-10 (#184): `0c6fab17` -> `eb32dd45`**, `fitted_digest` `df2c1948`
    -> `0590176f`. Both halves moved at once, which is unusual and is the whole shape of the
    change: the flex shares left `RosterConfig` -- so three float fields dropped out of the
    hashed schema -- and arrived in `hub.draft.board` as `FLEX_SHARES`, registered by hand in
    `FITTED_EXTRA`, so the same three numbers entered the fitted half.

    **This one is earned, and it is worth saying why against the two spurious moves earlier
    today.** #187's `_FACTOR_CACHE_MAX` and #199's `TYPE_CHECKING` moved the digest without
    any measurement changing: a cache bound and a standard-library import came to identify a
    model version because the sweep's rule is a naming convention. Nothing about a prediction
    was different afterwards. Here the *values* are also unchanged -- 0.45, 0.50, 0.05, byte
    for byte -- and the move is still real, because what changed is what can reach them.
    Before this commit `roster.flex_rb=0.9` on a Hydra command line moved the replacement
    index at RB, WR and TE and every VOR on the board, and produced a digest identical to a
    run that had not; after it, there is no such command line. A digest that did not move
    here would be claiming those two runs were the same model, which they never were.

    That the two halves move in opposite directions and cancel is not something to rely on:
    they are hashed as one string, so the arithmetic is not additive and the pairing above is
    a record of what happened rather than a property.

    **Moved again 2026-09-10 (#183): `eb32dd45` -> `6bdcb663`**, `fitted_digest` `0590176f`
    -> `8c248a6e`. One constant, `durability.MISSED_YOY_R = 0.407`, in a module already in
    `FITTED_MODULES`. It is not a new measurement -- the +0.407 has been in
    `docs/durability.md` and in this module's docstring since 2026-08-24 -- but it had never
    been a *name*, so nothing hashed it, and the simulator that now reads it had no concept
    of absence at all. Every P(win) this repo computes moves, because a player with an injury
    history now contributes zeros for the weeks he misses instead of a shrunken mean. The
    digest is doing exactly its job: two runs on either side of that are not the same model.

    `hub.season.pool`'s `field_concentration` (#152) landed between these two moves and does
    not appear in either, which is `config_digest`'s `pool` exclusion working as designed --
    a survivor rule no prediction can read must not issue a model version.

    **Moved again 2026-09-10 (#201): `6bdcb663` -> `3f96c0c2`**, `fitted_digest` `8c248a6e`
    -> `0e144a29`. **This one is a coverage correction and not a model change**, and the
    distinction matters because both kinds moved this digest today. Not one value in this
    repo is different on either side of it: no constant was refitted, no measurement was
    republished, and every prediction a run makes across this commit is identical. What
    changed is which constants the digest can see.

    Eight names entered, and they had all been invisible for the same reason -- the sweep was
    deciding coverage from how a line is *typed*:

      * `config.REG_SEASON_WEEKS`, `league.PLAYOFF_TEAMS`, `league.PLAYOFF_ROUNDS`,
        `optimize.DEFAULT_ROUNDS`, `cohort.ROUNDS`, `cohort.DRAFTS` -- the shape constants of
        #201's original finding. Each sets the extent of a random draw, and by #196 the extent
        of a draw fixes the seed-to-outcome map, so moving one re-prices every published Gate
        interval. They were out because they are `int` and the scan looked for `float`.
      * `volume._UNITS` and `regression._PHASES` -- out because of a leading underscore.
        `_PHASES` carries the four- and six-point scoring weights that `td_luck` is computed
        in, which is as much a model input as anything in `components.SCORING` beside it.

    Four names are now excluded *by name and with a reason* rather than by spelling, and none
    of them changes what the digest contains today -- they were already outside it, for the
    wrong reason. `predict._FACTORS`, `_FACTOR_CACHE_MAX`, `_EIG_FLOOR` and
    `_INDEPENDENT_FLOOR` are in `NOT_IN_DIGEST` with the argument for each. That closes the
    finding recorded against #187's move above: a cache bound stayed out of the model version
    because someone typed an underscore, and would have been in it otherwise.

    The one escape that is not closed is stated in `FITTED_EXTRA`'s comment: `n_draft_sims`
    and `n_season_sims` are function-signature defaults, so there is no name to register.

    **Moved again 2026-09-12 (#218): `e3d549ab` -> `08ceee28`**, `fitted_digest` `0df8d593`
    -> `90a4b5de`. A new module, `hub.models.quarterback`, registered wholesale. Its three
    numbers -- 0.132 spread points per quarterback value unit, 0.9 of the relative gap kept
    per game of a starter's tenure, and 7 days before a snapshot quote stops being a live
    price -- are **stated choices with their provenance beside them, not fitted constants**,
    and they are in the digest anyway, for `FLEX_SHARES`'s reason: coverage is owed by
    anything that changes a prediction, measured or not. Every game the staleness field
    marks as having no live price is now rated differently from the run before this commit,
    and a digest that did not move would be claiming those two runs were the same model.

    **Moved again 2026-09-12 (#268): `08ceee28` -> `fd28e5e5`**, `fitted_digest` `90a4b5de`
    -> `4152f399`. Two of the three left: `POINTS_PER_VALUE` (0.132) and `DECAY_PER_GAME`
    (0.9) went with the estimator they belonged to, and `ELO_PER_VALUE` (3.3), which was
    never in the digest's view as a number a prediction read, left with them. The
    quarterback adjustment is now the source's own `qb_adj` on the latest row over
    `ELO_PER_POINT` (25) and nothing else; the replaced construction subtracted an
    arrival-time baseline from a current value and decayed the difference by tenure, which
    on the 32 live teams was a mean absolute error of 0.4 spread points and worst 3.6. Every
    game rated without a live price moves, most by under half a point, and the digest says
    so. **This one is a model change**, not a coverage correction.

    **Moved again 2026-09-12 (#271): `fd28e5e5` -> `a1e669b9`**, `fitted_digest` `4152f399`
    -> `9be7844c`. One name entered through `FITTED_EXTRA`: `nfeloqb.COMMIT`, the commit of
    `greerreNFL/nfeloqb` the quarterback adjustment reads, so advancing the pin is an edit
    that moves the version and predictions under the new input are distinguishable from the
    old. **A coverage correction and not a model change**: the value is `None` -- the
    default branch, which is what the URL always was -- so nothing any run computes is
    different on either side of this commit. The first move of this name to a commit will be
    the model change, and it is the maintainer's to make from the next pull's stamp.

    **Unmoved 2026-09-12 (#253): `a1e669b9`, `fitted_digest` `9be7844c`.** The mechanism
    changed under the number and the number did not: the four lists (`FITTED_MODULES`,
    `FITTED_EXTRA`, `NOT_IN_DIGEST`, the `NOT_FITTED_BECAUSE` strings) are gone and every
    constant declares its own coverage where it is written (`hub.declare`), and the walk
    finds exactly the forty-eight the lists found. Recorded here because a pin that only
    speaks when it moves cannot say that a rewrite of what feeds it was a no-op.
    """
    assert config_digest(HubConfig()) == "a1e669b9"
    assert config_digest(config.resolved_config()) == config_digest(HubConfig())


def test_a_real_pin_is_accepted():
    """The stub above describes the contract; this is the record the fetch layer writes."""
    from hub.fetch.nflverse import Pin
    pin = Pin(source="ff_opportunity", as_of="2026-09-04", digest="aaaaaaaa", rows=1200)
    assert data_digest([pin]) == data_digest(
        [_StubPin("ff_opportunity", "2026-09-04", "aaaaaaaa")])


def test_the_data_digest_leaves_the_model_version_alone():
    """ADR-0006, stated as the test that would have caught the wrong choice. The archive
    moves weekly; the model version must move when and only when the code or a fitted
    constant does."""
    cfg = HubConfig()
    before = (config_digest(cfg), fitted_digest())
    moved = digests(cfg, [_StubPin("player_stats", "2026-09-04", "bbbbbbbb")])
    still = digests(cfg, [_StubPin("player_stats", "2026-09-04", "aaaaaaaa")])
    assert moved["data"] != still["data"], "the data digest did not notice the archive moving"
    assert moved["cfg"] == still["cfg"] == before[0]
    assert moved["fitted"] == still["fitted"] == before[1]


def test_digests_reports_all_three_and_names_them():
    """One call, so no gate has to decide for itself which digests identify a run."""
    got = digests(HubConfig(), [_StubPin("ff_opportunity", None, "aaaaaaaa")])
    assert set(got) == {"cfg", "fitted", "data"}
    assert got["cfg"] == config_digest(HubConfig())
    assert got["fitted"] == fitted_digest()


def test_an_imported_upper_case_name_is_not_a_fitted_constant():
    """A `typing` import identified the model version for the length of one merge.

    #199 added `from typing import TYPE_CHECKING` to `hub.draft.availability`, which is in
    `FITTED_MODULES`. `TYPE_CHECKING` is upper case, module level, public -- and `False`, which
    `isinstance(v, int)` accepts because `bool` subclasses `int`. So the sweep took it, the
    digest moved from `0c6fab17` to `e88c41d9`, and every prediction stamped in between claimed
    a model change that had not happened. ADR-0006 exists to stop exactly that.

    The old sweep's docstring already said "a re-export is not a number". It could not act on
    it, because the only question it asked was about type. This asks the source what the module
    assigns, so it also covers imports nobody has made yet.
    """
    got = config.fitted_constants()
    assert not [k for k in got if k.endswith(".TYPE_CHECKING")], (
        f"a typing import is in the model version: {sorted(got)}")
    # The general property since #253: nothing enters the digest by being *assigned* at
    # all. A name enters by being declared, and an import cannot be, because the walk reads
    # `NAME = fitted(...)` off the source and an import is not that.
    assert all(d.kind in ("fitted", "chosen") for d in declare.declarations() if d.covered)
