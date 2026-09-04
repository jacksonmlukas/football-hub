"""The config exists to make model versions honest. These tests pin that."""
from hub.config import (
    FITTED_EXTRA,
    FITTED_MODULES,
    NOT_FITTED,
    NOT_IN_DIGEST,
    DraftConfig,
    HubConfig,
    PollConfig,
    RosterConfig,
    config_digest,
    fitted_constants,
    fitted_digest,
    flex_capacity,
    flex_positions,
    flex_share,
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


def test_a_fitted_constant_outside_a_registered_module_is_still_covered():
    """`MIN_GAMES` decides who is eligible to set replacement level, so it moves every VOR
    on the board -- but it lives in a CLI module full of filesystem paths that must not be
    hashed. `FITTED_EXTRA` names it individually."""
    assert "board.MIN_GAMES" in fitted_constants()


def test_every_module_holding_a_fitted_constant_is_registered():
    """What stops the registry rotting. `FITTED_MODULES` is a list of modules rather than of
    constants so a new number is covered the day it lands -- but a whole new *module* still
    has to be added, and this is the line that notices.

    Known limitation, stated rather than hidden: this scans for module-level names holding a
    *float*. An integer threshold in an unregistered module still slips through, which is
    exactly how `MIN_GAMES` did until `FITTED_EXTRA` picked it up by hand. Widening the scan
    to ints would flag every structural count in the repo (`TEAMS`, `REG_SEASON_WEEKS`,
    `BOOTSTRAP`), so the line is drawn at floats and the exceptions are named.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    registered = {m.rsplit(".", 1)[-1] for m in FITTED_MODULES}
    registered |= {m.rsplit(".", 1)[-1] for m in NOT_FITTED}
    # A module can also be covered constant-by-constant rather than wholesale: registered
    # into the digest through FITTED_EXTRA, or deliberately excluded through NOT_IN_DIGEST.
    # `hub.models.components` is both -- three of its constants are live and four describe
    # code no prediction can reach.
    by_name = {f"{spec.split(':')[0].rsplit('.', 1)[-1]}.{spec.split(':')[1]}"
               for spec in FITTED_EXTRA} | set(NOT_IN_DIGEST)
    # Every module under `src/hub`, with no directory filter. It used to read
    # `searched = [src / "models", src / "draft"]`, which was a fourth exclusion mechanism
    # beside the three `config.py` names -- and a silent one, which is exactly what that
    # module says an exclusion must not be: "a decision on the record, not a module quietly
    # falling off FITTED_MODULES". A directory list inside a test is the quiet kind, and it
    # was hiding `lineup_gate.OPP_MU` and `OPP_SD`, in none of the three registries.
    #
    # Scanning everything rather than widening the list by one package is what stops it
    # coming back: a new package is covered the day it lands, and the only way out is a named
    # entry with a reason. Nothing outside `models/`, `draft/` and `season/` holds a
    # module-level float today, so the exhaustive scan costs nothing and closes the hole.
    missing = []
    for path in sorted(src.rglob("*.py")):
        if path.stem in registered or path.stem.startswith("_"):
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            for t in targets:
                if not (isinstance(t, ast.Name) and t.id.isupper()
                        and not t.id.startswith("_")):
                    continue
                # A fitted constant is a measured *number*. String and bool settings,
                # paths and column lists are not, and live in these modules legitimately.
                if _holds_a_float(node.value) and f"{path.stem}.{t.id}" not in by_name:
                    missing.append(f"{path.stem}.{t.id}")
    assert not missing, (
        f"fitted constants outside FITTED_MODULES, so config_digest does not cover them: "
        f"{sorted(set(missing))}")


def _holds_a_float(node) -> bool:
    """Whether a literal contains a float anywhere inside it."""
    import ast
    return any(isinstance(n, ast.Constant) and isinstance(n.value, float)
               for n in ast.walk(node))


def test_roster_reflects_three_wr_league():
    assert starters(RosterConfig())["WR"] == 3


def test_flex_shares_sum_to_one():
    assert abs(sum(flex_share(RosterConfig()).values()) - 1.0) < 1e-9


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
