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
    # Where fitted constants actually live: the prediction layer, plus the draft modules
    # that fit their own coefficients. Anything else is a CLI, a fetcher or a store.
    searched = [src / "models", src / "draft"]
    missing = []
    for d in searched:
        for path in sorted(d.glob("*.py")):
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
