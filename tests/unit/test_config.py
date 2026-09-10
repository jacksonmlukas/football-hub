"""The config exists to make model versions honest. These tests pin that."""
from dataclasses import dataclass, field

import pytest
from omegaconf.errors import ValidationError

from hub import config
from hub.config import (
    FITTED_EXTRA,
    FITTED_MODULES,
    NOT_IN_DIGEST,
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
    flex_share,
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
                 "buyback_restores_ledger", "double_pick_weeks", "tie_eliminates",
                 "field_size", "max_entries_per_person", "co_survivor_rule",
                 "playoff_continuation"):
        assert hasattr(pool, rule), rule


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
    # Generated from the declarations, not maintained beside them. Each not-fitted module
    # states its own reason in `NOT_FITTED_BECAUSE`, so the set of excluded modules is a
    # consequence of what the modules say rather than a second list that can disagree with
    # them -- the "compute it or stop claiming it" correction issue #109 asked for.
    registered |= {m.rsplit(".", 1)[-1] for m in not_fitted_modules()}
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
    """
    assert config_digest(HubConfig()) == "0c6fab17"
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


# --- the reason lives with the numbers (issue #109) ------------------------

def not_fitted_modules() -> dict[str, str]:
    """Every module under `hub` that declares itself not-fitted, and why.

    The registry this replaces lived in `config.py` and was the reason that module churned:
    a measured float anywhere meant editing it. The knowledge is about the constant, so it is
    kept with the constant, and any list of exclusions is derived from here.

    Import errors are swallowed rather than reported. A module that cannot import is a defect
    the rest of the suite fails on directly, and letting it break the exclusion list would
    turn one broken module into a wall of unrelated "unregistered float" failures.
    """
    import importlib
    import pkgutil

    import hub
    found: dict[str, str] = {}
    for info in pkgutil.walk_packages(hub.__path__, prefix="hub."):
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        said = getattr(mod, "NOT_FITTED_BECAUSE", None)
        if isinstance(said, str) and said.strip():
            found[info.name] = said
    return found


def test_every_not_fitted_module_gives_a_reason_worth_reading():
    """The declaration is the record now, so it has to carry what the registry carried.

    A bare marker would pass the exclusion check while saying nothing -- and the whole point
    of the registry it replaces was that an exclusion is "a decision on the record, not a
    module quietly falling off FITTED_MODULES". A module can exclude itself here, so the
    reason is the only thing standing between that and a silent opt-out.
    """
    said = not_fitted_modules()
    assert said, "no module declares itself not-fitted, so the exclusion list is empty"
    thin = {m: r for m, r in said.items() if len(r.split()) < 8}
    assert not thin, (
        f"these exclude themselves without saying enough to be checked later: {thin}")


def test_a_module_holding_an_unregistered_float_is_still_caught(tmp_path):
    """The property the registry existed for, asserted against the reshaped check.

    Proved by construction rather than by trusting the rewrite: a module with a module-level
    float and no declaration must be visible to the scan and absent from the exclusion set.
    This is what expand-then-contract was protecting -- if the reshaped check had the same
    gap as a missed declaration, nothing else here would notice.
    """
    import ast

    holder = tmp_path / "newthing.py"
    holder.write_text("SOME_COEFFICIENT = 0.42\n")
    tree = ast.parse(holder.read_text())
    floats = [t.id for node in tree.body if isinstance(node, ast.Assign)
              for t in node.targets
              if isinstance(t, ast.Name) and t.id.isupper() and _holds_a_float(node.value)]
    assert floats == ["SOME_COEFFICIENT"], "the scan no longer sees a module-level float"
    assert holder.stem not in {m.rsplit(".", 1)[-1] for m in not_fitted_modules()}, (
        "an undeclared module counts as registered, so the check cannot fail for it")


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
    from hub import config

    got = config.fitted_constants()
    assert not [k for k in got if k.endswith(".TYPE_CHECKING")], (
        f"a typing import is in the model version: {sorted(got)}")

    # The general property, not just the one name that got in. Every swept key must be a name
    # its own module assigns -- checked against the source rather than against a denylist,
    # because a denylist would need the next import added to it by hand.
    import importlib
    for key in got:
        short, attr = key.rsplit(".", 1)
        full = next((m for m in config.FITTED_MODULES if m.endswith(f".{short}")), None)
        if full is None:
            continue          # FITTED_EXTRA names its constants one at a time and by hand
        assigned = config._assigned_at_module_level(importlib.import_module(full))
        assert attr in assigned, (
            f"{key} is swept into the digest but {full} imports it rather than assigning it")


def test_a_module_whose_source_cannot_be_read_stops_the_digest():
    """The sweep must not answer "no constants" when it means "I could not look".

    An empty set here would drop that module's constants from `fitted_constants`, move
    `config_digest`, and say nothing -- the model version would change because a file became
    unreadable. Every module in `FITTED_MODULES` is a file in this repo, so the branch should
    be unreachable in practice; it is tested because an unreachable branch that fails open is
    how the other three defects in this file's history got in.
    """
    import sys
    import types

    from hub import config

    synthetic = types.ModuleType("hub.models.notondisk")   # a module with no source file
    sys.modules["hub.models.notondisk"] = synthetic
    try:
        with pytest.raises(RuntimeError, match="cannot read the source"):
            config._assigned_at_module_level(synthetic)
    finally:
        del sys.modules["hub.models.notondisk"]
