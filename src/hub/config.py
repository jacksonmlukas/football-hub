"""Structured Hydra config.

Every number that changes a prediction is *covered* by `config_digest()`, which folds into the
model version so two runs with different lambdas can never be mistaken for the same model in
the track record. The point is not tidiness; it is that the track record's claim about which
model made which prediction has to be true.

Covered is not the same as living here, and the difference is deliberate:

  * **Settings** are choices -- lambda, alpha, the roster shape. They live in the dataclasses
    below, where Hydra can override them from the command line.
  * **Fitted constants** are measurements with confidence intervals and write-ups in `docs/`.
    They live next to their provenance in the prediction modules listed in `FITTED_MODULES`,
    and `config_digest` hashes them from there. Making them Hydra-overridable would let a
    command-line flag quietly replace a measurement with a preference.

So: if you find yourself typing a float into a module, ask which kind it is. A choice belongs
in this file. A measurement belongs beside the evidence for it, in a module this file hashes.

There is a third kind, and it is deliberately *not* covered. The bytes a run was scored
against are not part of the model that scored them, and they move on their own -- so
`data_digest()` sits beside `config_digest()` and never inside it. `data_digest` argues why
at the point of the decision.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from hub import paths


@dataclass
class RosterConfig:
    """The league's shape. The one declaration of it.

    It used to be five: this class, `board.SLOTS`, `season.STARTERS`, `evaluate.STARTERS`,
    and a bare `< 7` inside `optimize._need_score` that was `rb + wr + te + flex` worked out
    by hand. Four of the five agreed, which is the problem -- nothing made them agree, so
    the first commissioner change would have moved some and not others.
    """
    teams: int = 12
    slot: int = 3                       # your snake position
    rounds: int = 16
    qb: int = 1
    rb: int = 2
    wr: int = 3                         # three-WR league, not the ESPN default
    te: int = 1
    flex: int = 1
    # Who may fill the flex. Separate from the shares below: eligibility is a league rule,
    # the shares are a projection of how the flex tends to get used.
    flex_from: list[str] = field(default_factory=lambda: ["RB", "WR", "TE"])
    # Flex allocation. With three required WR slots the top of the WR pool is already
    # consumed by starters, so the flex tilts back toward RB relative to a 2WR league.
    flex_rb: float = 0.45
    flex_wr: float = 0.50
    flex_te: float = 0.05


# The season being drafted for. One owner, because it was five: `board.SEASON_AHEAD`,
# `espn.league_settings`'s `League(year=...)`, `state.sync_from_espn`'s `year or 2026`,
# `playoff_sos`'s default, and `publish --season`'s default. Nothing made them agree, and
# two of them sat on the same call path: `player_market(season=...)` filtered stats by a
# season the League object had never been asked for, so passing 2027 fetched the 2026 league,
# matched no stat block, and returned proj_ppg as all-null with no error.
SEASON_AHEAD = 2026

# The last completed season -- the one every model is *fitted* on, as against the one being
# drafted for. Split out on 2026-08-27 because only half the SEASON_AHEAD fix had landed:
# `playoff_sos` read `SEASON_AHEAD` from here and then carried `dvp_season: int = 2025` as a
# literal beside it, so a rollover had to move two numbers in lockstep or the defence-adjusted
# ratios would silently come from a two-year-stale season. Six more sites said 2025 by hand.
SEASON_COMPLETED = SEASON_AHEAD - 1

# How long a fantasy regular season is, and the weeks it is played over as a tuple, derived
# rather than restated. `tuple(range(1, 15))` was written out three times -- in `weekly_screen`,
# in `weekly_gate` and here -- a literal 15 in three files for something with one owner. It sits
# in `config` rather than in `draft.season` because `models/` may not reach into `draft/`
# (`test_models_does_not_reach_into_draft`), and the screen and the Weekly projection both need
# it. Weeks 15-17 are the playoffs, reported apart rather than pooled; week 18 is meaningless in
# a league that ends at 17.
REG_SEASON_WEEKS = 14
FANTASY_WEEKS: tuple[int, ...] = tuple(range(1, REG_SEASON_WEEKS + 1))


@dataclass
class DraftConfig:
    # Fraction of the room drafting off ESPN's board. Fit from league history where possible.
    espn_weight: float = 0.5
    # Consensus adjustment strength. Multiplicative, so z=1 moves ~8% up the board.
    # 0.0 by evidence, not by default. Six holdouts (2019->20 .. 2024->25): five of six
    # negative at 0.08, and at 0.32 all six are negative at t = -6.83. No lambda is
    # significantly positive under any grouping. See docs/lambda-sweep.md.
    projection_lambda: float = 0.0
    z_clip: float = 3.0
    availability_sims: int = 5000
    # A pick with a longer wait than this after it is in "scarcity" mode.
    long_wait_threshold: int = 10
    # How far a fitted correction may move a player from his ADP, as a fraction of that ADP.
    # A *choice*, not a measurement, so it lives here rather than beside a coefficient --
    # ADR-0006's distinction. Fixed at 0.20 before anything was fitted: about one round at
    # ADP 60, and 0.6 picks at ADP 3 where consensus is tightest. See ADR-0011.
    correction_clamp_frac: float = 0.20


@dataclass
class ModelConfig:
    name: str = "market"
    seed: int = 0
    conformal_alpha: float = 0.2
    min_calibration_points: int = 20
    calibration_window_weeks: int = 6


@dataclass
class QuotaConfig:
    cfbd_monthly_budget: int = 1000
    cfbd_weekly_reserve: int = 40
    odds_monthly_credits: int = 500


@dataclass
class PollConfig:
    """Tiered cadence. See hub.fetch.espn.poll for why this shape."""
    scoreboard_interval: int = 45       # cheap: one request per league, all games
    summary_every_n_ticks: int = 4      # expensive: one request per game of interest
    max_games_of_interest: int = 12
    stale_after: int = 600              # watchdog threshold


@dataclass
class PoolConfig:
    """The survivor pool's rules, as the commissioner states them.

    Settings, not measurements: a rule is what someone decided, so a correction is a re-run
    rather than an edit. That is the whole reason these are here and not constants in
    `hub.season.survivor` -- three of them are still provisional, and a provisional number
    baked into a module is one nobody re-reads.

    **Confirmed** for 2026: winner-take-all at $20, buybacks at $20 running *through* week 6,
    and a buyback that restores the entry's used-team ledger rather than clearing it -- so
    re-entering late is worth less than re-entering early, because the teams are already
    spent. **Provisional**: the cap of four, and whether it counts per entry or per person.
    **Unconfirmed**: what happens when more than one entry survives, and whether the pool
    plays on past week 18. Both of those move every dollar figure, because every dollar
    figure divides a pot among survivors.

    `double_pick_weeks` is a tuple and not a set. `config_digest` builds a structured config
    and OmegaConf rejects a `set` annotation outright -- which is a startup error, not a
    survivor-path error, so the wrong type here would take down the board.
    """
    entry_fee: float = 20.0
    buyback_fee: float = 20.0
    buyback_cap: int = 4                     # provisional; 0 means no buybacks, not unlimited
    buyback_cutoff_week: int = 6             # inclusive -- buybacks run *through* this week
    buyback_restores_ledger: bool = True     # confirmed: used teams survive a re-entry
    double_pick_weeks: tuple[int, ...] = (13, 14, 15, 16, 17, 18)
    tie_eliminates: bool = True
    field_size: int = 21
    max_entries_per_person: int = 5
    co_survivor_rule: str = "split"          # unconfirmed: split | rollover | tiebreak
    playoff_continuation: bool = False       # unconfirmed for this pool


@dataclass
class HubConfig:
    roster: RosterConfig = field(default_factory=RosterConfig)
    draft: DraftConfig = field(default_factory=DraftConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    poll: PollConfig = field(default_factory=PollConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)


cs = ConfigStore.instance()
cs.store(name="hub_config", node=HubConfig)

# Where the overrides live. `hub.paths` is the one declaration of `ROOT` and imports nothing
# but `pathlib`, so taking the directory from it costs nothing and keeps the eighth copy of
# `parents[2]` from being written here.
CONF_DIR = paths.ROOT / "conf"


def resolved_config() -> HubConfig:
    """The configuration a run operates under: the dataclasses with `conf/` applied.

    The one thing `HubConfig()` is not. ADR-0004 says the digest folded into every model
    version is over the *resolved* config, and a bare `HubConfig()` is the resolved config
    only for as long as `conf/config.yaml` overrides nothing that diverges from a default --
    which is true today (both digests are `281b7b7a`, measured 2026-09-05) and is a
    coincidence rather than a property. The first divergent override would have a gate print
    a model version for a model nobody ran: provenance present in the schema and absent in
    the data, which is the defect `models/ratings.py` records from when `cfg_digest`
    defaulted to `"default"`.

    Composed through Hydra rather than by merging the YAML by hand, because "the way a run
    resolves one" is the claim being made, and a second resolution path would be a second
    answer to defend. `defaults` and the `hydra` block are consumed by the composer, so what
    comes back is the structured schema and nothing else; `to_object` makes it a real
    `HubConfig`, so a caller gets the same type either way and a type checker still sees the
    fields rather than the `Any` a bare `DictConfig` would hand it.

    Degrades to the defaults when `conf/` is absent -- an installed wheel, a partial
    checkout -- because a missing override file means "nothing to override" and a fetch
    that cannot print provenance is worse than one that prints the defaults it ran under.
    A `conf/` that exists and does not compose is *not* degraded past: that is the typo
    ADR-0004 chose structured configs to turn into a startup error, and swallowing it would
    stamp rows with a digest for a configuration the run never had.
    """
    if not CONF_DIR.is_dir():
        return HubConfig()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return cast(HubConfig, OmegaConf.to_object(compose(config_name="config")))


# Modules whose module-level constants were *measured* rather than chosen. They are not in
# `HubConfig` on purpose, and this is the one place that distinction is written down:
#
#   * A setting is a choice, and belongs in a dataclass above, where Hydra can override it
#     from the command line.
#   * A fitted constant is a measurement with a confidence interval and a write-up in
#     `docs/`. Making it Hydra-overridable would invite `draft.talent_cv=0.9` on a command
#     line -- silently replacing a measurement with a preference, which is the exact failure
#     the digest exists to catch. It also has 30 lines of provenance attached, which belongs
#     next to the number and not in a config schema.
#
# But the digest's claim -- "everything that can change a prediction" -- has to be true, and
# for a while it was not: refitting TALENT_CV from 0.35 to 0.42 changed every prediction in
# the repo and left the digest untouched.
#
# The registry is a list of *modules*, not of constants, so a new fitted number added to one
# of these files is covered the day it lands rather than the day someone remembers to
# register it. `test_config.py` holds the line that this list is complete.
FITTED_MODULES: tuple[str, ...] = (
    "hub.models.predict",
    "hub.models.market",
    "hub.models.volume",
    "hub.draft.availability",
    "hub.draft.durability",
    "hub.draft.regression",
)

# Individual fitted constants in modules that cannot be registered wholesale, as
# "module:NAME". `hub.draft.board` is a CLI: most of its module-level names are filesystem
# paths, re-exports and values derived from `RosterConfig`, and hashing those would make the
# digest depend on where the repo is checked out. But `MIN_GAMES` is a real fitted threshold
# -- it decides who is eligible to set replacement level, so it moves every VOR on the board
# -- and it was chosen from data ("the sign is stable from 8 games up").
# `hub.models.components` is deliberately NOT registered wholesale. Half of it --
# `sample_weeks`, `moments`, `project` and the four dispersions they read -- has no
# production caller: component-derived spread was measured worse than the fitted square-root
# law (1.365 against 1.140, P(better) 0.0%, see predict.py) and the losing measurement is
# kept per ADR-0007. Keeping it is right; letting it move the digest that identifies a
# *prediction* is not, because no prediction can reach it. The three constants that are
# reachable are named individually below.
FITTED_EXTRA: tuple[str, ...] = (
    "hub.draft.board:MIN_GAMES",
    "hub.models.components:SCORING",
    "hub.models.components:TD_RATE",
    "hub.models.components:FALLBACK_TD_RATE",
)

# Modules that hold measured floats which nonetheless must NOT move a model version, and why.
# Stated here rather than as a bare skip-list in a test, because each one is a claim about
# what the number does and the claim is what a future reader needs to check.
#
# The distinction throughout: a fitted constant is an *input* to a published prediction. A
# number that scores, tunes or illustrates predictions is not, and folding it in would make
# every version bump meaningless -- the same reason `poll` and `quota` are excluded above.
# Constants that sit in a module the digest otherwise touches and that deliberately do NOT
# move it, each with its reason. Named one by one on purpose: an exclusion should be a
# decision on the record, not a module quietly falling off FITTED_MODULES.
NOT_IN_DIGEST: dict[str, str] = {
    "components.PER_UNIT_CV": "read only by components.sample_weeks",
    "components.YARDS_PER_UNIT": "read only by components.sample_weeks",
    "components.COUNT_DISPERSION": "read only by components.sample_weeks",
    "components.TD_DISPERSION": "read only by components.sample_weeks",
}
# ... and why those four: component-derived spread lost its gate to the fitted square-root
# law (1.365 against 1.140, P(better) 0.0% -- see hub/models/predict.py). ADR-0007 says a
# measurement that steered a decision stays in the tree, so the code is kept. But a
# prediction cannot reach these numbers, so they must not identify one.



def _canonical(v: Any) -> str:
    """Deterministic text for a constant, whatever shape it has.

    Sorted by `repr` of the key rather than the key itself, because `TEAMMATE_RHO` is keyed
    by tuples and a plain sort across mixed key types raises.
    """
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: repr(kv[0]))
        return "{" + ",".join(f"{k!r}:{_canonical(x)}" for k, x in items) + "}"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(repr(x) for x in v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_canonical(x) for x in v) + "]"
    if isinstance(v, float):
        # repr() of a float round-trips exactly, so a refit that moves the last bit still
        # moves the digest.
        return repr(v)
    return repr(v)


def fitted_constants() -> dict[str, Any]:
    """Every fitted constant in the prediction modules, as {"module.NAME": value}.

    Public module-level names in upper case, which is the repo's own convention for a
    constant. Callables, modules and imported types are skipped -- a re-export is not a
    number.
    """
    from importlib import import_module

    out: dict[str, Any] = {}
    for name in FITTED_MODULES:
        mod = import_module(name)
        for attr in dir(mod):
            if attr.startswith("_") or not attr.isupper():
                continue
            v = getattr(mod, attr)
            if isinstance(v, (int, float, str, dict, set, frozenset, list, tuple)):
                out[f"{name.rsplit('.', 1)[-1]}.{attr}"] = v
    for spec in FITTED_EXTRA:
        mod_name, attr = spec.split(":")
        out[f"{mod_name.rsplit('.', 1)[-1]}.{attr}"] = getattr(import_module(mod_name), attr)
    return out


def fitted_digest() -> str:
    """Stable 8-char hash of the fitted constants alone."""
    text = "\n".join(f"{k}={_canonical(v)}" for k, v in sorted(fitted_constants().items()))
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def config_digest(cfg: HubConfig | object, *,
                  exclude: tuple[str, ...] = ("poll", "quota", "pool")) -> str:
    """Stable 8-char hash of everything that can change a prediction.

    Both halves of that: the settings in `cfg`, and the fitted constants in
    `FITTED_MODULES`. A model version that moved when you changed `conformal_alpha` but not
    when you refit `TALENT_CV` was recording the less important of the two.

    Operational settings are excluded on purpose: changing the poll interval must not
    invalidate a model version, or every version bump becomes meaningless noise.

    `pool` is excluded for a sharper version of the same reason. This digest is stamped on
    every prediction row and folded into `FitSpec`, so covering the survivor pool's rules
    would mean confirming the buyback cap invalidates every cached NFL fit and issues a new
    model version for the weekly prediction and the draft board -- neither of which can read
    a pool rule. `pool_digest` distinguishes two survivor runs instead, where the distinction
    belongs.
    """
    d = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True))
    for k in exclude:
        d.pop(k, None)
    canonical = OmegaConf.to_yaml(OmegaConf.create(d), sort_keys=True)
    return hashlib.sha256((canonical + fitted_digest()).encode()).hexdigest()[:8]


class DataPin(Protocol):
    """What identifies one loaded source, structurally.

    Read-only on purpose, so `hub.fetch.nflverse.Pin` -- a frozen dataclass -- satisfies it,
    and stated here rather than imported so no digest in this module has to reach a data
    layer to be computed. Every digest here answers from its arguments alone and can be
    computed from an empty checkout; making one import a fetch layer would be the wrong
    shape and a circular import besides, since `hub.fetch.nflverse` reads `SEASON_COMPLETED`
    from here. (`resolved_config` does read one file -- `conf/config.yaml`, and degrades to
    the defaults without it -- because resolving a config is the one job here that is about
    what is on disk. It is not on any digest's path: `config_digest` hashes the config it is
    handed.)
    """

    @property
    def source(self) -> str: ...

    @property
    def as_of(self) -> str | None: ...

    @property
    def digest(self) -> str: ...


def pin_fold(source: str, as_of: str | None, digest: str) -> str:
    """What one pin hashes as: content, folded with the source name and the as-of.

    The one statement of it. `nflverse.pin_digest` builds a pin's own digest from a frame,
    and `data_digest` below folds a set of pins into a run's; both hash this exact form, and
    until now each wrote it out with a comment saying the other matched. A comment is not a
    mechanism -- the two would have gone on agreeing right up until one of them was edited,
    and the failure would be a digest that no longer named the thing it was compared against.

    All three parts, and only these three. Hashing the labels alone would be invariant to
    exactly the drift the pins exist to catch -- two runs at one as-of that fetched different
    bytes would agree. Hashing content alone would make an archive that has not moved between
    two pins indistinguishable, and a pin is a claim about a date and a source as well as
    about rows. `pinned_at` is deliberately absent: it is a wall-clock stamp that moves on
    every refetch, so folding it in would report drift on rows that had not changed.
    """
    return f"{source}\n{as_of or ''}\n{digest}"


# What `data_digest` answers for a run that pinned nothing. Not a hash: sixteen modules
# besides `hub.fetch.nflverse` itself still `import nflreadpy` and reach the archive directly
# (counted 2026-09-05) -- `models/panel.py` at eight call sites on its own -- so the
# unpinned run is the common case until U2 of
# docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md routes them. A
# plausible-looking eight hex characters for that case would be provenance present in the
# schema and absent in the data, which is the defect `models/ratings.py` records when
# `cfg_digest` defaulted to "default" for every run under every configuration.
#
# So it shares a *width* with a digest and nothing else, and that is the whole of the answer
# to "should the sentinel and an eight-character digest share a shape". Eight characters so
# the three line up where they are printed together; not eight *hex* characters, so a reader
# who did not write this can tell which is which without being told -- `unpinned` is not a
# hexadecimal number, and `test_config.py` holds that line rather than trusting the word to
# stay unhexish through a rename. The shared return type is the same trade: a caller printing
# provenance gets one column of strings and never has to branch, and the only thing that
# would make a `None` better here is a caller that wants to *ask*, which none does -- the
# three digests are printed and compared, and a comparison against this sentinel means
# "these runs pinned nothing", which is true and is what it says.
UNPINNED = "unpinned"


def data_digest(pins: Iterable[DataPin]) -> str:
    """Stable 8-char hash of the data a run actually loaded, or `UNPINNED` for none.

    Deliberately a *sibling* of `config_digest` and not an ingredient of it. The two answer
    different questions and only one of them belongs in a model version:

      * `config_digest` answers "which model made this row". ADR-0006 requires that it move
        when, and only when, a setting or a fitted constant moves.
      * this answers "which bytes was it scored against". nflverse revises `player_stats` and
        `pbp` in place -- `nflverse._write_by_week` says so where it passes `replace=True` --
        so this moves on a Tuesday refetch with no line of code and no fitted constant having
        changed.

    Folding the second into the first would issue a new model version every time the archive
    was refetched. That is the mirror image of the failure ADR-0006 was written for: there,
    refitting TALENT_CV left the version byte-identical, so the record claimed one model had
    made both sets of rows; here the version would move while the model stood still, and it
    would stop discriminating hyperparameters, which is the one job ADR-0004 gave it. It would
    also make the version depend on the state of a cache directory, so a fresh clone would
    report a different model for identical code.

    So a gate prints both, and a moved archive shows up as a changed *data* digest beside an
    unchanged model version -- which is the distinction a reader needs to make.

    Each pin folds through `pin_fold`, which is the same call `nflverse.pin_digest` makes and
    argues there for what goes in and what stays out. Sorted and de-duplicated, so the answer
    does not depend on which source a gate happened to load first, or on one archive being
    reached through two call sites.
    """
    rows = sorted({pin_fold(p.source, p.as_of, p.digest) for p in pins})
    if not rows:
        return UNPINNED
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:8]


def digests(cfg: HubConfig | object, pins: Iterable[DataPin]) -> dict[str, str]:
    """The three digests that identify a run, as {"cfg", "fitted", "data"}.

    One call, so that no gate decides for itself which of the three name it -- and so the
    "beside, not inside" choice above is made in one place rather than re-argued at each
    output. `cfg` and `fitted` describe the code; `data` describes what the code read.
    """
    return {"cfg": config_digest(cfg), "fitted": fitted_digest(), "data": data_digest(pins)}


def pool_digest(cfg: PoolConfig) -> str:
    """Stable 8-char hash of the survivor pool's rules.

    Beside `config_digest`, never inside it. Two survivor runs under different pool rules are
    different runs and their outputs must say so; two *model* runs under different pool rules
    are the same model, and saying otherwise would move a version for a number the model
    cannot read.

    Not folded into `digests()` either, for the same reason stated from the other side: that
    helper answers "which run produced this row" for every gate in the repo, and a draft
    board has no pool. The survivor artifact and the decision journal call this one directly.
    """
    d = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
    canonical = OmegaConf.to_yaml(OmegaConf.create(d), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def starters(cfg: RosterConfig) -> dict[str, int]:
    """Every starting slot, flex included. What a draft board counts against."""
    return {"QB": cfg.qb, "RB": cfg.rb, "WR": cfg.wr, "TE": cfg.te, "FLEX": cfg.flex}


def required_starters(cfg: RosterConfig) -> dict[str, int]:
    """Starting slots by position, flex excluded.

    The flex is not a position, and a lineup optimiser that treats it as one will happily
    start a quarterback in it. `starters()` and this differ by exactly that key.
    """
    return {"QB": cfg.qb, "RB": cfg.rb, "WR": cfg.wr, "TE": cfg.te}


def flex_positions(cfg: RosterConfig) -> tuple[str, ...]:
    """Positions eligible for the flex."""
    return tuple(cfg.flex_from)


def drafted_positions(cfg: RosterConfig | None = None) -> tuple[str, ...]:
    """The positions this league actually drafts, in board order.

    Exactly `required_starters`' keys: a position with no starting slot is not drafted, and
    kickers and defences have none here on purpose (ADR-0008). The same tuple was written out
    twelve times under four names -- `DRAFTED_POSITIONS`, `SKILL`, `POSITIONS` and
    `SCORING_POSITIONS` -- which is the defect this module was written to fix for the roster
    shape, one level down: a superflex or a K/DST league would move `RosterConfig` and none of
    the twelve, and deleting any one copy breaks only its own module, which is how there came
    to be four names.
    """
    return tuple(required_starters(cfg or RosterConfig()))


# The league's own answer, for the modules that want a constant rather than a call. Callers
# inside `FITTED_MODULES` must use the function instead: an upper-case module-level name in
# one of those files is swept into `config_digest`, and the roster shape is a setting.
DRAFTED_POSITIONS: tuple[str, ...] = drafted_positions()


def flex_capacity(cfg: RosterConfig) -> int:
    """How many flex-eligible players a team can start at once.

    Required flex-eligible slots plus the flex itself -- 2 + 3 + 1 + 1 = 7 in this league.
    `optimize._need_score` carried that 7 as a literal, so a commissioner adding a second
    flex would have left the draft valuing depth against the old roster.
    """
    return sum(n for p, n in required_starters(cfg).items()
               if p in flex_positions(cfg)) + cfg.flex


def flex_share(cfg: RosterConfig) -> dict[str, float]:
    """Expected split of the flex slot across eligible positions."""
    return {"RB": cfg.flex_rb, "WR": cfg.flex_wr, "TE": cfg.flex_te}


# ESPN's own names for the slots this league starts, so a league read over the wire can be
# compared against the config. RB/WR and WR/TE flex variants exist; this league runs the
# plain one.
_ESPN_SLOT_NAMES: dict[str, str] = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "FLEX": "RB/WR/TE",
}


def roster_mismatch(league: dict[str, Any]) -> dict[str, tuple]:
    """Where a league's own roster slots disagree with `RosterConfig`, as {slot: (league, ours)}.

    The counterpart to `components.scoring_mismatch`, and it exists for the same reason: the
    roster shape belongs to the league, not to this repo. `espn.league_settings()` has always
    returned it and every caller discarded it, so a commissioner moving to two flex slots
    would have shifted replacement level, positional need and every VOR on the board without
    a word.

    Slots the league runs and this config does not (K, D/ST, bench, IR) are not a
    disagreement -- nothing here drafts them. A slot count that differs is.
    """
    ours = starters(RosterConfig())
    out: dict[str, tuple] = {}
    for slot, n in ours.items():
        theirs = league.get(_ESPN_SLOT_NAMES[slot])
        if theirs is not None and int(theirs) != int(n):
            out[slot] = (int(theirs), int(n))
    return out
