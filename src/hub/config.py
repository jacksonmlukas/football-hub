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
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, cast
import hashlib
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


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
class HubConfig:
    roster: RosterConfig = field(default_factory=RosterConfig)
    draft: DraftConfig = field(default_factory=DraftConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    poll: PollConfig = field(default_factory=PollConfig)


cs = ConfigStore.instance()
cs.store(name="hub_config", node=HubConfig)


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
    "hub.models.components",
    "hub.models.market",
    "hub.models.volume",
    "hub.draft.availability",
    "hub.draft.durability",
    "hub.draft.projection",
    "hub.draft.regression",
)

# Individual fitted constants in modules that cannot be registered wholesale, as
# "module:NAME". `hub.draft.board` is a CLI: most of its module-level names are filesystem
# paths, re-exports and values derived from `RosterConfig`, and hashing those would make the
# digest depend on where the repo is checked out. But `MIN_GAMES` is a real fitted threshold
# -- it decides who is eligible to set replacement level, so it moves every VOR on the board
# -- and it was chosen from data ("the sign is stable from 8 games up").
FITTED_EXTRA: tuple[str, ...] = (
    "hub.draft.board:MIN_GAMES",
)

# Modules that hold measured floats which nonetheless must NOT move a model version, and why.
# Stated here rather than as a bare skip-list in a test, because each one is a claim about
# what the number does and the claim is what a future reader needs to check.
#
# The distinction throughout: a fitted constant is an *input* to a published prediction. A
# number that scores, tunes or illustrates predictions is not, and folding it in would make
# every version bump meaningless -- the same reason `poll` and `quota` are excluded above.
NOT_FITTED: dict[str, str] = {
    "hub.draft.calibrate": "the recorded output of a fit, used by tests to guard the live "
                           "constants -- an assertion about predictions, not an input",
    "hub.draft.evaluate": "offline harness for scoring draft strategies against each other",
    "hub.draft.leverage": "a synthetic fixture league for the variance sweep",
    "hub.draft.live": "operational -- a refresh budget in seconds, same class as `poll`",
    "hub.draft.tune": "hyperparameter search grid, run offline and never at predict time",
    "hub.models.eval": "model-comparison harness; it reads predictions, never makes them",
}


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


def config_digest(cfg: HubConfig | object, *, exclude: tuple[str, ...] = ("poll", "quota")) -> str:
    """Stable 8-char hash of everything that can change a prediction.

    Both halves of that: the settings in `cfg`, and the fitted constants in
    `FITTED_MODULES`. A model version that moved when you changed `conformal_alpha` but not
    when you refit `TALENT_CV` was recording the less important of the two.

    Operational settings are excluded on purpose: changing the poll interval must not
    invalidate a model version, or every version bump becomes meaningless noise.
    """
    d = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True))
    for k in exclude:
        d.pop(k, None)
    canonical = OmegaConf.to_yaml(OmegaConf.create(d), sort_keys=True)
    return hashlib.sha256((canonical + fitted_digest()).encode()).hexdigest()[:8]


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
