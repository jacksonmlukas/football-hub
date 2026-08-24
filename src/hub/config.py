"""Structured Hydra config.

Every number that changes a prediction lives here, and nowhere else. The point is not tidiness:
it is that `config_digest()` folds the resolved config into the model version, so two runs with
different lambdas can never be mistaken for the same model in the track record.

If you find yourself typing a float into a module, it belongs in this file instead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, cast
import hashlib
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class RosterConfig:
    teams: int = 12
    slot: int = 3                       # your snake position
    rounds: int = 16
    qb: int = 1
    rb: int = 2
    wr: int = 3                         # three-WR league, not the ESPN default
    te: int = 1
    flex: int = 1
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
    # 0.0 by evidence, not by default. Across 2022->23, 2023->24 and 2024->25 the
    # adjustment's sign flips (-87, +184, -94 top-50 points at 0.08) and the pooled
    # effect is t = 0.01. See docs/lambda-sweep.md.
    projection_lambda: float = 0.0
    z_clip: float = 3.0
    availability_sims: int = 5000
    # A pick with a longer wait than this after it is in "scarcity" mode.
    long_wait_threshold: int = 10


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


def config_digest(cfg: HubConfig | object, *, exclude: tuple[str, ...] = ("poll", "quota")) -> str:
    """Stable 8-char hash of everything that can change a prediction.

    Operational settings are excluded on purpose: changing the poll interval must not
    invalidate a model version, or every version bump becomes meaningless noise.
    """
    d = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True))
    for k in exclude:
        d.pop(k, None)
    canonical = OmegaConf.to_yaml(OmegaConf.create(d), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def starters(cfg: RosterConfig) -> dict[str, int]:
    return {"QB": cfg.qb, "RB": cfg.rb, "WR": cfg.wr, "TE": cfg.te, "FLEX": cfg.flex}


def flex_share(cfg: RosterConfig) -> dict[str, float]:
    return {"RB": cfg.flex_rb, "WR": cfg.flex_wr, "TE": cfg.flex_te}
