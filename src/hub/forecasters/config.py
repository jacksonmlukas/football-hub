"""The study's pinned settings: four models, their call settings and prices, and the caps.

Everything here is fixed at `prereg-llm-v0` (`docs/prereg/llm-forecasters.md`, *Before v0*), and
any change after it is a row in the deviations log. The values live in `conf/forecasters.toml`
rather than in code so a v0 pin is one reviewed file; until it is filled the committed file
holds `"TBD"`, and `runner.require_pinned` refuses to call a model under it.

Temperature is `None` for a model whose API does not accept one -- the prereg's "temperature
where the API accepts it" -- and reasoning effort is `None` for a model that has none. The
prices are what the cost log and the spending guard multiply tokens by.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATH = Path(__file__).resolve().parents[3] / "conf" / "forecasters.toml"
UNSET = "TBD"


@dataclass(frozen=True)
class LlmSpec:
    name: str                       # the contestant's short name, e.g. "claude"
    provider: str
    model_version: str              # the exact version string the API is called with
    training_cutoff: str            # the provider's stated knowledge date, beside blind results
    temperature: float | None
    reasoning_effort: str | None
    max_output_tokens: int
    usd_per_mtok_in: float
    usd_per_mtok_out: float


@dataclass(frozen=True)
class RetrievalSpec:
    """The paired retrieval arm (#511): `paired_with` names the bare LLM it pairs with (same
    model, same settings, its line-aware packet plus news). Tavily credits are counted apart
    from dollars against `season_credit_cap`, and priced at `usd_per_credit` into the cost log
    (0 while the promo's credits are free)."""
    paired_with: str
    season_credit_cap: int
    usd_per_credit: float


@dataclass(frozen=True)
class StudyConfig:
    llms: tuple[LlmSpec, ...]
    season_cap_usd: float
    home_rate: float                # P(home win), the home-constant contestant
    home_rate_seasons: str          # the seasons the rate comes from
    est_input_tokens: int           # an upper estimate of one prompt, for the spending guard
    samples: int = 5
    max_parse_retries: int = 2
    min_valid: int = 3
    clip_lo: float = 0.01
    clip_hi: float = 0.99
    min_lead_minutes: int = 60      # a forecast finishing closer to kickoff is excluded
    retrieval: RetrievalSpec | None = None   # the paired retrieval arm; None runs without it


def _opt(v: Any) -> Any:
    return None if v in ("", "none", None) else v


def load(path: Path | None = None) -> StudyConfig:
    raw = tomllib.loads((path or PATH).read_text())
    llms = tuple(LlmSpec(name=m["name"], provider=m["provider"],
                         model_version=m["model_version"], training_cutoff=m["training_cutoff"],
                         temperature=_opt(m.get("temperature")),
                         reasoning_effort=_opt(m.get("reasoning_effort")),
                         max_output_tokens=int(m["max_output_tokens"]),
                         usd_per_mtok_in=float(m["usd_per_mtok_in"]),
                         usd_per_mtok_out=float(m["usd_per_mtok_out"]))
                 for m in raw.get("llm", []))
    s = raw["study"]
    r = raw.get("retrieval")
    retrieval = None if r is None else RetrievalSpec(
        paired_with=str(r["paired_with"]), season_credit_cap=int(r["season_credit_cap"]),
        usd_per_credit=float(r["usd_per_credit"]))
    return StudyConfig(llms=llms, season_cap_usd=float(s["season_cap_usd"]),
                       home_rate=float(s["home_rate"]),
                       home_rate_seasons=str(s["home_rate_seasons"]),
                       est_input_tokens=int(s["est_input_tokens"]), retrieval=retrieval)
