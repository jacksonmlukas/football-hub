"""The study's scoring: H1's encompassing test, H2's non-inferiority test, and the estimates.

`docs/prereg/llm-forecasters.md`, *Scoring and the confirmatory test*, implemented as written.
Every interval treats the NFL week as the unit, because games in one week share news shocks.

**H1 (confirmatory), the encompassing test** -- *Encompassing test (H1)*. For each line-aware
LLM, a logistic regression of the result on the betting market's log-odds and the model's
deviation from them,

    Pr(y = 1) = sigma(a + b * logit(m) + c * [logit(p) - logit(m)]),

with `c` the weight the deviations earn: 0 is noise, 1 is full weight. The standard error is a
leave-one-week-out jackknife, the reference is t on W - 1 degrees of freedom, the test is
one-sided `c > 0` at alpha = 0.05, and a Holm correction runs across the models in the H1
family (`h1_verdict`). What is reported per model is `c` with its one-sided 95% upper bound,
*not* adjusted for multiplicity -- a null reads "the deviations deserve at most X of full
weight".

**The H1 family is fixed at the burn-in** (#448, adopted 2026-10-22, item 3). A model whose
log-odds deviations from the betting market have a standard deviation below the v0 threshold
*on the burn-in week* is "not estimable (copies the betting market)" and leaves H1; the rest
are the Holm family, fixed before any scored game. The burn-in report (#451) writes that
measurement to `BURN_IN` (`docs/prereg/llm-forecasters.burn-in.json`, committed with
`prereg-llm-v1`), and this module reads it rather than re-measuring on the scored games --
which would let the outcomes' sample choose the family the outcomes are then tested in.
Scoring refuses, in one sentence, while that record is absent, was measured against a
different threshold than the pin, or does not name every line-aware model. A family member
whose fit fails on the scored games stays in the family with p = 1, so Holm's m never shrinks
after the fact.

**H2 (secondary), non-inferiority on log loss** -- *Log-loss difference (H2)*, *Intervals*,
*H2, alpha = 0.05*. Within each week, the mean over that week's games of
`l(p, y) - l(m, y)` (positive is worse than the betting market), averaged across weeks, so the
estimand is what the interval estimates. A week with no game for a contestant is dropped for
that contestant. A 95% t interval on the weekly values with W - 1 degrees of freedom; the
one-sided test rejects `Delta <= -delta`; the claim "no LLM beats the betting market by more
than delta" is made only if every line-aware model rejects (an intersection-union test, so no
multiplicity correction; `h2_verdict`). The matching 90% two-sided t interval -- the one
whose lower end is the one-sided 0.05 test's bound -- is reported beside the 95% (#448 item
4). TOST equivalence within +/- delta is reported beside it, and a week-block bootstrap only
as a robustness check. This module *is* the H2 implementation: it replaced the prereg's
"Parity" (#448, adopted 2026-10-22, item 1), pinned as `[analysis].h2_implementation =
"hub.forecasters.score@<commit>"`, and it refuses while that pin is unset.

**Beside them, no tests.** Brier the same way as Delta, each contestant's mean absolute gap
from the betting market (so a pass that comes from copying the line is visible), each
contestant's missing rate, and the same Delta estimate for the blind models, the hub model
and the two baselines -- estimates and intervals only (*Blind models and the hub model get
estimates and intervals only, no tests*).

**The clip.** *Forecast procedure*, step 4, as restated 2026-10-22 (#448 item 2): every
*contestant's* probability is clipped to [0.01, 0.99] before scoring. The runner already
writes clipped values; it is applied again here so a record carrying anything else is still
scored under the rule. The betting market's probability is the benchmark, not a contestant,
and is scored as de-vigged -- never clipped.

**Inputs** (#448 item 10). Each forecast record carries the forecast-time de-vigged betting
market probability from its run's snapshot (`market_home_win_prob`, written by the runner),
and the results come from nflverse's schedule cache (`hub.fetch.nflverse`, the frame the
forecast job already refreshes), so no hand-built games file is needed: the scored games are
every regular-season game of the burn-in record's season from the week after the burn-in
through Week 18, less those the runner logged as refused. `--games` stays as an override.

**Exclusions.** A tie is excluded from scoring; a game with no result yet is not scored; a
game with no forecast-time moneyline is left out of every comparison against the betting
market; a contestant record with no forecast (`p` null) or an exclusion (`excluded` set, e.g.
finished within 60 minutes of kickoff) is not a forecast. A game postponed into another week
counts in the week it is played: `Game.week` is the week played, and the caller supplies it.

**Refusal.** delta, the H1 threshold and the H2 implementation pin are fixed at
`prereg-llm-v0` and live in `conf/forecasters.toml`'s `[analysis]` table. While any is "TBD",
`require_pinned` raises `runner.NotPinned` and `main` exits 3 with one sentence -- the way
`runner.require_pinned` refuses a model call (which also refuses while delta or the H1
threshold is unset). Without the burn-in record `main` exits 4, also in one sentence.
Nothing here chooses them.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from hub.declare import decision, not_an_input
from hub.forecasters import config as study_config
from hub.forecasters.config import Analysis, StudyConfig, analysis_gaps
from hub.forecasters.packet import Condition
from hub.forecasters.runner import EXCLUSIONS, NotPinned
from hub.models import experiment

# The prereg's size for both tests: H1 one-sided `c > 0` at 0.05 with Holm across the four
# models, H2 one-sided non-inferiority at 0.05. Its own name rather than `experiment.ALPHA`,
# which is the walk-forward protocol's two-sided sizing pair.
ALPHA = not_an_input(
    0.05,
    "the prereg's one-sided size for H1 and H2, fixed in docs/prereg/llm-forecasters.md; "
    "a test's size, which no forecast reads")

# Newton-Raphson for the encompassing regression. A step this small is converged; a linear
# predictor this large on any game is separation (a probability within e^-30 of 0 or 1),
# where the maximum-likelihood `c` does not exist and the fit is reported as failed.
_NEWTON_TOL = not_an_input(
    1e-10, "the Newton step size at which the encompassing fit is converged; numerics only")
_ETA_LIMIT = not_an_input(
    30.0, "a linear predictor past which the encompassing fit is read as separated; numerics")
_NEWTON_MAX_ITER = 100

NOT_ESTIMABLE = "not estimable (copies the betting market)"


# The last regular-season week, the end of the prereg's scoring window (*Window*).
LAST_WEEK = 18

# The burn-in report's estimability record (#451 writes it; #448 item 3), committed beside
# the prereg so `prereg-llm-v1` pins it before the first scored week.
BURN_IN = Path(__file__).resolve().parents[3] / "docs" / "prereg" / "llm-forecasters.burn-in.json"


# --- the pinned analysis settings --------------------------------------------------------

def load_analysis(path: Path | None = None) -> Analysis:
    """`[analysis]` alone, from `conf/forecasters.toml` or `path`."""
    return study_config.load_analysis(path)


def require_pinned(analysis: Analysis, config: StudyConfig) -> None:
    """Refuse while delta, the H1 threshold or the H2 implementation pin is unset (the pin
    must read `hub.forecasters.score@<commit>`), or the study does not have the prereg's four
    models. The prereg fixes all of them at v0; none is chosen here."""
    gaps = analysis_gaps(analysis, implementation=True)
    if len(config.llms) != 4:
        gaps.append(f"study.llms ({len(config.llms)}, the prereg fixes four)")
    if gaps:
        raise NotPinned("conf/forecasters.toml is not pinned for scoring: " + ", ".join(gaps))


# --- inputs ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Game:
    """One game as scored: the week it was *played*, its result, and the forecast-time
    de-vigged home probability from the run's snapshot (None with no moneyline)."""
    game_id: str
    week: int
    home_points: int | None
    away_points: int | None
    market: float | None

    @property
    def outcome(self) -> int | None:
        """1 for a home win, 0 for an away win; None for a tie or a game not yet played."""
        if self.home_points is None or self.away_points is None:
            return None
        if self.home_points == self.away_points:
            return None
        return int(self.home_points > self.away_points)


Forecasts = Mapping[str, Mapping[str, float]]          # contestant -> game_id -> p


def clip(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The prereg's clip, [0.01, 0.99], from the study's config rather than restated. For a
    contestant's probability only: the betting market's is scored as de-vigged."""
    return np.clip(np.asarray(p, dtype=float), StudyConfig.clip_lo, StudyConfig.clip_hi)


def logit(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    q = np.asarray(p, dtype=float)
    return np.log(q) - np.log1p(-q)


def _records(out_dir: Path) -> Iterator[dict[str, Any]]:
    for f in sorted(out_dir.rglob("*.jsonl")):
        if f.name == EXCLUSIONS:
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def refused_games(out_dir: Path) -> set[str]:
    """Games the runner logged as refused: forecast by no one, so not scored."""
    path = out_dir / EXCLUSIONS
    if not path.is_file():
        return set()
    return {json.loads(x)["game_id"] for x in path.read_text().splitlines() if x.strip()}


def load_forecasts(out_dir: Path) -> dict[str, dict[str, float]]:
    """Every scorable record under the `forecasts` checkout: one p per (contestant, game).
    A record with no forecast or with an exclusion is not one; a second record for the same
    pair breaks the runner's write-once rule and is refused."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in _records(out_dir):
        if r.get("p") is None or r.get("excluded"):
            continue
        if r["game_id"] in out[r["contestant"]]:
            raise ValueError(f"{r['contestant']} has two forecasts for {r['game_id']}")
        out[r["contestant"]][r["game_id"]] = float(r["p"])
    return dict(out)


@dataclass(frozen=True)
class Markets:
    """Each game's forecast-time de-vigged home probability, read off its records.

    Every record of a run carries that run's snapshot, so one game normally has one. A run
    that crashed after its snapshot and was retried can leave a game's records on two: the
    benchmark is then the *earliest* run's snapshot (the first the study committed to), and
    the game is counted in `conflicts` so the report says how often that happened."""
    home: dict[str, float]
    conflicts: int


def load_markets(out_dir: Path) -> Markets:
    seen: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)   # game -> snap -> first
    for r in _records(out_dir):
        m = r.get("market_home_win_prob")
        if m is None:
            continue
        snap, start = str(r.get("market_snapshot_id")), str(r.get("run_start"))
        prior = seen[r["game_id"]].get(snap)
        if prior is None or start < prior[0]:
            seen[r["game_id"]][snap] = (start, float(m))
    home = {g: min(by.values())[1] for g, by in seen.items()}
    return Markets(home, sum(1 for by in seen.values() if len(by) > 1))


def games_from_schedule(frame: pl.DataFrame, markets: Mapping[str, float], *, season: int,
                        first_week: int, refused: AbstractSet[str] = frozenset()) -> list[Game]:
    """nflverse's schedule as the scored games: regular season, `season`, `first_week`
    through Week 18, by the week each game was *played* (nflverse moves a postponed game to
    the week it is played), less the games the runner refused. A game every contestant missed
    stays in, so it counts in every missing rate."""
    out = []
    for r in frame.select("game_id", "season", "game_type", "week", "home_score",
                          "away_score").iter_rows(named=True):
        if (r["game_type"] != "REG" or int(r["season"]) != season
                or not first_week <= int(r["week"]) <= LAST_WEEK or r["game_id"] in refused):
            continue
        hs, as_ = r["home_score"], r["away_score"]
        out.append(Game(game_id=str(r["game_id"]), week=int(r["week"]),
                        home_points=None if hs is None else int(hs),
                        away_points=None if as_ is None else int(as_),
                        market=markets.get(str(r["game_id"]))))
    return out


def cached_schedule(season: int) -> pl.DataFrame:                 # pragma: no cover - cache
    """The hub's nflverse schedule cache (the forecast job refreshes it every run)."""
    from hub.fetch import nflverse
    return nflverse.load("schedules", [season])


# --- the burn-in estimability record -----------------------------------------------------

class BurnInRefused(RuntimeError):
    """The burn-in estimability record is absent or does not fit the pinned analysis."""


@dataclass(frozen=True)
class BurnIn:
    """What the burn-in report (#451) measured: per line-aware model, the SD of its log-odds
    deviations from the betting market on the burn-in week, against `threshold`. Outcomes
    play no part in it."""
    season: int
    week: int
    threshold: float
    deviation_sd: Mapping[str, float]

    @property
    def first_scored_week(self) -> int:
        return self.week + 1                 # *Window*: the week after the burn-in


def load_burn_in(path: Path | None = None) -> BurnIn:
    """`{"season", "burn_in_week", "h1_min_deviation_sd", "deviation_sd": {contestant: sd}}`."""
    path = path or BURN_IN
    if not path.is_file():
        raise BurnInRefused(f"no burn-in estimability record at {path}: the H1 family is fixed "
                            f"by the burn-in report (#451) before any game is scored")
    raw = json.loads(path.read_text())
    return BurnIn(season=int(raw["season"]), week=int(raw["burn_in_week"]),
                  threshold=float(raw["h1_min_deviation_sd"]),
                  deviation_sd={str(k): float(v) for k, v in raw["deviation_sd"].items()})


def h1_family(burn_in: BurnIn, contestants: Sequence[str], threshold: float) -> frozenset[str]:
    """The Holm family: the line-aware models whose burn-in deviations clear the pinned
    threshold. Refused if the record was measured against another threshold or leaves a
    model out -- either would make the family a choice made here rather than at the burn-in."""
    if not math.isclose(burn_in.threshold, threshold):
        raise BurnInRefused(f"the burn-in record was measured against threshold "
                            f"{burn_in.threshold:g}, but the pin is {threshold:g}")
    missing = [c for c in contestants if c not in burn_in.deviation_sd]
    if missing:
        raise BurnInRefused(f"the burn-in record does not name {', '.join(missing)}")
    return frozenset(c for c in contestants if burn_in.deviation_sd[c] >= threshold)


def load_games(path: Path) -> list[Game]:
    """One JSON object per line: game_id, week (played), home_points, away_points, market."""
    return [Game(game_id=str(r["game_id"]), week=int(r["week"]),
                 home_points=r.get("home_points"), away_points=r.get("away_points"),
                 market=r.get("market"))
            for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())]


@dataclass(frozen=True)
class Paired:
    """One contestant's games against the betting market: both forecast, result known, no
    tie. The contestant's `p` clipped; the betting market's `m` as de-vigged."""
    week: npt.NDArray[np.int64]
    y: npt.NDArray[np.float64]
    p: npt.NDArray[np.float64]
    m: npt.NDArray[np.float64]

    @property
    def games(self) -> int:
        return int(self.y.size)

    @property
    def weeks(self) -> int:
        return int(np.unique(self.week).size)


def paired(games: Sequence[Game], forecasts: Mapping[str, float]) -> Paired:
    rows = [(g.week, g.outcome, forecasts[g.game_id], g.market) for g in games
            if g.outcome is not None and g.market is not None and g.game_id in forecasts]
    return Paired(week=np.array([r[0] for r in rows], dtype=np.int64),
                  y=np.array([r[1] for r in rows], dtype=float),
                  p=clip([r[2] for r in rows]), m=np.array([r[3] for r in rows], dtype=float))


# --- the weekly statistic ----------------------------------------------------------------

@dataclass(frozen=True)
class WeeklyMean:
    """The mean of weekly means, its 95% t interval on W - 1 df and the matching 90% one
    (whose ends are the one-sided 0.05 bounds H2 tests against), and the week-block
    bootstrap's percentile interval beside them (robustness only)."""
    estimate: float
    se: float
    lo: float
    hi: float
    weeks: int
    games: int
    boot_lo: float
    boot_hi: float
    lo90: float = float("nan")
    hi90: float = float("nan")


def weekly_mean(values: npt.ArrayLike, week: npt.ArrayLike, *, seed: int = 0,
                bootstrap: int = experiment.BOOTSTRAP) -> WeeklyMean:
    v, w = np.asarray(values, dtype=float), np.asarray(week)
    weeks = np.unique(w)
    means = np.array([v[w == k].mean() for k in weeks])
    n = int(means.size)
    if n == 0:
        nan = float("nan")
        return WeeklyMean(nan, nan, nan, nan, 0, 0, nan, nan)
    est = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = experiment.t_interval(est, se, n)
    half90 = (experiment.t_quantile(1.0 - ALPHA, n - 1) * se
              if n > 1 and math.isfinite(se) else float("nan"))
    draws = np.random.default_rng(seed).choice(means, size=(bootstrap, n), replace=True)
    boot = draws.mean(axis=1)
    return WeeklyMean(est, se, lo, hi, n, int(v.size),
                      float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)),
                      est - half90, est + half90)


def _upper_tail(t: float, df: int) -> float:
    """`P(T >= t)` on `df` degrees of freedom, from `experiment.two_sided_p`."""
    if df < 1 or not math.isfinite(t):
        return float("nan")
    half = experiment.two_sided_p(t, df) / 2.0
    return half if t >= 0 else 1.0 - half


def log_loss_terms(p: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    q, yy = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return -(yy * np.log(q) + (1.0 - yy) * np.log(1.0 - q))


# --- H2 and the estimates beside it ------------------------------------------------------

@dataclass(frozen=True)
class Scored:
    """One contestant against the betting market: Delta log loss, Delta Brier, and the mean
    absolute gap, each weekly; plus the missing rate over games with a result."""
    contestant: str
    log_loss: WeeklyMean
    brier: WeeklyMean
    abs_gap: WeeklyMean
    missing: int
    scorable: int


def describe(games: Sequence[Game], forecasts: Forecasts, contestant: str, *,
             seed: int = 0) -> Scored:
    mine = forecasts.get(contestant, {})
    pr = paired(games, mine)
    ll = log_loss_terms(pr.p, pr.y) - log_loss_terms(pr.m, pr.y)
    br = (pr.p - pr.y) ** 2 - (pr.m - pr.y) ** 2
    with_result = [g for g in games if g.outcome is not None]
    return Scored(contestant,
                  weekly_mean(ll, pr.week, seed=seed),
                  weekly_mean(br, pr.week, seed=seed),
                  weekly_mean(np.abs(pr.p - pr.m), pr.week, seed=seed),
                  missing=sum(1 for g in with_result if g.game_id not in mine),
                  scorable=len(with_result))


@dataclass(frozen=True)
class NonInferiority:
    """H2 for one line-aware model at margin delta, and TOST equivalence beside it."""
    scored: Scored
    delta: float
    t: float                    # (Delta + delta) / se
    p: float                    # one-sided, for H0: Delta <= -delta
    noninferior: bool           # rejects Delta <= -delta at ALPHA
    p_upper: float              # one-sided, for H0: Delta >= +delta (TOST's other half)
    equivalent: bool            # both halves reject: within +/- delta


def noninferiority(scored: Scored, delta: float) -> NonInferiority:
    ll = scored.log_loss
    df = ll.weeks - 1
    if df < 1 or not (math.isfinite(ll.se) and ll.se > 0):
        nan = float("nan")
        return NonInferiority(scored, delta, nan, nan, False, nan, False)
    crit = experiment.t_quantile(1.0 - ALPHA, df)
    t_lo = (ll.estimate + delta) / ll.se
    t_hi = (ll.estimate - delta) / ll.se
    return NonInferiority(scored, delta, t_lo, _upper_tail(t_lo, df), t_lo > crit,
                          1.0 - _upper_tail(t_hi, df), t_lo > crit and t_hi < -crit)


@dataclass(frozen=True)
class H2Result:
    rows: tuple[NonInferiority, ...]
    claim: bool                 # every line-aware model rejected Delta <= -delta
    sentence: str


@decision
def h2_verdict(rows: Sequence[NonInferiority]) -> H2Result:
    """The intersection-union claim: made only if every model still in H2 rejects on its
    own (*H2, alpha = 0.05*). No model in the family is no claim."""
    claim = bool(rows) and all(r.noninferior for r in rows)
    failed = [r.scored.contestant for r in rows if not r.noninferior]
    if not rows:
        sentence = "H2: no line-aware model to test; no claim"
    elif claim:
        sentence = (f"H2: no line-aware LLM beats the betting market by more than "
                    f"{rows[0].delta:g} nats a game (all {len(rows)} reject)")
    else:
        sentence = f"H2: not shown; {', '.join(failed)} did not reject Delta <= -delta"
    return H2Result(tuple(rows), claim, sentence)


# --- H1 ----------------------------------------------------------------------------------

def logistic_fit(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
                 ) -> npt.NDArray[np.float64] | None:
    """Maximum likelihood by Newton-Raphson; None if it does not converge or separates."""
    beta = np.zeros(x.shape[1])
    for _ in range(_NEWTON_MAX_ITER):
        eta = x @ beta
        if not np.all(np.isfinite(eta)) or np.max(np.abs(eta)) > _ETA_LIMIT:
            return None
        mu = 1.0 / (1.0 + np.exp(-eta))
        hess = x.T @ (x * (mu * (1.0 - mu))[:, None])
        try:
            step = np.linalg.solve(hess, x.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < _NEWTON_TOL:
            return beta if np.max(np.abs(x @ beta)) <= _ETA_LIMIT else None
    return None


def _design(pr: Paired, rows: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    lm = logit(pr.m[rows])
    return np.column_stack([np.ones(lm.size), lm, logit(pr.p[rows]) - lm])


@dataclass(frozen=True)
class Encompassing:
    """H1 for one line-aware model. `upper` is the unadjusted one-sided 95% upper bound."""
    contestant: str
    games: int
    weeks: int
    deviation_sd: float
    status: str                 # "estimated", NOT_ESTIMABLE, or why no estimate exists
    a: float
    b: float
    c: float
    se: float
    t: float
    p: float
    upper: float
    jackknife: tuple[float, ...]

    @property
    def estimable(self) -> bool:
        return self.status == "estimated"


def encompassing(games: Sequence[Game], forecasts: Forecasts, contestant: str, *,
                 in_family: bool) -> Encompassing:
    """The encompassing regression with a leave-one-week-out jackknife standard error:
    `se^2 = (W - 1) / W * sum_w (c_(-w) - mean c_(-w))^2`, t = c / se on W - 1 df.

    `in_family` is the burn-in's verdict (`h1_family`), not a measurement on these games: a
    model outside the family is NOT_ESTIMABLE whatever its scored deviations look like, and
    one inside is fitted however small they turn out. `deviation_sd` is the scored games'
    figure, reported beside it."""
    pr = paired(games, forecasts.get(contestant, {}))
    dev = logit(pr.p) - logit(pr.m)
    sd = float(dev.std(ddof=1)) if dev.size > 1 else float("nan")
    nan = float("nan")

    def none(status: str) -> Encompassing:
        return Encompassing(contestant, pr.games, pr.weeks, sd, status,
                            nan, nan, nan, nan, nan, nan, nan, ())
    if not in_family:
        return none(NOT_ESTIMABLE)
    if pr.games < 2:
        return none("fewer than two games against the betting market")
    if pr.weeks < 2:
        return none("fewer than two weeks: no jackknife")
    full = logistic_fit(_design(pr, np.ones(pr.games, dtype=bool)), pr.y)
    if full is None:
        return none("the encompassing fit did not converge")
    jack: list[float] = []
    for wk in np.unique(pr.week):
        keep = pr.week != wk
        fit = logistic_fit(_design(pr, keep), pr.y[keep])
        if fit is None:
            return none(f"the fit leaving out week {wk} did not converge")
        jack.append(float(fit[2]))
    w = len(jack)
    cj = np.array(jack)
    se = float(math.sqrt((w - 1) / w * float(np.sum((cj - cj.mean()) ** 2))))
    c = float(full[2])
    t = c / se if se > 0 else float("nan")
    upper = c + experiment.t_quantile(1.0 - ALPHA, w - 1) * se
    return Encompassing(contestant, pr.games, pr.weeks, sd, "estimated", float(full[0]),
                        float(full[1]), c, se, t, _upper_tail(t, w - 1), upper, tuple(jack))


@dataclass(frozen=True)
class H1Row:
    fit: Encompassing
    holm_p: float               # NaN for a model outside the family
    detected: bool              # rejects c <= 0 after Holm


@dataclass(frozen=True)
class H1Result:
    rows: tuple[H1Row, ...]
    sentence: str


def holm(p: Sequence[float]) -> list[float]:
    """Holm-adjusted p values, in the input's order: the step-down maximum of
    `(m - i + 1) * p_(i)`, capped at 1."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


@decision
def h1_verdict(fits: Sequence[Encompassing]) -> H1Result:
    """One-sided `c > 0` at ALPHA with Holm across the H1 family -- every model the burn-in
    found estimable (any status but NOT_ESTIMABLE). A family member whose fit does not exist
    on the scored games enters Holm with p = 1, so the family's size is the burn-in's, not
    what survived the fits. A model outside the family is reported with its status."""
    family = [f for f in fits if f.status != NOT_ESTIMABLE]
    tested = [f for f in family if f.estimable and math.isfinite(f.p)]
    ps = [f.p if f.estimable and math.isfinite(f.p) else 1.0 for f in family]
    adj = dict(zip((f.contestant for f in family), holm(ps), strict=True))
    rows = tuple(H1Row(f, adj.get(f.contestant, float("nan")),
                       f in tested and adj[f.contestant] <= ALPHA) for f in fits)
    found = [r.fit.contestant for r in rows if r.detected]
    if not tested:
        sentence = "H1: no line-aware model is estimable; no test"
    elif found:
        sentence = f"H1: deviations carry information for {', '.join(found)} (Holm, one-sided)"
    else:
        bounds = ", ".join(f"{f.contestant} c <= {f.upper:.2f}" for f in tested)
        sentence = f"H1: no detectable added information; at most {bounds} of full weight"
    return H1Result(rows, sentence)


# --- the whole analysis ------------------------------------------------------------------

@dataclass(frozen=True)
class Report:
    h1: H1Result
    h2: H2Result
    estimates: tuple[Scored, ...]                     # every contestant, no tests
    excluded: dict[str, int]
    h2_implementation: str | None = None
    burn_in: BurnIn | None = None


def analyse(games: Sequence[Game], forecasts: Forecasts, *, analysis: Analysis,
            config: StudyConfig, burn_in: BurnIn, seed: int = 0) -> Report:
    """The confirmatory and secondary analyses as pre-registered, plus the estimates."""
    require_pinned(analysis, config)
    delta, sd = analysis.delta, analysis.h1_min_deviation_sd
    assert delta is not None and sd is not None                 # require_pinned held
    aware = [f"{m.name}:{Condition.LINE_AWARE}" for m in config.llms]
    family = h1_family(burn_in, aware, sd)
    h1 = h1_verdict([encompassing(games, forecasts, c, in_family=c in family) for c in aware])
    h2 = h2_verdict([noninferiority(describe(games, forecasts, c, seed=seed), delta)
                     for c in aware])
    estimates = tuple(describe(games, forecasts, c, seed=seed) for c in sorted(forecasts))
    excluded = {
        "tie": sum(1 for g in games if g.home_points is not None
                   and g.home_points == g.away_points),
        "no_result": sum(1 for g in games if g.home_points is None or g.away_points is None),
        "no_moneyline": sum(1 for g in games if g.outcome is not None and g.market is None),
    }
    return Report(h1, h2, estimates, excluded, analysis.h2_implementation, burn_in)


def _num(x: float) -> float | None:
    return round(x, 6) if math.isfinite(x) else None


def summary(r: Report) -> dict[str, Any]:
    """The numbers, not the inputs: every one with its n and interval."""
    def wm(w: WeeklyMean) -> dict[str, Any]:
        return {"estimate": _num(w.estimate), "lo": _num(w.lo), "hi": _num(w.hi),
                "lo_90": _num(w.lo90), "hi_90": _num(w.hi90), "weeks": w.weeks, "games": w.games,
                "bootstrap_lo": _num(w.boot_lo), "bootstrap_hi": _num(w.boot_hi)}
    b = r.burn_in
    return {
        "h2_implementation": r.h2_implementation,
        "burn_in": None if b is None else {"season": b.season, "week": b.week,
                                           "threshold": b.threshold},
        "h1": {"sentence": r.h1.sentence, "models": [
            {"contestant": x.fit.contestant, "status": x.fit.status, "games": x.fit.games,
             "weeks": x.fit.weeks, "deviation_sd": _num(x.fit.deviation_sd),
             "c": _num(x.fit.c), "se": _num(x.fit.se), "upper_95": _num(x.fit.upper),
             "p": _num(x.fit.p), "holm_p": _num(x.holm_p), "detected": x.detected}
            for x in r.h1.rows]},
        "h2": {"sentence": r.h2.sentence, "claim": r.h2.claim, "models": [
            {"contestant": x.scored.contestant, "delta": x.delta,
             "log_loss_diff": wm(x.scored.log_loss), "p_noninferior": _num(x.p),
             "noninferior": x.noninferior, "p_equivalence_upper": _num(x.p_upper),
             "equivalent": x.equivalent, "abs_gap": wm(x.scored.abs_gap)}
            for x in r.h2.rows]},
        "estimates": [{"contestant": s.contestant, "log_loss_diff": wm(s.log_loss),
                       "brier_diff": wm(s.brier), "abs_gap": wm(s.abs_gap),
                       "missing": s.missing, "scorable": s.scorable} for s in r.estimates],
        "excluded": r.excluded,
    }


def main(argv: Sequence[str] | None = None, *, echo: Callable[[str], object] = print,
         config: StudyConfig | None = None, analysis: Analysis | None = None,
         schedule: Callable[[int], pl.DataFrame] = cached_schedule) -> int:
    """Exit codes: 0 ok, 2 an input that cannot be read, 3 the analysis is not pinned, 4 the
    burn-in estimability record is absent or does not fit the pin."""
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.score")
    ap.add_argument("--forecasts", type=Path, required=True, help="the forecasts checkout")
    ap.add_argument("--games", type=Path, default=None,
                    help="override: JSONL of game_id, week played, home_points, away_points, "
                         "market (default: nflverse's schedule cache and the records' market)")
    ap.add_argument("--burn-in", type=Path, default=None,
                    help=f"the burn-in estimability record (default: {BURN_IN.name})")
    ap.add_argument("--seed", type=int, default=0, help="the week-block bootstrap's seed")
    args = ap.parse_args(argv)
    cfg = config or study_config.load()
    an = analysis or cfg.analysis
    try:
        require_pinned(an, cfg)
    except NotPinned as exc:
        echo(f"refused: {exc}")
        return 3
    if not args.forecasts.is_dir() or (args.games is not None and not args.games.is_file()):
        echo(f"refused: cannot read {args.forecasts}"
             + ("" if args.games is None else f" and {args.games}"))
        return 2
    try:
        burn_in = load_burn_in(args.burn_in)
        h1_family(burn_in, [f"{m.name}:{Condition.LINE_AWARE}" for m in cfg.llms],
                  float(an.h1_min_deviation_sd or 0.0))
    except BurnInRefused as exc:
        echo(f"refused: {exc}")
        return 4
    markets = load_markets(args.forecasts)
    refused = refused_games(args.forecasts)
    games = (load_games(args.games) if args.games is not None else
             games_from_schedule(schedule(burn_in.season), markets.home, season=burn_in.season,
                                 first_week=burn_in.first_scored_week, refused=refused))
    report = analyse(games, load_forecasts(args.forecasts), analysis=an, config=cfg,
                     burn_in=burn_in, seed=args.seed)
    out = summary(report)
    out["excluded"]["packet_refused"] = len(refused)
    out["market_snapshot_conflicts"] = markets.conflicts
    echo(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())

