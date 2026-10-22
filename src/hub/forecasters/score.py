"""The study's scoring: H1's encompassing test, H2's non-inferiority test, and the estimates.

`docs/prereg/llm-forecasters.md`, *Scoring and the confirmatory test*, implemented as written.
Every interval treats the NFL week as the unit, because games in one week share news shocks.

**H1 (confirmatory), the encompassing test** -- *Encompassing test (H1)*. For each line-aware
LLM, a logistic regression of the result on the betting market's log-odds and the model's
deviation from them,

    Pr(y = 1) = sigma(a + b * logit(m) + c * [logit(p) - logit(m)]),

with `c` the weight the deviations earn: 0 is noise, 1 is full weight. The standard error is a
leave-one-week-out jackknife, the reference is t on W - 1 degrees of freedom, the test is
one-sided `c > 0` at alpha = 0.05, and a Holm correction runs across the four models
(`h1_verdict`). What is reported per model is `c` with its one-sided 95% upper bound, *not*
adjusted for multiplicity -- a null reads "the deviations deserve at most X of full weight".
A model whose log-odds deviations have a standard deviation below the v0 threshold is "not
estimable (copies the betting market)" and leaves H1.

**H2 (secondary), non-inferiority on log loss** -- *Log-loss difference (H2)*, *Intervals*,
*H2, alpha = 0.05*. Within each week, the mean over that week's games of
`l(p, y) - l(m, y)` (positive is worse than the betting market), averaged across weeks, so the
estimand is what the interval estimates. A week with no game for a contestant is dropped for
that contestant. A 95% t interval on the weekly values with W - 1 degrees of freedom; the
one-sided test rejects `Delta <= -delta`; the claim "no LLM beats the betting market by more
than delta" is made only if every line-aware model rejects (an intersection-union test, so no
multiplicity correction; `h2_verdict`). TOST equivalence within +/- delta is reported beside
it, and a week-block bootstrap only as a robustness check. The prereg names Parity as the
tool that runs this test at a pinned version; it is not in this repository, so this module
is the hand computation a test of Parity's output would be checked against, and it refuses
while the Parity version is unpinned (a question for the maintainer, #452).

**Beside them, no tests.** Brier the same way as Delta, each contestant's mean absolute gap
from the betting market (so a pass that comes from copying the line is visible), each
contestant's missing rate, and the same Delta estimate for the blind models, the hub model
and the two baselines -- estimates and intervals only (*Blind models and the hub model get
estimates and intervals only, no tests*).

**The clip.** *Forecast procedure*, step 4: every probability is clipped to [0.01, 0.99]
before scoring. The runner already writes clipped values; it is applied again here so a
record carrying anything else is still scored under the prereg's rule, and it is applied to
the betting market's probability too (the prereg's "every probability"; flagged on #452 as a
reading the maintainer should confirm).

**Exclusions.** A tie is excluded from scoring; a game with no result yet is not scored; a
game with no forecast-time moneyline is left out of every comparison against the betting
market; a contestant record with no forecast (`p` null) or an exclusion (`excluded` set, e.g.
finished within 60 minutes of kickoff) is not a forecast. A game postponed into another week
counts in the week it is played: `Game.week` is the week played, and the caller supplies it.

**Refusal.** delta, the H1 threshold and the Parity version are fixed at `prereg-llm-v0` and
live in `conf/forecasters.toml`'s `[analysis]` table. While any is "TBD", `require_pinned`
raises `runner.NotPinned` and `main` exits 3 with one sentence -- the way
`runner.require_pinned` refuses a model call. Nothing here chooses them.
"""
from __future__ import annotations

import argparse
import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hub.declare import decision, not_an_input
from hub.forecasters import config as study_config
from hub.forecasters.config import UNSET, StudyConfig
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


# --- the pinned analysis settings --------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    """`[analysis]` in `conf/forecasters.toml`: None where the committed file says "TBD"."""
    delta: float | None                  # H2's margin, in nats per game
    h1_min_deviation_sd: float | None    # H1's estimability threshold, on log-odds deviations
    parity_version: str | None           # the Parity version or commit that runs H2


def _pinned(v: Any) -> Any:
    return None if v in (UNSET, "", None) else v


def load_analysis(path: Path | None = None) -> Analysis:
    raw = tomllib.loads((path or study_config.PATH).read_text()).get("analysis", {})
    delta, sd = _pinned(raw.get("delta")), _pinned(raw.get("h1_min_deviation_sd"))
    parity = _pinned(raw.get("parity_version"))
    return Analysis(delta=None if delta is None else float(delta),
                    h1_min_deviation_sd=None if sd is None else float(sd),
                    parity_version=None if parity is None else str(parity))


def require_pinned(analysis: Analysis, config: StudyConfig) -> None:
    """Refuse while delta, the H1 threshold or the Parity version is unset, or the study does
    not have the prereg's four models. The prereg fixes all of them at v0; none is chosen
    here."""
    gaps = []
    if analysis.delta is None or not analysis.delta > 0:
        gaps.append("analysis.delta")
    if analysis.h1_min_deviation_sd is None or not analysis.h1_min_deviation_sd > 0:
        gaps.append("analysis.h1_min_deviation_sd")
    if analysis.parity_version is None:
        gaps.append("analysis.parity_version")
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
    """The prereg's clip, [0.01, 0.99], from the study's config rather than restated."""
    return np.clip(np.asarray(p, dtype=float), StudyConfig.clip_lo, StudyConfig.clip_hi)


def logit(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    q = np.asarray(p, dtype=float)
    return np.log(q) - np.log1p(-q)


def load_forecasts(out_dir: Path) -> dict[str, dict[str, float]]:
    """Every scorable record under the `forecasts` checkout: one p per (contestant, game).
    A record with no forecast or with an exclusion is not one; a second record for the same
    pair breaks the runner's write-once rule and is refused."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for f in sorted(out_dir.rglob("*.jsonl")):
        if f.name == EXCLUSIONS:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("p") is None or r.get("excluded"):
                continue
            if r["game_id"] in out[r["contestant"]]:
                raise ValueError(f"{r['contestant']} has two forecasts for {r['game_id']}")
            out[r["contestant"]][r["game_id"]] = float(r["p"])
    return dict(out)


def load_games(path: Path) -> list[Game]:
    """One JSON object per line: game_id, week (played), home_points, away_points, market."""
    return [Game(game_id=str(r["game_id"]), week=int(r["week"]),
                 home_points=r.get("home_points"), away_points=r.get("away_points"),
                 market=r.get("market"))
            for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())]


@dataclass(frozen=True)
class Paired:
    """One contestant's games against the betting market: both forecast, result known, no
    tie. Clipped."""
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
                  p=clip([r[2] for r in rows]), m=clip([r[3] for r in rows]))


# --- the weekly statistic ----------------------------------------------------------------

@dataclass(frozen=True)
class WeeklyMean:
    """The mean of weekly means, its t interval on W - 1 df, and the week-block bootstrap's
    percentile interval beside it (robustness only)."""
    estimate: float
    se: float
    lo: float
    hi: float
    weeks: int
    games: int
    boot_lo: float
    boot_hi: float


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
    draws = np.random.default_rng(seed).choice(means, size=(bootstrap, n), replace=True)
    boot = draws.mean(axis=1)
    return WeeklyMean(est, se, lo, hi, n, int(v.size),
                      float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))


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
                 min_deviation_sd: float) -> Encompassing:
    """The encompassing regression with a leave-one-week-out jackknife standard error:
    `se^2 = (W - 1) / W * sum_w (c_(-w) - mean c_(-w))^2`, t = c / se on W - 1 df."""
    pr = paired(games, forecasts.get(contestant, {}))
    dev = logit(pr.p) - logit(pr.m)
    sd = float(dev.std(ddof=1)) if dev.size > 1 else float("nan")
    nan = float("nan")

    def none(status: str) -> Encompassing:
        return Encompassing(contestant, pr.games, pr.weeks, sd, status,
                            nan, nan, nan, nan, nan, nan, nan, ())
    if pr.games < 2:
        return none("fewer than two games against the betting market")
    if not sd >= min_deviation_sd:
        return none(NOT_ESTIMABLE)
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
    holm_p: float               # NaN for a model that left H1
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
    """One-sided `c > 0` at ALPHA with Holm across the models still in H1 -- those whose
    deviations clear the estimability threshold and whose fit exists. A model that left
    H1 is reported with its status and is not in the family."""
    tested = [f for f in fits if f.estimable and math.isfinite(f.p)]
    adj = dict(zip((f.contestant for f in tested), holm([f.p for f in tested]), strict=True))
    rows = tuple(H1Row(f, adj.get(f.contestant, float("nan")),
                       f.contestant in adj and adj[f.contestant] <= ALPHA) for f in fits)
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


def analyse(games: Sequence[Game], forecasts: Forecasts, *, analysis: Analysis,
            config: StudyConfig, seed: int = 0) -> Report:
    """The confirmatory and secondary analyses as pre-registered, plus the estimates."""
    require_pinned(analysis, config)
    delta, sd = analysis.delta, analysis.h1_min_deviation_sd
    assert delta is not None and sd is not None                 # require_pinned held
    aware = [f"{m.name}:{Condition.LINE_AWARE}" for m in config.llms]
    h1 = h1_verdict([encompassing(games, forecasts, c, min_deviation_sd=sd) for c in aware])
    h2 = h2_verdict([noninferiority(describe(games, forecasts, c, seed=seed), delta)
                     for c in aware])
    estimates = tuple(describe(games, forecasts, c, seed=seed) for c in sorted(forecasts))
    excluded = {
        "tie": sum(1 for g in games if g.home_points is not None
                   and g.home_points == g.away_points),
        "no_result": sum(1 for g in games if g.home_points is None or g.away_points is None),
        "no_moneyline": sum(1 for g in games if g.outcome is not None and g.market is None),
    }
    return Report(h1, h2, estimates, excluded)


def _num(x: float) -> float | None:
    return round(x, 6) if math.isfinite(x) else None


def summary(r: Report) -> dict[str, Any]:
    """The numbers, not the inputs: every one with its n and interval."""
    def wm(w: WeeklyMean) -> dict[str, Any]:
        return {"estimate": _num(w.estimate), "lo": _num(w.lo), "hi": _num(w.hi),
                "weeks": w.weeks, "games": w.games,
                "bootstrap_lo": _num(w.boot_lo), "bootstrap_hi": _num(w.boot_hi)}
    return {
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
         config: StudyConfig | None = None, analysis: Analysis | None = None) -> int:
    """Exit codes: 0 ok, 2 an input that cannot be read, 3 the analysis is not pinned."""
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.score")
    ap.add_argument("--forecasts", type=Path, required=True, help="the forecasts checkout")
    ap.add_argument("--games", type=Path, required=True,
                    help="JSONL: game_id, week played, home_points, away_points, market")
    ap.add_argument("--seed", type=int, default=0, help="the week-block bootstrap's seed")
    args = ap.parse_args(argv)
    cfg = config or study_config.load()
    an = analysis or load_analysis()
    try:
        require_pinned(an, cfg)
    except NotPinned as exc:
        echo(f"refused: {exc}")
        return 3
    if not args.games.is_file() or not args.forecasts.is_dir():
        echo(f"refused: cannot read {args.games} and {args.forecasts}")
        return 2
    report = analyse(load_games(args.games), load_forecasts(args.forecasts), analysis=an,
                     config=cfg, seed=args.seed)
    echo(json.dumps(summary(report), sort_keys=True))
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())

