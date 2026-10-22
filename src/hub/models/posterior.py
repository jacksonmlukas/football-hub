"""The posterior over a gate's pooled effect, printed beside its verdict word (#375).

`docs/gate-power.md`, *Pre-registered 2026-10-18 -- ADOPTED: the posterior gate's constants,
decision rule and engine (#375)*, is the design; this module is its arithmetic and nothing
else. The model is the hierarchical normal

    gain_s  ~ Normal(delta_s, se_s)      # one per held-out season, off `per_season`
    delta_s ~ Normal(delta, tau)
    delta   ~ Normal(0, prior_sd)
    tau     ~ HalfNormal(tau_scale)

and the question it answers is the one a sign-and-interval word cannot: given these seasons,
what do we believe about the pooled effect `delta`, and which action does that belief's
expected utility recommend?

**The engine is quadrature, not sampling.** Given `tau`, `delta` is conjugate, and the
marginal likelihood of the k gains given `tau` is a k-variate normal whose determinant and
quadratic form close in O(k) (the matrix determinant lemma). So the posterior is a finite
mixture of normals over a grid on `tau`, and every number reported -- the mean, `P(delta > 0)`,
the interval -- is exact up to that grid. numpy only: no JAX, no compiled model, no chain to
diagnose, and deterministic, so `experiment.gate` stays pure. The pre-registration says why
this replaced the NumPyro extra the 2026-09-21 section had adopted.

**What decides is still the verdict word.** This is the transition's first landing: the
posterior's action is printed beside `ADOPT`/`REMOVE`/`SHOW`, never instead of it, until a
dated decision retires the word. Nothing here writes a ledger row or moves a published number.

Imported by `hub.models.experiment` alone. That module is `CLOSURE_EXEMPT` and the closure
walker does not descend into it, so this module is in no gate's arm closure, no ledger
`code_digest` and not the weekly forward pin (#456).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hub.declare import chosen, decision, not_an_input

# The frozen scale per gate: the sample sd (ddof=1) of the season gains in that gate's latest
# `state/gate-width.json` entry as of the pre-registration's commit (base d819fb9), 3 s.f.,
# each in the gate's own `Harness.unit`. `prior_sd = tau_scale = S_REF[gate]`. Frozen, so no
# run's prior moves with the data that run fits; data-informed all the same, and named so in
# the pre-registration. A gate absent here has no posterior, and its report says so.
S_REF: dict[str, float] = chosen({
    "draft": 4.54,
    "weekly": 0.648,
    "interval_shape": 0.00808,
    "player_spread": 0.0302,
    "injury_type": 0.0454,
    "component_calibration": 0.0706,
    "injury_retention": 0.125,
    "favourite_longshot": 0.00302,
    "state_space_rating": 0.0146,
    "td_dispersion": 0.00155,
})

# The gates whose action is a constant or refit swap inside a shipped module: their switching
# cost is zero and the posterior mean's sign is the action. Every other gate is costly.
REVERSIBLE: frozenset[str] = chosen(frozenset({"td_dispersion", "injury_type"}))

# The costly tier's bar: ADOPT above it, REMOVE below `1 - P_COSTLY`, SHOW between. The
# expected-utility action when a wrong switch costs nineteen times what a right one gains.
# 0.9 -> 0.95, ADOPTED 2026-10-21: 0.9's null ADOPT rate was 7-10% under the heterogeneity
# S_REF implies; 0.95's is 4.7% at the worst cell, and it clears the k = 3 saturation
# (0.9585) that any bar at or above it could never reach (docs/gate-power.md, #375).
P_COSTLY = chosen(0.95)

# The reversible tier's switching cost, in each gate's own unit: ADOPT when `E[delta] > it`,
# REMOVE when `< -it`. Zero is the linear-utility action at no friction (#364's form), as
# pre-registered; a gate whose swap has a real cost belongs in the costly tier instead.
SWITCHING_COST = chosen(0.0)

# The prior-sd multipliers every posterior is refitted under and printed beside; they decide
# nothing. `tau_scale` is not scaled.
SENSITIVITY: tuple[float, ...] = chosen((0.5, 0.25))

# The equal-tailed credible interval's mass.
CREDIBLE = chosen(0.95)

GRID = not_an_input(
    2001, "the number of quadrature nodes on tau; a numerical resolution of an exact "
          "integral, not a prior, and no reported digit moves with it at this size")
GRID_SPAN = not_an_input(
    8.0, "the tau grid's upper end, in units of the larger of tau_scale and the data's own "
         "reach; past it both prior and likelihood are below every printed digit")
_BISECT = not_an_input(
    80, "bisection steps for the credible interval's endpoints; each halves a bracket of "
        "twenty posterior sds, far below any printed digit")
_BRACKET = not_an_input(
    10.0, "the credible interval's bisection bracket, in posterior sds either side of the "
          "mixture's extreme components; it contains every endpoint, so no number moves")
_CHUNK = not_an_input(
    256, "rows fitted per vectorised block in fit_many; bounds memory, changes no number")


class Prior(NamedTuple):
    """One gate's prior and its tier. `costly` False is the reversible tier."""
    prior_sd: float
    tau_scale: float
    costly: bool


def prior_for(name: str) -> Prior | None:
    """The pre-registered prior for gate `name` (its `Harness.name`), or `None` if it has none."""
    s = S_REF.get(name)
    if s is None:
        return None
    return Prior(s, s, name not in REVERSIBLE)


class Posterior(NamedTuple):
    """The posterior on the pooled effect `delta`, plus the scatter `tau`'s posterior mean."""
    mean: float
    lo: float
    hi: float
    p_positive: float
    tau_mean: float


class GatePosterior(NamedTuple):
    """What a gate run reports beside its verdict word.

    `primary` is `None` exactly when `reason` says why nothing was computed. `action` is the
    verdict word itself when that word is VOID or NOT-RUNNABLE (the precondition is ahead of
    the posterior, as it is ahead of the sign-and-interval rule). `sensitivity` holds one
    `(multiplier, posterior, action)` per `SENSITIVITY` entry; `prior_sensitive` is whether any
    of their actions differs from the primary's.
    """
    prior: Prior | None
    primary: Posterior | None
    action: str | None
    sensitivity: tuple[tuple[float, Posterior, str], ...]
    prior_sensitive: bool
    reason: str | None


# --- the arithmetic ------------------------------------------------------------------------

def _phi(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standard normal CDF, vectorised (no scipy in the tree). Abramowitz & Stegun 7.1.26 for
    erf, absolute error under 1.5e-7 -- below every printed digit and every threshold here."""
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


class _Mixture(NamedTuple):
    """`p(delta | data) = sum_g w[g] N(m[g], sd[g])`, row-wise; `taus[n, g]` is each row's grid."""
    w: npt.NDArray[np.float64]
    m: npt.NDArray[np.float64]
    sd: npt.NDArray[np.float64]
    taus: npt.NDArray[np.float64]


def _mixture(y: npt.NDArray[np.float64], se: npt.NDArray[np.float64],
             prior_sd: float, tau_scale: float) -> _Mixture:
    """The posterior mixture for `n` independent fits at once: `y`, `se` are `(n, k)`.

    **The grid follows the data, not only the prior** (amended before any real fit; see the
    pre-registration's amendment). When the gains sit far outside `N(0, prior_sd)`, the model
    explains them with a large `tau` -- every season drawn high by chance -- and a grid ending
    at `GRID_SPAN * tau_scale` cuts that posterior off: at gains of `100 * S_REF` (k=4, se
    `S_REF`) it read a pooled mean of 5.8 where the integral is 2.02, with `tau` piled at the
    grid's edge. (At `20 * S_REF` the two agree to 1e-4.) So each row's grid ends at
    `GRID_SPAN` times the larger of `tau_scale` and that row's reach (`max |gain| + max se`),
    and its nodes are spaced as `u**2` on a uniform `u` -- dense near zero, where the prior's
    shape lives -- with the Jacobian `2 * upper * u` in the weights.
    """
    reach = np.abs(y).max(axis=1) + se.max(axis=1)
    upper = GRID_SPAN * np.maximum(tau_scale, reach)                # (n,)
    u = np.linspace(0.0, 1.0, GRID)
    taus = upper[:, None] * u[None, :] ** 2                         # (n, G)
    v = (taus ** 2)[:, :, None] + (se ** 2)[:, None, :]             # (n, G, k)
    log_det_d = np.log(v).sum(-1)
    iv = np.reciprocal(v, out=v)                                    # v is not read again
    a = iv.sum(-1)                                                  # sum 1/v
    b = np.matmul(iv, y[:, :, None])[..., 0]                        # sum y/v
    c = np.matmul(iv, (y ** 2)[:, :, None])[..., 0]                 # sum y^2/v
    prec = 1.0 / prior_sd ** 2 + a
    # log N(y; 0, diag(v) + prior_sd^2 11'): determinant lemma and Sherman-Morrison.
    loglik = -0.5 * (log_det_d + np.log1p(prior_sd ** 2 * a) + c - b ** 2 / prec)
    # Trapezoid in u with the Jacobian dtau/du = 2 * upper * u; the node at u = 0 has zero
    # weight (a single point of tau), so it is dropped rather than logged.
    jac = np.full(GRID, 2.0 * u)
    jac[-1] *= 0.5
    with np.errstate(divide="ignore"):
        lw = loglik - 0.5 * (taus / tau_scale) ** 2 + np.log(jac)[None, :] + np.log(upper)[:, None]
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return _Mixture(w, b / prec, 1.0 / np.sqrt(prec), taus)


def _cdf(mix: _Mixture, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The mixture's CDF at one `x` per row."""
    return (mix.w * _phi((x[:, None] - mix.m) / mix.sd)).sum(axis=1)


def _quantile(mix: _Mixture, q: float) -> npt.NDArray[np.float64]:
    lo = (mix.m - _BRACKET * mix.sd).min(axis=1)
    hi = (mix.m + _BRACKET * mix.sd).max(axis=1)
    for _ in range(_BISECT):
        mid = 0.5 * (lo + hi)
        below = _cdf(mix, mid) < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


class Fits(NamedTuple):
    """`fit_many`'s columns, one entry per row of its input."""
    mean: npt.NDArray[np.float64]
    lo: npt.NDArray[np.float64]
    hi: npt.NDArray[np.float64]
    p_positive: npt.NDArray[np.float64]
    tau_mean: npt.NDArray[np.float64]
    pit_delta: npt.NDArray[np.float64]
    pit_tau: npt.NDArray[np.float64]


def fit_many(y: npt.ArrayLike, se: npt.ArrayLike, prior_sd: float, tau_scale: float, *,
             interval: bool = True, delta_true: npt.ArrayLike | None = None,
             tau_true: npt.ArrayLike | None = None) -> Fits:
    """Fit `n` gates' worth of seasons at once: `y`, `se` are `(n, k)`, `se` finite and > 0.

    The calibration and size checks (`tests/unit/test_posterior.py`) run thousands of fits;
    `fit` below is the one-row case a gate reads. `interval=False` skips the bisection (a size
    check needs only the mean and `P(delta > 0)`; NaN in `lo`/`hi`). `delta_true`/`tau_true`,
    when given, return the posterior CDF at them (`pit_*`, the probability integral transform
    simulation-based calibration reads); NaN otherwise.
    """
    y2 = np.atleast_2d(np.asarray(y, dtype=float))
    se2 = np.atleast_2d(np.asarray(se, dtype=float))
    n = y2.shape[0]
    d_true = None if delta_true is None else np.asarray(delta_true, dtype=float).reshape(n)
    t_true = None if tau_true is None else np.asarray(tau_true, dtype=float).reshape(n)
    cols: list[list[npt.NDArray[np.float64]]] = [[] for _ in Fits._fields]
    nan = np.full(n, np.nan)
    for start in range(0, n, _CHUNK):
        sl = slice(start, start + _CHUNK)
        mix = _mixture(y2[sl], se2[sl], prior_sd, tau_scale)
        rows = mix.w.shape[0]
        mean = (mix.w * mix.m).sum(axis=1)
        p_pos = (mix.w * _phi(mix.m / mix.sd)).sum(axis=1)
        tau_mean = (mix.w * mix.taus).sum(axis=1)
        tail = (1.0 - CREDIBLE) / 2.0
        lo = _quantile(mix, tail) if interval else nan[:rows]
        hi = _quantile(mix, 1.0 - tail) if interval else nan[:rows]
        pit_d = nan[:rows] if d_true is None else _cdf(mix, d_true[sl])
        pit_t = (nan[:rows] if t_true is None
                 else (mix.w * (mix.taus <= t_true[sl][:, None])).sum(axis=1))
        for col, val in zip(cols, (mean, lo, hi, p_pos, tau_mean, pit_d, pit_t), strict=True):
            col.append(val)
    return Fits(*(np.concatenate(c) for c in cols))


def fit(gain: npt.ArrayLike, se: npt.ArrayLike, prior_sd: float,
        tau_scale: float) -> Posterior:
    """One gate's posterior from its season gains and their within-season SEs."""
    f = fit_many(np.asarray(gain, dtype=float)[None, :], np.asarray(se, dtype=float)[None, :],
                 prior_sd, tau_scale)
    return Posterior(float(f.mean[0]), float(f.lo[0]), float(f.hi[0]),
                     float(f.p_positive[0]), float(f.tau_mean[0]))


def action_codes(mean: npt.ArrayLike, p_positive: npt.ArrayLike, *,
                 costly: bool) -> npt.NDArray[np.int64]:
    """`posterior_action`, vectorised, as +1 ADOPT / -1 REMOVE / 0 SHOW."""
    stat = np.asarray(p_positive, dtype=float) - 0.5 if costly else np.asarray(mean, dtype=float)
    bar = P_COSTLY - 0.5 if costly else SWITCHING_COST
    return np.where(stat > bar, 1, np.where(stat < -bar, -1, 0)).astype(np.int64)


_WORD = {1: "ADOPT", -1: "REMOVE", 0: "SHOW"}


@decision
def posterior_action(post: Posterior, *, costly: bool) -> str:
    """The expected-utility action the pre-registration names, per tier.

    **Reversible** (`SWITCHING_COST`, zero): ADOPT when `E[delta] > 0`, REMOVE when `< 0`, SHOW
    at exactly zero -- the linear-utility action at no friction (#364's form). Its null ADOPT rate
    is 1/2 by construction, which is why its check is regret and not size.
    **Costly**: ADOPT when `P(delta > 0) > P_COSTLY`, REMOVE when `< 1 - P_COSTLY`, SHOW
    between -- the action under a loss where a wrong switch costs nineteen times a right
    one's gain, and the tier `test_the_posterior_rule_s_null_size_is_controlled` holds to `ALPHA`.
    """
    return _WORD[int(action_codes(post.mean, post.p_positive, costly=costly))]


_PRECONDITIONS = frozenset({"VOID", "NOT-RUNNABLE"})


def assess(gain: Sequence[float], se: Sequence[float], prior: Prior | None,
           verdict_word: str) -> GatePosterior:
    """The posterior block one gate run reports, or the reason it has none."""
    def missing(reason: str) -> GatePosterior:
        return GatePosterior(prior, None, None, (), False, reason)

    if prior is None:
        return missing("this gate has no pre-registered S_REF (docs/gate-power.md, #375)")
    if len(gain) == 0:
        return missing("no held-out season was measured")
    if not all(math.isfinite(s) and s > 0.0 for s in se):
        return missing("a season has no within-season SE (its within unit is a no-op or "
                       "has one cluster), and the likelihood is not imputed")
    primary = fit(gain, se, prior.prior_sd, prior.tau_scale)
    precondition = verdict_word in _PRECONDITIONS

    def act(p: Posterior) -> str:
        return verdict_word if precondition else posterior_action(p, costly=prior.costly)

    lead = act(primary)
    sens = tuple((m, p, act(p)) for m in SENSITIVITY
                 for p in (fit(gain, se, prior.prior_sd * m, prior.tau_scale),))
    return GatePosterior(prior, primary, lead, sens, any(a != lead for _, _, a in sens), None)


def posterior_report(gp: GatePosterior | None, verdict_word: str, *, unit: str,
                     places: int) -> list[str]:
    """The lines a gate's report prints after its per-season table."""
    head = "posterior on the pooled effect (#375; printed beside the verdict word, which decides)"
    if gp is None or gp.primary is None or gp.prior is None or gp.action is None:
        why = "not assessed" if gp is None else gp.reason
        return [f"{head}: not computed -- {why}."]
    pr, post = gp.prior, gp.primary

    def num(x: float) -> str:
        return f"{x:+.{places}f}"

    rule = (f"costly: ADOPT at P(effect > 0) > {P_COSTLY:g}, REMOVE below {1 - P_COSTLY:.1g}"
            if pr.costly else "reversible: the posterior mean's sign")
    lines = [
        f"{head}:",
        f"  prior N(0, {pr.prior_sd:g}), season scatter tau ~ HalfNormal({pr.tau_scale:g}); {rule}",
        f"  mean {num(post.mean)} {unit}, {CREDIBLE:.0%} credible "
        f"[{num(post.lo)}, {num(post.hi)}], "
        f"P(effect > 0) = {post.p_positive:.3f}, tau ~ {post.tau_mean:.{places}f}"
        f" -> {gp.action} (verdict word: {verdict_word})",
    ]
    sens = "; ".join(f"prior sd x{m:g}: mean {num(p.mean)}, P(effect > 0) = {p.p_positive:.3f}"
                     f" -> {a}" for m, p, a in gp.sensitivity)
    lines.append(f"  sensitivity -- {sens}" + ("  [prior-sensitive: not settled]"
                                               if gp.prior_sensitive else ""))
    return lines
