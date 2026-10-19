"""#375: the posterior gate's arithmetic, its calibration, and its decision rule's size.

`docs/gate-power.md`, *Pre-registered 2026-10-18 -- ADOPTED: the posterior gate's constants,
decision rule and engine (#375)*, names every check here before it was run: simulation-based
calibration with a misspecified-prior positive control, the costly tier's null size against
`ALPHA`, the reversible tier's regret against the t-interval rule, planted enormous effects in
both directions, and the sensitivity flag. Everything is scale-free -- `S_REF = 1` -- because
the model is location-scale equivariant: a gate's own unit changes no number below.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.models import experiment, posterior
from hub.models.posterior import Posterior, Prior, assess, fit, fit_many, posterior_action

SRC = Path(__file__).resolve().parents[2] / "src" / "hub"
S = 1.0


def _ks(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of `u` from Uniform(0, 1)."""
    x = np.sort(u)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(max((i / n - x).max(), (x - (i - 1) / n).max()))


def _simulate(rng: np.random.Generator, n: int, k: int, *, se: float, delta, tau):
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (n,))
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (n,))
    seasons = rng.normal(delta[:, None], tau[:, None], (n, k))
    return rng.normal(seasons, se), np.full((n, k), se)


# --- the engine against two independent computations -------------------------------------

def test_the_quadrature_matches_the_conjugate_limit():
    """`tau_scale -> 0` is the normal-normal model, whose posterior is closed-form."""
    y, se, p = np.array([0.3, -0.1, 0.5, 0.2]), np.array([0.2, 0.3, 0.25, 0.2]), 0.5
    got = fit(y, se, p, 1e-9)
    prec = 1 / p**2 + (1 / se**2).sum()
    m = (y / se**2).sum() / prec
    assert got.mean == pytest.approx(m, abs=1e-9)
    assert got.lo == pytest.approx(m - 1.959964 * prec**-0.5, abs=1e-6)
    assert got.p_positive == pytest.approx(0.5 * (1 + math.erf(m * prec**0.5 / math.sqrt(2))),
                                           abs=1e-6)


def test_the_quadrature_matches_a_brute_force_grid():
    """A dense 2-D grid over `(delta, tau)` that shares no code with the mixture."""
    y, se, p, t = np.array([0.3, -0.1, 0.5, 0.2]), np.array([0.2, 0.3, 0.25, 0.2]), 0.5, 0.4
    d = np.linspace(-3, 3, 3001)
    tt = np.linspace(0, 8 * t, 1601)
    dd, gg = np.meshgrid(d, tt, indexing="ij")
    v = gg[..., None] ** 2 + se**2
    ll = (-0.5 * np.log(v) - 0.5 * (y - dd[..., None]) ** 2 / v).sum(-1) \
        - 0.5 * dd**2 / p**2 - 0.5 * gg**2 / t**2
    w = np.exp(ll - ll.max())
    w /= w.sum()
    wd = w.sum(axis=1)
    got = fit(y, se, p, t)
    assert got.mean == pytest.approx(float((wd * d).sum()), abs=2e-3)
    assert got.p_positive == pytest.approx(float(wd[d > 0].sum()), abs=2e-3)
    assert got.tau_mean == pytest.approx(float((w.sum(axis=0) * tt).sum()), abs=2e-3)
    cdf = np.cumsum(wd)
    assert got.lo == pytest.approx(d[np.searchsorted(cdf, 0.025)], abs=5e-3)
    assert got.hi == pytest.approx(d[np.searchsorted(cdf, 0.975)], abs=5e-3)


# --- rule 16, check 1: simulation-based calibration --------------------------------------

_SBC_N = 1000
_KS_BAR = 1.63 / math.sqrt(_SBC_N)     # the 1% critical value


def _sbc(rng, k: int, se: float, *, generate_sd: float):
    delta = rng.normal(0.0, generate_sd, _SBC_N)
    tau = np.abs(rng.normal(0.0, S, _SBC_N))
    y, se2 = _simulate(rng, _SBC_N, k, se=se, delta=delta, tau=tau)
    f = fit_many(y, se2, S, S, interval=False, delta_true=delta, tau_true=tau)
    return _ks(f.pit_delta), _ks(f.pit_tau)


@pytest.mark.parametrize(("k", "se"), [(3, S), (4, S), (5, S), (10, S / 4)])
def test_simulation_based_calibration_is_uniform(k, se):
    """Draw `(delta, tau)` from the prior, data from the model, fit: the posterior CDF at the
    truth is Uniform(0, 1) for a calibrated posterior. k=10 at a quarter of the noise is #364's
    game-level shape; k=3-5 at the scatter's own noise is every season-level gate's."""
    ks_d, ks_t = _sbc(np.random.default_rng(375 + k), k, se, generate_sd=S)
    assert ks_d < _KS_BAR, f"delta's PIT is not uniform: KS {ks_d:.4f} >= {_KS_BAR:.4f}"
    assert ks_t < _KS_BAR, f"tau's PIT is not uniform: KS {ks_t:.4f} >= {_KS_BAR:.4f}"


def test_simulation_based_calibration_detects_a_misspecified_prior():
    """Rule 18's control for the check above: effects drawn four times wider than the prior
    the fit assumes must fail the same bar, or the bar cannot see miscalibration."""
    ks_d, _ = _sbc(np.random.default_rng(3750), 4, S, generate_sd=4 * S)
    assert ks_d >= _KS_BAR, f"a 4x misspecified prior passed SBC (KS {ks_d:.4f})"


# --- rule 16, check 2: the costly tier's null size ---------------------------------------

_NULL_N = 4000


def _null_size_failures(heterogeneous: bool) -> list[tuple]:
    """The pre-registered cells -- k in {3, 4, 5, 6}, the frozen scale at 1, 1.5 and 2 times
    the season noise, `tau` at zero or at all the heterogeneity that scale implies -- whose
    costly-tier ADOPT or REMOVE rate at `delta = 0` exceeds `ALPHA` plus two Monte Carlo SEs."""
    rng = np.random.default_rng(37501 + heterogeneous)
    bar = experiment.ALPHA + 2 * math.sqrt(experiment.ALPHA * (1 - experiment.ALPHA) / _NULL_N)
    failed = []
    for k in (3, 4, 5, 6):
        for ratio in (1.0, 1.5, 2.0):
            se = S / ratio
            tau = math.sqrt(max(S**2 - se**2, 0.0)) if heterogeneous else 0.0
            if heterogeneous and tau == 0.0:
                continue
            y, se2 = _simulate(rng, _NULL_N, k, se=se, delta=0.0, tau=tau)
            f = fit_many(y, se2, S, S, interval=False)
            codes = posterior.action_codes(f.mean, f.p_positive, costly=True)
            adopt, remove = float((codes == 1).mean()), float((codes == -1).mean())
            if adopt > bar or remove > bar:
                failed.append((k, ratio, round(tau, 3), adopt, remove))
    return failed


def test_the_posterior_rule_s_null_size_is_controlled():
    """Check 2 with no true heterogeneity: at `delta = 0` the costly tier's ADOPT rate, and by
    symmetry its REMOVE rate, is at most `ALPHA` plus two Monte Carlo SEs in every cell."""
    failed = _null_size_failures(heterogeneous=False)
    assert not failed, f"null ADOPT/REMOVE above the bar at (k, S/se, tau, adopt, remove): {failed}"


@pytest.mark.xfail(strict=True, reason=(
    "FINDING 2026-10-18 (#375), recorded not loosened: with the heterogeneity S_REF implies "
    "(tau > 0), the costly tier's 0.9 bar has a null ADOPT rate of 7-10% at k = 3-6, above "
    "ALPHA. Per the pre-registration this reopens P_COSTLY, which is the maintainer's; strict, "
    "so a change of bar that fixes it fails here until this marker is removed."))
def test_the_posterior_rule_s_null_size_is_controlled_under_heterogeneity():
    """Check 2's other half: the cells with `tau = sqrt(S_REF**2 - se**2) > 0`."""
    failed = _null_size_failures(heterogeneous=True)
    assert not failed, f"null ADOPT/REMOVE above the bar at (k, S/se, tau, adopt, remove): {failed}"


def test_the_reversible_rule_s_null_adopt_rate_is_one_half():
    """The property the pre-registration names rather than hides: at zero switching cost the
    posterior mean's sign is the sample's, so under the null the reversible tier adopts half
    the time. Its check is regret (below), never size."""
    y, se2 = _simulate(np.random.default_rng(37502), _NULL_N, 4, se=S, delta=0.0, tau=0.0)
    f = fit_many(y, se2, S, S, interval=False)
    rate = float((posterior.action_codes(f.mean, f.p_positive, costly=False) == 1).mean())
    assert abs(rate - 0.5) < 4 * math.sqrt(0.25 / _NULL_N), rate


@pytest.mark.parametrize("k", [3, 4, 5])
def test_the_reversible_rule_has_no_more_regret_than_the_t_interval_rule(k):
    """With effects drawn from the prior, expected regret -- the best action's gain minus the
    chosen one's, linear utility -- of the posterior-mean rule against `ADOPT only when the 95%
    t-interval over the k gains excludes zero`, the significance rule it replaces."""
    rng = np.random.default_rng(37503 + k)
    n = _NULL_N
    delta = rng.normal(0.0, S, n)
    tau = np.abs(rng.normal(0.0, S, n))
    y, se2 = _simulate(rng, n, k, se=S, delta=delta, tau=tau)
    f = fit_many(y, se2, S, S, interval=False)
    bayes = posterior.action_codes(f.mean, f.p_positive, costly=False) == 1
    t = experiment.t_quantile(0.975, k - 1)
    sig = y.mean(1) - t * y.std(1, ddof=1) / math.sqrt(k) > 0
    best = np.maximum(delta, 0.0)
    regret_bayes = float((best - np.where(bayes, delta, 0.0)).mean())
    regret_sig = float((best - np.where(sig, delta, 0.0)).mean())
    assert regret_bayes <= regret_sig, (regret_bayes, regret_sig)


# --- rule 18: the action fires on a planted effect, in both directions --------------------

@pytest.mark.parametrize("costly", [True, False])
def test_a_planted_enormous_effect_always_adopts(costly):
    """`delta = +-20 S_REF`, an effect no gate here has measured by accident: ADOPT, and its
    mirror REMOVE, on at least 99% of trials. Failing this means the harness is broken."""
    rng = np.random.default_rng(37504)
    for sign, want in ((1.0, 1), (-1.0, -1)):
        y, se2 = _simulate(rng, 1000, 4, se=S, delta=sign * 20 * S, tau=0.0)
        f = fit_many(y, se2, S, S, interval=False)
        rate = float((posterior.action_codes(f.mean, f.p_positive, costly=costly) == want).mean())
        assert rate >= 0.99, (sign, costly, rate)


def test_the_tau_grid_does_not_truncate_a_conflicting_posterior(monkeypatch):
    """Gains far outside the prior are explained by a large `tau`; the grid must reach it. The
    answer at the default grid equals the answer on a grid eight times as far, at 20, 100 and 1000
    `S_REF` -- where a prior-only grid read 5.8 for a mean of 2.02 at 100."""
    for g in (20.0, 100.0, 1000.0):
        here = fit([g] * 4, [S] * 4, S, S)
        # Eight times the span at eight times the nodes: the same resolution, further out.
        monkeypatch.setattr(posterior, "GRID_SPAN", 8 * posterior.GRID_SPAN)
        monkeypatch.setattr(posterior, "GRID", 8 * posterior.GRID)
        wide = fit([g] * 4, [S] * 4, S, S)
        monkeypatch.undo()
        assert here.mean == pytest.approx(wide.mean, abs=1e-3), (g, here, wide)
        assert here.p_positive == pytest.approx(wide.p_positive, abs=1e-3), (g, here, wide)


def test_a_consistent_effect_saturates_below_certainty():
    """Recorded, not a defect: under `N(0, S_REF)` a perfectly consistent effect of any size
    reads `P(effect > 0)` near 0.977 at k=4 (0.959 at k=3), because the model can always put
    the excess in `tau`. Pinned because a costly bar at or above this could never ADOPT."""
    assert fit([1000.0] * 3, [S] * 3, S, S).p_positive == pytest.approx(0.9585, abs=1e-3)
    assert fit([1000.0] * 4, [S] * 4, S, S).p_positive == pytest.approx(0.9774, abs=1e-3)


def test_posterior_action_reads_its_tier():
    """The scalar decision on both tiers' boundaries."""
    near = Posterior(mean=0.1, lo=-1.0, hi=1.0, p_positive=0.6, tau_mean=0.0)
    sure = Posterior(mean=2.0, lo=1.0, hi=3.0, p_positive=0.95, tau_mean=0.0)
    gone = Posterior(mean=-2.0, lo=-3.0, hi=-1.0, p_positive=0.05, tau_mean=0.0)
    assert posterior_action(near, costly=False) == "ADOPT"
    assert posterior_action(near, costly=True) == "SHOW"
    assert posterior_action(sure, costly=True) == "ADOPT"
    assert posterior_action(gone, costly=True) == "REMOVE"
    assert posterior_action(gone, costly=False) == "REMOVE"


# --- the report a gate prints ------------------------------------------------------------

_PRIOR = Prior(S, S, costly=True)


def test_sensitivity_flags_an_action_the_prior_decides():
    """A three-SE pooled gain clears 0.9 under the primary prior and not under s/4: printed
    prior-sensitive. A gain inside the noise is SHOW under all three and is not."""
    k, se = 4, [S] * 4
    borderline = assess([1.6] * k, se, _PRIOR, "SHOW")
    assert borderline.action == "ADOPT", borderline
    assert {a for _, _, a in borderline.sensitivity} != {"ADOPT"}
    assert borderline.prior_sensitive
    assert "prior-sensitive" in "\n".join(
        posterior.posterior_report(borderline, "SHOW", unit="u", places=2))
    settled = assess([0.1, -0.2, 0.15, 0.0], se, _PRIOR, "SHOW")
    assert settled.action == "SHOW" and not settled.prior_sensitive
    assert "prior-sensitive" not in "\n".join(
        posterior.posterior_report(settled, "SHOW", unit="u", places=2))


def test_a_precondition_word_is_the_action():
    """S6 and VOID stay ahead of the posterior: its numbers print, its action does not."""
    gp = assess([20.0] * 4, [S] * 4, _PRIOR, "NOT-RUNNABLE")
    assert gp.primary is not None and gp.primary.p_positive > posterior.P_COSTLY
    assert gp.action == "NOT-RUNNABLE"
    assert all(a == "NOT-RUNNABLE" for _, _, a in gp.sensitivity)


@pytest.mark.parametrize(("gain", "se", "prior", "why"), [
    ([1.0, 2.0], [1.0, float("nan")], _PRIOR, "within-season SE"),
    ([1.0, 2.0], [1.0, 0.0], _PRIOR, "within-season SE"),
    ([], [], _PRIOR, "no held-out season"),
    ([1.0], [1.0], None, "no pre-registered S_REF"),
])
def test_a_posterior_that_cannot_be_computed_says_why(gain, se, prior, why):
    gp = assess(gain, se, prior, "SHOW")
    assert gp.primary is None and gp.action is None
    line = posterior.posterior_report(gp, "SHOW", unit="u", places=2)
    assert len(line) == 1 and "not computed" in line[0] and why in line[0]


def _paired(gains: list[float], m: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"season": 2022 + i, "unit": u, "diff": float(g + rng.normal(0, 1))}
            for i, g in enumerate(gains) for u in range(m)]
    return pl.DataFrame(rows)


_ACTIONS = experiment.Actions(adopt="ADOPT.", remove="REMOVE.", show="SHOW.")


def test_the_gate_reports_a_posterior_beside_an_unchanged_verdict():
    """`gate(prior=...)` adds `GateRun.posterior`, read off the same per-season frame, and
    leaves the verdict, summary and seasons byte-identical to the run without one."""
    paired = _paired([0.6, 0.4, 0.7, 0.5])
    kw = {"cluster": experiment.SEASON_CLUSTER, "within": ("unit",),
          "ceiling": experiment.Ceiling("perfect", [1e6]), "actions": _ACTIONS, "bootstrap": 500}
    bare = experiment.gate(paired, **kw)
    with_prior = experiment.gate(paired, **kw, prior=Prior(0.5, 0.5, costly=True))
    assert bare.posterior is None
    assert with_prior.verdict == bare.verdict
    assert with_prior.summary == bare.summary
    assert with_prior.seasons.equals(bare.seasons)
    gp = with_prior.posterior
    assert gp is not None and gp.primary is not None
    assert gp.primary.lo < gp.primary.mean < gp.primary.hi
    assert gp.primary.p_positive > 0.99 and gp.action == "ADOPT"
    # The likelihood is the per-season frame's own gain and se, not a re-derivation.
    again = fit(bare.seasons["gain"].to_list(), bare.seasons["se"].to_list(), 0.5, 0.5)
    assert gp.primary == again


def test_every_prior_names_a_declared_harness():
    """`S_REF` and `REVERSIBLE` are keyed by `Harness.name`; a key that names no harness is a
    prior no gate will ever read, which is drift and not a decision."""
    names = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        for block in re.findall(r"Harness\((.*?)\)\n", text, flags=re.S):
            names.update(re.findall(r'name="([a-z_]+)"', block))
    assert names, "found no Harness declarations; the scan is broken"
    assert set(posterior.S_REF) <= names, set(posterior.S_REF) - names
    assert posterior.REVERSIBLE <= set(posterior.S_REF), posterior.REVERSIBLE - set(posterior.S_REF)
