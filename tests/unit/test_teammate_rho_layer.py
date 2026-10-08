"""`TEAMMATE_RHO` is a points-layer number; the two consumers must stay faithful to it (#369).

The constant is a Pearson r on standardised weekly *points* (`hub.models.correlate`). Two
things read it:

- `group_sd` applies it to the points variance directly, so its implied pair correlation is the
  constant exactly -- that is the layer the L1 coverage gate (72.9% -> 80.4%) validated.
- `correlated_normal` applies it to the *latent normal*, before `skewed()` (Cornish-Fisher and a
  clip at zero). The realised points correlation is therefore a little under the constant:
  measured 2026-10-08 at the shipped skews, QB-WR 0.232 -> 0.2236-0.2263 (mean 6..14),
  QB-TE 0.2245 -> 0.2166-0.2192, QB-RB 0.0533 -> 0.0520-0.0526 -- a 0.963-0.975 ratio. The
  adopted option C (docs/correlation.md, "Layer", 2026-10-08) records this as a known limit and
  moves no number. This test holds both facts so neither can drift unseen.
"""
import numpy as np
import pytest

from hub.models import predict

EDGES = [("QB", "WR"), ("QB", "TE"), ("QB", "RB")]
N = 300_000
# The measured attenuation is ~0.97 of the constant; sampling error at N is ~0.002 on the
# correlation. 0.006 absolute is ~3 standard errors and still sits below the 0.007 gap that
# removing the attenuation (a latent-to-points conversion, option A) would close.
EXPECTED_RATIO = 0.97
TOL = 0.006


def _realised(other: str, *, scale: float = 1.0, mu_other: float = 10.0, seed: int = 369) -> float:
    """Points correlation between a QB and `other` after correlated_normal + skewed()."""
    rng = np.random.default_rng(seed)
    pos = np.array(["QB", other])
    team = np.array(["A", "A"], dtype=object)
    z = predict.correlated_normal(rng, (N, 2), pos, team)
    # `scale` is a planted-failure knob: a latent built at k times the fitted correlation.
    # (Rescaling only the off-diagonal cannot be done on z after the fact, so mix instead.)
    if scale != 1.0:
        rho = predict.teammate_rho("QB", other) * scale
        rng2 = np.random.default_rng(seed)
        e = rng2.standard_normal((N, 2))
        z = np.stack([e[:, 0], rho * e[:, 0] + np.sqrt(1 - rho**2) * e[:, 1]], axis=1)
    mu = np.array([18.0, mu_other])
    sd = 0.55 * mu + 2.0
    skew = np.array([predict.WEEKLY_SKEW[p] for p in pos])
    x = predict.skewed(mu, sd, skew, z)
    return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])


@pytest.mark.parametrize(("a", "b"), EDGES)
def test_group_sd_implied_pair_correlation_is_teammate_rho_exactly(a, b):
    for x, y in ((a, b), (b, a)):
        s1, s2 = 7.0, 5.0
        var = predict.group_sd([(s1, x, "A"), (s2, y, "A")]) ** 2
        implied = (var - s1**2 - s2**2) / (2.0 * s1 * s2)
        assert implied == pytest.approx(predict.teammate_rho(a, b), abs=1e-12)


def test_group_sd_prices_unmeasured_pairs_and_strangers_at_zero():
    assert predict.teammate_rho("WR", "WR") == 0.0
    var = predict.group_sd([(4.0, "WR", "A"), (3.0, "WR", "A")]) ** 2
    assert var == pytest.approx(25.0, abs=1e-12)
    var = predict.group_sd([(4.0, "QB", "A"), (3.0, "WR", "B")]) ** 2
    assert var == pytest.approx(25.0, abs=1e-12)


@pytest.mark.parametrize(("a", "b"), EDGES)
@pytest.mark.parametrize("mu_other", [6.0, 10.0, 14.0])
def test_simulator_realises_the_measured_attenuation_of_teammate_rho(a, b, mu_other):
    rho = predict.teammate_rho(a, b)
    got = _realised(b, mu_other=mu_other)
    assert got < rho, "the transform attenuates; realising more than the constant is new"
    assert got == pytest.approx(EXPECTED_RATIO * rho, abs=TOL)


@pytest.mark.parametrize(("a", "b"), EDGES)
def test_control_the_band_rejects_a_doubled_latent_rho(a, b):
    """Rule 18: the band must be able to go red. A latent built at twice the constant -- the
    doubled-rho mistake -- is far outside it."""
    rho = predict.teammate_rho(a, b)
    got = _realised(b, scale=2.0)
    assert abs(got - EXPECTED_RATIO * rho) > TOL


def test_wr_wr_is_zero_in_the_latent_not_a_shared_factor():
    """The star topology: catchers share a quarterback and no direct edge (correlation.md)."""
    rng = np.random.default_rng(369)
    pos = np.array(["QB", "WR", "WR"])
    team = np.array(["A", "A", "A"], dtype=object)
    z = predict.correlated_normal(rng, (N, 3), pos, team)
    c = np.corrcoef(z.T)
    # Zero, not the +0.054 (0.232**2) a one-factor build through the quarterback would give:
    # the matrix is explicit and factored, so the unmeasured edge stays 0.
    assert c[1, 2] == pytest.approx(0.0, abs=0.006)
    assert c[0, 1] == pytest.approx(predict.teammate_rho("QB", "WR"), abs=0.006)
    assert predict.teammate_rho("WR", "WR") == 0.0
