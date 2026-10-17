"""`hub.screen_interval`: the interval a screen reports instead of "null" (#505).

**Rule 18 -- the control.** A planted screen with a known r and n must give the textbook
Fisher-z interval. The expected numbers below are worked by hand from the textbook formula,
not by calling the function under test: r = 0.5 on n = 28 with no controls has
z = atanh(0.5) = 0.549306 and se_z = 1/sqrt(25) = 0.2, so the 95% interval on z is
0.549306 -/+ 1.959964 * 0.2 = [0.157313, 0.941299], and tanh of those is [0.15603, 0.73582].
With one control the degrees of freedom drop by one (n - 3 - 1 = 24), which widens the
interval; both are pinned so that a mutation that forgets the control, or uses n - 3 for
a partial correlation, goes red.
"""
import math

import pytest

from hub import screen_interval as si


def test_the_planted_screen_gives_the_textbook_fisher_interval():
    lo, hi = si.fisher_interval(0.5, 28, controls=0)
    assert lo == pytest.approx(math.tanh(0.5493061443 - 1.959963985 * 0.2), abs=1e-9)
    assert hi == pytest.approx(math.tanh(0.5493061443 + 1.959963985 * 0.2), abs=1e-9)
    assert (round(lo, 5), round(hi, 5)) == (0.15603, 0.73582)


def test_a_partial_correlation_loses_a_degree_of_freedom_per_control():
    one = si.fisher_interval(0.5, 28, controls=1)
    none = si.fisher_interval(0.5, 28, controls=0)
    assert one[1] - one[0] > none[1] - none[0]
    half = 1.959963985 / math.sqrt(24)
    assert one[0] == pytest.approx(math.tanh(0.5493061443 - half), abs=1e-9)


def test_the_age_screen_reproduces_the_se_the_page_printed():
    """n = 2,140 rows over five seasons, one control: se_z = 0.0216, the 'se ~0.022'."""
    lo, hi = si.fisher_interval(0.0076, 2140)
    assert (lo, hi) == pytest.approx((-0.0348, 0.0500), abs=5e-4)


def test_the_interval_is_symmetric_on_z_and_brackets_r():
    lo, hi = si.fisher_interval(-0.3, 400)
    assert lo < -0.3 < hi
    lo2, hi2 = si.fisher_interval(0.3, 400)
    assert (lo2, hi2) == pytest.approx((-hi, -lo))


def test_the_t_interval_uses_the_t_quantile_on_k_minus_one_df():
    lo, hi = si.t_interval(0.0, 1.0, 4)
    assert hi == pytest.approx(3.182446, abs=1e-4) and lo == pytest.approx(-hi)
    lo, hi = si.t_interval(0.012, 0.052, 4)
    assert (round(lo, 3), round(hi, 3)) == (-0.153, 0.177)


@pytest.mark.parametrize("bad", [
    lambda: si.fisher_interval(0.1, 4, controls=1),
    lambda: si.fisher_interval(1.0, 100),
    lambda: si.t_interval(0.0, 0.1, 1),
])
def test_inputs_the_record_cannot_support_are_refused_not_invented(bad):
    with pytest.raises(ValueError):
        bad()


def test_render_is_the_form_the_ledger_reads():
    assert si.render(0.0076, (-0.0348, 0.05)) == \
        "partial r = +0.0076, 95% interval [-0.035, +0.050]"
