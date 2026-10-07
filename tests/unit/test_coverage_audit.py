"""#310: the claim is 0.80 (marginal), the verdict is taken at audits, and the weekly run reports
and never gates -- beyond a smoke alarm.

Written before the implementation (rule 18): each control plants the condition its check exists
to detect. The mutation each was seen red on is named in its docstring.
"""
import math
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.models import coverage, predict

ROOT = Path(__file__).resolve().parents[2]


def _player(pid, pos, season, points):
    return [{"player_id": pid, "position": pos, "season": season, "week": w + 1,
             "season_type": "REG", "fantasy_points_ppr": float(p)}
            for w, p in enumerate(points)]


def _stats(rows):
    return pl.DataFrame(rows, schema={
        "player_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32,
        "week": pl.Int32, "season_type": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _drawn(n_players=250, weeks=17, pos="WR", mu=12.0, seed=0, spread=1.0, seasons=(2023, 2024)):
    rng = np.random.default_rng(seed)
    sd = predict.WEEKLY_K[pos] * math.sqrt(mu) * spread
    sk = predict.WEEKLY_SKEW[pos]
    rows = []
    for season in seasons:
        for p in range(n_players):
            rows += _player(f"p{p}", pos, season, predict.skewed(mu, sd, sk, rng.standard_normal(weeks)))
    return _stats(rows)


def _row(group, n, cov):
    nominal = 0.80
    se = math.sqrt(2.0) * math.sqrt(nominal * (1 - nominal) / n)
    return {"group": group, "position": group if group != "all" else None, "n": n,
            "cov80": cov, "deviation": cov - nominal, "se": se, "sigma": (cov - nominal) / se,
            "n_cal": 300, "n_fallback": 0}


# --- the claim ------------------------------------------------------------

def test_the_claim_is_the_nominal_eighty_as_a_marginal_claim_and_the_band_is_unchanged():
    """Mutation: `CLAIMED_COV80 = 0.77` is red."""
    assert coverage.CLAIMED_COV80 == 0.80
    assert coverage.BAND == 0.02
    assert coverage.LEVELS[0] == (0.10, 0.90)


def test_the_pre_registered_constants_are_the_adopted_ones():
    assert coverage.LOOKS == 3
    assert coverage.ALPHA_PER_LOOK == pytest.approx(0.0167, abs=1e-4)
    assert coverage.Z_PER_LOOK == pytest.approx(2.394, abs=1e-3)
    assert coverage.N_MIN_GROUP == 8377
    assert coverage.SMOKE_BAND == 0.10


# --- the audit-time gate --------------------------------------------------

@pytest.mark.parametrize("cov,want", [
    (0.80, "COVERS"), (0.781, "COVERS"), (0.819, "COVERS"),
    (0.77, "UNDER-COVERS"), (0.83, "OVER-COVERS")])
def test_a_marginal_outside_eighty_plus_or_minus_two_fails_the_audit(cov, want):
    """The control the ticket asks for. Mutation: a band of 0.10 (the smoke alarm's) in
    `audit_verdict`, or reading the claim as 0.77, turns the 0.77/0.83 rows into COVERS/OVER."""
    got = coverage.audit_verdict([_row("all", 15000, cov)])
    assert got["marginal"] == want
    assert got["passes"] == (want == "COVERS")


def test_a_position_below_the_minimum_gets_no_verdict_only_its_deviation_and_sigma():
    """Mutation: lower `N_MIN_GROUP` to 1,000 and this QB row (1,805 weeks) gets a verdict."""
    rows = [_row("QB", 1805, 0.60), _row("all", 15000, 0.80)]
    got = coverage.audit_verdict(rows)
    qb = got["groups"][0]
    assert qb["verdict"] == "NOT-RUNNABLE" and qb["n_required"] == 8377
    assert qb["deviation"] == pytest.approx(-0.20) and qb["sigma"] < -10, "still reported"
    assert got["passes"], "a group that cannot rule does not fail the audit; the marginal binds"


def test_a_position_at_the_minimum_does_return_a_verdict_at_root_two_binomial_se():
    """The runnable branch, so NOT-RUNNABLE is not the only thing the rule can say (rule 16).
    At n = 8,377 the 95% interval at sqrt(2) x SE is +/- 0.0171; 0.77 is outside, 0.79 inside."""
    inside = coverage.audit_verdict([_row("QB", 8377, 0.79), _row("all", 20000, 0.80)])
    outside = coverage.audit_verdict([_row("QB", 8377, 0.77), _row("all", 20000, 0.80)])
    assert inside["groups"][0]["verdict"] == "COVERS"
    assert outside["groups"][0]["verdict"] == "UNDER-COVERS"
    assert outside["passes"] is False, "a runnable group that misses fails the audit"


def test_the_derivation_of_the_minimum_is_recorded_beside_the_adopted_number():
    """(z_alpha + z_0.8)^2 * 0.32 / 0.02^2 at alpha 0.0167 is 8,375.3; the adopted rule says
    8,377 and the adopted number is what binds. The derivation stays within three of it."""
    assert abs(coverage.derived_n_min() - coverage.N_MIN_GROUP) < 3


def test_an_audit_past_its_third_look_is_a_restatement_trigger_not_a_verdict(
        monkeypatch, capsys):
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn())
    assert coverage.main(["--audit", "--look", "4"]) == 1
    assert "restatement" in capsys.readouterr().err


def test_the_audit_prints_the_look_and_alpha_beside_the_verdict_and_gates_on_the_marginal(
        monkeypatch, capsys):
    """Calibrated draws cover 80%: the audit passes (exit 0) and says look 2 of 3, alpha, z."""
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(seed=3))
    assert coverage.main(["--audit", "--look", "2"]) == 0
    out = capsys.readouterr().out
    assert "look 2 of 3" in out and "0.0167" in out and "2.394" in out
    assert "NOT-RUNNABLE" in out, "no position has 8,377 weeks in a fixture this size"
    assert "season's own coverage" in out and "never gated" in out


def test_the_audit_exits_nonzero_on_a_marginal_the_window_has_not_seen(monkeypatch, capsys):
    a = _drawn(seasons=(2023,), seed=5)
    b = _drawn(seasons=(2024,), seed=6, spread=1.8)
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: pl.concat([a, b]))
    assert coverage.main(["--audit", "--look", "1"]) == 1
    assert "UNDER-COVERS" in capsys.readouterr().out


def test_the_audit_writes_the_claim_the_look_and_every_group_to_the_artifact(
        monkeypatch, tmp_path):
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(seed=3))
    monkeypatch.setattr(coverage, "ARTIFACT", tmp_path / "ic.json")
    assert coverage.main(["--audit", "--look", "1", "--write"]) == 0
    import json
    block = json.loads((tmp_path / "ic.json").read_text())["audit"]
    assert block["look"] == 1 and block["looks"] == 3 and block["claim"] == 0.80
    g = block["groups"][0]
    for key in ("position", "n", "n_cal", "cov80", "sigma", "verdict"):
        assert key in g
    # a later weekly write must not erase the audit
    assert coverage.main(["--measure", "--write"]) == 0
    assert json.loads((tmp_path / "ic.json").read_text())["audit"]["look"] == 1


# --- the weekly run reports and never gates --------------------------------

def test_the_weekly_run_never_exits_nonzero_inside_ten_points(monkeypatch, capsys):
    """Coverage 8 points under the claim is a miss the audit would call; the weekly gate step
    must still exit 0 -- reading it weekly is a sequential test (#310). Mutation: make the
    weekly step reuse `audit_verdict` and this exits 1."""
    a = _drawn(seasons=(2023,), seed=5)
    b = _drawn(seasons=(2024,), seed=6, spread=1.45)
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: pl.concat([a, b]))
    got = coverage.measure(pl.concat([a, b]), "prior")
    assert 0.02 < 0.80 - got["gate_cov80"] < 0.10, "the fixture misses by more than the band"
    assert coverage.main(["--gate"]) == 0
    assert "reports and never gates" in capsys.readouterr().out


def test_the_smoke_alarm_fires_on_a_broken_pipeline_beyond_ten_points():
    """Planted: a marginal of 0.65. Mutation: widen `SMOKE_BAND` to 1.0 and this is green."""
    rows = [_row("QB", 1000, 0.65), _row("all", 15000, 0.65)]
    assert coverage.smoke_alarm({"by_position": rows, "gate_cov80": 0.65,
                                 "gate_claim": 0.80})
    assert not coverage.smoke_alarm({"by_position": [_row("all", 15000, 0.72)],
                                     "gate_cov80": 0.72, "gate_claim": 0.80})


def test_the_smoke_alarm_fires_on_a_group_with_nothing_scored():
    empty = {"group": "TE", "position": "TE", "n": 0}
    reasons = coverage.smoke_alarm({"by_position": [empty, _row("all", 15000, 0.80)],
                                    "gate_cov80": 0.80, "gate_claim": 0.80})
    assert any("TE" in r and "calibration" in r for r in reasons)


def test_the_weekly_gate_exits_one_on_a_broken_pipeline(monkeypatch, capsys):
    """The CLI half of the alarm: draws 3x wider than the model, calibrated on a season at the
    model's own width, miss by far more than ten points."""
    a = _drawn(seasons=(2023,), seed=5)
    b = _drawn(seasons=(2024,), seed=6, spread=3.5)
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: pl.concat([a, b]))
    assert coverage.main(["--gate"]) == 1
    assert "smoke alarm" in capsys.readouterr().err


def test_the_weekly_gate_is_green_on_a_normal_week(monkeypatch, capsys):
    monkeypatch.setattr(coverage, "_stats", lambda seasons, cache: _drawn(seed=3))
    assert coverage.main(["--gate"]) == 0


def test_the_slate_runs_the_weekly_gate_not_the_audit_and_does_not_soft_fail_it():
    """The workflow half: the step is `--gate` (the smoke alarm) and never `--audit`, whose
    verdict is read at audits; and it is not soft-failed, so the alarm is a red run."""
    text = (ROOT / ".github" / "workflows" / "slate.yml").read_text()
    assert "hub.models.coverage --audit" not in text
    i = text.index("hub.models.coverage --gate")
    step = text[text.rfind("- name:", 0, i): i + 60]
    assert "continue-on-error" not in step
    assert re.search(r"smoke alarm", step, re.IGNORECASE), "the step says what it is"


def test_the_clip_split_is_a_diagnostic_beside_the_verdict_and_not_a_claim():
    """#310's 2026-10-07 decision. The split is reported; no verdict reads it. Mutation: add
    the strictly-positive row to `audit_verdict`'s inputs and the smoke test below is red."""
    got = coverage.measure(_drawn(seed=3), "prior")
    assert {r["group"] for r in got["floor_split"]} == {"clipped at zero", "strictly positive"}
    base = coverage.audit_verdict(got["by_position"])
    skewed_split = [dict(r, cov80=0.5) for r in got["floor_split"]]
    assert coverage.audit_verdict(got["by_position"] + skewed_split) == base, (
        "a clip-conditional miss must not move the adopted verdict")
    assert got["diagnostics"] == ["floor_split", "by_prior"]
