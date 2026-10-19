"""#502: P(win | spread) fitted on wins, with sd by the total, gated. Measurement only.

The pre-registration is `docs/margin-sd.md` ("Pre-registered 2026-10-18 ... (#502)"), committed
before any number existed. Everything here runs on synthetic spreads, totals and outcomes: no
test reads real data, and the real run is `python -m hub.models.total_sd --run`.

Rule 18's controls, each planted rather than assumed:

* a strong total dependence is ADOPTed;
* a null (outcomes from the incumbent) is not, with the ceiling held open so NOT-RUNNABLE
  cannot be the reason;
* a ceiling below the MDE is NOT-RUNNABLE;
* the walk-forward does not leak: flipping a season's outcomes moves no fit for that season or
  an earlier one;
* the fit recovers what was planted, from a start (the incumbent) that is not the answer.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hub.ledger import Ledger
from hub.models import total_sd as ts
from hub.models.market import MARGIN_SD

# Far above anything plausible, so the gate has something real to find.
REAL = ts.Fit(sd0=9.0, beta=2.5)
NULL = ts.Fit(sd0=MARGIN_SD, beta=0.0)
# Never binds: holds the NOT-RUNNABLE branch open so a null reads SHOW for the right reason.
OPEN = 1.0


def synth(seed: int, truth: ts.Fit = NULL, seasons=range(1999, 2026),
          games: int = 270) -> pl.DataFrame:
    """`games`-shaped frame: outcomes drawn from `truth`, spreads and totals on the half point."""
    rng = np.random.default_rng(seed)
    rows = []
    for yr in seasons:
        s = np.round(rng.normal(0.0, 6.5, games) * 2.0) / 2.0
        t = np.round(rng.normal(44.0, 5.0, games).clip(30.0, 60.0) * 2.0) / 2.0
        y = (rng.random(games) < ts.challenger_prob(s, t, truth)).astype(np.int64)
        rows.append(pl.DataFrame({"season": yr, "spread_line": s, "total_line": t,
                                  "home_won": y}))
    return pl.concat(rows)


def gated(paired: pl.DataFrame, *, ceiling: float | None = None):
    """The pure half of the rule on a walk-forward frame, bootstrap kept small."""
    frame = paired[["season", "diff", "ceiling_diff"]]
    if ceiling is not None:
        frame = frame.with_columns(pl.lit(ceiling).alias("ceiling_diff"))
    return ts.HARNESS.decide(frame, bootstrap=300)


def _fit(frame: pl.DataFrame, **kw) -> ts.Fit:
    return ts.fit_sigma(frame["spread_line"].to_numpy(), frame["total_line"].to_numpy(),
                        frame["home_won"].to_numpy(), **kw)


# --- the mapping and the fit -------------------------------------------------------------

def test_sigma_is_sd0_at_the_reference_total_and_grows_as_the_power():
    fit = ts.Fit(sd0=12.0, beta=0.5)
    got = ts.sigma(np.array([ts.T_REF, 4.0 * ts.T_REF]), fit)
    assert got == pytest.approx([12.0, 24.0])


def test_at_beta_zero_and_the_incumbent_sd_the_challenger_is_the_incumbent():
    s = np.array([-7.0, -3.0, 0.0, 2.5, 10.0])
    t = np.array([37.0, 41.0, 44.0, 48.5, 55.0])
    assert ts.challenger_prob(s, t, NULL) == pytest.approx(ts.incumbent_prob(s))
    assert ts.incumbent_prob(np.array([0.0]))[0] == pytest.approx(0.5)


def test_the_fit_recovers_the_planted_parameters():
    """Rule 15: the start is the incumbent (12.741, 0), which is not the planted point."""
    fit = _fit(synth(1, ts.Fit(sd0=11.0, beta=1.0), games=2000))
    assert fit.sd0 == pytest.approx(11.0, rel=0.06)
    assert fit.beta == pytest.approx(1.0, abs=0.35)


def test_a_null_fit_finds_no_total_dependence():
    fit = _fit(synth(2, NULL, games=2000))
    assert abs(fit.beta) < 0.35
    assert fit.sd0 == pytest.approx(MARGIN_SD, rel=0.06)


def test_scale_only_pins_beta_at_zero():
    fit = _fit(synth(3, REAL), by_total=False)
    assert fit.beta == 0.0
    assert fit.sd0 > 0


# --- the ceiling --------------------------------------------------------------------------

def test_the_bands_are_closed_below():
    assert ts.band(np.array([30.0, 40.5, 41.0, 46.5, 47.0, 60.0])).tolist() == [0, 0, 1, 1, 2, 2]


def test_the_oracle_is_the_favourite_rate_by_cell_and_a_pickem_is_a_coin():
    s = np.array([3.0, 3.0, -3.0, 3.0, 0.0])
    t = np.array([40.0, 40.0, 40.0, 50.0, 44.0])
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0])   # the away favourite at -3 won (home lost)
    rates = ts.oracle_rates(s, t, y)
    assert rates[(3.0, 0)] == pytest.approx(2 / 3)
    assert rates[(3.0, 2)] == pytest.approx(1.0)
    p = ts.oracle_prob(s, t, rates)
    assert p[0] == pytest.approx(2 / 3) and p[2] == pytest.approx(1 / 3)
    assert p[4] == 0.5


# --- the walk-forward ---------------------------------------------------------------------

def test_the_first_held_out_season_has_five_seasons_behind_it_and_incomplete_ones_are_skipped():
    frame = synth(4, NULL, seasons=range(1999, 2011))
    thin = frame.filter(pl.col("season") == 2009).head(100)
    frame = pl.concat([frame.filter(pl.col("season") != 2009), thin])
    out = ts.walk_forward(frame)
    assert out["season"].to_list() == [2004, 2005, 2006, 2007, 2008, 2010]


def test_no_held_out_season_is_an_empty_frame_with_the_columns():
    out = ts.walk_forward(synth(5, NULL, seasons=range(1999, 2003)))
    assert out.is_empty() and "diff" in out.columns and "ceiling_diff" in out.columns


def test_a_missing_total_is_dropped_from_both_arms_and_counted():
    frame = synth(6, NULL, seasons=range(1999, 2006))
    gap = frame.with_columns(
        pl.when((pl.col("season") == 2005) & (pl.int_range(pl.len()) % 10 == 0))
        .then(None).otherwise(pl.col("total_line")).alias("total_line"))
    out = ts.walk_forward(gap)
    row = out.filter(pl.col("season") == 2005).row(0, named=True)
    assert row["missing_total"] == pytest.approx(0.1, abs=0.01)
    assert row["n"] == gap.filter((pl.col("season") == 2005)
                                  & pl.col("total_line").is_not_null()).height


def test_the_walk_forward_does_not_leak():
    """Flip every outcome of one season: no fit for that season or an earlier one moves."""
    frame = synth(7, REAL)
    flipped = frame.with_columns(
        pl.when(pl.col("season") == 2012).then(1 - pl.col("home_won"))
        .otherwise(pl.col("home_won")).alias("home_won"))
    a, b = ts.walk_forward(frame), ts.walk_forward(flipped)
    early = a["season"] <= 2012
    for col in ("sd0", "beta"):
        assert a.filter(early)[col].to_list() == b.filter(early)[col].to_list()
        assert a.filter(~early)[col].to_list() != b.filter(~early)[col].to_list()


def test_the_trailing_window_is_ten_seasons():
    frame = synth(8, NULL)
    corrupt = frame.with_columns(
        pl.when(pl.col("season") == 2000).then(1 - pl.col("home_won"))
        .otherwise(pl.col("home_won")).alias("home_won"))
    a, b = ts.walk_forward(frame), ts.walk_forward(corrupt)
    late = a["season"] >= 2011
    assert a.filter(late)["sd0"].to_list() == b.filter(late)["sd0"].to_list()
    assert a.filter(~late)["sd0"].to_list() != b.filter(~late)["sd0"].to_list()


def test_the_ceiling_bounds_the_planted_effect():
    out = ts.walk_forward(synth(9, REAL))
    assert out["ceiling_diff"].to_numpy().mean() > out["diff"].to_numpy().mean() > 0


# --- the VOID conditions ------------------------------------------------------------------

def test_void_fires_on_too_few_seasons_and_on_a_season_missing_its_totals():
    few = ts.walk_forward(synth(10, NULL, seasons=range(1999, 2007)))
    assert "fewer than" in (ts.void_reason(few) or "")
    full = ts.walk_forward(synth(10, NULL, seasons=range(1999, 2012)))
    assert ts.void_reason(full) is None
    bad = full.with_columns(
        pl.when(pl.col("season") == 2008).then(0.11).otherwise(0.0).alias("missing_total"))
    assert "[2008]" in (ts.void_reason(bad) or "")
    ok = full.with_columns(pl.lit(ts.MAX_MISSING_TOTAL).alias("missing_total"))
    assert ts.void_reason(ok) is None


# --- rule 18: the controls ----------------------------------------------------------------

def test_a_planted_strong_total_dependence_is_adopted():
    paired = ts.walk_forward(synth(1, REAL))
    run = gated(paired)
    assert run.verdict[0] == "ADOPT", run.verdict
    assert run.summary["mean"] > run.summary["mde"] > 0
    assert int((paired["diff"] > 0).sum()) == paired.height


def test_a_null_is_not_adopted():
    for seed in (11, 12, 13):
        run = gated(ts.walk_forward(synth(seed, NULL)), ceiling=OPEN)
        assert run.verdict[0] in {"SHOW", "REMOVE"}, run.verdict


def test_a_ceiling_below_the_mde_is_not_runnable_and_reports_no_verdict():
    paired = ts.walk_forward(synth(1, NULL))
    run = gated(paired, ceiling=0.0)
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert run.summary["mde"] > run.summary["ceiling"]
    assert gated(paired, ceiling=OPEN).verdict[0] != "NOT-RUNNABLE"


def test_a_void_input_is_void_whatever_the_arms_say():
    paired = ts.walk_forward(synth(1, REAL))
    run = ts.HARNESS.decide(paired[["season", "diff", "ceiling_diff"]], bootstrap=300,
                            void="VOID: planted")
    assert run.verdict[0] == "VOID"


# --- the descriptive increment ------------------------------------------------------------

def test_the_increment_is_what_the_total_adds_over_a_constant_sigma():
    paired = ts.walk_forward(synth(1, REAL))
    mean, lo, hi, pos = ts.increment_summary(paired)
    assert lo < mean < hi and mean > 0 and pos > paired.height // 2
    assert paired["increment"].to_list() == pytest.approx(
        (paired["ll_scale_only"] - paired["ll_challenger"]).to_list())
    null = ts.walk_forward(synth(12, NULL))
    n_mean, n_lo, n_hi, _ = ts.increment_summary(null)
    assert n_lo < 0 < n_hi or abs(n_mean) < 0.002


# --- the entry point ----------------------------------------------------------------------

def _schedules(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("home_won") == 1).then(7).otherwise(-7).alias("result")
    ).drop("home_won")


def test_games_refuses_a_schedule_without_a_total():
    with pytest.raises(ValueError, match="total_line"):
        ts.games(_schedules(synth(1, seasons=range(1999, 2000))).drop("total_line"))


def test_main_without_the_flag_prints_help_and_runs_nothing(capsys):
    assert ts.main([]) == 0
    assert "--run" in capsys.readouterr().out


def test_main_says_so_when_the_schedules_cannot_be_read(capsys):
    def boom():
        raise OSError("no cache")

    ledger = Ledger(path=None)
    assert ts.main(["--run"], load=boom, ledger=ledger) == 1
    assert "hub.models.total_sd" in capsys.readouterr().err
    assert ledger._read() in (None, [])


def test_main_with_nothing_to_score_exits_nonzero(capsys):
    ledger = Ledger(path=None)
    assert ts.main(["--run"], ledger=ledger,
                   load=lambda: _schedules(synth(1, seasons=range(1999, 2003)))) == 1
    assert "nothing to gate" in capsys.readouterr().out
    assert ledger._read() in (None, [])


def test_main_runs_the_gate_once_and_prints_the_verdict(monkeypatch, capsys):
    monkeypatch.setattr(ts, "HARNESS", ts.HARNESS._replace(bootstrap=200))
    ledger = Ledger(path=None)
    assert ts.main(["--run"], load=lambda: _schedules(synth(1, REAL)), ledger=ledger) == 0
    out = capsys.readouterr().out
    assert "ADOPT" in out and "MDE" in out and "ceiling" in out
    assert "what the total adds" in out
    written = ledger._read()
    assert written is not None and len(written) == 1
    assert written[0].name == "total_sd"
