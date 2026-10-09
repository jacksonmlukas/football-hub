"""#465: the favourite-longshot location correction, gated. Measurement only.

The pre-registration is `docs/margin-sd.md` ("Pre-registered 2026-10-09 ... (#465)"), committed
before any number existed. Everything here runs on synthetic spreads and synthetic outcomes: no
test reads real data, and the real run is `python -m hub.models.favourite_longshot --run`.

Rule 18's controls, each planted rather than assumed:

* a shift that is real is ADOPTed;
* a shift that is null is not (SHOW), with the ceiling held open so NOT-RUNNABLE cannot be the
  reason;
* a ceiling below the MDE is NOT-RUNNABLE, whichever way the arms compare;
* the walk-forward does not leak: flipping a season's outcomes moves no fit for that season or
  an earlier one.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hub import cli
from hub.ledger import Ledger
from hub.models import favourite_longshot as fl
from hub.models.market import MARGIN_SD

# A shift far above anything the screen reported, so the gate has something real to find.
REAL = (-8.0, -3.0, 5.0, 6.0, 0.0)
NULL = (0.0, 0.0, 0.0, 0.0, 0.0)
# Never binds: holds the NOT-RUNNABLE branch open so a null reads SHOW for the right reason.
OPEN = 1.0


def synth(seed: int, delta=NULL, seasons=range(1999, 2026), games: int = 270) -> pl.DataFrame:
    """`residuals`-shaped frame: outcomes drawn from the shifted price, spreads on the half point."""
    rng = np.random.default_rng(seed)
    d = np.asarray(delta, dtype=float)
    rows = []
    for yr in seasons:
        s = np.round(rng.normal(0.0, 6.5, games) * 2.0) / 2.0
        p = fl._phi_cdf((s + np.sign(s) * d[fl.bucket(s)]) / MARGIN_SD)
        y = (rng.random(games) < p).astype(np.int64)
        rows.append(pl.DataFrame({"season": yr, "spread_line": s, "home_won": y}))
    return pl.concat(rows)


def gated(paired: pl.DataFrame, *, ceiling: float | None = None):
    """The pure half of the rule on a walk-forward frame, bootstrap kept small."""
    frame = paired[["season", "diff", "ceiling_diff"]]
    if ceiling is not None:
        frame = frame.with_columns(pl.lit(ceiling).alias("ceiling_diff"))
    return fl.HARNESS.decide(frame, bootstrap=300)


# --- the fit -----------------------------------------------------------------------------

def _truth(spread: np.ndarray, delta) -> np.ndarray:
    d = np.asarray(delta, dtype=float)
    return fl._phi_cdf((spread + np.sign(spread) * d[fl.bucket(spread)]) / MARGIN_SD)


GRID = np.arange(-14.0, 14.5, 0.5)


def test_the_fit_recovers_the_planted_price():
    """The price, not the parameters. Shift and sd are close to collinear on a binary outcome
    (the penalised likelihood at the planted values and at the fit differ by about a nat), so
    the parameters are not what the walk-forward scores and not what is asserted."""
    big = synth(1, REAL, games=1500)
    fit = fl.fit_shift(big["spread_line"].to_numpy(), big["home_won"].to_numpy())
    assert np.max(np.abs(fl.shifted_prob(GRID, fit) - _truth(GRID, REAL))) < 0.02
    # ...and the incumbent price is visibly not the planted one, so the check can fail.
    assert np.max(np.abs(fl.incumbent_prob(GRID) - _truth(GRID, REAL))) > 0.05


def test_a_null_fit_prices_like_the_incumbent():
    big = synth(2, NULL, games=1500)
    fit = fl.fit_shift(big["spread_line"].to_numpy(), big["home_won"].to_numpy())
    assert np.max(np.abs(fl.shifted_prob(GRID, fit) - fl.incumbent_prob(GRID))) < 0.02


def test_scale_only_pins_every_shift_at_zero():
    frame = synth(3, REAL)
    fit = fl.fit_shift(frame["spread_line"].to_numpy(), frame["home_won"].to_numpy(),
                       shifts=False)
    assert fit.delta == (0.0,) * fl.N_BUCKETS
    assert fit.sd > 0


def test_a_pickem_takes_no_shift_and_a_shift_moves_the_favourite_not_the_side():
    fit = fl.Fit(delta=(-2.0, 0.0, 0.0, 0.0, 0.0), sd=MARGIN_SD)
    got = fl.shifted_prob(np.array([0.0, 1.0, -1.0]), fit)
    assert got[0] == pytest.approx(0.5)
    # Home-favoured by one with a favourite shift of -2: the favourite is worse than priced,
    # and the mirror game (away favoured by one) is the exact complement.
    assert got[1] < fl.incumbent_prob(np.array([1.0]))[0]
    assert got[1] + got[2] == pytest.approx(1.0)


def test_an_empty_bucket_keeps_its_prior():
    s = np.array([1.0, 2.0, 1.5, -1.0, -2.5, 0.5] * 40)
    y = (np.arange(s.size) % 2).astype(float)
    fit = fl.fit_shift(s, y)
    assert fit.delta[2:] == (0.0, 0.0, 0.0)


# --- the walk-forward --------------------------------------------------------------------

def test_the_first_held_out_season_has_five_seasons_behind_it_and_incomplete_ones_are_skipped():
    frame = synth(4, NULL, seasons=range(1999, 2011))
    thin = frame.filter(pl.col("season") == 2009).head(100)
    frame = pl.concat([frame.filter(pl.col("season") != 2009), thin])
    out = fl.walk_forward(frame)
    assert out["season"].to_list() == [2004, 2005, 2006, 2007, 2008, 2010]
    assert 2009 not in out["season"].to_list()


def test_no_held_out_season_is_an_empty_frame_with_the_columns():
    out = fl.walk_forward(synth(5, NULL, seasons=range(1999, 2003)))
    assert out.is_empty() and "diff" in out.columns and "ceiling_diff" in out.columns


def test_the_walk_forward_does_not_leak():
    """Flip every outcome of one season: no fit for that season or an earlier one moves."""
    frame = synth(6, REAL)
    flipped = frame.with_columns(
        pl.when(pl.col("season") == 2012).then(1 - pl.col("home_won"))
        .otherwise(pl.col("home_won")).alias("home_won"))
    a, b = fl.walk_forward(frame), fl.walk_forward(flipped)
    early = a["season"] <= 2012
    assert a.filter(early)["sd"].to_list() == b.filter(early)["sd"].to_list()
    assert a.filter(~early)["sd"].to_list() != b.filter(~early)["sd"].to_list()


def test_the_trailing_window_is_ten_seasons():
    frame = synth(7, NULL)
    # Corrupt the oldest season only: a fit for 2020 reads 2010-2019, so it must not move.
    corrupt = frame.with_columns(
        pl.when(pl.col("season") == 2000).then(1 - pl.col("home_won"))
        .otherwise(pl.col("home_won")).alias("home_won"))
    a, b = fl.walk_forward(frame), fl.walk_forward(corrupt)
    late = a["season"] >= 2011
    assert a.filter(late)["sd"].to_list() == b.filter(late)["sd"].to_list()


def test_the_ceiling_is_the_pooled_oracle_against_the_incumbent():
    out = fl.walk_forward(synth(8, REAL))
    assert (out["ceiling_diff"] > out["diff"] - 0.02).all()
    assert float(out["ceiling_diff"].to_numpy().mean()) > 0


# --- rule 18: the three controls ----------------------------------------------------------

def test_a_planted_real_shift_is_adopted():
    paired = fl.walk_forward(synth(1, REAL))
    run = gated(paired)
    assert run.verdict[0] == "ADOPT", run.verdict
    assert run.summary["mean"] > run.summary["mde"] > 0
    assert int((paired["diff"] > 0).sum()) == paired.height


def test_a_null_shift_is_not_adopted():
    for seed in (11, 12, 13):
        run = gated(fl.walk_forward(synth(seed, NULL)), ceiling=OPEN)
        assert run.verdict[0] in {"SHOW", "REMOVE"}, run.verdict
        assert run.verdict[0] != "ADOPT"


def test_a_ceiling_below_the_mde_is_not_runnable_and_reports_no_verdict():
    paired = fl.walk_forward(synth(1, NULL))
    run = gated(paired, ceiling=0.0)
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert run.summary["mde"] > run.summary["ceiling"]
    # ...and the same frame with the ceiling held open is a verdict, so the control is the
    # ceiling and not something else about the frame.
    assert gated(paired, ceiling=OPEN).verdict[0] != "NOT-RUNNABLE"


# --- the entry point ----------------------------------------------------------------------

def _schedules(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("home_won") == 1).then(7).otherwise(-7).alias("result"))


def test_main_without_the_flag_prints_help_and_runs_nothing(capsys):
    assert fl.main([]) == 0
    assert "--run" in capsys.readouterr().out


def test_main_says_so_when_the_schedules_cannot_be_read(monkeypatch, capsys):
    from hub.fetch import nflverse

    def boom(*a, **k):
        raise OSError("no cache")

    monkeypatch.setattr(nflverse, "load", boom)
    assert fl.main(["--run"]) == 1
    assert "hub.models.favourite_longshot" in capsys.readouterr().err


def test_main_with_nothing_to_score_exits_nonzero(monkeypatch, capsys):
    from hub.fetch import nflverse

    monkeypatch.setattr(nflverse, "load", lambda *a, **k: _schedules(synth(1, seasons=range(1999, 2003))))
    monkeypatch.setattr(nflverse, "every_season", lambda: [1999])
    assert fl.main(["--run"]) == 1
    assert "nothing to gate" in capsys.readouterr().out


def test_main_runs_the_gate_once_and_prints_the_verdict(monkeypatch, capsys):
    from hub.fetch import nflverse

    monkeypatch.setattr(nflverse, "load", lambda *a, **k: _schedules(synth(1, REAL)))
    monkeypatch.setattr(nflverse, "every_season", lambda: list(range(1999, 2026)))
    ledger = Ledger(path=None)
    monkeypatch.setattr(fl, "HARNESS", fl.HARNESS._replace(ledger=ledger, bootstrap=200))
    assert fl.main(["--run"]) == 0
    out = capsys.readouterr().out
    assert "ADOPT" in out and "MDE" in out and "ceiling" in out
    written = ledger._read()
    assert written is not None and len(written) == 1
    assert written[0].name == "favourite_longshot"


def test_the_unavailable_helper_is_the_one_the_cli_surface_expects():
    assert callable(cli.unavailable)
