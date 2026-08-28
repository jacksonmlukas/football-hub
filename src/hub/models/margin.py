"""Fit the dispersion of game margin around the closing spread.

`hub.models.market` sets `MARGIN_SD = 13.5` under a comment reading "stable across decades of
NFL results". There was no fit, no interval and no write-up — and this is the number that turns
every closing spread into a win probability, so it is the most load-bearing constant in the NFL
path. It also sits in `config.FITTED_MODULES`, hashed into the model version *as though* it had
been fitted, which is ADR-0006's distinction landing on the wrong side.

The data to measure it has been in the fetch layer the whole time: `schedules` carries
`spread_line` and `result` back to 1999. `result` is home-relative (verified: it equals
`home_score - away_score` for every completed game), and `spread_line` is too, so the residual
is `result - spread_line` and needs no sign juggling.

**The dispersion is not constant, which is the interesting part.** Per-season sd runs from 11.5
to 14.4, and the trend is downward at -0.037 a year (-2.4 se) — the market has got sharper. So
"which window" is a real modelling choice, not a detail, and this module tests three candidates
rather than assuming one.

**The gate, fixed before any log-loss was computed.** A candidate must beat 13.5 on *held-out*
log-loss, walking forward one season at a time, fitting only on strictly earlier seasons. If
none does, 13.5 stays — and an asserted number that survives a fit is no longer asserted, which
is a result worth having.

    uv run python -m hub.models.margin --fit
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from math import erf, sqrt

import numpy as np
import polars as pl

# The incumbent. Imported rather than restated so the two cannot drift apart.
from hub.models.experiment import expanding_seasons
from hub.models.market import MARGIN_SD
from hub.models.scoring_rules import log_loss

# What the 2026-08-24 fit found, kept so a test can guard the live constant against it -- the
# same pattern `hub.draft.calibrate.FITTED_CI95` uses for TALENT_CV. A refit updates both.
FITTED_SD = 12.741
FITTED_SE = 0.164
FITTED_N = 3018
FITTED_WINDOW = "trailing 10 seasons, as of 2025"

# Candidate windows, named before the numbers.
#
# `all` is every season on record; `trailing10` reflects the measured downward trend, on the
# view that a decade-old market is a different market. Both are compared against the incumbent,
# and the incumbent wins ties -- an equal-performing new number is not an improvement, it is
# churn in a constant that is hashed into every model version.
CANDIDATES = ("incumbent", "all", "trailing10")
TRAILING = 10

# A tie is neither a home win nor an away win, and there is no sensible probability to score it
# against. 1999-2025 has a handful; dropping them is cleaner than inventing a convention.
DROP_TIES = True


def residuals(schedules: pl.DataFrame) -> pl.DataFrame:
    """(season, spread_line, result, resid) for every completed game with a closing spread."""
    keep = ("season", "spread_line", "result")
    if not set(keep) <= set(schedules.columns):
        missing = sorted(set(keep) - set(schedules.columns))
        raise ValueError(f"schedules is missing {missing}")
    out = (schedules.select(keep)
                    .drop_nulls(["spread_line", "result"])
                    .with_columns((pl.col("result") - pl.col("spread_line")).alias("resid")))
    if DROP_TIES:
        out = out.filter(pl.col("result") != 0)
    return out


def fit(resid: pl.DataFrame) -> dict[str, float]:
    """Sample dispersion of the residual, with a standard error.

    `se(sd) ≈ sd / sqrt(2(n-1))` — the large-sample standard error of a standard deviation.
    Reported because a point estimate of a dispersion invites being compared to another point
    estimate, and the whole question here is whether 13.5 is far enough away to matter.
    """
    r = resid["resid"].to_numpy().astype(float)
    n = r.size
    if n < 2:
        return {"n": float(n), "sd": float("nan"), "se": float("nan"), "mean": float("nan")}
    sd = float(r.std(ddof=1))
    return {"n": float(n), "sd": sd, "se": sd / sqrt(2 * (n - 1)), "mean": float(r.mean())}


def home_win_prob(spread: np.ndarray, sd: float) -> np.ndarray:
    """P(home team wins) given a home-relative closing spread and a margin dispersion."""
    z = np.asarray(spread, dtype=float) / (sd * sqrt(2.0))
    return 0.5 * (1.0 + np.array([erf(v) for v in z]))





def walk_forward(resid: pl.DataFrame, *, trailing: int = TRAILING) -> pl.DataFrame:
    """Held-out log-loss per season for each candidate, fitting only on earlier seasons.

    Expanding window, one season at a time, never peeking. The first season with any history
    is the first that can be scored, so the earliest season on record is used only for fitting.
    `min_past=2` because `fit` needs two residuals before it has a standard deviation.
    """
    rows = []
    for yr, past, now in expanding_seasons(resid, min_past=2):
        sds = {
            "incumbent": MARGIN_SD,
            "all": fit(past)["sd"],
            "trailing10": fit(past.filter(pl.col("season") >= yr - trailing))["sd"],
        }
        spread = now["spread_line"].to_numpy().astype(float)
        won = (now["result"].to_numpy().astype(float) > 0).astype(float)
        row: dict[str, float] = {"season": yr, "n": now.height}
        for name, sd in sds.items():
            row[f"sd_{name}"] = sd
            row[f"ll_{name}"] = log_loss(home_win_prob(spread, sd), won)
        rows.append(row)
    return pl.DataFrame(rows)


def _mean(df: pl.DataFrame, col: str) -> float:
    """Column mean as a plain float. Polars types `.mean()` as a union including None."""
    v = df[col].mean()
    return float(v) if isinstance(v, (int, float)) else float("nan")


def verdict(wf: pl.DataFrame) -> tuple[str, str]:
    """The pre-registered rule. Returns (winning candidate, the sentence explaining it).

    A candidate must beat the incumbent on **mean held-out log-loss**. Ties go to the
    incumbent: replacing a constant that is hashed into every model version, for no measured
    gain, is churn rather than improvement.
    """
    if wf.is_empty():
        return "incumbent", "no held-out seasons; 13.5 stands by default."
    means = {c: _mean(wf, f"ll_{c}") for c in CANDIDATES}
    base = means["incumbent"]
    challengers = {c: m for c, m in means.items() if c != "incumbent"}
    best = min(challengers, key=lambda c: challengers[c])
    if challengers[best] < base:
        gain = base - challengers[best]
        return best, (f"ADOPT '{best}': mean held-out log-loss {challengers[best]:.5f} against "
                      f"{base:.5f} for MARGIN_SD={MARGIN_SD}, an improvement of {gain:.5f}.")
    return "incumbent", (f"KEEP {MARGIN_SD}: no candidate beat it on held-out log-loss "
                         f"(best challenger {challengers[best]:.5f} against {base:.5f}). "
                         f"An asserted number that survives a fit is no longer asserted.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.margin",
        description="Fit the margin dispersion around the closing spread, and gate it.")
    ap.add_argument("--fit", action="store_true", help="run the walk-forward and report")
    ap.add_argument("--trailing", type=int, default=TRAILING)
    ap.add_argument("--out", default=None, help="write the per-season frame to this parquet")
    a = ap.parse_args(argv)
    if not a.fit:
        ap.print_help()
        return 0

    import nflreadpy as nfl

    print("  loading schedules ...")
    resid = residuals(nfl.load_schedules())
    whole = fit(resid)
    print(f"\n  full sample: n={int(whole['n'])}  sd={whole['sd']:.3f} +/-{whole['se']:.3f}  "
          f"mean residual {whole['mean']:+.3f}")
    print(f"  incumbent MARGIN_SD={MARGIN_SD} is "
          f"{(MARGIN_SD - whole['sd']) / whole['se']:+.1f} se from the full-sample fit")

    wf = walk_forward(resid, trailing=a.trailing)
    print(f"\n  Walk-forward, {wf.height} held-out seasons, fitting only on earlier ones.")
    print(f"  {'season':>6} {'n':>4}  " + "  ".join(f"{'ll_' + c:>13}" for c in CANDIDATES))
    for r in wf.tail(10).iter_rows(named=True):
        print(f"  {r['season']:>6} {r['n']:>4}  "
              + "  ".join(f"{r['ll_' + c]:>13.5f}" for c in CANDIDATES))
    print(f"  {'mean':>6} {'':>4}  "
          + "  ".join(f"{_mean(wf, 'll_' + c):>13.5f}" for c in CANDIDATES))

    winner, sentence = verdict(wf)
    print(f"\n  {sentence}")
    if winner != "incumbent":
        final = fit(resid if winner == "all" else
                    resid.filter(pl.col("season") >= resid["season"].max() - a.trailing))
        print(f"  Value to adopt: {final['sd']:.3f} +/-{final['se']:.3f} (n={int(final['n'])})")
    if a.out:
        wf.write_parquet(a.out)
        print(f"  wrote {wf.height} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
