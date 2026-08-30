"""Does any week-level feature predict a player's week *beyond what consensus already knows*?

A **screen**, in the sense `CONTEXT.md` defines: it asks "is this real?", never "is this better
than what it replaces?". Nothing here adopts anything. The gate that decides whether a Weekly
projection sets lineups is a different question with a different incumbent, and
[ADR-0015](../../../docs/adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md) records
why it has to be.

This module is the **statistic and the verdict**, and nothing else. What it measures *on* -- the
**Panel**, one row per player-week with every feature measured before its outcome -- is
`hub.models.panel`, which two other modules also read and were reaching in here to get.

Committed rather than left in a scratchpad because these numbers steer Phase 2 of
`docs/weekly-projection-plan.md`, and ADR-0007's trigger is citation.

THE DESIGN, PRE-REGISTERED in `docs/weekly-projection-plan.md` before the first run:

  * Outcome is player-week PPR points. Controls are season-to-date PPG measured strictly
    before the week, and that week's `weekly-op` consensus ECR.
  * **One partial correlation per (season, week) cell.** Every player appears at most once
    inside a cell, so no correlation contains repeated measures -- signal-screens.md protocol
    item 3, which turned noise into an apparent 4-sigma result once already. Pooling
    player-weeks would inflate every t here by roughly the square root of fourteen.
  * A feature clears only if its pre-stated sign holds in **every** season and the pooled
    statistic clears `MIN_SE`. A sign that flips between seasons is a bug, not a signal
    (protocol item 4), and it is the cheapest diagnostic available.

THE CONFOUND, which the first run found and which no available data removes: `weekly-op` is
FantasyPros' Monday ranking, scraped a median of six days before kickoff. Any feature carrying
Tuesday-to-Sunday news beats it for that reason alone. `LEAD_DAYS` reports the distribution so
the split is visible rather than assumed -- see `docs/weekly-screen.md`.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np
import polars as pl

from hub.config import FANTASY_WEEKS
from hub.models.experiment import MIN_SE
from hub.models.panel import (
    MIN_GAMES_BEFORE,
    OUTCOME,
    SCHEME,
    SEASONS,
    TREND_MIN_WEEK,
    USAGE,
    PanelSpec,
    build_panel,
)

# A cell smaller than this is a correlation on noise. 40 is roughly a tenth of a normal week.
MIN_CELL = 40


class Feature(NamedTuple):
    """A candidate, with the sign written down before the run."""
    name: str
    sign: str          # "+", "-", or "0" for a pre-stated null
    min_week: int


FEATURES: tuple[Feature, ...] = (
    Feature("implied_total", "+", 1),
    Feature("own_spread", "?", 1),
    Feature("dvp", "+", 1),
    Feature("wind", "-", 1),
    Feature("rest", "?", 1),
    Feature("inj_sev", "-", 1),
    Feature("td_rate_prior", "0", 1),
    Feature("snap_trend", "+", TREND_MIN_WEEK),
    Feature("tgt_trend", "+", TREND_MIN_WEEK),
)


# Screened 2026-08-29 and **not** in FEATURES, deliberately. `route_trend` clears on its own
# (+0.034 at 2.5 se, 5/5 seasons) and is a *null against `snap_trend`*: the two correlate at
# **0.917**, their underlying shares at **0.963**, and put in the same joint screen they
# annihilate each other and leave nothing. Snap share is the stronger of the two (+0.043
# against +0.034), so the literature's access-beats-presence distinction does not survive
# here. Kept in the tree with its harness per ADR-0007, out of the default screen because a
# collinear twin in the control set destroys a real signal. `--routes` reproduces it.
ROUTE_TREND = Feature("route_trend", "+", TREND_MIN_WEEK)


CONTROLS: tuple[str, ...] = ("ppg_before", "ecr")


# A verdict is one of three, not two. A pre-stated null that comes back significant in every
# season is a *finding* -- it is why the prediction was written down -- and folding it in with
# the rejections would lose the most informative outcome the screen can produce.
CLEARS, KILLED, NULL_BROKEN = "clears", "killed", "null-broken"


FINDINGS = (CLEARS, NULL_BROKEN)


def residual(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """`y` with the controls projected out. An intercept is added here, never by the caller."""
    x = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


# A residual smaller than this fraction of the original spread is floating-point dust, not a
# signal. Exact zero is the wrong test: a feature that is an exact linear function of a control
# residualises to ~1e-16 rather than to 0, and correlating two clouds of rounding error returns
# a number that looks like a finding.
DEGENERATE = 1e-10


def partial_r(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    """Correlation of `y` and `x` after both are residualised on the same controls.

    Both sides, which is what makes it partial rather than a regression coefficient: the
    quantity is scale-free and comparable across features that have no common units.

    Returns NaN when either residual has collapsed, so the caller drops the cell instead of
    reporting the correlation of two rounding errors.
    """
    ry, rx = residual(y, controls), residual(x, controls)
    if (rx.std() <= DEGENERATE * max(float(np.std(x)), 1.0)
            or ry.std() <= DEGENERATE * max(float(np.std(y)), 1.0)):
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def cell_correlations(panel: pl.DataFrame, feature: str, *, min_week: int = 1,
                      min_cell: int = MIN_CELL, outcome: str = OUTCOME,
                      controls: Sequence[str] = CONTROLS) -> pl.DataFrame:
    """One partial correlation per (season, week). No player appears twice inside a cell.

    `outcome` is a parameter because the same screen has to run against **Usage** -- targets,
    carries, attempts -- and not only against points. That is the premise of the multiplier
    form: a feature that moves points but not counts cannot be applied as a Usage multiplier,
    whatever it does to the total.
    """
    need = [outcome, feature, *controls]
    d = panel.filter(pl.col("week") >= min_week).drop_nulls(need)
    rows = []
    for (season, week), cell in d.group_by(["season", "week"]):
        if cell.height < min_cell:
            continue
        r = partial_r(cell[outcome].to_numpy().astype(float),
                      cell[feature].to_numpy().astype(float),
                      np.column_stack([cell[c].to_numpy().astype(float) for c in controls]))
        if not np.isnan(r):
            rows.append({"season": int(season), "week": int(week), "r": r, "n": cell.height})
    return pl.DataFrame(rows, schema={"season": pl.Int64, "week": pl.Int64,
                                      "r": pl.Float64, "n": pl.Int64})


def summarise(cells: pl.DataFrame) -> dict:
    """Pooled correlation, its standard error across cells, and the per-season means.

    The standard error is of the *cell* correlations, not of the pooled player-weeks. Cells
    within a season share players, so this is not fully independent either -- but it is the
    difference between a mild overstatement and the fourteen-fold one that pooling gives.
    """
    if cells.is_empty():
        return {"r": float("nan"), "se": float("nan"), "t": float("nan"),
                "cells": 0, "n": 0, "per_season": {}}
    r = cells["r"].to_numpy().astype(float)
    se = float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
    per = {int(s): float(cast(float, cells.filter(pl.col("season") == s)["r"].mean()))
           for s in sorted(cells["season"].unique().to_list())}
    return {"r": float(r.mean()), "se": se,
            "t": float(r.mean() / se) if se > 0 else 0.0,
            "cells": len(r), "n": int(cells["n"].sum()), "per_season": per}


def verdict(summary: dict, sign: str, *, min_se: float = MIN_SE) -> tuple[str, str]:
    """The pre-registered rule, both halves of it.

    A feature clears only if the sign it was given *before the run* holds in every season and
    the pooled statistic clears `min_se`. `sign="0"` is a pre-stated null: it clears when it
    behaves like one, and a null that comes back significant is reported as a finding against
    the pre-registration rather than quietly relabelled.
    """
    per = summary["per_season"]
    if not per:
        return KILLED, "nothing measured -- no cell reached the minimum"
    t = summary["t"]
    agree = sum((v > 0) == (summary["r"] > 0) for v in per.values())
    if sign == "0":
        if abs(t) < min_se:
            return CLEARS, f"null as pre-stated ({t:+.1f} se)"
        if agree == len(per):
            return NULL_BROKEN, (f"PRE-STATED NULL BROKEN: {summary['r']:+.4f} at {t:+.1f} se, "
                                 f"consistent in {agree}/{len(per)} seasons")
        return CLEARS, f"noisy, not a signal ({agree}/{len(per)} seasons agree)"
    held = agree if sign == "?" else (
        sum(v > 0 for v in per.values()) if sign == "+" else sum(v < 0 for v in per.values()))
    if held < len(per):
        return KILLED, (f"killed: sign holds in only {held}/{len(per)} seasons "
                        f"({summary['r']:+.4f} at {t:+.1f} se)")
    if abs(t) < min_se:
        return KILLED, f"killed: {held}/{len(per)} seasons but only {t:+.1f} se"
    return CLEARS, f"clears: {summary['r']:+.4f} at {t:+.1f} se, {held}/{len(per)} seasons"


def report(rows: Sequence[dict]) -> list[str]:
    """Lines, not prints -- the reason `hub.draft.report` exists."""
    out = ["", f"  {'feature':16} {'pre':>4} {'r':>9} {'t':>7} {'cells':>6}  verdict"]
    for row in rows:
        out.append(f"  {row['feature']:16} {row['sign']:>4} {row['r']:+9.4f} "
                   f"{row['t']:+7.2f} {row['cells']:6d}  {row['note']}")
    return out


SCHEME_TRENDS: tuple[Feature, ...] = tuple(
    Feature(f"{r}_trend", "?" if r != "nohuddle_rate" else "+", TREND_MIN_WEEK)
    for r in (*SCHEME, "pass_rate"))


def screen(panel: pl.DataFrame, features: Sequence[Feature] = FEATURES) -> pl.DataFrame:
    """Every feature, screened, with its pre-stated sign and its verdict."""
    rows = []
    for f in features:
        s = summarise(cell_correlations(panel, f.name, min_week=f.min_week))
        status, note = verdict(s, f.sign)
        rows.append({"feature": f.name, "sign": f.sign, "r": s["r"], "se": s["se"],
                     "t": s["t"], "cells": s["cells"], "n": s["n"],
                     "status": status, "note": note,
                     "per_season": str({k: round(v, 4) for k, v in s["per_season"].items()})})
    return pl.DataFrame(rows).sort("r", descending=True)


def screen_joint(panel: pl.DataFrame, survivors: Sequence[Feature]) -> pl.DataFrame:
    """Re-screen each survivor with the other survivors added to the controls.

    Without this the screen reports collinear features as separate findings. The first run
    found `own_spread` at +0.042 across all five seasons and `implied_total` at +0.055 across
    all five -- and `implied_total = total_line/2 + own_spread/2`, so they correlate at 0.83
    and are one finding wearing two hats. Controlled for the total, the spread leaves nothing.

    A feature that clears alone and dies here is not a signal; it is another signal's shadow.
    """
    rows = []
    for f in survivors:
        # Each feature keeps its OWN week range and is controlled only for survivors that
        # exist over it. Taking the widest min_week across the set instead would drop
        # `implied_total` from 54 cells to 35 purely because `snap_trend` starts at week 8,
        # and then report the lost power as a failed control.
        others = [g.name for g in survivors
                  if g.name != f.name and g.min_week <= f.min_week]
        s = summarise(cell_correlations(panel, f.name, min_week=f.min_week,
                                        controls=(*CONTROLS, *others)))
        status, note = verdict(s, f.sign)
        rows.append({"feature": f.name, "sign": f.sign, "r": s["r"], "t": s["t"],
                     "cells": s["cells"], "status": status,
                     "note": note, "controls": ", ".join(others) or "-"})
    return pl.DataFrame(rows).sort("r", descending=True)


def screen_usage(panel: pl.DataFrame, features: Sequence[Feature],
                 components: Sequence[str] = USAGE) -> pl.DataFrame:
    """Each feature against each Usage count, controlled for that count's own recent level.

    The controls are that count's own **season-to-date** mean and its **last three weeks**,
    plus consensus ECR. Not `ppg_before`: asking whether a feature predicts this week's targets
    beyond his season-to-date *points* would let a change in role show up as a target signal.
    And not the season-to-date mean alone, which lags -- see `recent_mean`.
    """
    rows = []
    for f in features:
        for c in components:
            cells = cell_correlations(panel, f.name, min_week=f.min_week, outcome=c,
                                      controls=(f"{c}_prior", f"{c}_recent", "ecr"))
            s = summarise(cells)
            status, note = verdict(s, f.sign)
            rows.append({"feature": f.name, "component": c, "sign": f.sign,
                         "r": s["r"], "t": s["t"], "cells": s["cells"],
                         "status": status, "note": note})
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:      # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.models.weekly_screen",
        description="Screen week-level features beyond weekly consensus.")
    ap.add_argument("--run", action="store_true", help="build the panel and screen")
    ap.add_argument("--scheme", action="store_true",
                    help="add the team scheme trends -- improvements.md #4")
    ap.add_argument("--routes", action="store_true",
                    help="add route_trend -- reproduces the null against snap_trend")
    ap.add_argument("--usage", action="store_true",
                    help="screen the survivors against Usage counts, not points")
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0
    seasons = [int(x) for x in a.seasons.split(",") if x]
    panel = build_panel(seasons, PanelSpec(routes=a.routes, scheme=a.scheme))
    sample = panel.filter(pl.col("week").is_in(list(FANTASY_WEEKS))
                          & (pl.col("games_before") >= MIN_GAMES_BEFORE))
    sample = sample.drop_nulls([OUTCOME, *CONTROLS])
    print(f"  {sample.height} player-weeks, {sample['player_id'].n_unique()} players, "
          f"seasons {sorted(sample['season'].unique().to_list())}")
    lead = panel["lead_days"]
    print(f"  consensus scraped a median {lead.median():.0f} days before kickoff "
          f"(the confound: see docs/weekly-screen.md)")
    extra = ((ROUTE_TREND,) if a.routes else ()) + (SCHEME_TRENDS if a.scheme else ())
    out = screen(sample, (*FEATURES, *extra))
    print("\n".join(report(out.to_dicts())))
    found = out.filter(pl.col("status").is_in(list(FINDINGS)))["feature"].to_list()
    print(f"\n  a signal on its own: {', '.join(found) if found else 'nothing'}")
    if len(found) > 1:
        pool = (*FEATURES, *extra)
        survivors = [f for f in pool if f.name in found]
        joint = screen_joint(sample, survivors)
        print("\n  each one, controlled for the others that exist over its weeks:")
        print("\n".join(report(joint.to_dicts())))
        left = joint.filter(pl.col("status").is_in(list(FINDINGS)))["feature"].to_list()
        print(f"\n  independent signals: {', '.join(left) if left else 'nothing'}")
        if a.usage and left:
            print("\n  and against Usage rather than points:")
            u = screen_usage(sample, [f for f in pool if f.name in left])
            for row in u.iter_rows(named=True):
                print(f"  {row['feature']:16} {row['component']:11} {row['r']:+7.4f} "
                      f"{row['t']:+6.2f}  {row['status']}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
