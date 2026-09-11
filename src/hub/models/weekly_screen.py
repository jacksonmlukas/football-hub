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
  * **The season is the unit of replication, so the standard error is over the seasons.**
    Cells are how the correlation is computed without repeating a player; they are not
    independent of each other, sharing a schedule, a rules year and one consensus source.
    The verdict rests on four or five season means and the interval has to rest on the same
    four or five -- an se over dozens of cells beside a verdict over seasons reports a
    precision the decision never had. Issue #169; `docs/method.md` rule 3, one level up
    from the pooling this screen was already built to avoid.

THE SCREEN'S MINIMUM WEEK is swept, not chosen -- #178. It used to be `panel.TREND_MIN_WEEK`,
which is 8 because 8 is the earliest of the anchors 4, 6, 8, 10, 12 that held when they were
tested *against the outcome*; using it here screened features on rows selected by a value
fitted to the outcome on the same data, and the write-up did not disclose the value. The model
threshold keeps that value because a threshold is what the measurement was for. The screen has
`SCREEN_TREND_ANCHORS` instead, runs at every one of them, and `sensitivity` names any verdict
that is not the same at all five. One is not: see `docs/weekly-screen.md`.

THE CONTROL BASIS is `(yds_prior, ecr)` -- #229, deciding what #179 found. The screen was
pre-registered on `(ppg_before, ecr)`, and `ppg_before` is PPR points, which contain touchdowns:
the control set held `td_rate_prior`'s own numerator, so at a fixed points total a higher
touchdown rate was arithmetically fewer yards. Splitting the control into its touchdown and
non-touchdown halves was tried first and made it worse -- it pinned the numerator, and the
partial correlation between the feature and prior yardage went from -0.114 to -0.401. Prior
yardage and consensus rank is the one basis that neither contains the numerator nor pins it,
and it is the basis #179's own issue body pre-registered. Under it `td_rate_prior` is -0.012
at -2.49 se with 2023 positive: 4/5 seasons, and `docs/method.md` rule 4 fires. **It is not a
finding.** `BASES` keeps `pooled` and `decomposed` reachable through `--basis` as the record
of what was tried; every figure taken on them is restated on this basis in
`docs/weekly-screen.md` under rule 13, all eight features and not the one the ticket was about.

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

from hub.cli import unavailable
from hub.config import FANTASY_WEEKS, digests, resolved_config
from hub.fetch.nflverse import pins_this_run, reads_of_one_run
from hub.models.experiment import MIN_SE
from hub.models.panel import (
    MIN_GAMES_BEFORE,
    OUTCOME,
    SCHEME,
    SEASONS,
    USAGE,
    PanelSpec,
    build_panel,
    require_features,
)

NOT_FITTED_BECAUSE = (
    "the Phase 1 screen for week-level features. DEGENERATE is a floating-point tolerance -- "
    "below it a residual is rounding error rather than a signal -- and nothing here predicts: "
    "it reads outcomes and reports correlations. See docs/weekly-screen.md "
)

# A cell smaller than this is a correlation on noise. 40 is roughly a tenth of a normal week.
MIN_CELL = 40


class Feature(NamedTuple):
    """A candidate, with the sign written down before the run."""
    name: str
    sign: str          # "+", "-", or "0" for a pre-stated null
    min_week: int


# **Not a week.** A trend feature has no single minimum week here any more -- #178 -- so the
# tuple below declares the sign, which is pre-registered, and leaves the row filter to be
# supplied by `at_anchor` from the sweep. Zero rather than a plausible week so that a caller
# who forgets is caught by `require_anchor` instead of quietly screening from week 0.
TREND_ANCHOR_UNSET = 0


# **The screen's minimum week, swept rather than chosen -- #178.** These are the five anchors
# `docs/snap-trend-signal.md` measured the trend at, and the range and the step are fixed here,
# in the commit that runs the sweep, rather than picked after seeing which range flatters the
# answer. That is the pre-registration the disposition on #178 adopted.
#
# The justification is deliberately **not** `panel.TREND_MIN_WEEK`'s. That constant is 8
# because 8 is the earliest anchor at which the trend held when the anchors were tested
# against the outcome, which is the right way to set a *model* threshold and the wrong way to
# select the rows a *screen* reads: it would screen features on rows chosen by a number fitted
# to the outcome on the same data. A sweep does not choose at all. It converts a hidden
# dependency into a published one, which is the honest answer when the constant was never
# derived for this job in the first place.
SCREEN_TREND_ANCHORS: tuple[int, ...] = (4, 6, 8, 10, 12)


# The anchor the published tables on `docs/weekly-screen.md` were taken at. Named so a re-run
# can reproduce the page, and it carries no claim of its own -- the sensitivity across
# `SCREEN_TREND_ANCHORS` is what the page now rests on.
PUBLISHED_ANCHOR = 8


# **Eight, not nine -- `wind` is gone and the family size moved with it (#170).** It was
# screened as a week-w pre-kickoff feature with a pre-stated negative sign, and the reading is
# taken *at* kickoff: game-time observed weather, read off the schedule as recorded conditions.
# That is `docs/method.md` rule 2 -- measure the predictor strictly before the outcome window --
# broken by the screen that exists to enforce it, which is the same shape as rule 3's second
# incident one paragraph up in that file.
#
# The null-filling compounded it. `panel.game_context` coded a dome or an absent reading as
# `0.0`, so "no measurement" and "no wind" were one number on a feature whose sign was
# pre-registered; on the frozen archive **175 of 416 game rows carry no reading and not one
# game has a measured zero**, so every zero in that column was a non-measurement. Both halves
# are fixed in `hub.models.panel`: the fill is gone and `wind` is `RECORDED` rather than
# `PRE_KICKOFF`, so `require_features` now refuses it here as it would refuse `targets`.
#
# **Removed from the family rather than quietly retained**, which is the ticket's third
# criterion: a screen that reports eight verdicts while nine features were tried is a multiple
# -comparison count that understates itself. `docs/weekly-screen.md` carries the restatement of
# what the wind row published, under rule 13.
#
# What is *not* claimed is that wind does not matter. It is an explanatory variable this panel
# can describe a week with and cannot screen; screening it needs a forecast published before
# kickoff, which no source in this repo carries.
FEATURES: tuple[Feature, ...] = (
    Feature("implied_total", "+", 1),
    Feature("own_spread", "?", 1),
    Feature("dvp", "+", 1),
    Feature("rest", "?", 1),
    Feature("inj_sev", "-", 1),
    Feature("td_rate_prior", "0", 1),
    Feature("snap_trend", "+", TREND_ANCHOR_UNSET),
    Feature("tgt_trend", "+", TREND_ANCHOR_UNSET),
)


# Screened 2026-08-29 and **not** in FEATURES, deliberately. `route_trend` clears on its own
# (+0.034 at 2.5 se, 5/5 seasons) and is a *null against `snap_trend`*: the two correlate at
# **0.917**, their underlying shares at **0.963**, and put in the same joint screen they
# annihilate each other and leave nothing. Snap share is the stronger of the two (+0.043
# against +0.034), so the literature's access-beats-presence distinction does not survive
# here. Kept in the tree with its harness per ADR-0007, out of the default screen because a
# collinear twin in the control set destroys a real signal. `--routes` reproduces it.
#
# Every figure in this comment is a pooled-basis figure, taken before #229 moved the default,
# and `--routes` now runs on the settled basis. The collinearity between the two shares is a
# property of the columns and does not depend on the controls; the two coefficients do, and
# have not been re-run here.
ROUTE_TREND = Feature("route_trend", "+", TREND_ANCHOR_UNSET)


def at_anchor(features: Sequence[Feature], anchor: int) -> tuple[Feature, ...]:
    """`features` with every trend feature's minimum week set to `anchor`.

    A trend feature is one whose name ends in `_trend`, which is every windowed feature the
    Panel carries and nothing else -- `snap_trend`, `tgt_trend`, `route_trend` and the five
    `SCHEME_TRENDS`. They are the features whose value is a change over a window, so they are
    the ones with a week before which there is not enough season to form the window; the
    pre-kickoff facts are the same quantity in week 2 as in week 12.
    """
    return tuple(f._replace(min_week=anchor) if f.name.endswith("_trend") else f
                 for f in features)


def require_anchor(features: Sequence[Feature]) -> None:
    """Refuse a feature set that still carries `TREND_ANCHOR_UNSET`.

    The failure this exists to catch is silent, which is the only reason it is a guard rather
    than a convention: `min_week=0` filters nothing, so a forgotten `at_anchor` would screen
    the trend features from week 1 and report a number rather than an error -- a number taken
    on rows the trend does not exist over, printed in the same column as the others.
    """
    unset = [f.name for f in features if f.min_week == TREND_ANCHOR_UNSET]
    if unset:
        raise ValueError(
            f"{', '.join(unset)} carry no anchor. The screen has no single minimum week for a "
            f"trend feature (#178) -- pass the set through `at_anchor` with one of "
            f"{SCREEN_TREND_ANCHORS}, or run the sweep.")


# **The control basis every surviving claim is conditional on -- #229.** Season-to-date
# *yardage* a game, strictly before week w, and that week's consensus rank. Not the
# pre-registration: that was `CONTROLS_POOLED` below, and it contains `td_rate_prior`'s own
# numerator. This is the set #179's issue body pre-registered and the maintainer adopted on
# 2026-09-07, before the decomposition replaced it; the decomposition was then measured and
# found to concentrate the confound rather than remove it, and the decision on #229 came back
# to this. It is the only basis of the three that neither contains the touchdown count nor
# pins it.
#
# What it costs is stated rather than hidden. `yds_prior` is a weaker player control than
# `ppg_before` -- it is the yardage half of prior scoring, not the whole of it -- so the
# eight features are being asked a slightly different question than they were asked on the
# published basis, and `docs/weekly-screen.md` reports all eight on it rather than the one the
# ticket was about. Moving the basis for one feature and not the rest would itself be a choice.
CONTROLS: tuple[str, ...] = ("yds_prior", "ecr")


# **The pre-registration, and the basis every figure published before #229 rests on.** Kept
# reachable through `--basis pooled` because those figures are the record and a re-run has to
# be able to reproduce them; not the default, because `ppg_before` is PPR points, PPR points
# contain touchdowns, and `td_rate_prior` is `tds_prior / yds_prior` -- so at a fixed points
# total a higher touchdown rate is arithmetically fewer yards, and the -0.040 this basis
# returned could not say whether it was efficiency or yardage.
CONTROLS_POOLED: tuple[str, ...] = ("ppg_before", "ecr")


# **The alternative #179 pre-registered and ran, kept as the record of what it showed.**
# `td_ppg_before + nontd_ppg_before == ppg_before` on every row, so this set **spans** the
# pooled one and the only constraint relaxed is that a point of touchdown scoring and a point
# of everything else carry the same slope. That constraint turned out not to bind (the halves
# want +0.328 and +0.289, a difference at t +1.75), and holding the touchdown half fixed pins
# the feature's numerator, so what varies in `tds_prior / yds_prior` is very nearly the
# denominator alone: the partial correlation between the feature and `yds_prior` goes from
# -0.114 on the pooled basis to -0.401 here. The instrument built to remove the yardage
# confound concentrated it, which is why #229 did not adopt it.
CONTROLS_DECOMPOSED: tuple[str, ...] = ("td_ppg_before", "nontd_ppg_before", "ecr")


BASES: dict[str, tuple[str, ...]] = {"yardage": CONTROLS, "pooled": CONTROLS_POOLED,
                                     "decomposed": CONTROLS_DECOMPOSED}
"""The control bases a run may be taken on, by the name `--basis` takes.

Each name says what the basis *does* with prior scoring rather than which one is current:
`yardage` holds the yards and leaves the touchdowns free, `pooled` holds the PPR total as one
number, `decomposed` holds its two halves apart. `DEFAULT_BASIS` names the one `CONTROLS` is.
"""


DEFAULT_BASIS = "yardage"
"""The name `--basis` defaults to, and the entry of `BASES` that is `CONTROLS` -- #229."""


# A verdict is one of three, not two. A pre-stated null that comes back significant in every
# season is a *finding* -- it is why the prediction was written down -- and folding it in with
# the rejections would lose the most informative outcome the screen can produce.
CLEARS, KILLED, NULL_BROKEN = "clears", "killed", "null-broken"


def is_signal(status: str, sign: str) -> bool:
    """Whether a verdict is a finding the joint screen should carry forward.

    `CLEARS` means the pre-registration held. For a signed feature that is a signal. For a
    pre-stated null it is the **absence** of one -- the null behaved as a null -- and carrying
    it into the joint screen as a survivor would print "noisy, not a signal" under the heading
    "independent signals" and control every other survivor for a quantity the screen had just
    said carries nothing. `NULL_BROKEN` is the null feature's finding, and the only one it has.

    Latent until #229. `td_rate_prior` is the only pre-stated null in `FEATURES`, and it was
    `NULL_BROKEN` on every basis until the yardage one, so a null that cleared had never
    reached the survivor filter and `(CLEARS, NULL_BROKEN)` was the whole test.
    """
    return status == NULL_BROKEN or (status == CLEARS and sign != "0")


def signals(rows: pl.DataFrame, status: str) -> list[str]:
    """The feature names in `rows` whose verdict in column `status` is a signal."""
    return [d["feature"] for d in rows.select("feature", "sign", status).to_dicts()
            if is_signal(d[status], d["sign"])]


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

    **This is the seam where a column becomes a feature, so this is where the Panel's rule is
    asked.** `feature` and every control must be measured before week w or published for it;
    `outcome` is the one parameter that may be week w's own play, and naming it there is how a
    caller says it means the realised column *as an outcome*. Before #203 the difference was a
    suffix convention and `FEATURES` below was the only thing keeping it -- which made the
    screen correct by the care of whoever last edited a tuple. `screen`, `screen_joint` and
    `screen_usage` all route through here, so asking once covers the three.
    """
    require_features(panel, [feature, *controls])
    need = [outcome, feature, *controls]
    # Sorted before grouping, and grouped in order. `group_by` promises neither the order of
    # the groups nor the order of rows within one, and both reach floating-point sums here:
    # the rows become numpy arrays a correlation is taken over, and the correlations become a
    # pooled mean. Two runs over one panel returned 0.029259338518170086 and then
    # 0.02925933851817009 -- float addition is not associative, so a reordering shows up in
    # the last bit. That is immaterial to any verdict and fatal to a reproducibility claim:
    # `docs/track-record.md` rests on a published run being re-derivable, and "re-derivable to
    # within a rounding error" is a weaker promise than the one this repo makes.
    d = (panel.filter(pl.col("week") >= min_week).drop_nulls(need)
              .sort(["season", "week", "player_id"]))
    rows = []
    for (season, week), cell in d.group_by(["season", "week"], maintain_order=True):
        if cell.height < min_cell:
            continue
        r = partial_r(cell[outcome].to_numpy().astype(float),
                      cell[feature].to_numpy().astype(float),
                      np.column_stack([cell[c].to_numpy().astype(float) for c in controls]))
        if not np.isnan(r):
            rows.append({"season": int(season), "week": int(week), "r": r, "n": cell.height})
    return pl.DataFrame(rows, schema={"season": pl.Int64, "week": pl.Int64,
                                      "r": pl.Float64, "n": pl.Int64}
                        ).sort(["season", "week"])


def summarise(cells: pl.DataFrame) -> dict:
    """The season means, their standard error, and the pooled correlation over them.

    **The unit is the season, because the season is what the verdict evaluates.** `verdict`
    requires the pre-stated sign to hold in every season -- it reads `per_season` and
    nothing else -- so the decision rests on four or five numbers. The standard error used
    to be taken across the *cells*: dozens of season-weeks, and the reported precision came
    from those while the decision came from the seasons. The interval was too narrow for
    the rule reading it, and a t built on one unit beside a verdict built on another is two
    claims about different quantities printed on one line. Issue #169.

    That is `docs/method.md` **rule 3** -- repeated measures are not independent
    observations -- broken by the screen that exists to enforce it. Cells inside a season
    share players, a schedule, a rules year and one consensus source; the module docstring
    above says pooling player-weeks would inflate every t by roughly sqrt(14), and taking
    the season-weeks as independent is the same error one level up. The old docstring knew:
    it called cells "not fully independent either" and kept them anyway.

    Clustering by averaging within the unit and treating the unit vector as the sample is
    `experiment.summarise`'s `cluster` argument, in the module that names getting this wrong
    as the most expensive mistake in the repo's record.

    `cells` and `n` are still reported. They describe the sample and are worth printing;
    they are simply not what the interval is built from -- which was the confusion.
    """
    if cells.is_empty():
        return {"r": float("nan"), "se": float("nan"), "t": float("nan"),
                "cells": 0, "n": 0, "seasons": 0, "per_season": {}}
    per = {int(s): float(cast(float, cells.filter(pl.col("season") == s)["r"].mean()))
           for s in sorted(cells["season"].unique().to_list())}
    # The season means, in season order. Sorted for the reason `cell_correlations` sorts:
    # a float mean that depends on group order is not re-derivable, and `docs/track-record.md`
    # rests on a published run being reproducible bit for bit.
    seasons = np.array([per[s] for s in sorted(per)], dtype=float)
    se = (float(seasons.std(ddof=1) / np.sqrt(len(seasons)))
          if len(seasons) > 1 else 0.0)
    # The mean *of the season means*, not of the cells: with unequal cells per season the
    # two differ, and a t whose numerator and denominator come from different units is the
    # defect in another form. Balanced seasons make them identical.
    r = float(seasons.mean())
    return {"r": r, "se": se,
            "t": float(r / se) if se > 0 else 0.0,
            "cells": cells.height, "n": int(cells["n"].sum()),
            "seasons": len(seasons), "per_season": per}


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
    """Lines, not prints -- the reason `hub.draft.report` exists.

    Both counts are printed, and the header says which one the `t` is built from. A reader
    seeing 55 cells beside a t of 5.7 will read the t as a 55-unit statistic unless the
    column tells them otherwise, and that misreading is exactly what #169 corrected.
    """
    out = ["", f"  {'feature':16} {'pre':>4} {'r':>9} {'t':>7} {'cells':>6} {'szn':>4}"
               "  verdict", "  (t is over the seasons; cells are how each season is built)"]
    for row in rows:
        out.append(f"  {row['feature']:16} {row['sign']:>4} {row['r']:+9.4f} "
                   f"{row['t']:+7.2f} {row['cells']:6d} {row['seasons']:4d}  {row['note']}")
    return out


SCHEME_TRENDS: tuple[Feature, ...] = tuple(
    Feature(f"{r}_trend", "?" if r != "nohuddle_rate" else "+", TREND_ANCHOR_UNSET)
    for r in (*SCHEME, "pass_rate"))


def screen(panel: pl.DataFrame, features: Sequence[Feature] = FEATURES,
           controls: Sequence[str] = CONTROLS) -> pl.DataFrame:
    """Every feature, screened, with its pre-stated sign and its verdict.

    `controls` is a parameter for the same reason `outcome` is one on `cell_correlations`:
    which basis a partial correlation is taken on is a property of the question, and #179 is
    the ticket that found out how much of an answer it can carry. The default is the basis
    #229 decided, `CONTROLS`; the pre-registered set is `CONTROLS_POOLED` and is what every
    figure published before that decision rests on.
    """
    require_anchor(features)
    rows = []
    for f in features:
        s = summarise(cell_correlations(panel, f.name, min_week=f.min_week,
                                        controls=controls))
        status, note = verdict(s, f.sign)
        rows.append({"feature": f.name, "sign": f.sign, "r": s["r"], "se": s["se"],
                     "t": s["t"], "cells": s["cells"], "seasons": s["seasons"],
                     "n": s["n"], "status": status, "note": note,
                     "per_season": str({k: round(v, 4) for k, v in s["per_season"].items()})})
    return pl.DataFrame(rows).sort("r", descending=True)


def screen_joint(panel: pl.DataFrame, survivors: Sequence[Feature],
                 controls: Sequence[str] = CONTROLS) -> pl.DataFrame:
    """Re-screen each survivor with the other survivors added to the controls.

    Without this the screen reports collinear features as separate findings. The first run
    found `own_spread` at +0.042 across all five seasons and `implied_total` at +0.055 across
    all five -- and `implied_total = total_line/2 + own_spread/2`, so they correlate at 0.83
    and are one finding wearing two hats. Controlled for the total, the spread leaves nothing.

    A feature that clears alone and dies here is not a signal; it is another signal's shadow.
    """
    require_anchor(survivors)
    rows = []
    for f in survivors:
        # Each feature keeps its OWN week range and is controlled only for survivors that
        # exist over it. Taking the widest min_week across the set instead would drop
        # `implied_total` from 54 cells to 35 purely because `snap_trend` starts at week 8,
        # and then report the lost power as a failed control.
        others = [g.name for g in survivors
                  if g.name != f.name and g.min_week <= f.min_week]
        s = summarise(cell_correlations(panel, f.name, min_week=f.min_week,
                                        controls=(*controls, *others)))
        status, note = verdict(s, f.sign)
        rows.append({"feature": f.name, "sign": f.sign, "r": s["r"], "t": s["t"],
                     "cells": s["cells"], "seasons": s["seasons"], "status": status,
                     "note": note, "controls": ", ".join(others) or "-"})
    return pl.DataFrame(rows).sort("r", descending=True)


def sweep(panel: pl.DataFrame, features: Sequence[Feature] = FEATURES,
          anchors: Sequence[int] = SCREEN_TREND_ANCHORS,
          controls: Sequence[str] = CONTROLS) -> pl.DataFrame:
    """The whole screen, re-run at every anchor. One row per (anchor, feature) -- #178.

    The screen's minimum week used to be `panel.TREND_MIN_WEEK`, a value chosen by testing
    these same five anchors *against the outcome*. Rows selected by a number fitted to the
    outcome are not a neutral sample to screen features on, and the screen was not disclosing
    which number it had used. Sweeping does not choose: it reports the verdict at each anchor
    and lets `sensitivity` say which verdicts depend on the choice.

    Both halves are re-run at each anchor, not just the first. The anchor sets which rows the
    trend features are measured on **and** which survivors may act as controls in
    `screen_joint`, so a verdict that moves could move by either route.

    `final` is the verdict the write-up reports: the joint one where the feature reached the
    joint screen, and the alone one where it did not. A feature killed alone never enters the
    joint screen, and calling that a joint result would read as a stronger rejection than the
    run performed.
    """
    rows = []
    for anchor in anchors:
        pool = at_anchor(features, anchor)
        alone = screen(panel, pool, controls)
        found = signals(alone, "status")
        survivors = [f for f in pool if f.name in found]
        # Mirrors `main`: one survivor has nothing to be controlled for, so there is no joint
        # screen to run and its alone verdict is the one that stands.
        joint = ({d["feature"]: d for d in screen_joint(panel, survivors, controls).to_dicts()}
                 if len(survivors) > 1 else {})
        for d in alone.to_dicts():
            j = joint.get(d["feature"])
            rows.append({
                "anchor": anchor, "feature": d["feature"], "sign": d["sign"],
                "min_week": next(f.min_week for f in pool if f.name == d["feature"]),
                "alone_r": d["r"], "alone_t": d["t"], "cells": d["cells"],
                "seasons": d["seasons"], "alone": d["status"], "alone_note": d["note"],
                "joint_r": j["r"] if j else None, "joint_t": j["t"] if j else None,
                "joint_cells": j["cells"] if j else None,
                "joint": j["status"] if j else None,
                "joint_note": j["note"] if j else None,
                "joint_controls": j["controls"] if j else None,
                "final": j["status"] if j else d["status"]})
    return pl.DataFrame(rows, schema={
        "anchor": pl.Int64, "feature": pl.Utf8, "sign": pl.Utf8, "min_week": pl.Int64,
        "alone_r": pl.Float64, "alone_t": pl.Float64, "cells": pl.Int64,
        "seasons": pl.Int64, "alone": pl.Utf8, "alone_note": pl.Utf8,
        "joint_r": pl.Float64, "joint_t": pl.Float64, "joint_cells": pl.Int64,
        "joint": pl.Utf8, "joint_note": pl.Utf8, "joint_controls": pl.Utf8,
        "final": pl.Utf8}).sort(["feature", "anchor"])


def surviving(swept: pl.DataFrame, anchor: int) -> list[str]:
    """The feature names that come out of the screen as findings at one anchor."""
    return sorted(signals(swept.filter(pl.col("anchor") == anchor), "final"))


def sensitivity(swept: pl.DataFrame) -> pl.DataFrame:
    """Per feature: whether its verdict is the same at every anchor, and where it holds.

    This is the object #178 asks for. A feature whose verdict is identical across the sweep is
    reported as it stands and the anchor never mattered to it; a feature whose verdict changes
    anywhere is named as depending on the choice and reported with the anchors over which it is
    a finding, rather than as a single verdict taken at whichever anchor is tidiest.

    `stable` is over the *verdict*, not over `r`. Every `r` moves a little with the sample --
    that is what changing the row filter does, and it is not what the pre-registration asked
    about. What it asked is whether the screen's conclusions are conditional on a constant the
    screen was not disclosing.
    """
    rows = []
    for name in swept["feature"].unique(maintain_order=True).to_list():
        d = swept.filter(pl.col("feature") == name).sort("anchor")
        verdicts = d["final"].to_list()
        anchors = d["anchor"].to_list()
        sign = d["sign"][0]
        holds = [a for a, v in zip(anchors, verdicts, strict=True) if is_signal(v, sign)]
        rows.append({
            "feature": name,
            "stable": len(set(verdicts)) == 1,
            "verdict": verdicts[0] if len(set(verdicts)) == 1 else "DEPENDS ON THE ANCHOR",
            "finding_at": ", ".join(str(a) for a in holds) or "-",
            "r_lo": float(min(d["alone_r"].to_list())),
            "r_hi": float(max(d["alone_r"].to_list())),
            "by_anchor": " ".join(f"{a}:{v}" for a, v in
                                  zip(anchors, verdicts, strict=True))})
    return pl.DataFrame(rows, schema={
        "feature": pl.Utf8, "stable": pl.Boolean, "verdict": pl.Utf8, "finding_at": pl.Utf8,
        "r_lo": pl.Float64, "r_hi": pl.Float64, "by_anchor": pl.Utf8})


def sweep_report(swept: pl.DataFrame, sens: pl.DataFrame) -> list[str]:
    """Lines: the surviving set at each anchor, then the features whose verdict moved."""
    anchors = swept["anchor"].unique(maintain_order=False).sort().to_list()
    out = ["", "  the surviving feature set at each anchor:"]
    for a in anchors:
        survived = surviving(swept, a)
        cells = swept.filter((pl.col("anchor") == a)
                             & (pl.col("min_week") == a))["cells"].to_list()
        out.append(f"  week >= {a:<3} {', '.join(survived) if survived else 'nothing':60}"
                   f"  ({min(cells) if cells else 0} trend cells)")
    moved = sens.filter(~pl.col("stable"))
    out += ["", f"  {'feature':16} {'r range':>18}  verdict across the sweep"]
    for row in sens.iter_rows(named=True):
        out.append(f"  {row['feature']:16} {row['r_lo']:+8.4f} {row['r_hi']:+8.4f}  "
                   f"{row['verdict']}"
                   + ("" if row["stable"] else f"  [{row['by_anchor']}]"))
    out.append("")
    out.append("  every verdict holds at all five anchors -- the screen's minimum week "
               "changed nothing" if moved.is_empty() else
               f"  CONDITIONAL ON THE ANCHOR: {', '.join(moved['feature'].to_list())}")
    return out


def every_season_null(panel: pl.DataFrame, feature: Feature, controls: Sequence[str] = CONTROLS,
                      *, draws: int = 2000, seed: int = 0, effects: Sequence[float] = (),
                      min_cell: int = MIN_CELL, outcome: str = OUTCOME) -> dict:
    """How often the every-season half fails under the null of no effect -- #238.

    The feature is permuted **within each (season, week) cell** and re-residualised on the
    controls, so the null keeps everything about the outcome and the controls and breaks only
    the feature's link to the outcome. That is the placebo `docs/weekly-screen.md` reports, run
    `draws` times, and read through the rule rather than through the `t`: the fraction of draws
    in which at least one season mean has the wrong sign is the every-season half's answer on
    pure noise, with the observed number of cells per season. It has to be about **1 - 2^-k**
    for k seasons at every anchor, and it is -- so one season crossing zero says nothing about
    the anchor on its own, and the anchor question is about power, not size.

    `effects` asks that: each is a true partial correlation assumed in every cell, added to the
    null noise, and the result is how often a real effect of that size would fail the
    every-season half at this anchor's cell counts. That is the calculation the ticket names as
    the alternative to a permutation, taken from the permutation's own noise rather than a
    normal approximation.

    A pre-stated null has no sign to hold, so `feature.sign` must be `+`, `-` or `?`; for `?`
    the sign held is the observed one, as `verdict` reads it.
    """
    if feature.sign == "0":
        raise ValueError(f"{feature.name} is a pre-stated null; the every-season half reads a "
                         f"sign it has to hold and a null has none.")
    require_features(panel, [feature.name, *controls])
    rng = np.random.default_rng(seed)
    d = (panel.filter(pl.col("week") >= feature.min_week)
              .drop_nulls([outcome, feature.name, *controls])
              .sort(["season", "week", "player_id"]))
    cells: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for (season, _), cell in d.group_by(["season", "week"], maintain_order=True):
        if cell.height < min_cell:
            continue
        y = cell[outcome].to_numpy().astype(float)
        x = cell[feature.name].to_numpy().astype(float)
        c = np.column_stack([np.ones(len(y)),
                             *[cell[k].to_numpy().astype(float) for k in controls]])
        h = c @ np.linalg.pinv(c)                 # projection onto the controls, intercept in
        ry = y - h @ y
        cells.append((int(season), x, h, ry / np.linalg.norm(ry)))
    seasons = sorted({s for s, *_ in cells})
    if not seasons:
        return {"cells": 0, "seasons": 0, "per_season_cells": {}, "r": float("nan"),
                "p_any_wrong_sign": float("nan"), "p_every_season": float("nan"),
                "p_value": float("nan"), "cell_sd": float("nan"), "alternatives": {}}
    idx = {s: [i for i, (cs, *_) in enumerate(cells) if cs == s] for s in seasons}
    observed = np.array([float(ry @ (x - h @ x) / np.linalg.norm(x - h @ x))
                         for _, x, h, ry in cells])
    obs_means = np.array([observed[idx[s]].mean() for s in seasons])
    r = float(obs_means.mean())
    want = {"+": 1.0, "-": -1.0}.get(feature.sign, float(np.sign(r)) or 1.0)
    null = np.empty((draws, len(cells)))
    for j, (_, x, h, ry) in enumerate(cells):
        xp = np.stack([rng.permutation(x) for _ in range(draws)])
        rx = xp - xp @ h.T
        null[:, j] = (rx @ ry) / np.linalg.norm(rx, axis=1)
    means = np.column_stack([null[:, idx[s]].mean(axis=1) for s in seasons])
    r_null = means.mean(axis=1)
    return {
        "cells": len(cells), "seasons": len(seasons),
        "per_season_cells": {s: len(idx[s]) for s in seasons},
        "r": r, "per_season": {s: float(m) for s, m in zip(seasons, obs_means, strict=True)},
        "p_any_wrong_sign": float((np.sign(means) != want).any(axis=1).mean()),
        "p_every_season": float((np.sign(means) == want).all(axis=1).mean()),
        "p_value": float((np.abs(r_null) >= abs(r)).mean()),
        "cell_sd": float(null.std(axis=0).mean()),
        "alternatives": {float(e): float((np.sign(means + want * e) != want).any(axis=1).mean())
                         for e in effects},
    }


def null_report(name: str, anchor: int, n: dict) -> list[str]:
    """Lines for one `every_season_null` result."""
    out = [f"  {name} from week {anchor}: {n['cells']} cells, per season "
           f"{n['per_season_cells']}; observed r {n['r']:+.4f}, two-sided permutation "
           f"p {n['p_value']:.3f}",
           f"    under the null, P(at least one season has the wrong sign) = "
           f"{n['p_any_wrong_sign']:.3f}; P(every season holds) = {n['p_every_season']:.3f}",
           f"    sd of one cell's r under the null {n['cell_sd']:.4f}"]
    for e, p in n["alternatives"].items():
        out.append(f"    if the true effect were {e:+.4f} in every cell: P(at least one "
                   f"season crosses zero) = {p:.3f}")
    return out


def screen_usage(panel: pl.DataFrame, features: Sequence[Feature],
                 components: Sequence[str] = USAGE) -> pl.DataFrame:
    """Each feature against each Usage count, controlled for that count's own recent level.

    The controls are that count's own **season-to-date** mean and its **last three weeks**,
    plus consensus ECR. Not `ppg_before`: asking whether a feature predicts this week's targets
    beyond his season-to-date *points* would let a change in role show up as a target signal.
    And not the season-to-date mean alone, which lags -- see `recent_mean`.
    """
    require_anchor(features)
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
    ap.add_argument("--permute", action="append", default=[], metavar="FEATURE",
                    help="permute this feature within its cells at each anchor and report "
                         "how often the every-season half fails under the null, and under a "
                         "true effect of the size it was published at -- #238")
    ap.add_argument("--basis", choices=sorted(BASES), default=DEFAULT_BASIS,
                    help="the control basis: 'yardage' holds season-to-date yards a game "
                         "(the default -- #229, and what every surviving claim is conditional "
                         "on); 'pooled' holds season-to-date PPG as one number (the "
                         "pre-registration, and every figure published before #229); "
                         "'decomposed' holds its touchdown and non-touchdown halves apart "
                         "(#179)")
    ap.add_argument("--trend-min-week", dest="trend_min_week", type=int, default=None,
                    metavar="N",
                    help="run the screen at this one anchor instead of sweeping "
                         f"{SCREEN_TREND_ANCHORS}. {PUBLISHED_ANCHOR} reproduces the tables "
                         "on docs/weekly-screen.md -- #178")
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    ap.add_argument("--as-of", dest="as_of", default=None, metavar="YYYY-MM-DD",
                    help="bound the consensus archive at this date, inclusive of the day "
                         "itself; two runs at one as-of read the same rankings")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0
    # The screen's reads are the screen's own, whichever way it was invoked (#192). Run
    # as a process this changes nothing: the scope opens on an empty set, exactly as the
    # process-global one was. Called in-process -- a harness, a notebook -- it is what
    # stops the line below naming bytes the enclosing run read and this screen never
    # touched. The reads still reach the enclosing run on the way out. It encloses the
    # Panel build and the line both, because a scope around one of the two is the defect
    # wearing a different hat.
    with reads_of_one_run():
        seasons = [int(x) for x in a.seasons.split(",") if x]
        try:
            panel = build_panel(seasons, PanelSpec(routes=a.routes, scheme=a.scheme), as_of=a.as_of)
        except Exception as e:
            return unavailable("hub.models.weekly_screen", "the sources the Panel is built from", e)
        # What the run read, printed where the run is read. Until the archive was routed through
        # the fetch layer there was nothing to print: two screens a week apart disagreed and
        # nothing said whether the code or the rankings had moved -- which is the correction
        # `docs/weekly-screen.md` already records having to make once.
        #
        # **Every source the Panel loaded, and not the rankings entry alone (#192).** Until
        # then this line read `panel.consensus_pin(a.as_of)` -- one cache entry, named by its
        # key and looked up on disk -- while the Panel also loads `ff_opportunity` and
        # `player_stats` through the same layer, and `participation` and `ftn_charting` under
        # `--scheme`. A digest over one of three sources is the false one #165 argues against:
        # it compares equal to a run that read only that one. So the line now says what this
        # run read, from the run's own record, and says `unpinned` when any of it cannot be
        # named -- which is what a Panel built on an entry written before pinning existed will
        # print, and is the truth about it. No published figure carries the superseded form:
        # `docs/weekly-screen.md` prints no data digest.
        pins = pins_this_run()
        # `resolved_config()`, not `HubConfig()`: this line exists to say what the run read,
        # and a bare dataclass prints the defaults whatever `conf/` says -- a model version
        # for a model nobody ran, which is what #74 removed from the other three stamps.
        d = digests(resolved_config(), pins)
        print(f"  cfg {d['cfg']} | fitted {d['fitted']} | data {d['data']} over {len(pins)} "
              f"pinned source(s)"
              + ("" if a.as_of else "   (no --as-of: the digest names the bytes this run "
                                    "happened to read, not a date it can be re-read at)"))
        controls = BASES[a.basis]
        sample = panel.filter(pl.col("week").is_in(list(FANTASY_WEEKS))
                              & (pl.col("games_before") >= MIN_GAMES_BEFORE))
        # The **union** of every basis, not just the one being run, so that a `--basis` run is on
        # the same rows as any other and the movement between two of them is attributable to the
        # basis alone. `docs/method.md` rule 13 asks for a re-run rather than an argument, and a
        # re-run on a different sample is an argument with a number attached. Sorted so the drop
        # is the same list in the same order whichever basis was asked for.
        sample = sample.drop_nulls([OUTCOME, *sorted({c for b in BASES.values() for c in b})])
        print(f"  {sample.height} player-weeks, {sample['player_id'].n_unique()} players, "
              f"seasons {sorted(sample['season'].unique().to_list())}")
        print(f"  controls: {', '.join(controls)}   (--basis {a.basis})")
        lead = panel["lead_days"]
        print(f"  consensus scraped a median {lead.median():.0f} days before kickoff "
              f"(the confound: see docs/weekly-screen.md)")
        extra = ((ROUTE_TREND,) if a.routes else ()) + (SCHEME_TRENDS if a.scheme else ())
        pool = (*FEATURES, *extra)
        # The sweep is the default, and one anchor is the special case -- #178. The screen has no
        # single minimum week to fall back on: the value it used to fall back on was fitted to the
        # outcome on these rows, which is what the sweep exists to stop it claiming silently.
        anchors = [a.trend_min_week] if a.trend_min_week else list(SCREEN_TREND_ANCHORS)
        print(f"  trend anchors: {', '.join(str(x) for x in anchors)}"
              + ("   (the whole sweep -- #178)" if len(anchors) > 1 else
                 f"   (one anchor; the sweep is {SCREEN_TREND_ANCHORS})"))
        swept = sweep(sample, pool, anchors, controls)
        for anchor in anchors:
            d = swept.filter(pl.col("anchor") == anchor)
            print(f"\n  === trend features from week {anchor} ===")
            print("\n".join(report([{**r, "r": r["alone_r"], "t": r["alone_t"],
                                     "note": r["alone_note"]}
                                    for r in d.sort("alone_r", descending=True).to_dicts()])))
            found = signals(d, "alone")
            print(f"\n  a signal on its own: {', '.join(found) if found else 'nothing'}")
            j = d.filter(pl.col("joint").is_not_null()).sort("joint_r", descending=True)
            if not j.is_empty():
                print("\n  each one, controlled for the others that exist over its weeks:")
                print("\n".join(report([{**r, "r": r["joint_r"], "t": r["joint_t"],
                                        "cells": r["joint_cells"], "note": r["joint_note"]}
                                       for r in j.to_dicts()])))
            left = surviving(swept, anchor)
            print(f"\n  independent signals: {', '.join(left) if left else 'nothing'}")
            if a.usage and left:
                print("\n  and against Usage rather than points:")
                u = screen_usage(sample, [f for f in at_anchor(pool, anchor) if f.name in left])
                for row in u.iter_rows(named=True):
                    print(f"  {row['feature']:16} {row['component']:11} {row['r']:+7.4f} "
                          f"{row['t']:+6.2f}  {row['status']}")
            for name in a.permute:
                f = next(g for g in at_anchor(pool, anchor) if g.name == name)
                # The alternatives are the sizes the feature has been published at, so the
                # power figure is about the claim on the page and not about a round number.
                alt = (0.0382, 0.0356) if name == "snap_trend" else ()
                n = every_season_null(sample, f, controls, effects=alt)
                print("\n  the every-season half under the null -- #238:")
                print("\n".join(null_report(name, anchor, n)))
        if len(anchors) > 1:
            print("\n".join(sweep_report(swept, sensitivity(swept))))
        return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
