"""The walk-forward paired experiment, which this repo runs twice.

Both `hub.draft.backtest` and `hub.season.lineup_gate` do the same six things: build a board
for each season *as it stood before that season opened*, load what actually happened, play two
arms against identical inputs, summarise the paired difference by bootstrap, print an interval,
and optionally write the rows. Only the middle step -- what the two arms are -- differs.

The two copies had drifted the way copies do. `lineup_gate` imports `play` and
`market_strategy` from `backtest` and then re-declares its own `compare` and `verdict`; the
season-setup loop was byte-identical in both; and each asked `nflverse` for its own slice of
`player_stats`, which the cache keys on the column set, so the same table was fetched twice.

What lives here is the protocol, not the experiment. The arms stay with the harness that owns
the question -- a gate is only meaningful against a specific incumbent, and hiding that behind
a generic interface would be the mistake `CONTEXT.md` warns about when it separates a screen
from a gate.

`hub.models.measure` was folded in on 2026-08-27. It held `realised_ppg` and `summarise`, and
every caller of one was a caller of the other, on adjacent import lines -- the split ran along
the line an earlier extraction happened to stop at rather than along any invariant. Its two
functions are used only inside this experiment.

`expanding_seasons`, `expanding_weeks` and `paired_gain` live here. Both are the *mechanics*
of a walk-forward gate rather than any gate's decision. The split loop was written out four
times -- `margin.walk_forward`, `injury.walk_forward`, `injury.walk_forward_type`,
`spread.walk_forward` -- and had already drifted three ways; it is `docs/method.md` rule #2,
the invariant the repo records violating at 7.4 sigma, and four hand-written copies is four
places a `<` can become a `<=` silently, because a leaking model does not crash. The paired
statistic was written twice, three of five lines byte-identical, under two names for one bar.

What still deliberately does NOT live here: `verdict`. Its branches name a specific incumbent,
so it belongs to the harness asking the question. Every gate writes its own, which is the
point -- a pre-registered rule is specific to what it decides, and a shared one would drift
toward being decorative. The line is between the *statistic* and the *rule*: `paired_gain`
returns numbers and decides nothing, and each `verdict` still spells out its own thresholds,
its own incumbent and its own sentences.
"""
from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
import polars as pl

from hub.config import (
    NO_FRAMES,
    commit,
    config_digest,
    data_digest,
    frames_digest,
    resolved_config,
)
from hub.declare import chosen, not_an_input
from hub.jsonio import stamp as _now
from hub.names import player_key
from hub.paths import STATE_DIR

# One column list, so both harnesses hit one cache entry. `nflverse._cache_path` keys on the
# sorted column set -- deliberately, so a caller asking for six columns is never served an
# earlier caller's five -- which means two callers asking for different slices of the same
# table download it twice.
PLAYER_STATS_COLS: tuple[str, ...] = (
    "player_id", "player_display_name", "position", "season", "week", "fantasy_points_ppr",
)


class CorrectionReport(Protocol):
    """The half of a Board's build report a Gate reads.

    Structural, and that is the whole reason it is declared rather than imported:
    `hub.draft.board.BuildReport` satisfies it without this module reaching into `hub.draft`.
    Six `draft/` modules import `models/` and nothing points back -- `board_as_of` was moved
    out of here for exactly that, and `test_experiment_does_not_reach_into_draft` holds the
    direction. The parameter below used to be typed `object`, which is the same non-import
    with nothing said about what a Gate needs from what it is handed.
    """

    def corrections_missing(self) -> tuple[str, ...]:
        """Correction terms this Board's own ranking was computed without."""
        ...


class ReportedFrame(Protocol):
    """A Board as a Gate is handed one: the frame with its build report attached (#295).

    Structural for the same reason `CorrectionReport` is -- `hub.draft.board.Board` satisfies
    it without this module reaching into `hub.draft` -- and read-only, because the two are one
    object and a Gate that could swap the report from under the frame would be the defect
    the pair exists to end. `walk_forward_inputs` returns these rather than frames so that
    every Gate reads the report off the Board it plays; the stamping half takes the frames
    off them, once, in `stamped_for_publication`.
    """

    @property
    def frame(self) -> pl.DataFrame:
        """The frame every draft-night decision reads."""
        ...

    @property
    def report(self) -> CorrectionReport:
        """What built it."""
        ...


# The Board type a Gate is handed, so a Gate gets back the type it built rather than the
# protocol: `hub.draft.board.Board` in, `dict[int, Board]` out.
ReportedT = TypeVar("ReportedT", bound=ReportedFrame)


class CorrectionMissing(RuntimeError):
    """A Board reached a Gate short a Correction its ranking is computed from."""


def require_corrections(season: int, report: CorrectionReport) -> None:
    """Refuse a Board whose ranking was computed from a subset of the Corrections.

    `hub.draft.board.build` degrades stage by stage on purpose, and one of those stages leaves
    a column a **Correction** reads. Absorbing it does not leave a thinner Board:
    `correct_projection` returns the frame untouched when its column is absent, so what comes
    out is a Corrected ADP computed from a subset of the terms and reported as having run.
    Issue #121 measured it -- absorbing durability alone moves 350 of 457 players by up to
    36.8 picks, and 134 of the first 192 change rank.

    Touchdown luck was the second such stage until #48 emptied its coefficients. It is no
    longer in `CORRECTION_COLUMN`, so a Board built without it is a thinner Board and is not
    refused here -- which is right, because with nothing multiplied its absence moves no
    ranking at all.

    **Refuse rather than record, and the reason is what a Gate is.** CLAUDE.md's degradation
    rule is written for the live path: on draft night an hours-old ADP beats a stack trace,
    and `hub.draft.report` therefore *records* this in the terminal, beside the board, where
    an operator on the clock is reading and the alternative is no board at all. A Gate is not
    that path. It is a harness nobody runs on a clock, and what it produces is a published
    interval that `docs/track-record.md` makes commit-dated -- so serving a degraded season
    here keeps nobody working, it publishes a number measured on two different arms and dates
    it as one. ADR-0019 sharpens that: a Gate adopts only when the sign holds in **every**
    held-out season, which is a comparison of the seasons *to each other*, and a season built
    without a Correction is a different arm rather than a thinner one. A footnote under an
    interval does not survive being quoted; a run that stopped does.

    Refusing costs the operator nothing they cannot see: both harnesses already wrap
    `walk_forward_inputs` in a `try` that ends in `hub.cli.unavailable`, so this arrives as a
    named input, one sentence and a non-zero exit rather than a traceback.

    Nothing is refused for having no Corrected ADP at all. `board_as_of` builds every Board a
    Gate scores today and ESPN publishes ADP for the current season only, so those Boards rank
    on consensus and have no corrected ranking to be short a term -- `corrections_missing`
    says so by returning nothing, and a rule that read "no ADP" as "no durability" would
    refuse every backtest in the repo for a Correction none of them applies.
    """
    missing = report.corrections_missing()
    if not missing:
        return
    raise CorrectionMissing(
        f"the {season} board was built without {' and '.join(missing)}, and its Corrected "
        f"ADP was computed anyway -- so that season is a different ranking from the others' "
        f"and not a thinner board. Issue #121: absorbing durability alone moves 350 of "
        f"457 players by up to 36.8 picks. Rebuild {season} with every Correction, or run "
        f"the seasons that carry them and say which those were.")


def walk_forward_inputs(
    seasons: Sequence[int],
    build_board: Callable[[int], ReportedT],
    *,
    load_stats: Callable[[int], pl.DataFrame] | None = None,
    on_season: Callable[[int], None] | None = None,
) -> tuple[dict[int, ReportedT], dict[int, pl.DataFrame]]:
    """`(boards, realised)` per season. The two injectable seams are what make this testable.

    `build_board` is required rather than defaulted. It used to default to a `board_as_of`
    defined here, which put draft-domain knowledge under `models/` and inverted the tree's one
    consistent direction -- and needed a function-local import to do it. That function now
    lives in `hub.draft.board`, beside the `build` whose rule it states.

    **`build_board` returns the Board with its report attached, and this used to take
    `[0]`.** Every Gate in the repo reaches its Boards through here, so that one subscript
    was where a Board built while a Correction stage was absorbed became indistinguishable
    from a whole one. The report is read rather than dropped -- `require_corrections` says
    what is done with it and why -- and since #295 it is returned *on* the Board rather
    than kept beside it, so the Gate that drafts a Cohort from a season's frame reads the
    same report this refused on, off the same object.

    `on_season` is a progress hook rather than a print, so a caller under a line cap can stay
    quiet and this module stays free of stdout.
    """
    boards: dict[int, ReportedT] = {}
    realised: dict[int, pl.DataFrame] = {}
    for yr in seasons:
        if on_season is not None:
            on_season(yr)
        board = build_board(yr)
        # GUARD a-gate-refuses-a-season-short-a-correction: deleting it puts the season back
        # in the pool, and one season whose Corrected ADP came from a subset of the terms is
        # a second arm inside an interval published as one.
        require_corrections(yr, board.report)
        # /GUARD
        boards[yr] = board
        realised[yr] = realised_ppg((load_stats or _stats)(yr))
    return boards, realised



# The repo's usual significance bar, in standard errors of the paired difference. One name,
# because it was two -- `spread.MIN_SE` and `injury.TYPE_MIN_SE`, both 2.0, both commented
# "the repo's usual bar". A gate wanting a different bar passes its own; what it must not do
# is declare a second 2.0.
MIN_SE = not_an_input(
    2.0,
    "the significance bar every gate reads: a setting, declared here so it is "
    "declared once, and a decision about which runs are published rather than what "
    "one says")

# #335 -- below this many within-season clusters, a season's win/tie/loss test falls back to
# the sign alone rather than trusting a bootstrap SE built on too few units to mean anything.
# Chosen, not fitted: the relative error of a sample standard deviation is approximately
# `1 / sqrt(2 * (m - 1))` under normality, and 12 puts that near 20% -- a stated choice, the
# way `hub.declare.chosen` marks it, and never placed on any one gate's default so an
# off-by-one in an unrelated flag cannot flip the rule's shape for every gate at once.
TIE_MIN_CLUSTERS = chosen(12)

# The paired bootstrap, matching `hub.models.eval.compare`. Declared here, ahead of
# `paired_gain`, `summarise` and `_cluster_se`, which all default to it.
BOOTSTRAP = 4000


# What one independent observation is, for every gate in this repo. One name, declared once,
# for the same reason `MIN_SE` is: it was three different answers in three harnesses --
# `backtest.compare` and `lineup_gate.compare` took the row, `weekly_gate.compare` took
# `("season", "roster")` -- and nobody chose that either.
#
# **The claim is about the data, not about the power.** Within a season the rows share a
# board, a player pool, a schedule and one realisation of the year; what varies independently
# between them is the season. `docs/gate-power.md` fixes the consequence before the numbers:
# the cluster "is either true or false about the data, not chosen to suit the power", and it
# does not suit the power -- it costs it. Four seasons is four observations, the interval that
# admits it is wide, and the gates that cannot clear their own ceiling under it say so through
# the not-runnable branch in `gate` rather than publishing a null they were never able to find.
SEASON_CLUSTER: tuple[str, ...] = ("season",)

# Two-sided 5%, 80% power -- the pre-registered pair in `docs/gate-power.md`.
POWER = not_an_input(
    0.80,
    "the power the walk-forward protocol's sample-size arithmetic is stated at; it "
    "sizes an experiment and nothing here predicts")
ALPHA = not_an_input(
    0.05,
    "the size the walk-forward protocol's sample-size arithmetic is stated at; it "
    "sizes an experiment and nothing here predicts")

# At or below this many clusters the percentile bootstrap is reported *beside* a t interval
# rather than alone. A nonparametric percentile bootstrap resamples the units it was given,
# and over four of them the resample space is 4**4 = 256 distinct multisets -- it cannot
# express a tail it never drew, so it under-covers, and it does so silently and narrowly.
# The t interval makes a distributional assumption the percentile one does not; neither is
# right, and printing both is the honest render of a disagreement that only exists because
# the cluster count is small.
SMALL_CLUSTERS = 8


def _t_cdf(t: float, df: int) -> float:
    """`P(T <= t)` for Student's t on *integer* degrees of freedom, from the standard library.

    Exact rather than approximate, and no dependency: for integer `df` the t CDF closes in
    elementary functions of `theta = arctan(t / sqrt(df))`, with one recursion for odd `df`
    and another for even. `hub.draft.tune` records the same trade -- "scipy is not a declared
    dependency" -- and `hub.models.coverage._z` takes the normal quantile out of `statistics`
    for the same reason. `df` is always a cluster count minus one here, so integer is not a
    restriction this has to apologise for.
    """
    th = math.atan(t / math.sqrt(df))
    cos = math.cos(th)
    if df == 1:
        return 0.5 + th / math.pi
    if df % 2 == 1:
        coef, total = 1.0, 0.0
        for j in range(1, (df - 1) // 2 + 1):
            if j > 1:
                coef *= (2 * j - 2) / (2 * j - 1)
            total += coef * cos ** (2 * j - 1)
        return 0.5 + (th + math.sin(th) * total) / math.pi
    coef, total = 1.0, 0.0
    for j in range(1, df // 2 + 1):
        if j > 1:
            coef *= (2 * j - 3) / (2 * j - 2)
        total += coef * cos ** (2 * j - 2)
    return 0.5 + 0.5 * math.sin(th) * total


def t_quantile(p: float, df: int) -> float:
    """The `p` quantile of Student's t on `df` degrees of freedom.

    Bisection on `_t_cdf`, which is monotone, to a tolerance far below anything a printed
    interval shows. Verified against the published table at
    `tests/unit/test_experiment.py::test_the_t_quantile_matches_the_table`.
    """
    if df < 1:
        raise ValueError(f"t has no {df} degrees of freedom")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def two_sided_p(t: float, df: int) -> float:
    """`P(|T| >= |t|)` for Student's t on `df` degrees of freedom -- the p a screen's `t` is.

    Two-sided because the screen's bar is `abs(t) < MIN_SE`, and on the same `_t_cdf` the
    quantile above comes from, so the p printed beside a `t` and the interval printed beside
    a gain cannot disagree about the reference distribution. NaN when there are no degrees of
    freedom -- one season has a mean and nothing to test it against -- or no `t`.
    """
    if df < 1 or not math.isfinite(t):
        return float("nan")
    return 2.0 * (1.0 - _t_cdf(abs(t), df))


# **The false-discovery rate a screen's family is read at -- a stated choice, #37.** Not a
# rule: every screen's verdict is its pre-registered rule and nothing else, and a feature
# clearing that rule while sitting above this threshold is reported as clearing, with the
# adjusted result beside it. What the threshold does is make the multiplicity visible. A
# screen over eight features at `MIN_SE` runs eight tests, and a reader of eight verdicts
# is owed the count and what the count does to the bar; until #37 the count was quoted in a
# docstring and moved silently when a feature left the family (#170). 0.10 rather than 0.05
# because a screen asks "is this real?" ahead of a gate that will ask again with a different
# incumbent; a false discovery here costs a gate run, not a shipped model.
FDR_Q = not_an_input(
    0.10,
    "the false-discovery rate a screen's family is read at, a stated choice (#37) "
    "printed beside the pre-registered rule and deciding nothing a prediction reads")


class FalseDiscovery(NamedTuple):
    """One family's Benjamini-Hochberg reading, in the caller's order."""
    tests: int                     # the family size: every test run, measured or not
    q: float                       # the rate it was read at
    threshold: float               # the largest p rejected; 0.0 when nothing is
    adjusted: tuple[float, ...]    # the step-up adjusted p per test; NaN where p was NaN
    rejected: tuple[bool, ...]     # p <= threshold, per test; never for a NaN


def false_discovery(p: Sequence[float], q: float = FDR_Q) -> FalseDiscovery:
    """Benjamini-Hochberg over one family of p-values, controlling the FDR at `q`.

    Step-up: sorted ascending, the threshold is the largest `p_(i)` with `p_(i) <= i q / m`,
    and every p at or below it is rejected -- including one that fails its own step, which
    is what makes it step-up rather than a stop at the first failure. The adjusted p is the
    running minimum from the right of `m p_(j) / j`, capped at one, so an adjusted p at or
    below `q` is exactly a rejection. A family of one is unadjusted: its adjusted p is its p.

    **`m` counts every test the caller ran**, and a NaN -- a feature the screen could not
    measure -- stays in the count: it was a test, and dropping it would shrink the family by
    exactly the tests that came back empty. It ranks last, is never rejected, and its
    adjusted p is NaN rather than a number nobody computed.
    """
    m = len(p)
    ranked = [(float("inf") if math.isnan(v) else v) for v in p]
    order = sorted(range(m), key=lambda i: ranked[i])
    threshold = 0.0
    for rank, i in enumerate(order, start=1):
        if ranked[i] <= rank * q / m:
            threshold = ranked[i]
    adjusted = [float("nan")] * m
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        if math.isfinite(ranked[i]):
            running = min(running, m * ranked[i] / rank)
            adjusted[i] = running
    rejected = tuple(math.isfinite(v) and v <= threshold for v in ranked)
    return FalseDiscovery(m, q, threshold, tuple(adjusted), rejected)


def minimum_detectable_effect(se: float, clusters: int) -> float:
    """The smallest effect this run had 80% power to detect, two-sided at 5%.

    `(t(0.975, k-1) + z(0.80)) * se`, where `k` is the number of clusters and `se` is the
    standard deviation of the bootstrap draws that produced the interval -- the same
    bootstrap, not a second one that happens to agree. That half of the rule is
    `docs/gate-power.md`'s and has not moved.

    **The quantile is a t and this is a correction, dated 2026-09-07.** The criterion first
    written for #45 said "not a t approximation", against the failure mode of a second
    bootstrap; it also ruled out the t *quantile*, and at four clusters the normal quantile is
    simply the wrong reference distribution. It does not make the MDE wrong by a rounding --
    it makes the published SE and the published interval consistent with each other rather
    than correct. At four clusters `t(0.975, 3)` is 3.1824 against `z(0.975)`'s 1.9600, so the
    normal understates the MDE by **1.44x**: on the draft gate's published season-clustered SE
    of 3.67 it reads 10.29 where the honest number is 14.77.

    Only the 0.975 quantile is a t. The 0.80 power term stays normal: it is a statement about
    where the alternative's sampling distribution sits, not about estimating a variance from
    `k` observations, and that is the standard form rather than a convenience.

    Returns NaN when there is no dispersion to speak of -- one cluster has no degrees of
    freedom for a t, and a run with a single independent observation has no power to report
    rather than infinite power.
    """
    if clusters < 2 or not math.isfinite(se):
        return float("nan")
    return (t_quantile(1.0 - ALPHA / 2.0, clusters - 1)
            + statistics.NormalDist().inv_cdf(POWER)) * se


def t_interval(mean: float, se: float, clusters: int) -> tuple[float, float]:
    """The t-based interval `mean +/- t(1 - alpha/2, clusters - 1) * se`.

    **Issue #357 (S1).** `gate`'s ADOPT half used to read the *percentile* bootstrap's `lo`
    -- the 2.5th percentile of resampled cluster means. Under `SEASON_CLUSTER` a nonparametric
    percentile bootstrap can only resample the `k` cluster means it was handed: if all `k` are
    positive, every resample is a convex combination of positive numbers, so `lo > 0` follows
    from `all(gain > 0)` **by construction**, with no reference to how large the gains are
    relative to their spread. `won == total` and `lo > 0` were, on that half, the same
    condition twice -- a sign test of size `2**-k`, not the two-part rule ADR-0019 documents.

    This interval is not implied by the sign the same way. It is a **distributional**
    assumption -- that the cluster means are drawn from something close enough to normal for
    a t reference to apply -- rather than a **resampling** one, and whether it excludes zero
    depends on the *magnitude* of `mean` relative to `se`, not merely on every cluster's sign
    agreeing. `k` seasons that are all barely positive, with a `se` that swamps their mean,
    produce a percentile `lo` above zero every time (rule construction) and a `t_lo` below
    zero most of the time (the classic textbook remedy for `k=4`; `gate-power.md`'s own
    docstring on `minimum_detectable_effect` already renders this exact interval for small
    cluster counts, computed and printed and never read by the rule -- this is that
    computation, now read).

    NaN both, matching `minimum_detectable_effect`'s guard: a t interval needs a degrees of
    freedom to have a shape, and there is none below two clusters.
    """
    if clusters < 2 or not math.isfinite(se):
        return float("nan"), float("nan")
    half = t_quantile(1.0 - ALPHA / 2.0, clusters - 1) * se
    return mean - half, mean + half


def achieved_power(mean: float, se: float, clusters: int) -> float:
    """The power this run's design achieved against the effect actually observed.

    **Issue #357 (S1), acceptance (b).** `minimum_detectable_effect` fixes power at `POWER`
    (0.80) and solves for the effect: `delta = (t_crit + z(power)) * se`. This inverts it --
    fixes the effect at `abs(mean)`, the one this run actually saw, and solves for power:
    `power = Phi(abs(mean) / se - t_crit)`. Same construction, same `t_crit`, same `se` the
    interval and the MDE were drawn from, so the three numbers cannot disagree about which
    bootstrap produced them.

    **A diagnostic, not a rule -- the same status `docs/method.md` gives the FDR threshold
    and CRPS.** It says what this design's power was against what it happened to see, which
    is not the same claim as power against a pre-registered effect and is not read by `gate`.
    Its purpose is narrower and specific to S1: a SHOW is a null, and a null from an
    underpowered design and a null from a well-powered one look identical on the page unless
    the power sits beside them -- "a SHOW that does not say what it could have detected is not
    a result."

    NaN under the same conditions `t_interval` returns NaN, plus a non-positive `se`, which
    would make `abs(mean) / se` either undefined or a direction-free infinity.
    """
    if clusters < 2 or not math.isfinite(se) or se <= 0:
        return float("nan")
    z = abs(mean) / se - t_quantile(1.0 - ALPHA / 2.0, clusters - 1)
    return statistics.NormalDist().cdf(z)


def expanding_seasons(
    df: pl.DataFrame, *, min_past: int = 1, season_col: str = "season",
) -> Iterator[tuple[int, pl.DataFrame, pl.DataFrame]]:
    """`(season, past, now)` for each season that has enough history to be scored.

    The one statement of `docs/method.md` rule #2 in code: `past` is strictly earlier than
    `now`, never the same season, so nothing fitted can have seen what it is scored on. The
    earliest season on record is only ever training data.

    `min_past` is a row count, not a season count -- `margin` needs two residuals before
    `fit` has a standard deviation to give, the others only need one row. A season whose
    `past` falls short is skipped rather than raising, because a walk-forward that stopped at
    the first thin year would report nothing at all.
    """
    for yr in sorted(df[season_col].unique().to_list()):
        past = df.filter(pl.col(season_col) < yr)
        now = df.filter(pl.col(season_col) == yr)
        if past.height < min_past or now.is_empty():
            continue
        yield int(yr), past, now


def expanding_weeks(
    df: pl.DataFrame, *, min_past: int = 1, season_col: str = "season",
    week_col: str = "week",
) -> Iterator[tuple[int, int, pl.DataFrame, pl.DataFrame]]:
    """`(season, week, past, now)` for each week that has enough history to be scored.

    The within-season sibling of `expanding_seasons`, and the same invariant one grain finer:
    `past` is everything strictly earlier in `(season, week)` order -- every earlier season in
    full, plus this season's earlier weeks -- and `now` is exactly that week. A weekly model
    has seventeen times the surface for the leakage that made depth-chart climb read at 7.4
    sigma, so the split belongs here rather than in each harness.

    Ordering is lexicographic on `(season, week)`, not on week alone: week 3 of 2024 is later
    than week 14 of 2023, and a naive sort on the week column would put them the other way
    round and quietly train on the future.

    A caller wanting *within-season* history only filters `past` itself. That is deliberately
    not a flag: the two rules differ by one `filter`, and a boolean parameter that silently
    changes what counts as the past is the kind of thing this function exists to prevent.
    """
    keys = (df.select(season_col, week_col).unique()
              .sort([season_col, week_col]).rows())
    for season, week in keys:
        earlier = (pl.col(season_col) < season) | (
            (pl.col(season_col) == season) & (pl.col(week_col) < week))
        past = df.filter(earlier)
        now = df.filter((pl.col(season_col) == season) & (pl.col(week_col) == week))
        if past.height < min_past or now.is_empty():
            continue
        yield int(season), int(week), past, now


class Gain(NamedTuple):
    """The numbers a two-half gate reads. Numbers only -- it decides nothing.

    `wins`, `ties` and `losses` since #335: a season is a *tie* rather than a win or a loss
    when its gain does not clear its own within-season noise (`_disposition`), and a tie
    blocks both directions -- it is not a win for ADOPT and not a loss for REMOVE. `wins +
    ties + losses == seasons` always.
    """
    mean: float
    se: float
    t: float
    wins: int
    seasons: int
    ties: int = 0
    losses: int = 0


def _bootstrap_se(units: npt.NDArray[np.float64], *, bootstrap: int, seed: int) -> float:
    """The nonparametric percentile bootstrap's own by-product: the standard deviation of the
    resampled means of `units`. NaN below two units -- no spread to resample a shape from.

    The one piece `_cluster_se` (row-array callers) and `per_season` (frame callers, which
    already have their own grouped units) share, so clustering a DataFrame by column names and
    clustering a raw array by a parallel label array do not duplicate the resampling itself.
    """
    m = len(units)
    if m < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(bootstrap, m))
    draws = units[idx].mean(axis=1)
    return float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")


def _cluster_se(diff: npt.NDArray[np.float64], within: npt.NDArray, *,
                bootstrap: int, seed: int) -> tuple[float, int]:
    """Bootstrap SE of `diff`'s mean, clustered on `within`, over these rows alone.

    Used by `paired_gain`, whose `within` is a row-parallel array of labels rather than a
    DataFrame's column names (`per_season` groups its own frame directly and calls
    `_bootstrap_se` on the result) -- one computation for "how much does this season's own
    repeated-measure unit say the mean could have moved", read the two ways each caller's
    data already arrives in. Returns `(se, m)`: `m` is the cluster count the SE was computed
    over, and is always returned even when `se` comes back NaN, because `TIE_MIN_CLUSTERS`
    reads `m` on its own.
    """
    keys = pl.DataFrame({"within": within, "diff": diff})
    units = (keys.group_by("within").agg(pl.col("diff").mean().alias("_u"))
                 ["_u"].to_numpy().astype(float))
    return _bootstrap_se(units, bootstrap=bootstrap, seed=seed), len(units)


def _disposition(gain: float, se: float, m: int) -> str:
    """`"win"`, `"tie"` or `"loss"` for one season, per ADR-0019's #335 amendment.

    **The rule.** A season is a win if `gain >= 2 * se` over its within-season clusters, a
    loss if `gain <= -2 * se`, a tie between. **Below `TIE_MIN_CLUSTERS`** -- too few clusters
    for a bootstrap SE to mean anything, `m` included -- **the test falls back to the sign
    alone**: win if strictly positive, loss if strictly negative, tie at exactly zero (the
    same three-way split, read off the sign rather than the bootstrap).
    """
    if not math.isfinite(se) or m < TIE_MIN_CLUSTERS:
        return "win" if gain > 0 else "loss" if gain < 0 else "tie"
    if gain >= 2.0 * se:
        return "win"
    if gain <= -2.0 * se:
        return "loss"
    return "tie"


def paired_gain(base_err: npt.ArrayLike, arm_err: npt.ArrayLike, *,
                season: npt.ArrayLike, within: npt.ArrayLike,
                bootstrap: int = BOOTSTRAP, seed: int = 0) -> Gain:
    """Mean paired gain of `arm` over `base`, its standard error, t, and the every-season half.

    Both halves of the repo's usual gate come from here: `t` for the significance half and
    `wins`/`ties`/`losses` for the every-season half. Positive `mean` means the arm has the
    smaller error, since the difference is taken as base minus arm.

    The errors are *per observation* and paired -- the same player-week scored by both arms --
    which is why the standard error is of the difference rather than of either arm, and why
    the two arms' seasonal composition cannot contaminate the comparison.

    **`season` and `within`, since #335, row-parallel to `base_err`/`arm_err` and required --
    no default, for the reason `summarise`'s `cluster` and `per_season`'s `within` have none:
    guessing the repeated-measure unit is the mistake, not a convenience a caller can skip.**
    `season` is which held-out season a row belongs to; `within` is that gate's own
    repeated-measure unit inside a season (`docs/method.md` rule 3), the same one its
    `run_gate` call declares. A season's disposition is `_disposition` on its own mean gain
    and its own within-season bootstrap SE (`_cluster_se`) -- exactly what `per_season`
    computes for the gates that go through `gate`, so a verdict that bypasses `gate` (this
    module's three callers) reads the seasons the same way one that does not would.
    """
    d = np.asarray(base_err, dtype=float) - np.asarray(arm_err, dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    t = float(d.mean() / se) if se > 0 else 0.0
    season_arr = np.asarray(season)
    within_arr = np.asarray(within)
    wins = ties = losses = 0
    for yr in sorted(set(season_arr.tolist())):
        mask = season_arr == yr
        d_s = d[mask]
        gain_s = float(d_s.mean()) if len(d_s) else 0.0
        se_s, m_s = _cluster_se(d_s, within_arr[mask], bootstrap=bootstrap, seed=seed)
        disp = _disposition(gain_s, se_s, m_s)
        wins += disp == "win"
        ties += disp == "tie"
        losses += disp == "loss"
    return Gain(float(d.mean()) if len(d) else 0.0, se, t, wins, wins + ties + losses,
               ties, losses)

def _stats(season: int) -> pl.DataFrame:                        # pragma: no cover - network
    from hub.fetch import nflverse
    return nflverse.load("player_stats", [season], cols=list(PLAYER_STATS_COLS))


# What a summary says about a field it has nothing to put in -- and why that is a *shape*
# rather than a value.
#
# The first version of this was `ABSENT = float("nan")` with a `present()` predicate, and its
# comment cited `.github/scripts/heartbeat.sh` as the reason: `jq -r '.ts // 0'` made a missing
# timestamp an age of `now - 0`, so the watchdog reported 56.7 years of staleness on every run
# and could never reach the branch that closes an incident -- "one sentinel standing for
# unreachable, unreadable and stale", three failures whose fix was three separate words.
#
# It then reproduced exactly that defect. `summarise` already returns NaN for its mean, its
# bounds and its probability on an empty frame, meaning *the experiment had no rows*; `ABSENT`
# was the same NaN in the same dictionary meaning *this summary predates the field*. Measured
# 2026-09-05 on `summarise(pl.DataFrame())`: `present(s["mean"])` and `present(s["ceiling"])`
# both answered `False`, and the two values were both `float("nan")`. Those are different facts
# with different responses -- a gate over an empty frame has nothing to say, a gate over a full
# frame whose ceiling was never computed has plenty to say and one missing line -- and nothing
# a reader was given could separate them.
#
# So the states are carried apart the way `heartbeat.sh` carries its three: a field with no
# value at all is *not in the summary*, which leaves NaN meaning exactly one thing, no data.
# The mapping stays `dict[str, float]`, which is what makes this cheap -- widening the values
# to admit `None` was re-measured on 2026-09-05 at 30 pyrefly errors across six files, four of
# them inside the three harnesses this prefactor exists in order not to touch.
#
# The other half of the first version was right and is kept: a ceiling of zero -- a
# perfect-foresight arm gaining nothing at all over the incumbent -- is the strongest finding a
# ceiling can carry, and it has to reach the reader as one. `reading` is the only way to ask,
# rather than a truthiness check each reader writes for itself and one of them writes as
# `if s["ceiling"]:`.


class Field(Enum):
    """What a summary has to say about one of its fields. Three states, three answers.

    `NO_SLOT` names a dictionary key that is not there. That is this summary's own slot and
    not `CONTEXT.md`'s draft **Slot**, which is a pick position.
    """

    VALUE = "value"        # a number, a measured zero included
    NO_DATA = "no data"    # the field is there and the experiment scored nothing into it
    NO_SLOT = "no slot"    # this summary's producer does not compute the field at all


def reading(summary: Mapping[str, float], field: str) -> Field:
    """Which of the three a summary gives for `field`.

    `reading(s, "mean")` on an empty frame is `NO_DATA`; `reading(s, "ceiling")` on a summary
    written before anything measured one is `NO_SLOT`. That the same question works on every
    field is the point -- the two fields waiting on their producers are not a special case with
    a private predicate, they are ordinary fields whose producer has not landed yet.
    """
    if field not in summary:
        return Field.NO_SLOT
    return Field.NO_DATA if math.isnan(summary[field]) else Field.VALUE


def paired_report(s: dict, *, arm_a: str, arm_b: str,
                  unit: str = "points per team game", places: int = 2,
                  show_n: bool = True,
                  ceiling_arm: str | None = "perfect foresight") -> list[str]:
    """The n / interval / P(better) block, as lines rather than prints.

    Returned rather than printed for the reason `hub.draft.report` exists: a block that prints
    cannot be composed, capped, or asserted on.

    `places` because the weekly gate's effect is a tenth the size of the draft gate's and
    rounds to +0.22 at two -- while every doc and ADR quotes it as +0.215. `show_n` because
    that gate prints its own roster-week count, with the cluster count beside it.

    `mde` and `ceiling` render only when they carry a value. Since #45 every real run computes
    an `mde`, so every caller gains that one line and gains it in the same place; a caller
    that hands in no ceiling still prints no ceiling line -- not a placeholder, not a blank,
    not a line reading `nan`. A field with a slot and no data prints nothing either:
    `nan` set against a unit is the watchdog's 56.7 years, and silence is the honest render of
    a number that was not computed. Order is deliberate: the effect, the interval around it,
    the smallest effect the run could have resolved, how much power the run actually achieved
    against what it saw, and then how much there was to resolve. Each line is read against the
    one above it.

    **`power`, since #357 (S1).** A SHOW is a null, and a null from an underpowered run and a
    null from a well-powered one render identically unless the power sits beside them --
    printed here rather than only inside `gate`, because every verdict shares this block and
    a SHOW is the one that most needs it read.
    """
    head = f"n={int(s['n'])}  " if show_n else ""
    lines = [
        f"\n  {head}{arm_a} - {arm_b} = {s['mean']:+.{places}f} {unit}",
        f"  95% CI [{s['lo']:+.{places}f}, {s['hi']:+.{places}f}]   "
        f"P({arm_a} better) {s['p_better'] * 100:.1f}%",
    ]
    # `reading` rather than `s["mde"]`: hand-built summaries reach this block too, and one
    # with no such key has nothing to say about its MDE. This block renders neither that nor a
    # slot holding no data, but they are different nothings, and `gate` is where the
    # difference will be acted on -- which is why the block asks a three-answer question
    # rather than a predicate that folds them together.
    if reading(s, "mde") is Field.VALUE:
        lines.append(f"  MDE at 80% power {s['mde']:+.{places}f} {unit}")
    if reading(s, "power") is Field.VALUE:
        lines.append(f"  achieved power against the observed effect {s['power'] * 100:.1f}% "
                     f"(a diagnostic: `gate` does not read it)")
    # `ceiling_arm` is the caller's, for the same reason `unit` is. Two of the three gates
    # bound *perfect foresight* and the lineup gate bounds a *perfect spread*, and a line that
    # called the second one foresight would be the exact confusion
    # `lineup_gate.foresight_lineup_points` exists as a separate function to prevent -- a
    # ceiling that had quietly become full foresight makes an underpowered gate look powered.
    # `None` says no line at all. Until #135 the weekly and lineup gates passed it because
    # each rendered its own `ceiling_report`; now `run_gate` hands in the declared arm's name
    # and this is the one renderer, so `None` is what a run with no ceiling passes.
    if reading(s, "ceiling") is Field.VALUE and ceiling_arm is not None:
        # The caveat travels with the number. Stage 2 reads three gates' ceilings and holds
        # each against its own MDE; three numbers in three units under one word is the
        # tabulation the arm's name and this clause together prevent.
        lines.append(f"  ceiling ({ceiling_arm}) {s['ceiling']:+.{places}f} {unit}, measured "
                     f"on this gate's own harness -- not comparable with another gate's")
    return lines


def small_sample_report(s: Mapping[str, float], seasons: pl.DataFrame | None = None, *,
                        unit: str = "points per team game", places: int = 2,
                        small: int = SMALL_CLUSTERS) -> list[str]:
    """What a reader needs when the interval above rests on very few units. Lines, not prints.

    Nothing at all above `small` clusters, so a gate with a hundred rosters prints exactly
    what it printed before and this block is confined to the case that motivates it.

    **The t interval beside the percentile one**, because a nonparametric percentile bootstrap
    over four units under-covers: it can only resample the four numbers it was handed, so its
    tails are drawn from a space of 256 multisets and it is narrow in a way that looks like
    precision. The t interval assumes a normal sampling distribution the bootstrap does not.
    Neither is right. Printing one would be choosing; printing both says the two disagree and
    by how much, which is the fact.

    **And the cluster means themselves**, when a frame of them is handed in. Four numbers is
    few enough to read, and an interval over four replications is a claim a reader should be
    able to check by eye -- one season carrying the whole effect is visible in the four and
    invisible in the interval. `per_season` produces exactly this frame, and under
    `SEASON_CLUSTER` the season means *are* the cluster means rather than a second grouping
    that happens to agree.
    """
    k = int(s.get("clusters", 0))
    if not k or k > small or reading(s, "se") is not Field.VALUE:
        return []
    half = t_quantile(1.0 - ALPHA / 2.0, k - 1) * s["se"] if k > 1 else float("nan")
    lines = [f"\n  {k} clusters -- few enough that the percentile bootstrap under-covers, so "
             f"both intervals are shown"]
    if math.isfinite(half):
        lines.append(f"  95% t CI [{s['mean'] - half:+.{places}f}, "
                     f"{s['mean'] + half:+.{places}f}] {unit}   "
                     f"(percentile [{s['lo']:+.{places}f}, {s['hi']:+.{places}f}])")
    if seasons is not None and not seasons.is_empty():
        means = "  ".join(f"{int(r['season'])} {r['gain']:+.{places}f}"
                          for r in seasons.iter_rows(named=True))
        lines.append(f"  the {seasons.height} replications the interval rests on: {means}")
    return lines


# Where a gate's last season-clustered interval width is remembered between runs. Under
# `state/` with the odds poller and the CFBD quota, which is where this repo keeps the small
# facts one run leaves for the next -- not under `data/processed/`, which is measured output.
WIDTH_STATE = STATE_DIR / "gate-width.json"


class Narrowing(NamedTuple):
    """Whether this run's interval narrowed against the last, and what to say about it."""
    ratio: float
    requires_review: bool
    lines: list[str]


def narrowing(width: float, previous: float | None, *, places: int = 2) -> Narrowing:
    """Did the season-clustered interval get *narrower* than last time? Pure; the IO is below.

    **Narrower is not automatically wrong, but it must be visible.** #45's clustering argument
    predicts widening -- within-season rows are near-identical, so pooling them into one
    reading per season should cost precision. On 2026-09-07 the weekly screen (#169) ran it
    and five of nine intervals came back *narrower*, which is the opposite of what the
    argument predicts and was noticed only because someone happened to compare. A prediction
    that fails silently is not a prediction.

    So the run says so itself, with the ratio, and records that it requires review. It does
    not fail and it does not refuse: a narrower interval has real causes -- a cluster mean is
    an average and averaging removes within-cluster noise, so a gate whose variance was mostly
    *within* season honestly tightens -- and a rule that treated it as an error would be
    pre-judging the very thing that wants looking at.
    """
    if previous is None or not (math.isfinite(width) and math.isfinite(previous)) \
            or previous <= 0:
        return Narrowing(float("nan"), False, [])
    ratio = width / previous
    if ratio >= 1.0:
        return Narrowing(ratio, False, [
            f"  interval width {width:.{places}f} against the previous run's "
            f"{previous:.{places}f} -- {ratio:.2f}x, wider as clustering predicts"])
    return Narrowing(ratio, True, [
        f"\n  REQUIRES REVIEW: this run's season-clustered interval is NARROWER than the "
        f"previous one -- {width:.{places}f} against {previous:.{places}f}, a ratio of "
        f"{ratio:.2f}.",
        "  Clustering on the season predicts widening. Narrower is not automatically wrong "
        "-- a gate whose variance sat within the season honestly tightens -- but it "
        "contradicts the argument the cluster was chosen on, and #169 found five of nine "
        "intervals doing this unnoticed. Read it before quoting the interval."])


def _width_state(path: Path) -> list[dict]:
    """Every entry on disk, oldest first, or nothing. Never raises -- CLAUDE.md's degradation
    rule: a gate that could not read its own history still has a verdict to report; what it
    loses is one comparison line, and losing that is strictly better than a harness that dies
    because a JSON file is half-written.

    **#362 (S5): a list under `"entries"`, not a `{gate: record}` dict.** The dict shape was
    overwritten on every run -- one record per gate, no matter how many times it had been run
    -- so nothing on disk could say which run produced a number or whether a run had ever
    happened between two published figures. Reads the pre-#362 dict shape too, as a single
    entry per gate with no `config_digest`/`data_digest`/`timestamp`/`verdict`, so a file this
    function has not yet rewritten does not read as empty.
    """
    try:
        got = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if isinstance(got, dict) and isinstance(got.get("entries"), list):
        return [e for e in got["entries"] if isinstance(e, dict)]
    if isinstance(got, dict):
        # The shape before #362: `{gate: {width, clusters, lo, hi, requires_review}}`.
        return [{"gate": gate, **rec} for gate, rec in got.items() if isinstance(rec, dict)]
    return []


def review_width(name: str, summary: Mapping[str, float], *, verdict: str,
                 config_digest: str, data_digest: str, path: Path = WIDTH_STATE,
                 places: int = 2, write: bool = True,
                 seasons: pl.DataFrame | None = None) -> list[str]:
    """Compare this run's interval width with the last one under `name`, and append this one.

    The comparison is against the *most recent previous entry for this gate*, read back to
    front so three gates' histories interleaved in one file do not confuse each other.
    `requires_review` is written into the record as well as printed, because the printed line
    scrolls past and the record is what a later reader has.

    **Append-only, since #362 (S5).** Every call that writes adds one entry rather than
    overwriting the one this gate already had -- keyed by `(gate, config_digest, data_digest,
    timestamp)`, with the verdict recorded alongside the width, so two runs of the same gate
    are two rows a later reader can tell apart rather than one row silently replaced by the
    other. `config_digest`/`data_digest` are the caller's -- `run_gate` reads them off the
    same `stamped_for_publication` call every other stamp comes from, so this ledger's digests
    cannot disagree with the row's own.

    **`seasons`, since #382, optional and additive.** When the caller hands in the frame
    `per_season` produced, the entry gains a `seasons` field -- `_season_records`' list of
    per-season `season`/`gain`/`se`/`m`/`disposition` dicts -- so a tie (or a loss, or a win)
    is readable from the ledger afterwards and not only from the run's stdout, per #381's open
    question about whether a tie is abstaining on noisy seasons or small effects. `None` (the
    default) omits the field entirely, which is what every pre-#382 entry -- and every call
    this repo's own test suite makes without a `seasons` frame -- still writes, so the
    pre-#362 dict-shape read and the append-only shape both keep reading exactly as before.
    """
    width = float(summary["hi"]) - float(summary["lo"])
    entries = _width_state(path)
    previous = None
    for e in reversed(entries):
        if e.get("gate") == name and isinstance(e.get("width"), int | float):
            previous = float(e["width"])
            break
    said = narrowing(width, previous, places=places)
    if write:
        entry: dict[str, object] = {
            "gate": name,
            "config_digest": config_digest,
            "data_digest": data_digest,
            "timestamp": _now(),
            "width": width,
            "clusters": float(summary.get("clusters", 0)),
            "lo": float(summary["lo"]), "hi": float(summary["hi"]),
            "verdict": verdict,
            "requires_review": said.requires_review,
        }
        if seasons is not None:
            entry["seasons"] = _season_records(seasons)
        entries.append(entry)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"entries": entries}, indent=2, sort_keys=True) + "\n")
        except OSError:
            pass
    return said.lines


def realised_ppg(stats: pl.DataFrame) -> pl.DataFrame:
    """Realised fantasy points per player per week, from nflverse weekly player stats.

    Returns (player, week, points). `player` is normalised with `state.player_key`, the same key
    the board joins on, because nflverse and FantasyPros disagree about suffixes and
    punctuation and an exact join drops the disagreements silently.
    """
    out = stats.select(
        pl.col("player_display_name").map_elements(player_key, return_dtype=pl.Utf8)
          .alias("player"),
        pl.col("week").cast(pl.Int64),
        pl.col("fantasy_points_ppr").fill_null(0.0).cast(pl.Float64).alias("points"),
    )
    return out.group_by(["player", "week"]).agg(pl.col("points").sum())


def summarise(paired: pl.DataFrame, *, cluster: Sequence[str] | None = None,
              bootstrap: int = BOOTSTRAP, seed: int = 0,
              ceiling: float | None = None) -> dict[str, float]:
    """Mean paired difference, a bootstrap interval, and P(arm A better).

    Bootstrapped over *paired* observations rather than over each arm separately, matching
    `hub.models.eval.compare`: the arms share a room and a seed, so resampling them
    independently would throw away the pairing that the design exists to create.

    **`cluster` names what one independent observation is**, and getting it wrong is the most
    expensive mistake in this repo's record -- repeated measures once turned noise into an
    apparent 4-sigma result (signal-screens.md protocol item 3). Pass the columns identifying
    a cluster and each is averaged to a single reading before resampling; pass nothing and the
    row is the unit. There is no safe default, so callers state it:

      * `backtest.compare` -- one row per (season, draft), independent rooms: no cluster.
      * `lineup_gate.compare` -- one row per roster: no cluster.
      * `weekly_gate.compare` -- one row per roster-*week*, fourteen readings sharing a
        roster's players, bye and draft: `cluster=("season", "roster")`. Resampling rows there
        would report an interval about sqrt(14) too narrow.

    The clusters are **sorted** before resampling. They used to arrive in `.unique()` order,
    and since the bootstrap indexes into that order a permutation moved the interval while
    leaving the mean alone -- which is why `docs/weekly-blend-gate.md` records a CI of
    [-0.249, +0.659] against a re-run's [-0.251, +0.663] for an identical +0.215. Same defect
    as improvements #18, one layer down.

    **`se` and `mde` are computed here, from this bootstrap; `ceiling` still arrives from the
    caller.** They are the two numbers a gate needs before a null it reports means anything:
    the smallest effect this run could have resolved at 80% power, and the largest one there
    was to find -- what a perfect arm gains over this gate's own incumbent, on this gate's own
    harness, in this gate's own units. `mde` is `minimum_detectable_effect(se, clusters)` over
    the same draws that give `lo` and `hi`, which is what makes the standard error under the
    interval the interval's own. `ceiling` cannot be computed here, because measuring it means
    playing an extra arm and only the harness knows what its arms are -- so it appears exactly
    when a caller hands one in.

    A key's *presence* is this producer's claim to compute the field; a NaN inside one is its
    claim to have computed nothing this time. So `mde` is always present and is NaN on an
    empty frame or a single cluster -- no rows, or no degrees of freedom for a t -- while
    `ceiling` is absent entirely unless a caller supplied one. A block renders neither NaN,
    and `gate` reads both: an MDE above a ceiling is NOT-RUNNABLE, ahead of every branch but
    VOID, and a gate with no ceiling still decides on the two halves ADR-0019 fixed.
    """
    # A ceiling the caller measured is a fact about its harness rather than about these
    # rows, so it is carried onto the empty summary too -- that pairing, a real ceiling beside
    # a mean of no data, is the one the old sentinel could not express. No ceiling means no
    # key: a NaN here would be indistinguishable from the mean directly above it.
    carried = {} if ceiling is None else {"ceiling": ceiling}
    if paired.is_empty():
        return {"n": 0, "clusters": 0, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_better": float("nan"), "se": float("nan"),
                "mde": float("nan"), "t_lo": float("nan"), "t_hi": float("nan"),
                "power": float("nan")} | carried
    if cluster:
        keys = list(cluster)
        units = (paired.group_by(keys).agg(pl.col("diff").mean().alias("_unit"))
                       .sort(keys)["_unit"].to_numpy().astype(float))
    else:
        units = paired["diff"].to_numpy().astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(bootstrap, len(units)))
    draws = units[idx].mean(axis=1)
    # The interval's own bootstrap, read twice. `se` is the standard deviation of these draws
    # and `mde` is a quantile pair times it -- so the standard error under the MDE is by
    # construction the one the interval above it was drawn from, which is the half of
    # `docs/gate-power.md`'s rule that a second bootstrap agreeing by luck would fake.
    se = float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")
    mean = float(units.mean())
    t_lo, t_hi = t_interval(mean, se, len(units))
    return {"n": float(paired.height), "clusters": float(len(units)),
            "mean": mean,
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "p_better": float((draws > 0).mean()),
            "se": se,
            "mde": minimum_detectable_effect(se, len(units)),
            # #357 (S1): the t interval and the achieved power, off the same `mean`/`se`/
            # `clusters` triple the MDE reads -- see `t_interval` and `achieved_power`.
            "t_lo": t_lo,
            "t_hi": t_hi,
            "power": achieved_power(mean, se, len(units))} | carried


# --- the Gate ---------------------------------------------------------------

class Actions(NamedTuple):
    """What a gate does with each verdict, in that gate's own words.

    The rule below is shared; these are not. "The optimiser sets your Week 1 lineup" and
    "championship equity returns to the headline" are different actions and have to read that
    way, so each gate supplies its three sentences and the rule supplies the evidence.
    """
    adopt: str
    remove: str
    show: str


def per_season(paired: pl.DataFrame, *, within: Sequence[str],
               bootstrap: int = BOOTSTRAP, seed: int = 0) -> pl.DataFrame:
    """Mean paired difference per held-out season -- the every-season half of the bar.

    An empty frame answers with the empty table. `weekly_gate.compare` returns a frame with
    no rows *and no columns* when nothing is covered, and a `group_by` on a season column
    that is not there is a traceback where the verdict already knows what to say -- nothing
    was measured.

    **`within`, since #335, no default -- for the reason `summarise`'s `cluster` has none.**
    ADR-0019's amendment: a season only counts as a *win* if its gain clears its own noise,
    and "its own noise" is a bootstrap over the within-season repeated-measure unit --
    `docs/method.md` rule 3's unit, named per gate in `run_gate`'s own call site. Adds `se`
    (`_bootstrap_se` over that season's own `within`-clustered unit means) and `m` (the
    cluster count it was computed over) beside `gain` and `n`; `gate` reads both through
    `_disposition` rather than `gain`'s sign alone. `se` is NaN when a season has fewer than
    two `within` clusters -- `_bootstrap_se`'s own guard -- and `_disposition` reads `m`
    regardless of whether `se` is a number, since `m < TIE_MIN_CLUSTERS` is its own reason to
    fall back to the sign.
    """
    schema = {"season": pl.Int64, "gain": pl.Float64, "n": pl.UInt32,
             "se": pl.Float64, "m": pl.Int64}
    if paired.is_empty():
        return pl.DataFrame(schema=schema)
    base = (paired.group_by("season")
                  .agg(pl.col("diff").mean().alias("gain"), pl.len().alias("n"))
                  .sort("season"))
    keys = list(within)
    se_col, m_col = [], []
    for yr in base["season"].to_list():
        rows = paired.filter(pl.col("season") == yr)
        # **Sorted**, the same fix `summarise` already carries (issue #45's second bug): a
        # `group_by` without `maintain_order` arrives in an order that can vary call to call,
        # and the bootstrap indexes into that order positionally, so a permutation of `units`
        # with the same seed draws a different resample -- silently, and it moved
        # `docs/weekly-blend-gate.md`'s own CI once. `.sort(keys)` makes this deterministic.
        units = (rows.group_by(keys).agg(pl.col("diff").mean().alias("_u"))
                     .sort(keys)["_u"].to_numpy().astype(float))
        se_col.append(_bootstrap_se(units, bootstrap=bootstrap, seed=seed))
        m_col.append(len(units))
    return base.with_columns(pl.Series("se", se_col, dtype=pl.Float64),
                             pl.Series("m", m_col, dtype=pl.Int64))


def _season_records(seasons: pl.DataFrame) -> list[dict]:
    """One dict per season -- `season`, `gain`, `se`, `m`, `disposition` -- the fields #382
    adds beside `_seasons_won_tied_lost`'s tally so a tie is readable from the record and not
    only from a run's stdout. #381's open question (is the tie rule abstaining on *noisy*
    seasons rather than *small* ones?) cannot be checked from a run that only ever printed the
    tally; this is what a later reader needs to check it, per season rather than pooled.

    **Same backward-compatible reading as `_seasons_won_tied_lost`.** A frame with no `se`/`m`
    columns reads `m = 0`, always below `TIE_MIN_CLUSTERS`, so `_disposition` falls back to the
    sign alone -- every hand-built `seasons` frame in this repo's test suite included.
    """
    has_se = "se" in seasons.columns and "m" in seasons.columns
    out = []
    for r in seasons.iter_rows(named=True):
        se = float(r["se"]) if has_se else float("nan")
        m = int(r["m"]) if has_se else 0
        gain = float(r["gain"])
        out.append({"season": int(r["season"]), "gain": gain, "se": se, "m": m,
                    "disposition": _disposition(gain, se, m)})
    return out


def per_season_report(seasons: pl.DataFrame, *, places: int = 2) -> list[str]:
    """The per-season table #382 prints beside the tally -- `season`, `gain`, `se`, `m` and
    the disposition each drove, for every season and every gate run, not only the ones under
    `SMALL_CLUSTERS` `small_sample_report` already covers.

    Nothing on an empty frame -- a run with nothing measured has no seasons to break out.
    `(sign-only)` is appended to a season's disposition when it was read off the raw sign
    rather than the `2 * se` test -- `se` non-finite or `m` below `TIE_MIN_CLUSTERS` -- so a
    reader does not mistake a sign-only reading for the bootstrap-backed one at a glance.
    """
    if seasons.is_empty():
        return []
    lines = ["\n  per-season:"]
    for rec in _season_records(seasons):
        se, m = rec["se"], rec["m"]
        se_str = f"{se:.{places}f}" if math.isfinite(se) else "nan"
        sign_only = not math.isfinite(se) or m < TIE_MIN_CLUSTERS
        tag = rec["disposition"] + (" (sign-only)" if sign_only else "")
        lines.append(f"    {rec['season']}  gain {rec['gain']:+.{places}f}  se {se_str}  "
                     f"m {m}  {tag}")
    return lines


def _seasons_won_tied_lost(seasons: pl.DataFrame) -> tuple[int, int, int]:
    """`(won, tied, lost)` over every row of `seasons`, per `_disposition`.

    **Backward compatible with a `seasons` frame that has no `se`/`m` columns**, which every
    hand-built summary in this repo's test suite is -- `_disposition` itself falls back to the
    sign alone when `m < TIE_MIN_CLUSTERS`, and a frame with no `m` column at all is read as
    `m = 0`, which is always below the floor. So a caller that has not adopted `per_season`'s
    `within` yet reads exactly the pre-#335 sign test, and a `seasons["gain"] == 0` row -- the
    boundary the old rule folded into "not won" -- is a tie under both the old and new
    reading, which is why it was never a REMOVE-blocking case before either.
    """
    has_se = "se" in seasons.columns and "m" in seasons.columns
    won = tied = lost = 0
    for r in seasons.iter_rows(named=True):
        se = r["se"] if has_se else float("nan")
        m = int(r["m"]) if has_se else 0
        disp = _disposition(float(r["gain"]), se, m)
        won += disp == "win"
        tied += disp == "tie"
        lost += disp == "loss"
    return won, tied, lost


def gate(summary: dict, seasons: pl.DataFrame, actions: Actions,
         *, void: str | None = None) -> tuple[str, str]:
    """Did this beat the simplest thing that already works? The rule, in one place.

    `CONTEXT.md` defines a **Gate** as exactly that question, and three modules answered it
    with three copies of these branches. They had diverged: the weekly gate required the sign
    to hold in every held-out season, the lineup gate and the draft backtest adopted on the
    pooled interval alone. Nobody chose that.

    **Both halves are required, and that is ADR-0019.**
    An interval excluding zero says the pooled effect is unlikely to be noise; it says nothing
    about whether one lucky season carried it. `hub.models.spread` had already been bitten:
    its verdict's own docstring records that an earlier version checked the seasons alone and
    "would have adopted a model on a gain too small to distinguish from noise". That copy was
    strengthened and the others were not, because nothing connected them.

    The asymmetry is deliberate and pre-registered in every plan that uses this: the arm under
    test is the complicated thing and the burden sits on it. The middle branch therefore
    carries an *action* rather than being a disappointment to explain away.

    `void` is the caller's own precondition, already phrased -- the weekly gate voids above a
    join-failure rate. A gate whose inputs are broken has no verdict to read, and what counts
    as broken is specific to the gate, so this honours the condition rather than defining it.

    **NOT-RUNNABLE comes second, ahead of every branch but VOID**, and that order is the whole
    point of it. `docs/gate-power.md` stage 2: a gate whose MDE exceeds its own measured
    ceiling cannot separate a real effect from a perfect one, so every branch below it would
    be reading noise with a decimal point -- including, and especially, the middle one. A null
    published from an underpowered gate is the failure this branch exists to prevent, and a
    null is what the middle branch prints. Ordering this after SHOW would let exactly the
    verdict that must not be published be published first.

    VOID stays above it because a void gate's inputs are broken, which makes its MDE and its
    ceiling untrustworthy too -- there is nothing to compare.

    It fires only when *both* numbers are present as values. A gate that measured no ceiling
    has not shown that it cannot run; it has shown nothing, and `Field.NO_SLOT` is that
    third state rather than a licence to guess. Every gate in the repo today that hands in no
    ceiling therefore reaches precisely the verdict it reached before.

    The comparison is signed rather than absolute. A ceiling of zero -- a perfect-foresight
    arm gaining nothing over the incumbent -- is the strongest finding a ceiling can carry,
    and any positive MDE exceeds it, which is the correct reading and not an edge case.

    **The interval half reads `t_lo`/`t_hi`, not `lo`/`hi`, since #357 (S1).** Every gate here
    clusters on the season (`SEASON_CLUSTER`), so `summarise`'s bootstrap resamples exactly
    the `k` season means `seasons["gain"]` also holds. A *percentile* bootstrap over `k`
    clusters can only resample the `k` numbers it was handed: when every one is positive,
    every resample is a convex combination of positive numbers, so its `lo > 0` follows from
    `won == total` **by construction**, independent of how large the gains are next to their
    spread. The ADOPT conjunction was, on that half, `won == total` read twice -- a one-sided
    sign test of size `2**-k` wearing two names. `t_lo`/`t_hi` (`t_interval`) is a
    distributional claim rather than a resampling one and is not implied by the seasons'
    signs the same way: see `t_interval`'s own docstring for the full argument and
    `tests/unit/test_experiment.py::test_the_fixed_rule_s_null_size_is_not_degenerate` (and
    its sibling planted against the *old* rule) for the simulation that checks it rather than
    arguing it.

    **The every-season half is tie-aware, since #335.** ADOPTED (A) with (i): a season is a
    *win* only if its gain clears `2 * se` over its own within-season clusters (`_disposition`,
    reading `seasons["se"]`/`seasons["m"]` when `per_season` computed them, falling back to
    the sign alone otherwise or below `TIE_MIN_CLUSTERS`); a *tie* is neither a win nor a loss
    and blocks **both** directions, symmetrically -- it is not a win ADOPT needs in every
    season, and it is not a loss REMOVE needs in every season either.

    **A gate that measured no ceiling is NOT-RUNNABLE in both directions, since #363 (S6),
    option 1.** Before this, the precondition fired only when *both* the MDE and the ceiling
    were present as values, so a gate that never measured a ceiling reached the verdict it
    would have reached anyway -- protected from publishing a null (SHOW) or from adopting,
    and not protected from REMOVE, because the two excluding branches are self-limiting (an
    underpowered gate rarely produces an interval that excludes zero) and REMOVE is one of
    them. That asymmetry is S6's own finding: an underpowered design was never shown *safe*
    to REMOVE on, only never *caught* removing on. Requiring the ceiling first closes it, at
    the cost every reachable branch now pays: nothing below this line runs without one.
    `docs/adr/0019-a-gate-requires-every-season.md`'s dated amendment names which gates that
    leaves NOT-RUNNABLE today, and #376 by number as what ends it.
    """
    if void:
        return "VOID", void
    has_data = bool(summary.get("clusters"))
    if has_data and reading(summary, "ceiling") is not Field.VALUE:
        return "NOT-RUNNABLE", (
            "NOT RUNNABLE: this gate has not measured a ceiling. Per ADR-0019's #363 (S6) "
            "amendment, a gate that measured no ceiling is NOT-RUNNABLE in both directions -- "
            "not only when its MDE happens to exceed one, since REMOVE was never shown safe "
            "on an unmeasured design, only never caught. Run with the ceiling this gate "
            "declares, or read `docs/gate-power.md` for which ceilings are measured and "
            "#376 for what ends this for the rest.")
    if has_data and reading(summary, "mde") is Field.VALUE and summary["mde"] > summary["ceiling"]:
        return "NOT-RUNNABLE", (
            f"NOT RUNNABLE: the smallest effect this gate could resolve at 80% power is "
            f"{summary['mde']:+.3f}, against a ceiling of {summary['ceiling']:+.3f} -- the "
            f"largest effect there was to find. Over {int(summary.get('clusters', 0))} "
            f"independent clusters this design cannot tell a real effect from a perfect one, "
            f"so no verdict below is reported. `docs/gate-power.md` stage 2: this is *not "
            f"planned*, not *failed* -- the arm did not lose, the question cannot be answered "
            f"with the data that exists.")
    if not has_data:
        return "SHOW", f"{actions.show} Nothing measured -- no paired observation."
    won, tied, lost = _seasons_won_tied_lost(seasons)
    total = seasons.height
    t_lo = summary.get("t_lo", float("nan"))
    t_hi = summary.get("t_hi", float("nan"))
    has_interval = reading(summary, "t_lo") is Field.VALUE
    tally = f"won {won}, tied {tied}, lost {lost} of {total} seasons"
    if has_interval and t_lo > 0 and won == total:
        return "ADOPT", (f"{actions.adopt} It won in every held-out season ({tally}) "
                         f"and the t interval [{t_lo:+.3f}, {t_hi:+.3f}] excludes zero.")
    if has_interval and t_hi < 0 and lost == total:
        return "REMOVE", (f"{actions.remove} Worse in every held-out season ({tally}) "
                          f"and the t interval [{t_lo:+.3f}, {t_hi:+.3f}] excludes zero.")
    if not has_interval:
        why = "too few clusters for a t interval"
    elif t_lo <= 0 <= t_hi:
        why = "the interval contains zero"
    else:
        why = "the interval excludes zero but the sign is not consistent across seasons"
    return "SHOW", (f"{actions.show} {tally} and {why} -- absence of evidence, not evidence "
                    f"of equivalence.")


# --- one gate run: the sequence around the rule, written once (issue #135) -----------------
#
# `gate` above unified the three *verdict* branches (ADR-0019). What it did not unify is the
# sequence around it -- summarise, break out by season, take the verdict, render the report,
# stamp the output -- which every entry point wrote for itself. The consequences arrived as
# tickets: only one gate could void (#46), only one stamped its output (#133), only one could
# reach its ceiling (#134). Two of the eight tickets in the modelling programme were "apply
# this gate property in the other gates too", which is the signature of a function that
# should exist. This is it. Each entry point assembles its arms, calls this, and prints what
# comes back.
#
# What is deliberately still the caller's: the cluster unit (see `run_gate`), the three
# sentences (`Actions`), the void condition, and the ceiling arm's *name* -- because two of
# the three gates bound perfect foresight and the lineup gate bounds a perfect spread, and a
# run that named the arm itself would be the confusion `lineup_gate.foresight_lineup_points`
# exists as a separate function to prevent.


class Ceiling(NamedTuple):
    """A gate's declared ceiling arm: what it is called where it is printed, and its rows.

    `arm` is the gate's own `CEILING_ARM` -- *perfect foresight, the season known in advance*
    for the draft gate, *perfect foresight* for the weekly gate, *a perfect spread* for the
    lineup gate -- and `diff` is one paired difference per row, the ceiling arm minus the
    incumbent, on that gate's own harness. The run takes the mean; the caller does not, so
    the number the rule reads and the number the line prints cannot be two numbers.
    """

    arm: str
    diff: npt.ArrayLike


class GateRun(NamedTuple):
    """What one gate run returns, in the order an entry point used to assemble them."""

    summary: dict[str, float]
    seasons: pl.DataFrame
    verdict: tuple[str, str]
    lines: list[str]
    stamped: pl.DataFrame


def ceiling_check(summary: Mapping[str, float], *, places: int = 2) -> list[str]:
    """A loud line when the ceiling does not bound the effect it was measured to bound.

    A ceiling below the effect is not a tight result, it is a broken one: either the ceiling
    arm is not seeing what it was handed or the arm under test is being scored on something
    else. Said loudly rather than published quietly, because the number's whole use is as an
    upper bound in `docs/gate-power.md`, and a bound that does not bound reads exactly like a
    tight one. This sentence was written three times -- `backtest.with_ceiling` at two
    places, `lineup_gate.ceiling_report` at two, `weekly_gate.ceiling_report` at three -- and
    is now written once, at the caller's places.

    Nothing when there is no ceiling to check, and nothing on an empty frame: `nan` compares
    false and an experiment that scored nothing has no effect to fall below.
    """
    if reading(summary, "ceiling") is not Field.VALUE:
        return []
    top, mean = summary["ceiling"], summary["mean"]
    if top < mean:
        return [f"\n  CEILING BELOW THE EFFECT: {top:+.{places}f} < {mean:+.{places}f}. One "
                f"of the two is measuring something the other is not; do not read the "
                f"interval above as bounded."]
    return []


def stamped_for_publication(paired: pl.DataFrame,
                            boards: Mapping[int, ReportedFrame] | None = None,
                            ) -> tuple[pl.DataFrame, str]:
    """The paired frame carrying what produced it, and the lines a reader gets.

    **Four stamps, and until #196 there were two.** `cfg_digest` says which model, `data_digest`
    says which upstream bytes -- and neither says which *Board*, which is the object a gate's
    `compare` is actually handed, or which *code* read it.

    `board_digest` closes the first gap. The Board is built from those bytes by `board_as_of`,
    through joins, an as-of boundary, a `MIN_GAMES` filter and an xFP imputation, and any of
    them can change its membership without a source byte moving -- `90a9bbb` dropped 814 of
    1,372 players from 2025 at an unchanged data digest. `backtest.compare`'s docstring says
    *"Pure: takes frames, returns a frame, touches no network"*, and purity with respect to the
    network had been getting read as reproducibility. It is not: it is a statement about what
    the function does not touch, and the frames it does take were unrecorded.

    `commit` closes the second. `docs/gate-power.md` records the draft gate's effect moving
    8.07 points across 270 commits at an identical data digest, with no owning commit --
    a movement nobody can attribute because the runs recorded which bytes they read and never
    which tree read them. `docs/track-record.md` rule 1 makes these numbers commit-dated.

    `boards` is optional and defaults to `NO_FRAMES`, which is a sentinel and not a hash, so a
    caller that did not hand its frames over says so rather than publishing eight
    legitimate-looking characters that name nothing. They are the Boards the run played, as
    `walk_forward_inputs` returned them; the digest is over their frames, taken off them
    here, since the report is what built a frame and not part of what was played.

    `resolved_config()`, not `HubConfig()`: ADR-0007 keeps this file so a later reader can tell
    which configuration produced these rows, and the defaults are that only while `conf/`
    overrides nothing that diverges from one. Same call as the fetch layer's provenance line
    and `ratings.live_config`, so a run's three stamps cannot disagree about what a run was.

    `data_digest` beside it is the pinning layer's premise arriving: every gate output should
    name the data it scored against, so an archive that moved shows up as a changed digest
    rather than as a silently different number. The digest existed and nothing computed one,
    so until 2026-09-04 a moved archive was exactly the silent case (issue #71).

    **Here rather than in `hub.draft.backtest`, where it was written**, because the lineup
    gate reached it through a draft module (#133) and the weekly gate did not reach it at all.
    It is the stamping half of `run_gate`, and a stamping rule that every gate applies is a
    property of one function rather than of three entry points a network has to reach.
    """
    from hub.fetch.nflverse import pins_this_run

    pins = pins_this_run()
    data = data_digest(pins)
    played = (NO_FRAMES if boards is None
              else frames_digest({yr: b.frame for yr, b in boards.items()}))
    made_by = commit()
    stamped = paired.with_columns(
        pl.lit(config_digest(resolved_config())).alias("cfg_digest"),
        pl.lit(data).alias("data_digest"),
        pl.lit(played).alias("board_digest"),
        pl.lit(made_by).alias("commit"))
    if paired.is_empty():
        # A literal broadcasts to one row over a frame with no columns at all, which is what
        # `weekly_gate.compare` returns when nothing is covered. No rows scored, no rows stamped.
        stamped = stamped.clear()
    # Said as well as stored, because the reader deciding whether two runs are comparable is
    # usually reading the terminal, not the parquet.
    said = (f"  data: {data} over {len(pins)} pinned source(s)"
            + ("" if pins else " -- nothing was loaded through the pinning layer")
            + f"\n  board: {played}"
            + ("" if boards else " -- the run did not hand over the frames it played")
            + f"\n  commit: {made_by}"
            + ("-- a dirty tree; this run is not that commit" if made_by.endswith("-dirty")
               else ""))
    return stamped, said


def run_gate(paired: pl.DataFrame, *, cluster: Sequence[str] | None, within: Sequence[str],
             actions: Actions, name: str, arm_a: str, arm_b: str,
             unit: str = "points per team game", places: int = 2, show_n: bool = True,
             void: str | None = None, ceiling: Ceiling | None = None, seed: int = 0,
             bootstrap: int = BOOTSTRAP, boards: Mapping[int, ReportedFrame] | None = None,
             width_path: Path = WIDTH_STATE, record_width: bool = True) -> GateRun:
    """One gate run: summarise, break out by season, take the verdict, render, stamp.

    **`cluster` has no default, and that is the most important line of the signature.**
    `summarise`'s docstring says what one independent observation is has no safe default and
    that getting it wrong is the most expensive mistake in this repo's record. A shared run
    that guessed one would make that mistake in every gate at once, silently, with the same
    interval shape a correct run produces. So each gate states its own at its own call site,
    and `tests/contracts/test_gates_cluster_on_the_season.py` reads the argument off the call.

    **`within` has no default either, since #335, for the same reason.** It is
    `per_season`'s own repeated-measure unit -- `docs/method.md` rule 3's unit, named per gate:
    `draft` for the draft backtest, `roster` for the weekly and lineup gates, `player_id` for
    the coverage gate, the event (a declared no-op) for the quarterback gate. See
    `tests/contracts/test_gates_tie_test_names_its_within_season_unit.py`.

    `name` keys the interval-width history in `WIDTH_STATE`, so three gates do not overwrite
    each other's; `record_width=False` is for a test, which has no history to keep.

    `void` is the caller's precondition, already phrased -- the weekly gate voids above a
    join-failure share, and since #46 so does the draft gate. `gate` honours it ahead of every
    branch. `ceiling` is the caller's declared arm with its rows; its mean enters the summary
    the verdict reads (so NOT-RUNNABLE can fire) and its name enters the line a reader sees.

    **The weekly gate's ceiling now reaches the rule, and this is the one place the shape
    change is not a no-op.** Its `main` handed the ceiling to a report and never to
    `summarise`, so `docs/weekly-blend-gate.md` records that its NOT-RUNNABLE branch "does not
    fire and cannot". The lineup gate had already been wired the other way (#134). One run
    means one wiring, and it is the lineup gate's. No published weekly verdict was reached
    with a ceiling in hand, so none moves.

    The report is the block, the per-season table (#382), the small-sample lines, the width
    review, the ceiling check and the stamp -- in that order for every gate, so a reader of one
    gate's output can read another's. The stamp is said on every run and not only when a frame
    is written, for the reason `stamped_for_publication` gives: the reader deciding whether two
    runs compare is at the terminal.
    """
    top = None if ceiling is None else float(np.asarray(ceiling.diff, dtype=float).mean())
    summary = summarise(paired, cluster=cluster, bootstrap=bootstrap, seed=seed, ceiling=top)
    seasons = per_season(paired, within=within, bootstrap=bootstrap, seed=seed)
    verdict = gate(summary, seasons, actions, void=void)
    stamped, stamp = stamped_for_publication(paired, boards)
    # The ledger's own digests come off the same stamp every other row in `stamped` carries,
    # so `review_width`'s record cannot name a config or a data byte this run did not read.
    # `stamped` is cleared to no rows on an empty `paired` (`stamped_for_publication`'s own
    # empty-frame rule), so the columns exist but there is no row to read them off; the
    # digests are recomputed the same way in that one case.
    if stamped.height:
        row = stamped.row(0, named=True)
        cfg_dig, data_dig = row["cfg_digest"], row["data_digest"]
    else:
        from hub.fetch.nflverse import pins_this_run
        cfg_dig = config_digest(resolved_config())
        data_dig = data_digest(pins_this_run())
    lines = [
        *paired_report(summary, arm_a=arm_a, arm_b=arm_b, unit=unit, places=places,
                       show_n=show_n, ceiling_arm=None if ceiling is None else ceiling.arm),
        *per_season_report(seasons, places=places),
        *small_sample_report(summary, seasons, unit=unit, places=places),
        *review_width(name, summary, verdict=verdict[0], config_digest=cfg_dig,
                      data_digest=data_dig, path=width_path, places=places,
                      write=record_width, seasons=seasons),
        *ceiling_check(summary, places=places),
        "",
        *stamp.split("\n"),
    ]
    return GateRun(summary, seasons, verdict, lines, stamped)
