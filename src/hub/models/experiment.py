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

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import polars as pl

from hub.names import player_key

# One column list, so both harnesses hit one cache entry. `nflverse._cache_path` keys on the
# sorted column set -- deliberately, so a caller asking for six columns is never served an
# earlier caller's five -- which means two callers asking for different slices of the same
# table download it twice.
PLAYER_STATS_COLS: tuple[str, ...] = (
    "player_id", "player_display_name", "position", "season", "week", "fantasy_points_ppr",
)


def walk_forward_inputs(
    seasons: Sequence[int],
    build_board: Callable[[int], tuple[pl.DataFrame, object]],
    *,
    load_stats: Callable[[int], pl.DataFrame] | None = None,
    on_season: Callable[[int], None] | None = None,
) -> tuple[dict[int, pl.DataFrame], dict[int, pl.DataFrame]]:
    """`(boards, realised)` per season. The two injectable seams are what make this testable.

    `build_board` is required rather than defaulted. It used to default to a `board_as_of`
    defined here, which put draft-domain knowledge under `models/` and inverted the tree's one
    consistent direction -- and needed a function-local import to do it. That function now
    lives in `hub.draft.board`, beside the `build` whose rule it states.

    `on_season` is a progress hook rather than a print, so a caller under a line cap can stay
    quiet and this module stays free of stdout.
    """
    boards: dict[int, pl.DataFrame] = {}
    realised: dict[int, pl.DataFrame] = {}
    for yr in seasons:
        if on_season is not None:
            on_season(yr)
        boards[yr] = build_board(yr)[0]
        realised[yr] = realised_ppg((load_stats or _stats)(yr))
    return boards, realised



# The repo's usual significance bar, in standard errors of the paired difference. One name,
# because it was two -- `spread.MIN_SE` and `injury.TYPE_MIN_SE`, both 2.0, both commented
# "the repo's usual bar". A gate wanting a different bar passes its own; what it must not do
# is declare a second 2.0.
MIN_SE = 2.0


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
    """The four numbers a two-half gate reads. Numbers only -- it decides nothing."""
    mean: float
    se: float
    t: float
    wins: int
    seasons: int


def paired_gain(base_err: npt.ArrayLike, arm_err: npt.ArrayLike, *,
                base_mae: npt.ArrayLike, arm_mae: npt.ArrayLike) -> Gain:
    """Mean paired gain of `arm` over `base`, its standard error, t, and seasons won.

    Both halves of the repo's usual gate come from here: `t` for the significance half and
    `wins`/`seasons` for the every-season half. Positive `mean` means the arm has the smaller
    error, since the difference is taken as base minus arm.

    The errors are *per observation* and paired -- the same player-week scored by both arms --
    which is why the standard error is of the difference rather than of either arm, and why
    the two arms' seasonal composition cannot contaminate the comparison. The per-season means
    come in separately rather than being regrouped here, because each caller already has them
    for its own printing.
    """
    d = np.asarray(base_err, dtype=float) - np.asarray(arm_err, dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    t = float(d.mean() / se) if se > 0 else 0.0
    per = np.asarray(arm_mae, dtype=float)
    wins = int((per < np.asarray(base_mae, dtype=float)).sum())
    return Gain(float(d.mean()) if len(d) else 0.0, se, t, wins, len(per))

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
                  show_n: bool = True) -> list[str]:
    """The n / interval / P(better) block, as lines rather than prints.

    Returned rather than printed for the reason `hub.draft.report` exists: a block that prints
    cannot be composed, capped, or asserted on.

    `places` because the weekly gate's effect is a tenth the size of the draft gate's and
    rounds to +0.22 at two -- while every doc and ADR quotes it as +0.215. `show_n` because
    that gate prints its own roster-week count, with the cluster count beside it.

    `mde` and `ceiling` render only when they carry a value. Nothing computes either yet, so
    every caller today gets exactly the two lines it got before -- not a placeholder, not a
    blank, not a line reading `nan`. A field with a slot and no data prints nothing either:
    `nan` set against a unit is the watchdog's 56.7 years, and silence is the honest render of
    a number that was not computed. Order is deliberate: the effect, the interval around it,
    the smallest effect the run could have resolved, and then how much there was to resolve.
    Each line is read against the one above it.
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
    if reading(s, "ceiling") is Field.VALUE:
        lines.append(f"  ceiling (perfect foresight) {s['ceiling']:+.{places}f} {unit}")
    return lines


# The paired bootstrap, matching `hub.models.eval.compare`.
BOOTSTRAP = 4000


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

    **`mde` and `ceiling` are the two fields whose producers have not landed, so a summary
    carries each only when something filled it -- and today nothing computes an `mde` and no
    caller hands in a `ceiling`.** They are the two numbers a gate needs before a null it
    reports means anything: the smallest effect this run could have resolved at 80% power,
    and the largest one there was to find -- what a perfect-foresight arm gains over this
    gate's own incumbent, on this gate's own harness, in this gate's own units. `mde` will be
    computed here, from the same cluster-mean vector the interval comes from, and the key
    appears when it does. `ceiling` arrives from the caller instead, because measuring it
    means playing an extra arm and only the harness knows what its arms are -- so it appears
    exactly when a caller hands one in.

    A key's *presence* is this producer's claim to compute the field; a NaN inside one is its
    claim to have computed nothing this time. Neither field is read by anything yet, and that
    is deliberate: they exist now so the units that compute them change no call site's
    signature. Until then a block that has neither prints neither, and `gate` decides on the
    same two halves ADR-0019 fixed.
    """
    # A ceiling the caller measured is a fact about its harness rather than about these
    # rows, so it is carried onto the empty summary too -- that pairing, a real ceiling beside
    # a mean of no data, is the one the old sentinel could not express. No ceiling means no
    # key: a NaN here would be indistinguishable from the mean directly above it.
    carried = {} if ceiling is None else {"ceiling": ceiling}
    if paired.is_empty():
        return {"n": 0, "clusters": 0, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_better": float("nan")} | carried
    if cluster:
        keys = list(cluster)
        units = (paired.group_by(keys).agg(pl.col("diff").mean().alias("_unit"))
                       .sort(keys)["_unit"].to_numpy().astype(float))
    else:
        units = paired["diff"].to_numpy().astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(bootstrap, len(units)))
    draws = units[idx].mean(axis=1)
    return {"n": float(paired.height), "clusters": float(len(units)),
            "mean": float(units.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "p_better": float((draws > 0).mean())} | carried


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


def per_season(paired: pl.DataFrame) -> pl.DataFrame:
    """Mean paired difference per held-out season -- the every-season half of the bar."""
    return (paired.group_by("season")
                  .agg(pl.col("diff").mean().alias("gain"), pl.len().alias("n"))
                  .sort("season"))


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
    """
    if void:
        return "VOID", void
    if not summary.get("clusters"):
        return "SHOW", f"{actions.show} Nothing measured -- no paired observation."
    won = int((seasons["gain"] > 0).sum())
    total = seasons.height
    if summary["lo"] > 0 and won == total:
        return "ADOPT", (f"{actions.adopt} It won in every held-out season ({won}/{total}) "
                         f"and the interval excludes zero.")
    if summary["hi"] < 0 and won == 0:
        return "REMOVE", (f"{actions.remove} Worse in every held-out season ({total}/{total}) "
                          f"and the interval excludes zero.")
    why = ("the interval contains zero" if summary["lo"] <= 0 <= summary["hi"]
           else "the interval excludes zero but the sign is not consistent across seasons")
    return "SHOW", (f"{actions.show} Won {won}/{total} seasons and {why} -- absence of "
                    f"evidence, not evidence of equivalence.")
