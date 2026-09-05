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
from collections.abc import Callable, Iterator, Sequence
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


# What a summary field says when it has no value on it.
#
# `float("nan")` rather than `None` because `summarise` is declared `dict[str, float]` and
# `backtest.verdict` and `lineup_gate.verdict` both declare that same type on the way in.
# Widening it would push a signature change through three harnesses that these two fields are
# specifically meant not to touch.
#
# The one thing ABSENT must never be confusable with is a *computed* value, and a computed
# zero above all. The repo has paid for that twice. ESPN's historical ADP returns 169 for
# "undrafted" -- 78%/73%/69% of players in 2022-24 and 100% in 2025 -- and `docs/decisions.md`
# records that it makes the series unusable, because nothing downstream can tell a 169th pick
# from no pick. And `.github/scripts/heartbeat.sh` carries the closer one: `jq -r '.ts // 0'`
# made a missing timestamp an age of `now - 0`, so the watchdog reported 56.7 years of
# staleness on every run and could never reach the branch that closes an incident -- "one
# sentinel standing for unreachable, unreadable and stale", three failures with three
# different fixes. NaN is outside the range of both fields by construction and cannot be
# arithmetic'd into a reading.
#
# A ceiling of zero -- a perfect-foresight arm gaining nothing at all over the incumbent -- is
# the strongest finding a ceiling can carry, and it has to reach the reader as one. So
# `present` is the only way to ask, rather than a truthiness check each reader writes for
# itself and one of them writes as `if s["ceiling"]:`.
ABSENT = float("nan")


def present(value: float) -> bool:
    """Whether a summary field carries a value at all. `present(0.0)` is True; that is why
    this exists rather than a `!= 0` or a truthiness test at each reader."""
    return not math.isnan(value)


def paired_report(s: dict, *, arm_a: str, arm_b: str,
                  unit: str = "points per team game", places: int = 2,
                  show_n: bool = True) -> list[str]:
    """The n / interval / P(better) block, as lines rather than prints.

    Returned rather than printed for the reason `hub.draft.report` exists: a block that prints
    cannot be composed, capped, or asserted on.

    `places` because the weekly gate's effect is a tenth the size of the draft gate's and
    rounds to +0.22 at two -- while every doc and ADR quotes it as +0.215. `show_n` because
    that gate prints its own roster-week count, with the cluster count beside it.

    `mde` and `ceiling` render only when present. Nothing computes either yet, so every caller
    today gets exactly the two lines it got before -- not a placeholder, not a blank, not a
    line reading `nan`. Order is deliberate: the effect, the interval around it, the smallest
    effect the run could have resolved, and then how much there was to resolve. Each line is
    read against the one above it.
    """
    head = f"n={int(s['n'])}  " if show_n else ""
    lines = [
        f"\n  {head}{arm_a} - {arm_b} = {s['mean']:+.{places}f} {unit}",
        f"  95% CI [{s['lo']:+.{places}f}, {s['hi']:+.{places}f}]   "
        f"P({arm_a} better) {s['p_better'] * 100:.1f}%",
    ]
    # `.get` rather than `s["mde"]`: hand-built summaries reach this block too, and a dict
    # without the slot has exactly as much to say about its MDE as one carrying ABSENT.
    mde = s.get("mde", ABSENT)
    if present(mde):
        lines.append(f"  MDE at 80% power {mde:+.{places}f} {unit}")
    ceiling = s.get("ceiling", ABSENT)
    if present(ceiling):
        lines.append(f"  ceiling (perfect foresight) {ceiling:+.{places}f} {unit}")
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
              ceiling: float = ABSENT) -> dict[str, float]:
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

    **`mde` and `ceiling` are slots, and today both are `ABSENT`.** They are the two numbers
    a gate needs before a null it reports means anything: the smallest effect this run could
    have resolved at 80% power, and the largest one there was to find -- what a
    perfect-foresight arm gains over this gate's own incumbent, on this gate's own harness, in
    this gate's own units. `mde` will be computed here, from the same cluster-mean vector the
    interval comes from. `ceiling` arrives from the caller instead, because measuring it means
    playing an extra arm and only the harness knows what its arms are.

    Neither is read by anything yet, and that is deliberate: they exist now so the units that
    compute them change no call site's signature. Until then a block that has neither prints
    neither, and `gate` decides on the same two halves ADR-0019 fixed.
    """
    if paired.is_empty():
        return {"n": 0, "clusters": 0, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_better": float("nan"),
                "mde": ABSENT, "ceiling": ceiling}
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
            "p_better": float((draws > 0).mean()),
            "mde": ABSENT, "ceiling": ceiling}


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
