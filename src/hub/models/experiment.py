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

`expanding_seasons` and `paired_gain` were pulled in on 2026-08-27. Both are the *mechanics*
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


def paired_report(s: dict, *, arm_a: str, arm_b: str,
                  unit: str = "points per team game") -> list[str]:
    """The n / interval / P(better) block, as lines rather than prints.

    Returned rather than printed for the reason `hub.draft.report` exists: a block that prints
    cannot be composed, capped, or asserted on.
    """
    return [
        f"\n  n={int(s['n'])}  {arm_a} - {arm_b} = {s['mean']:+.2f} {unit}",
        f"  95% CI [{s['lo']:+.2f}, {s['hi']:+.2f}]   "
        f"P({arm_a} better) {s['p_better'] * 100:.1f}%",
    ]


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


def summarise(paired: pl.DataFrame, *, bootstrap: int = BOOTSTRAP,
              seed: int = 0) -> dict[str, float]:
    """Mean paired difference, a bootstrap interval, and P(optimizer better).

    Bootstrapped over *paired* observations rather than over each arm separately, matching
    `hub.models.eval.compare`: the arms share a room and a seed, so resampling them
    independently would throw away the pairing that the design exists to create.
    """
    d = paired["diff"].to_numpy()
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_better": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(bootstrap, n))
    means = d[idx].mean(axis=1)
    return {"n": float(n), "mean": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "p_better": float((means > 0).mean())}
