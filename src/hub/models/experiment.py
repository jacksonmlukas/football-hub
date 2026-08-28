"""The walk-forward paired experiment, which this repo runs twice and had never named.

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
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl

from hub.models.measure import realised_ppg

# One column list, so both harnesses hit one cache entry. `nflverse._cache_path` keys on the
# sorted column set -- deliberately, so a caller asking for six columns is never served an
# earlier caller's five -- which means two callers asking for different slices of the same
# table download it twice.
PLAYER_STATS_COLS: tuple[str, ...] = (
    "player_id", "player_display_name", "position", "season", "week", "fantasy_points_ppr",
)


def board_as_of(season: int) -> tuple[pl.DataFrame, object]:
    """The board for `season`, built from the last consensus scrape before it opened.

    The temporal rule lives here rather than in each harness: a strategy scored against
    rankings published after the season is hindsight wearing a backtest's clothes.
    """
    from hub.draft.board import build
    return build(season=season - 1, season_ahead=season, as_of=f"{season}-09-01")


def walk_forward_inputs(
    seasons: Sequence[int], *,
    build_board: Callable[[int], tuple[pl.DataFrame, object]] | None = None,
    load_stats: Callable[[int], pl.DataFrame] | None = None,
    on_season: Callable[[int], None] | None = None,
) -> tuple[dict[int, pl.DataFrame], dict[int, pl.DataFrame]]:
    """`(boards, realised)` per season. The two injectable seams are what make this testable.

    `on_season` is a progress hook rather than a print, so a caller under a line cap can stay
    quiet and this module stays free of stdout.
    """
    boards: dict[int, pl.DataFrame] = {}
    realised: dict[int, pl.DataFrame] = {}
    for yr in seasons:
        if on_season is not None:
            on_season(yr)
        boards[yr] = (build_board or board_as_of)(yr)[0]
        realised[yr] = realised_ppg((load_stats or _stats)(yr))
    return boards, realised


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
