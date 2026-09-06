"""Assembly for `hub.season.weekly_gate`: rosters, realised points, and both arms' scores.

Separate from the gate for the reason `backtest` and `lineup_gate` split the same way -- the
statistics have to be testable without a network, and a gate that only runs against live
nflverse is one nobody re-runs.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.fetch import nflverse
from hub.league import REG_SEASON_WEEKS
from hub.models.experiment import realised_ppg
from hub.models.panel import PanelSpec, build_panel, weekly_consensus
from hub.names import player_key
from hub.season.weekly_gate import UNRANKED, GateInputs

NOT_FITTED_BECAUSE = (
    "assembly for Gate B: rosters, realised points and both arms' scores. It reads outcomes "
    "and never predicts. "
)


def preseason_ranks(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Each player's **August** consensus rank, from the board as it stood before the season.

    Deliberately not the weekly ranking: this is the market's opinion four months before the
    week, which is a different quantity from the Friday page Gate B measures against. It is
    what the market-implied shrinkage regresses toward.
    """
    from hub.draft.board import board_as_of
    frames = []
    for yr in seasons:
        b = board_as_of(yr)[0]
        frames.append(b.select(
            pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
            pl.lit(yr).cast(pl.Int64).alias("season"),
            pl.col("ecr").cast(pl.Float64).alias("preseason_ecr")))
    return (pl.concat(frames).drop_nulls("preseason_ecr")
              .group_by(["key", "season"]).agg(pl.col("preseason_ecr").min()))


WEEKS = REG_SEASON_WEEKS


def _one_scale(cons: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The arm under test, scored in points throughout -- including where it must fall back.

    **The defect this replaces.** The fallback was `np.where(np.isnan(mu), cons, mu)`, which
    put two incommensurable things in one column: `cons` is negated ECR, roughly -1 to -300,
    and `mu` is fantasy points, roughly 0 to 30. So every player the model could project
    outranked every player it could not, whatever either was worth -- a projected two-point
    scrub sorted above a rank-1 star. The fallback was meant to stop the arm being handicapped
    and instead handed it a preference unrelated to either estimate (#44).

    **Why a fallback at all, rather than a sentinel.** Benching everyone the model cannot price
    would hand consensus information the arm under test lacks, and a gate whose arms see
    different universes is the defect that made the first `lineup_gate` unable to fail. The
    fallback's intent was right; only its units were wrong.

    **What replaces it, and what it assumes.** Per week, among players carrying both a rank and
    a projection, consensus rank and points are paired observations of the same players. An
    unprojected player is scored at the points those paired players carry *at his rank* --
    linear interpolation through them, which is monotone in rank and reads off nothing but the
    week's own numbers. It assumes only that consensus ordering carries information about
    points, which is the premise of using consensus as the incumbent at all: if that were
    false the gate would have no control arm.

    A week with no paired player has nothing to calibrate against, so its unprojected players
    keep `UNRANKED` -- unscoreable rather than guessed at.
    """
    out = np.where(np.isnan(mu), UNRANKED, mu)
    for w in range(cons.shape[1]):
        col_c, col_m = cons[:, w], mu[:, w]
        paired = ~np.isnan(col_m) & (col_c > UNRANKED)
        need = np.isnan(col_m) & (col_c > UNRANKED)
        if not paired.any() or not need.any():
            continue
        order = np.argsort(col_c[paired])
        out[need, w] = np.interp(col_c[need], col_c[paired][order], col_m[paired][order])
    return out


def _matrix(keys: Sequence[str], lookup: dict[tuple[str, int], float],
            default: float) -> np.ndarray:
    """(roster, weeks) from a {(player key, week): value} map, with a stated default."""
    out = np.full((len(keys), WEEKS), default, dtype=float)
    for i, k in enumerate(keys):
        for w in range(1, WEEKS + 1):
            v = lookup.get((k, w))
            if v is not None:
                out[i, w - 1] = float(v)
    return out


def assemble_universe(seasons: Sequence[int], *, drafts: int = 20, seed: int = 0,
                      shrink: str | None = None, expected: bool = False):
    # pragma: no cover - network
    """Rosters, realised points and both arms' scores, over the whole board.

    A roster becomes a list of indices into the season's universe so it can *change*: the
    waiver arm adds and drops, and a per-roster matrix cannot represent a player who was not
    on the roster when the matrix was built.

    Returns a `GateInputs`: nine aligned collections that used to be a positional tuple.
    """
    from collections.abc import Sequence as _Seq

    from hub.draft.board import board_as_of
    from hub.draft.cohort import cohort
    from hub.models.experiment import PLAYER_STATS_COLS, expanding_seasons
    from hub.models.weekly import (
        VOLUME,
        fit_multiplier,
        fit_shrink,
        positional_sd,
        project,
        standard_error,
    )
    want_ranks = shrink is not None and "market" in shrink
    panel = build_panel(seasons, PanelSpec(
        consensus=False, expected=expected,
        ranks=preseason_ranks(seasons) if want_ranks else None))
    ecr = weekly_consensus(seasons)

    projected = []
    for _season, past, now in expanding_seasons(panel):
        coefs = {c: fit_multiplier(past, c) for c in VOLUME}
        sigma = positional_sd(past)
        # Fitted on `past` only, like the multiplier. A shrinkage fitted on the season it is
        # scored against would be the treatment arm reading its own answer sheet.
        sh = None if shrink is None else fit_shrink(
            past, coefs, objective=shrink.split("-")[0],
            target=("market-only" if shrink == "market-only"
                    else "market" if shrink is not None and "market" in shrink else "position"))
        projected.append(project(now, coefs, shrink=sh)
                         .with_columns(pl.Series("se", standard_error(now, sigma)))
                         .select("key", "season", "week", "mu", "se"))
    proj = pl.concat(projected) if projected else pl.DataFrame(
        schema={"key": pl.Utf8, "season": pl.Int64, "week": pl.Int64,
                "mu": pl.Float64, "se": pl.Float64})

    rosters: dict[int, list[list[int]]] = {}
    pos: dict[int, _Seq[str]] = {}
    realised: dict[int, np.ndarray] = {}
    consensus: dict[int, np.ndarray] = {}
    weekly: dict[int, np.ndarray] = {}
    pool: dict[int, list[list[int]]] = {}
    addable: dict[int, np.ndarray] = {}
    se: dict[int, np.ndarray] = {}

    for yr in sorted(set(proj["season"].unique().to_list()) & set(seasons)):
        print(f"  building the {yr} board as of {yr}-09-01 ...", flush=True)
        # `board_as_of` is reproducible as of improvements.md #18 -- it sorts on
        # (ecr, player) and its DvP aggregation no longer hands a hash-ordered
        # frame to a mean -- so the workaround that used to sort here is gone.
        board = board_as_of(yr)[0]
        names = board["player"].to_list()
        keys = [player_key(n) for n in names]

        # Routed (#36), the same call `models.experiment` already makes for this source. The
        # loader drops the ~22 rows a season that belong to no player and refuses if any of
        # them scored -- so what was a silent inclusion of unattributable rows in a points
        # aggregation is now either absent or an upstream break that says so.
        stats = nflverse.load("player_stats", [yr], cols=list(PLAYER_STATS_COLS))
        pts_of = {(r["player"], int(r["week"])): float(r["points"])
                  for r in realised_ppg(stats).iter_rows(named=True)}
        ecr_of = {(r["key"], int(r["week"])): -float(r["ecr"])
                  for r in ecr.filter(pl.col("season") == yr).iter_rows(named=True)}
        rows = list(proj.filter(pl.col("season") == yr).iter_rows(named=True))
        mu_of = {(r["key"], int(r["week"])): float(r["mu"]) for r in rows}
        se_of = {(r["key"], int(r["week"])): float(r["se"]) for r in rows}

        realised[yr] = _matrix(keys, pts_of, 0.0)
        cons = _matrix(keys, ecr_of, UNRANKED)
        consensus[yr] = cons
        mu = _matrix(keys, mu_of, float("nan"))
        weekly[yr] = _one_scale(cons, mu)
        # Addable only where BOTH arms can score him. See the pre-registration in
        # docs/weekly-projection-plan.md: consensus ranks 35.8% of the pool, so an unmasked
        # pool would hand the arm under test six hundred players the incumbent cannot see.
        addable[yr] = (cons > UNRANKED) & ~np.isnan(mu)
        # A player with no projection has no standard error either; zero means the lower
        # confidence bound leaves the consensus fallback exactly where it was.
        se[yr] = np.nan_to_num(_matrix(keys, se_of, float("nan")), nan=0.0)

        # One recipe, one home. This was written out here and again inside
        # `lineup_gate.main`, with the seed formula copied by hand into both.
        made = cohort(board, yr, drafts=drafts, seed=seed)
        # Positions come from the Cohort too, rather than being read off the board a second
        # time -- two readings of one frame is how they come to disagree.
        rosters[yr], pool[yr], pos[yr] = made.rosters, made.pool, made.pos
    return GateInputs(rosters, pos, realised, consensus, weekly, pool, addable, se,
                      covered_weeks(ecr))


def covered_weeks(ecr: pl.DataFrame) -> set[tuple[int, int]]:
    """The `(season, week)` pairs the consensus page actually has a scrape for."""
    return {(int(r["season"]), int(r["week"]))
            for r in ecr.select("season", "week").unique().iter_rows(named=True)}
