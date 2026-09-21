"""Assembly for `hub.season.weekly_gate`: rosters, realised points, and both arms' scores.

Separate from the gate for the reason `backtest` and `lineup_gate` split the same way -- the
statistics have to be testable without a network, and a gate that only runs against live
nflverse is one nobody re-runs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import polars as pl

from hub.fetch import nflverse
from hub.league import REG_SEASON_WEEKS
from hub.models.experiment import realised_ppg, require_corrections
from hub.models.panel import PanelSpec, build_panel, weekly_consensus
from hub.models.weekly import Shrink, fit_shrink, positional_sd, project, standard_error
from hub.names import player_key
from hub.season.weekly_gate import UNRANKED, GateInputs


class SeasonDropped(RuntimeError):
    """A requested season produced no scored row, and it is not the walk-forward buffer.

    `assemble_universe` walk-forwards on `expanding_seasons`, which needs a season strictly
    *before* the one it scores to train on -- so the earliest season in `seasons` is always
    consumed as training only and never itself scored. That drop is expected and every caller
    of this module has to be able to tell it apart from a real one: a partition missing on
    disk, a `VOID` condition upstream, or a join that silently lost a season's rows.

    #378: a `--ceiling` run named four seasons, `docs/weekly-blend-gate.md` and the ADR-0019
    amendment's rule-16 table both read this gate at k=4, and the run came back with three
    clusters and no indication anything had gone missing -- season 2022 was simply not in the
    per-season table it printed. The run was not naming the walk-forward buffer; it was
    reusing `weekly_gate`'s own `--seasons` default, which named four seasons with none of
    them set aside to train the first one on, so `expanding_seasons` dropped the earliest of
    the four rather than a fifth season nobody asked it to score. This exception is what turns
    that into a refusal instead of a quietly thinner table.
    """


# No cycle: `hub.models.weekly` reaches `hub.cli`, `hub.config`, `hub.declare` and three
# sibling `hub.models` modules and nothing under `hub.season` or `hub.draft`, so this import
# was never lazy for a cycle -- it was lazy because it sat inside `assemble_universe` beside
# `board_as_of`, `cohort` and `applied`, which *are* fold-of-a-different-kind lazy (see the
# in-body imports still in `assemble_universe`: none of those has a cycle back here either, on
# the same check, but moving them is not this ticket -- #342 is the projection arm only). This
# one has to be a top-level import regardless of the cycle question, because `project` is now
# a default *argument value*, and a default is evaluated when the module loads, not when the
# function runs -- an in-body import cannot supply one.


def preseason_ranks(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Each player's **August** consensus rank, from the board as it stood before the season.

    Deliberately not the weekly ranking: this is the market's opinion four months before the
    week, which is a different quantity from the Friday page Gate B measures against. It is
    what the market-implied shrinkage regresses toward.

    The board's report is read even though only `ecr` is selected here, and no **Correction**
    moves that column. The board short a term is the same board `assemble_universe` drafts a
    **Cohort** from later in the same run, so the question is not whether this column is
    affected but whether the run should start at all -- and refusing at the first board that
    is short a term costs an operator one build rather than four and a discarded interval.
    `hub.models.experiment.require_corrections` holds the rule and the reason.
    """
    from hub.draft.board import board_as_of
    frames = []
    for yr in seasons:
        b, report = board_as_of(yr)
        require_corrections(yr, report)
        frames.append(b.select(
            pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
            pl.lit(yr).cast(pl.Int64).alias("season"),
            pl.col("ecr").cast(pl.Float64).alias("preseason_ecr")))
    return (pl.concat(frames).drop_nulls("preseason_ecr")
              .group_by(["key", "season"]).agg(pl.col("preseason_ecr").min()))


WEEKS = REG_SEASON_WEEKS


class ProjectionArm(Protocol):
    """The projection arm `assemble_universe` scores, over one fold.

    Structural rather than a subclass relationship -- the same reason
    `hub.models.experiment.CorrectionReport` is a `Protocol` -- so a test's second arm has to
    match only this call shape, not inherit from anything. `hub.models.weekly.project` already
    has exactly this signature and is the default without `assemble_universe` doing anything
    to adapt it.

    The return only has to carry `now`'s columns plus `mu`, which is what `project` returns
    (`now.with_columns(...)`) and what the loop below already assumed of whatever `project`
    returned before this was a parameter: it reads `key`, `season`, `week` and `mu` off the
    result and adds `se` itself. `se` is not this callable's to supply -- it is
    `positional_sd(past)` and `standard_error(now, sigma)`, read from the fold's *training*
    half, which a callable handed only `now` has no way to reach. It stayed assembled outside
    the arm rather than being threaded in as a third parameter no test needs to vary.
    """

    def __call__(self, now: pl.DataFrame, *, shrink: Shrink | None = None) -> pl.DataFrame: ...


def _one_scale(cons: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The arm under test, scored in points throughout -- including where it must fall back.

    **The defect this replaces.** The fallback was `np.where(np.isnan(mu), cons, mu)`, which
    put two incommensurable things in one column: `cons` is negated ECR, roughly -1 to -300,
    and `mu` is fantasy points, roughly 0 to 30. So every player the model could project
    outranked every player it could not, whatever either was worth -- a projected two-point
    scrub sorted above a rank-1 star. The fallback was meant to stop the arm being handicapped
    and instead handed it a preference unrelated to either estimate (#44).

    **Why a fallback at all, rather than a sentinel -- and why the question stopped being the
    right one.** Benching everyone the model cannot price would hand consensus information the
    arm under test lacks, and a gate whose arms see different universes is the defect that
    made the first `lineup_gate` unable to fail. That argument is correct and it is an argument
    against *one arm* benching him. Both arms declining to score him is not a sentinel, it is
    a smaller slate, and that is what the gate does now: `weekly_gate.priced_by_both` keeps
    every cell below out of every arm's lineup, so the guess this function makes reaches no
    result. #206, and the reason the units question was never the whole of it -- a fallback
    with the right units is still an estimator whose error the arm under test carries.

    **What replaces it, and what it assumes.** Per week, among players carrying both a rank and
    a projection, consensus rank and points are paired observations of the same players. An
    unprojected player is scored at the points those paired players carry *at his rank* --
    linear interpolation through them, which is monotone in rank and reads off nothing but the
    week's own numbers. It assumes only that consensus ordering carries information about
    points, which is the premise of using consensus as the incumbent at all: if that were
    false the gate would have no control arm.

    A week with no paired player has nothing to calibrate against, so its unprojected players
    keep `UNRANKED` -- unscoreable rather than guessed at.

    **This is one of three treatments and the gate reports under none of them.** The other two
    are derivable from what this returns beside `~np.isnan(mu)`, and
    `hub.season.weekly_gate.TREATMENTS` holds all three with the standing of each: the column
    below is the one the assembly carries (`COLUMN_TREATMENT`), an unprojected player scored
    `UNRANKED` throughout is the *unscoreable* comparison, and the superseded mixed scale is
    `cons` in those cells. #207 made a run report the spread across all three, which is why
    the `projected` mask leaves here at all; #206 then removed the choice by removing the
    cells from what is scored.

    **So why interpolate at all now.** Because a run still scores all three, as a check that
    they cannot differ, and two of the three are rebuilt from this column -- so a column that
    had already collapsed them to `UNRANKED` could only check `mixed scale` against itself.
    The arithmetic below is what makes the -1.004 in `docs/weekly-blend-gate.md` reachable
    from the code that produced it rather than transcribable only from the page. It is not on
    any path the verdict is read off.
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
                      shrink: str | None = None, arm: ProjectionArm = project,
                      expected: bool = False, holdout: bool = False):
    # pragma: no cover - network
    """Rosters, realised points and both arms' scores, over the whole board.

    A roster becomes a list of indices into the season's universe so it can *change*: the
    waiver arm adds and drops, and a per-roster matrix cannot represent a player who was not
    on the roster when the matrix was built.

    `holdout` drafts each season's Cohort under `conf/holdout/{season}.json`, the constants
    fitted without that season (#294): the draft simulation is where the eight held-out
    constants reach this gate, and the wrap is around that call and nothing else, so the
    shipped values are back before the next season is read. The run line naming each set is
    `weekly_gate.main`'s to print.

    `arm` is the projection under test, accepted rather than created (#342). Every fold calls
    it the same way `weekly_gate.main` always called `hub.models.weekly.project` -- `now` and
    the `shrink` already fit on that fold's `past` -- so `project` is the default and every
    existing caller sees the identical byte-for-byte universe it always did. Before this the
    import sat inside this function's body and the arm under test was a name only this module
    bound, so scoring anything else meant monkeypatching `hub.models.weekly.project` from
    outside and hoping the patch landed on the module this file actually reads from -- which
    it once did not, and scored the shipped arm twice with zero flips to show for it. A second
    adapter now just is a second value for this parameter.

    Returns a `GateInputs`: ten aligned collections that used to be a positional tuple.
    """
    from collections.abc import Sequence as _Seq
    from contextlib import nullcontext

    from hub.draft.board import board_as_of
    from hub.draft.cohort import cohort
    from hub.holdout import applied
    from hub.models.experiment import PLAYER_STATS_COLS, expanding_seasons
    want_ranks = shrink is not None and "market" in shrink
    panel = build_panel(seasons, PanelSpec(
        consensus=False, expected=expected,
        ranks=preseason_ranks(seasons) if want_ranks else None))
    ecr = weekly_consensus(seasons)

    projected = []
    for _season, past, now in expanding_seasons(panel):
        # No multiplier is fitted here since #248: the weekly projection is the `f = 1`
        # rebuild, and its Usage counts are the priors. The shrinkage still is, on `past`
        # only. A shrinkage fitted on the season it is scored against would be the treatment
        # arm reading its own answer sheet.
        sigma = positional_sd(past)
        sh = None if shrink is None else fit_shrink(
            past, objective=shrink.split("-")[0],
            target=("market-only" if shrink == "market-only"
                    else "market" if shrink is not None and "market" in shrink else "position"))
        projected.append(arm(now, shrink=sh)
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
    # Not `projected`: that name is the list of per-season projection frames thirty lines up,
    # and rebinding it here would leave one name for two things in one function.
    priced: dict[int, np.ndarray] = {}

    for yr in sorted(set(proj["season"].unique().to_list()) & set(seasons)):
        print(f"  building the {yr} board as of {yr}-09-01 ...", flush=True)
        # `board_as_of` is reproducible as of improvements.md #18 -- it sorts on
        # (ecr, player) and its DvP aggregation no longer hands a hash-ordered
        # frame to a mean -- so the workaround that used to sort here is gone.
        board, report = board_as_of(yr)
        # The Cohort below is drafted from this board, so a season whose Corrected ADP came
        # from a subset of the Corrections is a season drafted off a different ranking rather
        # than a thinner one -- and the gate would pool it with the others and publish one
        # interval over two arms. `require_corrections` holds the rule and the reason.
        require_corrections(yr, report)
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
        # Which cells of that column are the model's own number, carried out rather than left
        # behind. Everything the fallback did is `~projected`, so the gate can re-score the
        # same rows under the other two treatments of it without re-reading the network -- and
        # the mixture it prints is counted off this mask instead of being discovered by
        # probing (#207). `addable` is the *other* half of the same `np.isnan` and is a pool
        # mask, not a cell classification; deriving one from the other would tie the waiver
        # rule to the fallback report.
        priced[yr] = ~np.isnan(mu)
        # Addable only where BOTH arms can score him. See the pre-registration in
        # docs/weekly-projection-plan.md: consensus ranks 35.8% of the pool, so an unmasked
        # pool would hand the arm under test six hundred players the incumbent cannot see.
        #
        # Since #206 this mask is read twice and the second reading is the load-bearing one:
        # `weekly_gate.priced_by_both` is this, and it is now what either arm may *start* as
        # well as what it may add. It is left as one array rather than split, because the two
        # readings are one quantity -- "both arms can price him this week" -- and two arrays
        # spelling it would be two things to keep in agreement.
        addable[yr] = (cons > UNRANKED) & ~np.isnan(mu)
        # A player with no projection has no standard error either; zero means the lower
        # confidence bound leaves the consensus fallback exactly where it was.
        se[yr] = np.nan_to_num(_matrix(keys, se_of, float("nan")), nan=0.0)

        # One recipe, one home. This was written out here and again inside
        # `lineup_gate.main`, with the seed formula copied by hand into both.
        #
        # The report goes with the board (#231). It is the one unpacked above and already read
        # by `require_corrections`; before this it stopped here, and `simulate_remaining_draft`
        # derived a fresh one from the frame's columns instead. The two agree on a historical
        # board, which is why nothing ranked differently -- and why passing the recorded answer
        # rather than relying on the re-derivation agreeing is worth doing while they do.
        with applied(yr) if holdout else nullcontext():
            made = cohort(board, yr, drafts=drafts, seed=seed, report=report)
        # Positions come from the Cohort too, rather than being read off the board a second
        # time -- two readings of one frame is how they come to disagree.
        rosters[yr], pool[yr], pos[yr] = made.rosters, made.pool, made.pos

    # #378: loud rather than silent. Exactly one season -- the earliest requested -- is
    # supposed to vanish here, consumed by `expanding_seasons` as the walk-forward buffer for
    # the one after it; anything else missing is a real drop and would otherwise surface only
    # as a per-season table one row short, the way #376's `--ceiling` run did. Requested and
    # scored are compared as sets because `seasons` is not required to arrive sorted or deduped
    # and dict keys already collapsed duplicates on the scored side.
    requested = sorted({int(s) for s in seasons})
    scored = sorted(rosters)
    buffer_season = requested[0] if requested else None
    unexpected = [s for s in requested if s not in rosters and s != buffer_season]
    if unexpected:
        raise SeasonDropped(
            f"requested seasons {requested}, scored {scored} -- season(s) {unexpected} "
            f"produced no row and {'is' if len(unexpected) == 1 else 'are'} not the "
            f"walk-forward buffer ({buffer_season}). A partition missing on disk, a VOID "
            f"condition, or a join dropping the season -- find out which before trusting "
            f"anything scored on the seasons that did come back.")

    return GateInputs(rosters, pos, realised, consensus, weekly, pool, addable, se,
                      covered_weeks(ecr), priced)


def covered_weeks(ecr: pl.DataFrame) -> set[tuple[int, int]]:
    """The `(season, week)` pairs the consensus page actually has a scrape for."""
    return {(int(r["season"]), int(r["week"]))
            for r in ecr.select("season", "week").unique().iter_rows(named=True)}
