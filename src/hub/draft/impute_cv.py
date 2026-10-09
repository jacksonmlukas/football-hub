"""IMPUTE_CV measured on the players the board actually imputes: rookies (#277).

`hub.models.predict.IMPUTE_CV` / `IMPUTE_CV_BY_POS` say how wrong `board._impute_xfp`'s
rank curve is, as a share of the projection it invents. What shipped before 2026-09-16 was
measured leave-one-out on observed **veterans** inside the top 200: each was blanked and
re-imputed from the rest. That was a lower bound on the error the flag is carrying, because
the players the board imputes are **rookies** -- no prior NFL season, so no prior-season
xFP -- and a veteran's rank was set partly by the very production the curve predicts, where
a rookie's rank carries no such information. Since #298 the constants **are this module's
measurement**: pooled 0.315 season-clustered, RB 0.359, WR 0.288, and QB and TE at the
pooled value because eight and six rookies are too thin to split.

**The population, as #277's disposition fixed it.** Rookies are players with no prior NFL
season who appear inside the top 200 by consensus on that season's board; the realised
season is scored against the curve's prediction, and the residual CV is taken per position
and pooled -- the same statistic as the shipped constant, `sd(realised / imputed - 1)`, so
the two are comparable. Two departures from the disposition's wording, each stated:

* *Top 200 by consensus rank, not by ADP.* ESPN publishes ADP for the current season only
  (ADR-0010; `backtest.LIMITATIONS`), so a past board has no ADP to rank on. Consensus is
  the rank the curve reads, which is the rank the residual is a function of.
* *"No prior NFL season" is read as "first season with any weekly line in nflverse player
  stats".* nflverse's `rookie_year` lives on its rosters table, which no loader in this
  repo carries; the weekly stats are cached 2019 onward, so a 2021 board looks back two
  seasons and a 2025 board six. A player who was rostered but recorded nothing for a whole
  season before his first line would be mis-read as a rookie; inside the top 200 that is
  rare and it is said rather than hidden.

Realised production comes from `board.expected_points(season)` -- the season's own
`ff_opportunity` rows -- as **both** realised PPR points per game (`ppg`, the disposition's
quantity) and the season's own xFP per game (`xfp_pg`, the quantity the curve
imputes and the one the veteran measurement compared against). Both CVs are printed; the
first is what the ticket asked for, the second is the like-for-like beside the shipped
number.

Nothing here writes a shipped constant. The decision that moved the values to this
measurement was #298 (2026-09-16), taken on the five-season run; a re-run prints the numbers
beside the shipped ones.

**The hold-out binds both constants (#320, option B, adopted 2026-10-08).** Until then a
season's set recorded `IMPUTE_CV` and `IMPUTE_CV_BY_POS` as *not refitted*, which left the
rookie measurement in-sample on the hold-out draft path -- the half of the #376 width
discrepancy the maintainer's 2026-09-21 comment names. `--exclude-season N --point-in-time
--record` now refits them on the seasons **strictly before N** (method.md rule 2) and records
the values:

* `predict.IMPUTE_CV` is the shipped statistic on the fit seasons: the mean of the per-season
  pooled CVs against the season's own xFP (the season-clustered estimate, what #298 adopted);
  on a single fit season (2022's, which has only 2021) that mean is the one season's pooled CV.
* `predict.IMPUTE_CV_BY_POS` gives a position its own CV (over its rookies, as the shipped
  RB 0.359 and WR 0.288 are) only where it has at least `MIN_POS_N` of them, and the refit
  pooled value otherwise -- the rule that put the shipped QB and TE (eight and six rookies) at
  the pooled value, stated as a threshold fixed before the run. A position the refit cannot
  split says so in the printed line; it is not a missing measurement.

The earlier-seasons fit is deliberately not the leave-one-season-out of the other fitting
scripts: a 2023 board's constants refitted with 2024-25 in them would read seasons that had
not happened, which is the leakage rule 2 names. The cost is fewer clusters on the early
seasons (2022 fits on one), said in the run line, not hidden.

**Like with like (#327).** The shipped basis imputes a rookie from the *previous* season's
curve and realises him on his own season, so its ratio carries a year of drift on top of the
interpolation error -- and `IMPUTE_CV` is added in quadrature to `TALENT_CV`, which carries that
drift already. `measure` therefore also imputes each rookie from his **own season's**
veterans (`like_for_like_imputed`) and reports the 2x2 (sd and RMS relative error, on the
drift-carrying and the like-for-like basis), the median residual beside the RMS, and the drift
component separated out. It reads the `BuildReport` of every board and refuses one that did
not build every stage a historical build can run (`require_built`). Rookies are keyed on
`player_id` from regular-season lines, so a name two players share is tallied as ambiguous
rather than read as the earlier of them, and a debut a season late is tallied as delayed.
Nothing here writes the constant: that is #322's decision, taken on the like-for-like RMS.

    uv run python scripts/fit_impute_cv.py
    uv run python scripts/fit_impute_cv.py --seasons 2021,2022,2023,2024,2025 --min-games 8
    uv run python scripts/fit_impute_cv.py --exclude-season 2024
    uv run python scripts/fit_impute_cv.py --exclude-season 2024 --point-in-time --record
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from hub import holdout
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.declare import not_an_input
from hub.names import player_key

# The board depth the disposition named and the veteran measurement used: inside rank 200.
TOP = not_an_input(
    200,
    "the board depth this measurement is taken inside, the same depth the shipped "
    "veteran measurement used; a rule of the fit and not an input to a prediction")
# Fewest games before a per-game rate is a rate. Eight is half a season and the sample rule
# docs/weekly-spread.md fitted under; the board's own MIN_GAMES (10) is a replacement-level
# rule and would drop a rookie hurt in November whose ten weeks are a measurement.
MIN_GAMES = not_an_input(
    8,
    "a sample threshold of this measurement: fewest games before a rookie's per-game "
    "rate is read, and nothing a board or a simulation computes from")
# The boards a rookie residual can be measured on from the archive: the consensus archive
# starts 2019-12-27, so no 2019 preseason board exists, and 2020's as-of is not cached.
DEFAULT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

# Fewest rookies at a position before the hold-out refit gives it a CV of its own. Fixed
# before the refit was run (method.md rule 1). The shipped table splits RB (36) and WR (42)
# and pools QB (8) and TE (6); anything between 8 and 36 reproduces that split on the full
# data, and twenty is the round number in the gap.
MIN_POS_N = not_an_input(
    20,
    "fewest rookies a position needs before the hold-out refit gives it its own CV rather "
    "than the refit pooled value: the shipped table's own split (RB 36 and WR 42 own, QB 8 "
    "and TE 6 pooled) put as a threshold; a rule of the fit and not an input to a prediction")

FIRST_STATS_SEASON = not_an_input(
    2019,
    "the first season the cached weekly stats reach: the look-back a rookie proxy has, "
    "a fact about the archive and not an input to any prediction")


REGULAR_SEASON = not_an_input(
    "REG",
    "nflverse's `season_type` for the regular season: a rookie's first season is the first "
    "regular-season line, because a postseason-only line is not a season played (#327); a "
    "fact about the archive and not an input to any prediction")

# The stages `build` runs on a historical board are the advisory ones that are not live-only;
# `require_built` reads that off the declaration rather than naming them. One board is short a
# stage for a reason no run can cure: the durability stage reads the *previous* preseason's
# consensus, and the archive starts 2020-10-16, so the 2021 board cannot build it (the stage
# raises `ContractViolation: ... a season before 2021 cannot be replayed`). It is named here,
# with the season it applies to, so the exemption is a fact a test can pin and not a loosened
# rule; any other degraded stage on any board is still refused.
ARCHIVE_UNBUILDABLE = not_an_input(
    {2021: ("durability",)},
    "the stages a historical board cannot build because the consensus archive begins "
    "2020-10-16, by season: a fact about the archive, and a stage `rookie_rows` never reads "
    "(durability prices absence; the imputation reads rank and xFP), not an input to a prediction")


class BoardNotBuilt(RuntimeError):
    """A board reached the measurement short a stage a historical build can run."""


def require_built(season: int, report: Any) -> None:
    """Refuse a board whose `BuildReport` shows a stage that should have run not having run.

    `board_as_of` degrades stage by stage on purpose (CLAUDE.md, graceful degradation), and
    that is right for the live path. This is a measurement whose number becomes a shipped
    constant, and the fitting script used to discard the report altogether (`board_as_of(yr)[0]`)
    although `board.board_as_of` names it as not optional: every caller of it is a Gate. A
    measurement on a board built without, say, its durability stage would be a number about a
    different board, printed as the same one.

    "Every stage" is every stage a historical build can run: the advisory ones that are not
    `live_only`. The scoring and roster checks are live-only (ESPN publishes both for the
    current season only) and the ADP stage is the one stage that legitimately does not
    apply to an as-of board, so none of the three is required here. The one stage the archive
    itself cannot build (`ARCHIVE_UNBUILDABLE`, the 2021 board's durability) is excused by name.
    """
    required = {s.name for s in report.stages if s.advisory and s.live_only is None}
    missing = [n for n in report.degraded()
               if n in required and n not in ARCHIVE_UNBUILDABLE.get(season, ())]
    if missing:
        raise BoardNotBuilt(
            f"the {season} board did not build {missing}; a measurement on a board short a "
            f"stage is a number about a different board, so none is taken (#327)")


def _key() -> pl.Expr:
    return pl.col("player_display_name").map_elements(player_key, return_dtype=pl.Utf8)


def first_seasons(stats: pl.DataFrame) -> pl.DataFrame:
    """`(player_id, first_season)`: the first regular season each player has any weekly line.

    The proxy for nflverse `rookie_year` -- see the module docstring. Keyed on `player_id`
    (#327), not on the player key: two players who share a name used to take the earliest
    season across both and a designed collision silently turned a rookie into a veteran. Regular
    season only, so a postseason line is not a season played; a `min` per id, so the answer
    cannot depend on row order.
    """
    return (stats.filter((pl.col("season_type") == REGULAR_SEASON)
                         & pl.col("player_id").is_not_null())
                 .group_by("player_id")
                 .agg(pl.col("season").cast(pl.Int64).min().alias("first_season")))


def name_ids(stats: pl.DataFrame) -> pl.DataFrame:
    """`(player, player_id, ambiguous)`: the id a board name resolves to, through the stats' own
    normalised names, over the fantasy positions only (a linebacker named like a quarterback is
    not a candidate). A name that resolves to more than one id is `ambiguous` and carries the
    smallest, only so the frame has one row per name: the caller excludes and tallies it."""
    named = (stats.filter((pl.col("season_type") == REGULAR_SEASON)
                          & pl.col("player_id").is_not_null()
                          & pl.col("position").is_in(DRAFTED_POSITIONS))
                  .select(_key().alias("player"), "player_id").unique())
    return (named.group_by("player")
                 .agg(pl.col("player_id").min().alias("player_id"),
                      (pl.col("player_id").n_unique() > 1).alias("ambiguous")))


def like_for_like_imputed(keyed: pl.DataFrame, ids: pl.DataFrame,
                          real: pl.DataFrame) -> pl.DataFrame:
    """`(player, imputed_lfl)`: the board's own rank curve, drawn from the board's own season.

    The shipped basis fills a rookie from the curve of last season's xFP per game (the as-of
    board's `xfp_per_game`) and is scored on his own season. Here each *observed* player keeps
    his rank and carries his **own-season** xFP per game instead; a veteran with no line that
    season, every imputed row and every ambiguous name goes null, and the shipped
    `board._impute_xfp` is run unchanged on the result. Same smoothing, same rank, same
    position split: only the season the curve is drawn from differs from the drift-carrying
    basis, which is what makes the difference between the two bases the drift (#327).
    """
    # Reached as an attribute and not imported by name: the curve under measurement is the
    # board's own private `_impute_xfp`, and making it public would edit `hub.draft.board`, a
    # module the weekly forward pin hashes (#456), for a measurement that changes nothing the
    # arm computes. The name is the board's to expose if it ever wants to.
    from hub.draft import board as board_mod

    amb = pl.col("ambiguous").fill_null(False)
    frame = (keyed.join(ids, on="player", how="left")
                  .join(real.select("player_id", "xfp_pg"), on="player_id", how="left")
                  .with_columns(pl.when(~pl.col("xfp_imputed") & ~amb)
                                  .then(pl.col("xfp_pg")).otherwise(None)
                                  .cast(pl.Float64).alias("xfp_per_game"))
                  .select("player", "pos", "ecr", "xfp_per_game"))
    filled = board_mod._impute_xfp(frame)
    return filled.select("player", pl.col("xfp_per_game").alias("imputed_lfl"))


def rookie_rows(board: pl.DataFrame, stats: pl.DataFrame, realised: pl.DataFrame, *,
                season: int, top: int = TOP,
                min_games: int = MIN_GAMES) -> tuple[pl.DataFrame, dict[str, int]]:
    """One row per rookie the board imputed and the season measured, and the counts on the way.

    The counts are the population rules applied in order: inside the top `top` by consensus;
    a rookie (first regular-season line is `season`, by player id); imputed by the board -- a
    rookie carrying a prior xFP is a join collision and is counted under `collisions`, not
    measured; holding a season line at all; and at least `min_games` games played. Three
    players the proxy cannot classify are counted apart and never folded into either side:
    `no_line` (imputed, no regular-season line at a fantasy position in any season),
    `ambiguous` (imputed, and the name resolves to more than one player id) and `delayed`
    (imputed, and the first line comes in a season *after* this board's -- a rookie drafted
    here who did not debut, who is a rookie on the later board and not on this one).

    Each row carries `imputed` (the board's, from last season's curve) and `imputed_lfl` (the
    same rank on this season's curve), so the two bases are scored on the identical rookies.
    """
    # The five columns the rules read, and no others: a served board also carries a prior
    # season's `games`, `fp` and `xfp`, null for every rookie, and a join that let them
    # through would filter on the wrong `games`.
    keyed = (board.select("player", "pos", "ecr", "xfp_per_game", "xfp_imputed")
                  .with_columns(pl.col("player").map_elements(player_key, return_dtype=pl.Utf8)
                                .alias("player")))
    top_n = keyed.filter((pl.col("ecr") <= top) & pl.col("pos").is_in(DRAFTED_POSITIONS))
    ids = name_ids(stats)
    first = first_seasons(stats)
    tagged = (top_n.join(ids, on="player", how="left")
                   .with_columns(pl.col("ambiguous").fill_null(False))
                   .join(first, on="player_id", how="left"))
    clear = tagged.filter(~pl.col("ambiguous"))
    rookies = clear.filter(pl.col("first_season") == season)
    collisions = rookies.filter(~pl.col("xfp_imputed"))
    imputed = rookies.filter(pl.col("xfp_imputed"))
    # One realised row per player id, the one with the most games and then by position and name,
    # so no row order can pick a different survivor (it was `unique(keep="first")` on a name).
    real = (realised.select(
                "player_id", "position", "full_name",
                pl.col("games").cast(pl.Int64),
                (pl.col("fp") / pl.col("games")).alias("ppg"),
                (pl.col("xfp") / pl.col("games")).alias("xfp_pg"))
            .filter(pl.col("games") > 0)
            .sort(["games", "position", "full_name"], descending=[True, False, False])
            .unique(subset=["player_id"], keep="first", maintain_order=True))
    lined = imputed.join(real.drop("position", "full_name"), on="player_id", how="inner")
    played = lined.filter(pl.col("games") >= min_games)
    top_imputed = tagged.filter(pl.col("xfp_imputed"))
    lfl = like_for_like_imputed(keyed, ids, real)
    rows = (played.join(lfl, on="player", how="left")
                  .select(pl.lit(season, dtype=pl.Int64).alias("season"), "player", "player_id",
                          "pos", "ecr", pl.col("xfp_per_game").alias("imputed"), "imputed_lfl",
                          "games", "ppg", "xfp_pg")
                  .sort("ecr", "player"))
    counts = {"top200": top_n.height, "rookies": rookies.height, "imputed": imputed.height,
              "with_line": lined.height, "played": played.height,
              "collisions": collisions.height,
              "no_line": top_imputed.filter(pl.col("player_id").is_null()).height,
              "ambiguous": top_imputed.filter(pl.col("ambiguous")).height,
              "delayed": top_imputed.filter(~pl.col("ambiguous")
                                            & (pl.col("first_season") > season)).height}
    return rows, counts


def _ratio(rows: pl.DataFrame, against: str, imputed: str = "imputed") -> np.ndarray:
    return rows.select((pl.col(against) / pl.col(imputed) - 1.0).alias("r"))["r"].to_numpy()


def _spread(r: np.ndarray, stat: str) -> float:
    """`sd` (ddof 1, the shipped constant's statistic) or `rms` (the root mean square of the
    relative error, which carries the bias the sd centres away)."""
    if stat == "rms":
        return float(np.sqrt(np.mean(r ** 2)))
    return float(np.std(r, ddof=1))


def residual_cv(rows: pl.DataFrame, against: str,
                imputed: str = "imputed") -> dict[str, dict[str, Any]]:
    """`sd` and `rms` of `realised / imputed - 1` per position and pooled, with n, the median and
    the mean (the bias).

    `cv` is the sd: the shipped constant's statistic, so the rookie number reads beside it.
    `rms` is the same residual about zero rather than about its mean, `sqrt(sd^2 + bias^2)`
    up to the n/(n-1) factor. A position with fewer than two rows has no spread and carries
    `None` rather than a zero that would read as a measurement.
    """
    out: dict[str, dict[str, Any]] = {}
    for pos in (*DRAFTED_POSITIONS, "pooled"):
        sub = rows if pos == "pooled" else rows.filter(pl.col("pos") == pos)
        r = _ratio(sub, against, imputed)
        out[pos] = {
            "n": int(r.size),
            "cv": _spread(r, "sd") if r.size >= 2 else None,
            "rms": _spread(r, "rms") if r.size >= 2 else None,
            "mean": float(np.mean(r)) if r.size else None,
            "median": float(np.median(r)) if r.size else None,
        }
    return out


def hold_out(rows: pl.DataFrame, exclude: int | None) -> tuple[pl.DataFrame, str]:
    """The rows with one season held out, and the sentence that says so (#294)."""
    seasons = sorted(rows["season"].unique().to_list())
    if exclude is None:
        return rows, f"  seasons {seasons}, nothing held out"
    kept = rows.filter(pl.col("season") != exclude)
    remain = sorted(kept["season"].unique().to_list())
    return kept, f"  season {exclude} held out; fitted on {remain}"


def season_clustered(rows: pl.DataFrame, against: str, *, shipped: float,
                     imputed: str = "imputed", stat: str = "sd") -> dict[str, Any] | None:
    """The interval, on the unit that varies independently: the season.

    Rookies inside one season share a board, a curve fitted on that year's veterans and one
    realisation of the year, so the rows are not independent observations of the residual
    and an interval over them reads too narrow -- the error `docs/gate-power.md` names.
    The pooled spread (`stat`: the sd, or the RMS) is measured once per season, and the
    estimate is the mean of those `k` readings with `se = sd / sqrt(k)` under a t on `k - 1`
    degrees of freedom, the form the gates use. `t_vs_shipped` is the distance from the shipped
    constant on that unit and `clears` says whether it passes the same two-sided 95% bar; a
    season with fewer than two rookies has no spread and is not a cluster, for either
    statistic, so the two read over the same seasons. None with fewer than two clusters.
    """
    from hub.models.experiment import t_quantile

    per: dict[int, dict[str, Any]] = {}
    for season in sorted(rows["season"].unique().to_list()):
        sub = rows.filter(pl.col("season") == season)
        r = _ratio(sub, against, imputed)
        if r.size >= 2:
            per[int(season)] = {"cv": _spread(r, stat), "n": int(r.size)}
    k = len(per)
    if k < 2:
        return None
    vals = np.array([v["cv"] for v in per.values()])
    mean, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(k))
    crit = t_quantile(0.975, k - 1)
    t = (mean - shipped) / se if se > 0 else float("inf")
    return {"k": k, "per_season": per, "mean": mean, "se": se, "t_crit": crit,
            "lo": mean - crit * se, "hi": mean + crit * se,
            "t_vs_shipped": t, "clears": abs(t) >= crit}


def basis_table(rows: pl.DataFrame, imputed: str, *, shipped: float) -> dict[str, Any]:
    """One basis of the 2x2, against the season's own xFP per game: the residual by position and
    pooled (sd, RMS, bias, median), and the season-clustered sd and RMS pooled and by position."""
    def clustered(sub: pl.DataFrame, stat: str) -> dict[str, Any] | None:
        return season_clustered(sub, "xfp_pg", shipped=shipped, imputed=imputed, stat=stat)

    return {
        "by_position": residual_cv(rows, "xfp_pg", imputed),
        "clustered_sd": clustered(rows, "sd"),
        "clustered_rms": clustered(rows, "rms"),
        "position_rms": {p: clustered(rows.filter(pl.col("pos") == p), "rms")
                         for p in DRAFTED_POSITIONS},
    }


def drift_component(rows: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """The year passing, as the curve sees it: `imputed_lfl / imputed - 1` on the same rookies,
    the distance between the rank's value on this season's curve and on last season's. By
    position and pooled: n, sd, RMS, mean and median. Not a residual against an outcome -- the
    realised value cancels out of it -- so it is the drift with no interpolation error in it."""
    out: dict[str, dict[str, Any]] = {}
    for pos in (*DRAFTED_POSITIONS, "pooled"):
        sub = rows if pos == "pooled" else rows.filter(pl.col("pos") == pos)
        d = sub.select((pl.col("imputed_lfl") / pl.col("imputed") - 1.0).alias("d"))["d"].to_numpy()
        out[pos] = {"n": int(d.size),
                    "sd": _spread(d, "sd") if d.size >= 2 else None,
                    "rms": _spread(d, "rms") if d.size >= 2 else None,
                    "mean": float(np.mean(d)) if d.size else None,
                    "median": float(np.median(d)) if d.size else None}
    return out


def level_shift_check(rows: pl.DataFrame) -> dict[str, float]:
    """Why the like-for-like sd can sit *above* the drift-carrying one: a relative error divides
    by the imputed value.

    `1 + r_drift = (1 + r_like) * q` row by row, with `q = imputed_lfl / imputed`, an identity.
    If the drift were only a uniform level shift (`q` the same for every rookie) the drift-
    carrying sd would be `q * sd(1 + r_like)` exactly: a curve that sits above the outcome by
    11% shrinks every *relative* error by 11%, sd included. So "removing drift can only lower
    the number" holds for an additive drift and not for a multiplicative one, and the tripwire
    (#322) has to be read with this beside it. Returns the mean `q`, the like-for-like sd, that
    sd scaled by the mean `q` (what a pure level shift would leave) and the drift-carrying sd
    actually observed, over the rookies pooled.
    """
    q = (rows["imputed_lfl"] / rows["imputed"]).to_numpy()
    like = (rows["xfp_pg"] / rows["imputed_lfl"]).to_numpy()
    drift = (rows["xfp_pg"] / rows["imputed"]).to_numpy()
    return {"mean_q": float(np.mean(q)), "sd_like_for_like": float(np.std(like, ddof=1)),
            "sd_if_level_shift_only": float(np.mean(q) * np.std(like, ddof=1)),
            "sd_drift_carrying": float(np.std(drift, ddof=1))}


def remainder(drift_basis_rms: float, like_for_like_rms: float) -> float:
    """`sqrt(max(0, a^2 - b^2))`: the RMS the drift adds to the like-for-like error if the two
    were independent. An approximation and labelled one: they are not independent (both are
    functions of the same rank and the same player-season)."""
    return float(np.sqrt(max(0.0, drift_basis_rms ** 2 - like_for_like_rms ** 2)))


def player_bootstrap_se(rows: pl.DataFrame, against: str, *, draws: int = 2000,
                        seed: int = 0) -> float | None:
    """Standard error of the pooled CV over players -- the unit the shipped number quoted
    its se on (0.010 over players), kept as a labelled secondary so the two read side by
    side. It is not the interval: the season is the cluster (`season_clustered`)."""
    r = rows.select((pl.col(against) / pl.col("imputed") - 1.0).alias("r"))["r"].to_numpy()
    if r.size < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(draws, r.size))
    return float(np.std(np.std(r[idx], axis=1, ddof=1), ddof=1))


def refit(measured: dict[str, Any], *, min_pos_n: int = MIN_POS_N,
          ) -> tuple[float, dict[str, float], list[str]] | None:
    """`(IMPUTE_CV, IMPUTE_CV_BY_POS, positions_with_their_own)` from one `measure` result, or
    None when it holds no pooled spread (no fit season carried two rookies).

    The pooled value is the season-clustered mean where there are two fit seasons or more and
    the one season's pooled CV where there is one -- the same statistic, the mean of the
    per-season pooled CVs. A position keeps its own CV over its rookies when it has
    `min_pos_n` of them and takes the pooled value otherwise. Values are rounded to the three
    places the shipped table carries.
    """
    xfp = measured["xfp_pg"]
    pooled = (xfp["clustered"]["mean"] if xfp["clustered"] is not None
              else xfp["by_position"]["pooled"]["cv"])
    if pooled is None:
        return None
    pooled = round(float(pooled), 3)
    by_pos: dict[str, float] = {}
    own: list[str] = []
    for pos in DRAFTED_POSITIONS:
        c = xfp["by_position"][pos]
        if c["cv"] is not None and c["n"] >= min_pos_n:
            by_pos[pos] = round(float(c["cv"]), 3)
            own.append(pos)
        else:
            by_pos[pos] = pooled
    return pooled, by_pos, own


def _f(x: float | None, spec: str = ".3f") -> str:
    return f"{x:{spec}}" if x is not None else "n/a"


def like_with_like_lines(result: dict[str, Any], shipped: float) -> list[str]:
    """The #327 report: the 2x2, the median beside the RMS, the drift component, the tripwire."""
    drift, lfl = result["drift_basis"], result["like_for_like"]
    dr = result["drift_component"]
    lines = [f"\n  LIKE WITH LIKE (#327): the same {result['n']} rookies over {result['clusters']} "
             f"seasons, scored against the season's own xFP per game, imputed from last "
             f"season's curve (drift-carrying, what shipped) and from his own season's "
             f"(like-for-like)"]
    cs = {b: (t["clustered_sd"], t["clustered_rms"]) for b, t in (("d", drift), ("l", lfl))}
    pool = {b: t["by_position"]["pooled"] for b, t in (("d", drift), ("l", lfl))}
    lines.append("\n  the 2x2, pooled; season-clustered mean of the per-season values "
                 "[over all rookies]")
    lines.append(f"  {'':>22} {'drift-carrying':>22} {'like-for-like':>22}")
    for label, idx, key in (("sd  (shipped statistic)", 0, "cv"), ("RMS relative error", 1, "rms")):
        cells = []
        for b in ("d", "l"):
            c = cs[b][idx]
            cells.append(f"{_f(c['mean'] if c else None)} [{_f(pool[b][key])}]")
        lines.append(f"  {label:>22} {cells[0]:>22} {cells[1]:>22}")
    lines.append(f"  {'median residual':>22} {_f(pool['d']['median'], '+.3f'):>22} "
                 f"{_f(pool['l']['median'], '+.3f'):>22}")
    lines.append(f"  {'mean residual (bias)':>22} {_f(pool['d']['mean'], '+.3f'):>22} "
                 f"{_f(pool['l']['mean'], '+.3f'):>22}")
    for b, name in (("d", "drift-carrying"), ("l", "like-for-like")):
        for idx, stat in ((0, "sd"), (1, "RMS")):
            c = cs[b][idx]
            if c is None:
                continue
            seasons_s = ", ".join(f"{yr} {v['cv']:.3f} (n {v['n']})"
                                  for yr, v in c["per_season"].items())
            lines.append(f"    {name} {stat}: 95% [{c['lo']:.3f}, {c['hi']:.3f}] (k = {c['k']}, "
                         f"se {c['se']:.3f}); {c['t_vs_shipped']:+.1f} t from the shipped "
                         f"{shipped:.3f}, which {'clears' if c['clears'] else 'does not clear'} "
                         f"t(0.975, {c['k'] - 1}) = {c['t_crit']:.2f}; per season: {seasons_s}")
    lines.append("\n  by position, like-for-like (the drift-carrying value in brackets)")
    lines.append(f"  {'pos':>6} {'n':>4} {'sd':>14} {'RMS':>14} {'median':>8} {'bias':>8}  "
                 f"clustered RMS (own interval)")
    for pos in (*DRAFTED_POSITIONS, "pooled"):
        a, b = lfl["by_position"][pos], drift["by_position"][pos]
        own = (lfl["clustered_rms"] if pos == "pooled" else lfl["position_rms"][pos])
        own_s = (f"{own['mean']:.3f} [{own['lo']:.3f}, {own['hi']:.3f}] k = {own['k']}"
                 if own else "n/a (fewer than two seasons with two rookies)")
        lines.append(f"  {pos:>6} {a['n']:>4} "
                     f"{_f(a['cv']) + ' [' + _f(b['cv']) + ']':>14} "
                     f"{_f(a['rms']) + ' [' + _f(b['rms']) + ']':>14} "
                     f"{_f(a['median'], '+.2f'):>8} {_f(a['mean'], '+.2f'):>8}  {own_s}")
    p = dr["pooled"]
    lines.append(f"\n  the drift component, pooled: imputed(this season's curve) / imputed(last "
                 f"season's) - 1 at the same rank: sd {_f(p['sd'])}, RMS {_f(p['rms'])}, "
                 f"mean {_f(p['mean'], '+.3f')}, median {_f(p['median'], '+.3f')}")
    lines.append("  " + ", ".join(f"{pos} sd {_f(dr[pos]['sd'])} mean {_f(dr[pos]['mean'], '+.2f')}"
                                  for pos in DRAFTED_POSITIONS))
    rem = result["drift_remainder"]
    lines.append(f"  the RMS the drift adds on top of the like-for-like error, as "
                 f"sqrt(RMS_drift-carrying^2 - RMS_like-for-like^2) (an approximation, the two "
                 f"are not independent): {_f(rem['clustered'])} season-clustered, "
                 f"{_f(rem['rookies'])} over rookies")
    sd_l = cs["l"][0]["mean"] if cs["l"][0] else None
    rms_l = cs["l"][1]["mean"] if cs["l"][1] else None
    if sd_l is not None:
        verdict = ("ABOVE the shipped value: a bug until shown otherwise (method.md rule 9)"
                   if sd_l > shipped else "not above the shipped value, as removing drift should")
        lines.append(f"\n  tripwire (#322): like-for-like sd {sd_l:.3f} against the shipped "
                     f"{shipped:.3f}: {verdict}")
    ls = result["level_shift"]
    if ls is not None:
        lines.append(f"  read with the identity 1 + r_drift = (1 + r_like) * q, q = this "
                     f"season's curve over last season's at the rank: mean q "
                     f"{ls['mean_q']:.3f}, so a level shift alone would leave the sd at "
                     f"{ls['sd_if_level_shift_only']:.3f} "
                     f"(= {ls['mean_q']:.3f} x like-for-like {ls['sd_like_for_like']:.3f}); the "
                     f"drift-carrying sd over rookies is {ls['sd_drift_carrying']:.3f}")
    if rms_l is not None:
        lines.append(f"  adoption reads the like-for-like RMS {rms_l:.3f} (median residual "
                     f"{_f(pool['l']['median'], '+.3f')} beside it); the constant is not changed "
                     f"by this run")
    return lines


def measure(seasons: Sequence[int], *, exclude: int | None = None, min_games: int = MIN_GAMES,
            build_board=None, load_stats=None, load_realised=None) -> tuple[dict, list[str]]:
    """The measurement over `seasons`, and the lines a script prints. The three loaders
    are seams for a test; the defaults read the archive through the repo's own readers.

    The default board loader keeps the `BuildReport` `board_as_of` returns and refuses a board
    that did not build every stage (`require_built`); a seam handed in is the test's own."""
    from hub.draft.board import board_as_of, expected_points
    from hub.fetch import nflverse
    from hub.models import predict

    if build_board is None:
        def build_board(yr):
            board, report = board_as_of(yr)
            require_built(yr, report)
            return board
    load_realised = load_realised or expected_points
    if load_stats is None:
        def load_stats():
            span = list(range(FIRST_STATS_SEASON, max(seasons) + 1))
            return nflverse.load("player_stats", span, cols=list(nflverse.PLAYER_STATS_COLS))
    stats = load_stats()
    lines: list[str] = []
    frames, tallies = [], {}
    for yr in seasons:
        rows, counts = rookie_rows(build_board(yr), stats, load_realised(yr),
                                   season=yr, min_games=min_games)
        frames.append(rows)
        tallies[yr] = counts
        lines.append(f"  {yr}: top {TOP} {counts['top200']}, rookies {counts['rookies']}, "
                     f"imputed {counts['imputed']} (collisions {counts['collisions']}, "
                     f"imputed with no line in any season {counts['no_line']}, "
                     f"ambiguous name {counts['ambiguous']}, "
                     f"debut delayed to a later season {counts['delayed']}), "
                     f"with a season line {counts['with_line']}, "
                     f">= {min_games} games {counts['played']}")
    rows = pl.concat(frames)
    kept, said = hold_out(rows, exclude)
    lines.append(said)
    result: dict[str, Any] = {"seasons": list(seasons), "exclude": exclude,
                              "n": kept.height, "clusters": kept["season"].n_unique(),
                              "tallies": tallies, "min_games": min_games}
    for against, label in (("ppg", "realised PPR points per game"),
                           ("xfp_pg", "the season's own xFP per game")):
        cv = residual_cv(kept, against)
        se = player_bootstrap_se(kept, against)
        clustered = season_clustered(kept, against, shipped=predict.IMPUTE_CV)
        result[against] = {"by_position": cv, "se": se, "clustered": clustered}
        lines.append(f"\n  residual CV against {label}: n = {kept.height} rookies over "
                     f"{result['clusters']} seasons (the clusters)")
        lines.append(f"  {'pos':>6} {'n':>4} {'rookie cv':>10} {'median':>8} {'shipped':>8}")
        for pos in (*DRAFTED_POSITIONS, "pooled"):
            c = cv[pos]
            shipped = (predict.IMPUTE_CV if pos == "pooled"
                       else predict.IMPUTE_CV_BY_POS.get(pos, predict.IMPUTE_CV))
            cv_s = f"{c['cv']:.3f}" if c["cv"] is not None else "n/a"
            md_s = f"{c['median']:+.2f}" if c["median"] is not None else "n/a"
            lines.append(f"  {pos:>6} {c['n']:>4} {cv_s:>10} {md_s:>8} {shipped:>8.3f}")
        if clustered is not None:
            c = clustered
            seasons_s = ", ".join(f"{yr} {v['cv']:.3f} (n {v['n']})"
                                  for yr, v in c["per_season"].items())
            lines.append(f"  the interval, clustered on the season (k = {c['k']}, t on "
                         f"{c['k'] - 1} df): mean of the per-season pooled CVs {c['mean']:.3f}, "
                         f"se {c['se']:.3f}, 95% [{c['lo']:.3f}, {c['hi']:.3f}]; "
                         f"{c['t_vs_shipped']:+.1f} t from the shipped "
                         f"{predict.IMPUTE_CV:.3f}, which "
                         f"{'clears' if c['clears'] else 'does not clear'} t(0.975, "
                         f"{c['k'] - 1}) = {c['t_crit']:.2f}")
            lines.append(f"    per season: {seasons_s}")
        else:
            lines.append("  no season-clustered interval: fewer than two seasons with a spread")
        if se is not None:
            lines.append(f"  secondary, not the interval: pooled se over players {se:.3f} "
                         f"(the unit the shipped number quoted; rows within a season are not "
                         f"independent)")
    result["drift_basis"] = basis_table(kept, "imputed", shipped=predict.IMPUTE_CV)
    result["like_for_like"] = basis_table(kept, "imputed_lfl", shipped=predict.IMPUTE_CV)
    result["drift_component"] = drift_component(kept)
    result["level_shift"] = level_shift_check(kept) if kept.height >= 2 else None
    d_cl, l_cl = result["drift_basis"]["clustered_rms"], result["like_for_like"]["clustered_rms"]
    result["drift_remainder"] = {
        "clustered": (remainder(d_cl["mean"], l_cl["mean"]) if d_cl and l_cl else None),
        "rookies": remainder(result["drift_basis"]["by_position"]["pooled"]["rms"] or 0.0,
                             result["like_for_like"]["by_position"]["pooled"]["rms"] or 0.0)}
    lines.extend(like_with_like_lines(result, predict.IMPUTE_CV))
    return result, lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts/fit_impute_cv.py",
        description="Measure IMPUTE_CV on rookies -- the players the board imputes (#277).")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    ap.add_argument("--min-games", type=int, default=MIN_GAMES)
    ap.add_argument("--point-in-time", action="store_true",
                    help="with --exclude-season N: fit on the seasons strictly before N "
                         "(rule 2) instead of every season but N; --record needs it (#320)")
    holdout.add_arguments(ap)
    a = ap.parse_args(argv)
    from hub.models import predict

    if a.point_in_time and a.exclude_season is None:
        raise SystemExit("--point-in-time needs --exclude-season N: the season fitted for")
    if a.record and not a.point_in_time:
        raise SystemExit("--record needs --point-in-time: a set records the refit on the "
                         "seasons strictly before the excluded one (#320)")
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    note = holdout.recording(a, f"uv run python scripts/fit_impute_cv.py "
                                f"--exclude-season {a.exclude_season} --point-in-time --record")
    if a.point_in_time:
        seasons = [s for s in seasons if s < a.exclude_season]
        if not seasons:
            return _not_refitted(note, f"no board before {a.exclude_season} in --seasons to "
                                       f"fit on")
    try:
        result, lines = measure(seasons, exclude=None if a.point_in_time else a.exclude_season,
                                min_games=a.min_games)
    except Exception as e:
        return unavailable("scripts/fit_impute_cv.py", "the boards and seasons of the archive", e)
    print("\n".join(lines))
    print(f"\n  the shipped IMPUTE_CV {predict.IMPUTE_CV:.3f} is this measurement on all five "
          f"seasons "
          f"(#298, 2026-09-16); a run prints beside it and does not rewrite it")
    if not a.point_in_time:
        return 0
    # Recorded as a refit (#320): the pre-#320 "not refitted: fewer clusters than the decision
    # was taken on" disqualified every hold-out fit by construction, since one season out is
    # always one cluster fewer than the full fit.
    fitted_on = f"seasons {result['seasons']} (strictly before {a.exclude_season})"
    got = refit(result)
    if got is None:
        return _not_refitted(note, f"too few rookies to measure on {fitted_on}")
    pooled, by_pos, own = got
    split = ", ".join(f"{p} {v:.3f}" + ("" if p in own else f" (pooled: < {MIN_POS_N} rookies)")
                      for p, v in by_pos.items())
    print(f"\n  refit for {a.exclude_season}, on {fitted_on}, {result['clusters']} "
          f"cluster(s), n = {result['n']}: IMPUTE_CV {pooled:.3f}; by position {split}")
    note("predict.IMPUTE_CV", pooled)
    note("predict.IMPUTE_CV_BY_POS", by_pos)
    return 0


def _not_refitted(note, reason: str) -> int:
    """Record both keys as not refitted, with the reason, when the fit has nothing to fit."""
    why = f"IMPUTE_CV is not refitted: {reason}"
    for key in ("predict.IMPUTE_CV", "predict.IMPUTE_CV_BY_POS"):
        note(key, None, why_not=why)
    print(f"  {why}")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
