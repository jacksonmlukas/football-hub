"""Full-PPR draft board for a 12-team ESPN redraft league.

Three signals, joined:
  1. FantasyPros ECR      -- consensus rank, with ecr_sd/best/worst as a free uncertainty band
  2. ffopportunity xFP    -- expected fantasy points, separating opportunity from efficiency
  3. ESPN ADP             -- what YOUR room will actually do (optional; needs cookies)

The output column that matters is `edge`: consensus rank minus ESPN ADP. Positive means
your leaguemates, drafting off ESPN's default board, will let this player fall.

Runs without cookies (ECR + xFP only). Add ESPN creds for the ADP diff.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import nflreadpy as nfl
import numpy as np
import polars as pl

from hub import jsonio, store
from hub.config import (
    DRAFTED_POSITIONS,
    SEASON_AHEAD,
    SEASON_COMPLETED,
    RosterConfig,
    flex_positions,
    flex_share,
    starters,
)
from hub.contracts import DRAFT_BOARD, ContractViolation
from hub.draft import adp_history, durability
from hub.draft import regression as td_regression
from hub.draft import report as report_mod
from hub.draft import state as state_mod
from hub.draft.availability import DEFAULT_ESPN_WEIGHT, pick_value
from hub.draft.picks import MY_SLOT, TEAMS, draft_mode, my_picks, next_two
from hub.draft.playoff_sos import attach_sos, playoff_sos
from hub.draft.state import DraftState, remaining
from hub.models import components
from hub.models.predict import blend
from hub.names import player_key
from hub.paths import BOARD_JSON, BOARD_PARQUET, ROOT

if TYPE_CHECKING:                      # `optimize` imports from here, so runtime would cycle
    pass

# From the leaf, not declared here. `adp_history` and `adherence` used to import these
# from this module -- a 780-line board builder -- purely to learn a Path.
# improvements.md #17.
OUT = ROOT / "site" / "data"

# A `PPR` scoring table used to sit here, keyed `pass_yd`/`rec_td`, with nothing reading it.
# `hub.models.components.SCORING` is the live one, keyed by nflverse's own column names
# (`passing_yards`, `receiving_tds`) and checked against the league by `scoring_mismatch`.
# Two vocabularies for one league's scoring, one of them unread, is how the wrong one gets
# picked up later.

# Starters per team. THREE WR slots plus a flex -- confirmed for this league, not the ESPN
# default. Derived from `hub.config.RosterConfig`, which is the one declaration of the
# league's shape.
#
# This used to say "Overridden by league_settings() when cookies are present", which was not
# true: `espn.league_settings()` does return the league's own roster slots, but all four
# callers spell it `lg, _ = league_settings()` and throw them away. `roster_mismatch` below
# is what actually checks them now, the same way `scoring_mismatch` checks the scoring.
ROSTER = RosterConfig()
SLOTS = starters(ROSTER)
FLEX_ELIGIBLE = flex_positions(ROSTER)


def expected_points(season: int = SEASON_COMPLETED) -> pl.DataFrame:
    """Season-total expected vs actual PPR points, and the components behind the expected half.

    The gap between the two totals is the regression signal. The components are new: this step
    used to keep two columns of the twenty-three `ff_opportunity` publishes and drop the rest,
    so a disagreement with any other projection was a scalar and could not be attributed to a
    stat. They are carried under an `exp_` prefix and are **per game**, matching `xfp_per_game`.

    **`xfp_per_game` is unchanged and deliberately not rebuilt from them.** The component sum
    reproduces the published total to about 0.017 points a player-week, which would move
    `proj_blend` by half that -- immaterial as a projection, and `docs/improvements.md` #18
    records that a Board change of that order reorders simulated drafts and moved the frozen
    weekly gate from +0.711 to +0.215. So the parts ride alongside the total rather than
    replacing it; `xfp_components_per_game` carries the rebuild for comparison.

    The realised side stays here rather than moving to the shared aggregation, because `fp` and
    `rec_act` are what happened, not what was expected, and that function is about expected
    stats. Only the expected half goes through it.
    """
    o = nfl.load_ff_opportunity(seasons=[season], stat_type="weekly")
    agg = o.group_by(["player_id", "full_name", "position"]).agg([
        pl.col("receptions_exp").sum().alias("rec_exp"),
        pl.col("receptions").sum().alias("rec_act"),
        pl.col("total_fantasy_points_exp").sum().alias("xfp"),
        pl.col("total_fantasy_points").sum().alias("fp"),
        pl.len().alias("games"),
    ])
    comps = components.from_opportunity(o, by="player_id")
    comps = comps.rename({c: f"exp_{c}" for c in components.EXPECTED if c in comps.columns}
                         | {"xfp_per_game": "xfp_components_per_game"}).drop("games")
    return (agg.join(comps, on="player_id", how="left")
               .with_columns([
                   (pl.col("fp") - pl.col("xfp")).alias("fp_over_expected"),
                   (pl.col("xfp") / pl.col("games")).alias("xfp_per_game"),
               ]))


# `load_ff_rankings("draft")` stacks 31 FantasyPros pages into one frame -- redraft,
# dynasty, best-ball, superflex, IDP -- each with its OWN ecr scale. Picking a player's
# row without filtering draws from 27 of them, which is how kickers ended up with ECRs of
# 11 and topped the edge list. This page is /nfl/rankings/ppr-cheatsheets.php: redraft,
# full PPR, no superflex, no IDP. It is the only one that matches this league.
CONSENSUS_PAGE = "redraft-overall"

# The season being drafted for. xFP describes the season just gone; the simulation has to
# be about the one ahead, or a strategy ranked on last year's numbers scores against last
# year's truth and reads as clairvoyant.


# On the board because you draft them off it. K and DST are rostered but taken late off
# ESPN's own list, and their presence distorts every rank below the skill players.


def _select_consensus(r: pl.DataFrame) -> pl.DataFrame:
    """Reduce the multi-page rankings frame to this league's single comparable board."""
    if "page_type" not in r.columns:
        raise ContractViolation(
            "ff_rankings: no `page_type` column; cannot tell which ranking page each row "
            "came from, and blending pages silently produces a mongrel ECR scale")
    # FantasyPros calls the consensus rank's spread `sd`. It is renamed on the way in
    # because `sd` is also what `hub.models.predict.moments` calls a player's *weekly points*
    # spread, and both land on frames derived from this board -- one measured in picks, one
    # in points, three times apart in magnitude and neither obviously wrong on inspection.
    # `_sigma` in `hub.draft.availability` reads this one and would have silently read the
    # other the first time anyone ran an availability sim on a scored frame.
    keep = [c for c in ("player", "pos", "team", "ecr", "sd", "best", "worst") if c in r.columns]
    out = (r.filter(pl.col("page_type") == CONSENSUS_PAGE)
            .select(keep)
            .rename({"sd": "ecr_sd"} if "sd" in keep else {})
            .filter(pl.col("ecr").is_not_null())
            .filter(pl.col("pos").is_in(DRAFTED_POSITIONS))
            .unique(subset=["player"], keep="first")
            .sort("ecr"))
    if out.is_empty():
        raise ContractViolation(
            f"ff_rankings: page `{CONSENSUS_PAGE}` returned no {'/'.join(DRAFTED_POSITIONS)} "
            f"rows; FantasyPros likely renamed the page")
    return out


def consensus(as_of: str | None = None) -> pl.DataFrame:
    """FantasyPros ECR. `ecr_sd` is a crude prior on uncertainty until conformal lands.

    `as_of` is an ISO date. Given one, this returns the board as it stood *before* that date
    -- the latest scrape per player -- rather than today's. That is what makes a historical
    replay honest: the room in 2022 could only see what had been published by then, and
    scoring a 2022 draft against 2026 rankings would be hindsight wearing a backtest's
    clothes.

    The current path is untouched and still reads the small `draft` table. `all` is 1.8M rows
    across 2020-10-16 onward, which is fine once per backtest and wrong on draft night.
    """
    if as_of is None:
        return _select_consensus(nfl.load_ff_rankings("draft"))
    allr = nfl.load_ff_rankings("all")
    snap = (allr.filter((pl.col("page_type") == CONSENSUS_PAGE)
                        # scrape_date is an ISO string, which sorts correctly as text;
                        # casting it just to compare would be waste. Same technique as
                        # `hub.draft.availability.historical_picks`.
                        & (pl.col("scrape_date") < as_of)
                        & (pl.col("ecr").is_not_null()))
                .sort("scrape_date", descending=True)
                .unique(subset=["player"], keep="first"))
    if snap.is_empty():
        raise ContractViolation(
            f"ff_rankings: no `{CONSENSUS_PAGE}` rows scraped before {as_of}; the archive "
            f"starts 2020-10-16, so a season before 2021 cannot be replayed")
    return _select_consensus(snap)


# A per-game rate needs a real denominator. Without this, a player with one big game
# outranks every genuine starter and occupies a slot that defines replacement level --
# which is how WR replacement came out ABOVE RB (11.14 vs 11.10), contradicting the
# three-WR effect. At >= 10 games WR drops to 10.02 and the effect reappears; the sign
# is stable from 8 games up, so it is not fitted to the threshold.
MIN_GAMES = 10


def replacement_levels(position: pl.Series, points: pl.Series, games: pl.Series,
                       teams: int = 12,
                       min_games: int = MIN_GAMES) -> dict[str, float]:
    """Replacement = the points of the last startable player at each position.

    In full PPR with a 12-team flex, the flex is almost always a WR, which pushes WR
    replacement meaningfully deeper than a naive teams*slots calculation suggests.

    Only players with at least `min_games` are eligible to set the level: replacement is
    a statement about the quality freely available all season, and a one-game sample says
    nothing about that. A null `games` is therefore not eligible either, which is what
    keeps a rookie with no prior season from setting a level he has no evidence for -- he
    is still *priced* against it, he just does not define it.

    Three Series rather than a frame. The signature used to be `(df, teams, min_games)`, but
    the real interface was "a frame with columns named exactly `position`, `games` and
    `xfp_per_game`" -- so `build()` renamed `pos` to `position_` and `proj_blend` to
    `xfp_per_game_`, then aliased both back, to satisfy a function forty lines above it in
    the same file. Naming the three inputs is the whole job.
    """
    share = flex_share(ROSTER)
    df = pl.DataFrame({"position": position, "points": points, "games": games})
    levels = {}
    for pos in DRAFTED_POSITIONS:
        n = teams * SLOTS.get(pos, 0)
        n += round(teams * SLOTS.get("FLEX", 0) * share.get(pos, 0)) if pos in FLEX_ELIGIBLE else 0
        pool = (df.filter((pl.col("position") == pos) & (pl.col("games") >= min_games))
                  .sort("points", descending=True))
        levels[pos] = float(pool["points"][min(n, pool.height) - 1]) if pool.height else 0.0
    return levels


def espn_adp(league_size: int = 12, season: int = SEASON_AHEAD) -> pl.DataFrame | None:
    """ESPN's own ADP -- the thing your room is actually drafting off. Optional.

    Was built on espn_api's `posRank`, which is a *positional* rank (WR5 -> 5) parsed from
    a `positionalRanking` key that free-agent payloads omit entirely. It therefore returned
    empty lists, and subtracting a positional rank from an overall consensus rank would not
    have meant anything even if it had not. Real overall ADP now comes from the fetch layer.
    """
    try:
        from hub.fetch.espn import player_market
        adp = player_market(season=season)
    except Exception as e:
        print(f"  ESPN ADP unavailable ({type(e).__name__}); running ECR-only mode.")
        return None

    # Shape is not substance. The predecessor of this function returned a frame of 400 rows
    # whose every value was an empty list: not null, so it passed every null check, and
    # `edge` shipped as a column of empty lists with no crash and no warning. Keep asserting
    # on dtype and height here even though the fetch layer is now typed, because the failure
    # this catches is ESPN quietly changing a field, which is not a hypothetical.
    adp = adp.filter(pl.col("adp").is_not_null())
    if adp.is_empty() or not adp["adp"].dtype.is_numeric():
        print(f"  ESPN ADP present but unusable "
              f"(dtype={adp['adp'].dtype}, rows={adp.height}); running ECR-only mode.")
        return None
    return adp


def _adp_saturation_cutoff(adp: pl.Series, teams: int) -> float | None:
    """The ADP value at which ESPN stops actually knowing anything.

    ESPN does not leave ADP null for players nobody drafts -- it parks them in a jittered
    band at the tail. Observed Aug 2026: bins 160-167 hold 1-10 players each, then 168
    holds 16 and 169 holds 191, almost all distinct values. Counting exact repeats finds
    the spike (169.97 x29) but lets the shoulder through, which prices a dozen deep TEs
    on sentinel ADPs and invents a large negative edge for each.

    So bin to whole picks. Genuine ADP averages roughly one player per pick, so the first
    bin holding a full league's worth of players is where ESPN stopped knowing anything.
    Derived from the data rather than hardcoded, so it self-tunes if ESPN moves the band
    or the league changes size.
    """
    if adp.is_empty():
        return None
    bins = (adp.to_frame()
               .with_columns(pl.col(adp.name).floor().alias("_bin"))
               .group_by("_bin").len().sort("_bin"))
    hits = bins.filter(pl.col("len") >= teams)
    return float(hits["_bin"][0]) if hits.height else None


def _attach_edge(board: pl.DataFrame, adp: pl.DataFrame, teams: int) -> pl.DataFrame:
    """Join ADP and compute edge on a scale where the subtraction means something.

    Both operands have to be pick numbers over the SAME population. `consensus_rank`
    spans every row on the board (~1474) while ESPN ADP covers only the draftable pool
    (~460, saturating near 170); subtracting those made edge a disguised restatement of
    -consensus_rank, which sorts the board almost exactly backwards from the intent.

    So: drop the saturated tail, then rank consensus *within* the surviving pool. Positive
    edge then means what the docstring has always claimed -- your room lets him fall.
    """
    board = board.join(adp, on="player", how="left")
    cut = _adp_saturation_cutoff(board["adp"].drop_nulls(), teams)
    if cut is not None:
        board = board.with_columns(
            pl.when(pl.col("adp") >= cut).then(None).otherwise(pl.col("adp")).alias("adp"))
    pool = (board.filter(pl.col("adp").is_not_null())
                 .select(["player", "ecr"])
                 .with_columns(pl.col("ecr").rank(method="min").cast(pl.Int64)
                                 .alias("consensus_pick"))
                 .select(["player", "consensus_pick"]))
    return (board.join(pool, on="player", how="left")
                 .with_columns((pl.col("adp") - pl.col("consensus_pick")).alias("edge")))


def _join_expected_points(ecr: pl.DataFrame, xp: pl.DataFrame) -> pl.DataFrame:
    """Attach expected points to the consensus board, matching on normalised names.

    FantasyPros and ffopportunity disagree about suffixes and punctuation, and an exact
    join silently drops the disagreements: 20 of the top 168 by ADP arrived with no xFP,
    James Cook III and Michael Pittman Jr. among them. A null xFP is not a small gap --
    it nulls VOR, so every VOR-ranked view skips the player, and the season simulation
    scores him zero, turning a first-round RB into an empty roster slot.
    """
    norm = pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("_key")
    right = (xp.with_columns(pl.col("full_name").map_elements(player_key, return_dtype=pl.Utf8)
                               .alias("_key"))
               .unique(subset=["_key"], keep="first")
               # `position` duplicates `pos`, which the board already has from FantasyPros.
               # It is dropped rather than carried because it is the *worse* of the two: it
               # arrives through this left join, so it is null for exactly the players the
               # join missed, while `pos` is always populated. Two columns for one concept,
               # one of them silently sparser, is a trap for whoever reaches for the
               # familiar-looking name.
               .drop("full_name", "position"))
    return ecr.with_columns(norm).join(right, on="_key", how="left").drop("_key")


def _impute_xfp(board: pl.DataFrame) -> pl.DataFrame:
    """Fill missing expected points from consensus rank, within position.

    Rookies have no prior-season xFP, but the market drafts them inside the top 168, so
    the market plainly expects production. Leaving them null told every downstream model
    they were worth zero: VOR skipped them and the season simulation treated a
    second-round rookie RB as an empty roster slot, which is what put P(win) at 85%.

    Consensus rank is the signal we have for what a player is expected to do. Interpolate
    within position, because a TE and a WR at the same rank are not the same asset, and
    smooth first so a single outlier does not set a rookie's projection.
    """
    if board["xfp_per_game"].is_null().sum() == 0:
        return board
    filled = board["xfp_per_game"].to_list()
    ecr_all = board["ecr"].to_list()
    for pos in board["pos"].unique().to_list():
        rows = [i for i, p in enumerate(board["pos"].to_list()) if p == pos]
        known = [(ecr_all[i], filled[i]) for i in rows if filled[i] is not None]
        gaps = [i for i in rows if filled[i] is None]
        if not gaps:
            continue
        if not known:
            for i in gaps:
                filled[i] = 0.0
            continue
        known.sort()
        xs = np.array([k[0] for k in known], dtype=float)
        ys = np.array([k[1] for k in known], dtype=float)
        # Rolling median, then enforce monotone decline: a worse rank must never impute
        # a better projection, which raw interpolation on noisy data will happily do.
        win = max(3, min(15, len(ys) // 4 * 2 + 1))
        sm = np.array([np.median(ys[max(0, j - win // 2): j + win // 2 + 1])
                       for j in range(len(ys))])
        sm = np.minimum.accumulate(sm)
        for i in gaps:
            filled[i] = float(np.interp(ecr_all[i], xs, sm))
    return board.with_columns(pl.Series("xfp_per_game", filled, dtype=pl.Float64))




def board_age_hours(path: Path, now: float) -> float:
    """How old the board file is, in hours. Separate so it can be tested without a clock."""
    return max(now - path.stat().st_mtime, 0.0) / 3600.0


def last_good(path: Path | None = None,
              now: float | None = None) -> tuple[pl.DataFrame, float]:
    """The board as last written to disk, with its age in hours.

    Two callers make this a real seam rather than a hypothetical one: the poller, which has
    always read the board rather than building it, and `main`'s offline fallback below.
    """
    import time
    path = path if path is not None else BOARD_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"no board at {path}. Run `make draft` first -- the poller reads the board, "
            f"it does not build one.")
    return pl.read_parquet(path), board_age_hours(
        path, time.time() if now is None else now)


def published(site: Path | None = None) -> pl.DataFrame:
    """The Board as last published to the site: the top 300 by consensus, as committed JSON.

    Narrower than the parquet and deliberately so -- it is the draft-night fallback
    `docs/draft-night.md` names, sized to cover all 192 picks. For an in-season roster that is
    ample; a player outside consensus 300 comes through unprojected, which `build` already
    renders as "carries no projection" rather than as a zero.
    """
    path = (site / "draft_board.json") if site else BOARD_JSON
    if not path.exists():
        raise FileNotFoundError(f"no published board at {path}")
    return pl.DataFrame(json.loads(path.read_text()))


def readable(path: Path | None = None,
             site: Path | None = None) -> tuple[pl.DataFrame, str]:
    """The best Board available here, and which one it was.

    The parquet where it exists, because it is wider. The published artifact otherwise --
    which is what a scheduled run has, since `data/processed/` is gitignored as redistributed
    third-party data and a runner therefore never holds a parquet. Both scheduled runs on
    2026-09-04 reported the roster panel stale for exactly that reason.

    Nothing new is published to make this work: the board artifact has been committed and
    public since the repo flipped. What changes is that a reader can reach it.

    The source is returned rather than logged, so a caller can say which board its numbers
    came from instead of leaving a reader to assume the richer one.
    """
    try:
        board, _age = last_good(path)
        return board, "local parquet"
    except FileNotFoundError:
        pass
    try:
        return published(site), "published artifact (top 300 by consensus)"
    except FileNotFoundError:
        # Both places, and the action -- a message naming only the artifact would point a
        # reader at the site when what they need is to build a board.
        raise FileNotFoundError(
            f"no board here. Looked for a built one at {path or BOARD_PARQUET} and a "
            f"published one at {(site / 'draft_board.json') if site else BOARD_JSON}. "
            f"Run `make draft` to build one, or check out a commit that has the artifact."
        ) from None


def build_or_last_good(league_size: int = 12, season: int = SEASON_COMPLETED, *,
                       path: Path | None = None, now: float | None = None,
                       ) -> tuple[pl.DataFrame, BuildReport, float | None]:
    """Build the board, and if the build cannot happen at all, serve the last good one.

    `build` degrades stage by stage, but only for *advisory* stages. The two spine fetches --
    ffopportunity and the consensus ranks -- have no fallback inside `build`, because a board
    with neither expected points nor consensus ranks is not a board. Until now that meant a
    `ConnectionError` traceback and no recommendation.

    That is the wrong failure on the one night this repo exists for. CLAUDE.md:

        If a fetch fails, serve last-good state from `data/processed/` rather than erroring.
        Systems that need an operator die in October.

    The runbook already says to rebuild the board on the day, so the board on disk at 9pm is
    hours old, not weeks. Hours-old ADP beats a stack trace.

    Returns the age in hours when serving last-good, and `None` when the build succeeded --
    so the caller can say which of the two happened, loudly.

    The exception is deliberately broad. Any failure to build is a failure to build, and
    falling back to a board that is *known* to have been good is safe in a way that guessing
    at which exception types are transient is not.
    """
    try:
        board, report = build(league_size, season)
    except Exception as exc:
        board, age = last_good(path, now)
        print(f"\n  BUILD FAILED: {type(exc).__name__}: {exc}")
        print(f"  serving the last good board instead, built {age:.1f}h ago.")
        print("  ADP is that old. Everything else on it is a season-long number "
              "and does not move.")
        return board, BuildReport(), age
    return board, report, None


@dataclass
class BuildReport:
    """Which optional stages made it into the board.

    `build` degrades on purpose: a board that will not build because one advisory column is
    unavailable is the operator-dependence CLAUDE.md warns about. But the report layer used
    to *infer* what had happened by sniffing for columns -- `"td_luck" not in board.columns`
    at one site, `"adp" not in board.columns` at another, `"injury_status" ... and "missed"`
    at a third. Four independent `except Exception` handlers, four separate guesses at what
    they had done, and no single place that said what the board actually contains.

    Sniffing gets the common case right and the interesting case wrong: a stage that ran and
    returned an all-null column is indistinguishable from one that never ran, and the
    difference is exactly what an operator on the clock needs to know.
    """
    sos: bool = False
    td_luck: bool = False
    durability: bool = False
    adp: bool = False
    scoring_checked: bool = False
    roster_checked: bool = False

    def degraded(self) -> tuple[str, ...]:
        """Stages that did not make it, in declaration order."""
        return tuple(k for k, v in vars(self).items() if not v)


def _check_scoring(board: pl.DataFrame) -> None:
    """Compare the league's own scoring weights against this repo's."""
    from hub.fetch import espn as espn_fetch
    from hub.models.components import scoring_mismatch
    bad = scoring_mismatch(espn_fetch.scoring_settings())
    if bad:
        print("  SCORING MISMATCH -- the league scores differently from hub.models."
              "components.SCORING:")
        for k, (theirs, ours) in sorted(bad.items()):
            print(f"    {k}: league {theirs}, this repo {ours}")
        print("  Every projection below is scored on the wrong weights until that is fixed.")


def _check_roster(board: pl.DataFrame) -> None:
    """Compare the league's own starting slots against `hub.config.RosterConfig`."""
    from hub.config import roster_mismatch
    from hub.fetch import espn as espn_fetch
    bad_slots = roster_mismatch(espn_fetch.league_settings().roster_slots or {})
    if bad_slots:
        print("  ROSTER MISMATCH -- the league starts a different lineup from "
              "hub.config.RosterConfig:")
        for k, (theirs, ours) in sorted(bad_slots.items()):
            print(f"    {k}: league {theirs}, this repo {ours}")
        print("  Replacement level and every VOR below assume this repo's shape.")


def _stage(board: pl.DataFrame, report: BuildReport, flag: str, label: str,
           run: Callable[[pl.DataFrame], pl.DataFrame | None], *, live: bool = True,
           skip_note: str | None = None,
           on_fail: str = "board built without it.") -> pl.DataFrame:
    """Run one optional build stage under the repo's degradation policy.

    A board that will not build because one advisory column is unavailable is the
    operator-dependence CLAUDE.md warns about, so every stage here fails soft: say what
    broke, leave the flag false, carry on.

    This was written out five times -- 48 lines, 31% of `build()` -- and the cost was not the
    duplication. `BuildReport` exists because the *consumers* used to infer what had happened
    by sniffing for columns, and with five producers and no shared shape the consumers' guards
    drifted anyway: on 2026-08-27 one read `durability or adp` and another read `td_luck`, so a
    board built without an ESPN key raised `ColumnNotFoundError` before printing THE PICK.

    `run` returns the new board, or None when the stage only checks something.
    `skip_note` marks a stage ESPN publishes for the current season only: outside a live
    build it is announced and skipped rather than attempted and caught.
    """
    if skip_note is not None and not live:
        print(f"  {label} skipped: {skip_note}")
        return board
    try:
        out = run(board)
        setattr(report, flag, True)
        return board if out is None else out
    except Exception as e:
        print(f"  {label} unavailable ({type(e).__name__}); {on_fail}")
        return board


def _attach_market(board: pl.DataFrame, adp: pl.DataFrame, *, league_size: int,
                   season: int, season_ahead: int) -> pl.DataFrame:
    """Everything the market contributes once ADP is in hand.

    Extracted so it can run under `_stage` like the other five. It used to sit inline
    under no `try` at all -- forty-odd lines including `_attach_edge`, `proj_blend`, both
    corrections and `corrected_adp` -- while `report.adp = True` was set by hand above it.
    So the one stage whose flag two renderers read was the one stage the degradation
    policy did not cover, and a failure anywhere in here did not degrade: it propagated to
    `build_or_last_good` and served YESTERDAY'S board, which is a far bigger hammer than
    degrading for a stage that is advisory by design.

    On failure `_stage` returns the board it was handed, so the `adp` column is not on it
    either -- the outcome is exactly ECR-only mode, which is why the failure note uses the
    same words `espn_adp` uses when the fetch itself comes back empty. One outcome, one
    vocabulary.
    """
    board = _attach_edge(board, adp, league_size)
    # The forecast the optimiser plays on: the market's forward projection, nudged by
    # the xFP regression signal. Keeping VOR on xFP alone would have the greedy rank
    # players on last season while the simulation scores them on this one -- the two
    # must share a basis or the "edge" is just the gap between the two signals.
    board = board.with_columns(blend())
    # Mark quarterbacks down for last season's touchdown luck. ESPN's projection carries
    # the same bias the draft room does, but only at QB (-0.540 points per point of
    # luck, 99.5%; see docs/td-luck.md). This has to happen here rather than in the
    # display, because `hub.draft.optimize` scores seasons against proj_blend -- a bias
    # left in this column is a bias in every P(win) the optimiser reports.
    #
    # The size of the correction is kept, not just its effect. `proj_correction` is what
    # our measurements say the market is wrong by, in points per game, and it is what
    # `optimize.corrected_adp` converts into a pick shift. Recovering it as a delta
    # rather than recomputing it means the number THE PICK ranks on and the number
    # printed beside it can never disagree.
    raw = board["proj_blend"]
    board = td_regression.correct_projection(board)
    # And for his own availability history, where the market leaves a residual: QB and
    # WR. Running backs are left alone -- the market already prices their durability.
    board = durability.correct_projection(board)
    board = board.with_columns(
        (pl.col("proj_blend") - raw).alias("proj_correction"))
    # What THE PICK ranks on: ADP moved by the correction, bounded. See
    # `optimize.corrected_adp` and ADR-0011.
    from hub.draft.optimize import corrected_adp
    board = board.with_columns(corrected_adp(board).alias("adp_corrected"))
    # Replacement on the projection scale rather than the xFP one, so `vor_proj` and
    # `proj_blend` are the same currency.
    pb = replacement_levels(board["pos"], board["proj_blend"], board["games"],
                            league_size)
    board = board.with_columns(
        (pl.col("proj_blend")
         - pl.col("pos").replace_strict(pb, default=0.0)).alias("vor_proj"))
    n = int(board["edge"].is_not_null().sum())
    print(f"  ESPN ADP: {n} drafted players priced; edge on a common scale")

    return board


def build(league_size: int = 12, season: int = SEASON_COMPLETED, *,
          season_ahead: int = SEASON_AHEAD,
          as_of: str | None = None) -> tuple[pl.DataFrame, BuildReport]:
    """The draft board. `season` is the season just gone; `season_ahead` is the one drafted for.

    `as_of` reconstructs the board as it stood before an ISO date, for replaying a past
    draft. It is one parameter rather than two because the ESPN skip *follows* from it: ESPN
    publishes ADP, projections, scoring and roster slots for the current season only, so
    asking for a 2022 board and then reading 2026 ADP onto it is not a configuration choice,
    it is a contradiction. Historical mode is therefore consensus-and-nflverse only, and
    `proj_blend` never forms -- `moments()` falls back to `xfp_per_game`, which is the prior
    season's expected points and contemporaneous by construction.
    """
    live = as_of is None
    print("  loading ffopportunity ...")
    xp = expected_points(season)
    print(f"  loading consensus rankings{'' if live else f' as of {as_of}'} ...")
    ecr = consensus(as_of)
    levels = replacement_levels(xp["position"], xp["xfp_per_game"], xp["games"],
                                league_size)
    print("  replacement (xFP/gm): " + ", ".join(f"{k}={v:.1f}" for k, v in levels.items()))

    # Impute before VOR, not after: a rookie with no prior-season xFP still has a
    # consensus rank, and leaving him null propagates a zero all the way into the season
    # simulation. VOR is then computed from the filled values so every drafted player is
    # priced on the same basis.
    board = _impute_xfp(_join_expected_points(ecr, xp))
    board = board.with_columns(
        (pl.col("xfp_per_game")
         - pl.col("pos").replace_strict(levels, default=0.0)).alias("vor"),
        pl.col("ecr").rank().alias("consensus_rank"),
    )

    # Every optional stage goes through `_stage`, which owns the degradation policy: try it,
    # flag it if it worked, say what broke if it did not, and never let an advisory column
    # stop the board building. That rule used to be written out five times, and the guards
    # its consumers read had already drifted apart -- see `_stage`.
    report = BuildReport()

    # Weeks 15-17 strength of schedule. A tiebreaker column, not a ranking: the fantasy
    # playoffs are three known games against known defences and nobody drafting off the
    # ESPN app prices them, but last season's defence is a noisy guide to this one.
    board = _stage(board, report, "sos", "weeks 15-17 SoS",
                   lambda b: attach_sos(b, playoff_sos(season_ahead=season_ahead,
                                                       dvp_season=season)))

    # Touchdown luck from last season's actuals. Reaches nflverse, so it can fail on its own.
    board = _stage(board, report, "td_luck", "touchdown luck",
                   lambda b: td_regression.attach(b, td_regression.prior_season(season)))

    # The league owns the scoring weights, so check ours against them rather than assuming.
    # Fantasy points are an aggregate of real stats; if the commissioner moves to half-PPR,
    # every projection and every pick is silently mis-scored until someone notices.
    board = _stage(board, report, "scoring_checked", "scoring check", _check_scoring,
                   live=live, skip_note="ESPN publishes settings for the current season only.",
                   on_fail="assuming full PPR.")

    # The league owns the roster shape too, and for a long time nothing checked it: the slots
    # came back from `league_settings()` on every call and every caller discarded them. A
    # second flex would move replacement level at every position.
    board = _stage(board, report, "roster_checked", "roster check", _check_roster,
                   live=live, skip_note="ESPN publishes slots for the current season only.",
                   on_fail=f"assuming {SLOTS}.")

    # Availability as a per-player trait.
    board = _stage(board, report, "durability", "durability",
                   lambda b: durability.attach(b, durability.prior_season(season)))

    adp = espn_adp(league_size, season_ahead) if live else None
    if adp is not None:
        board = _stage(board, report, "adp", "market corrections",
                       lambda b: _attach_market(b, adp, league_size=league_size,
                                                season=season, season_ahead=season_ahead),
                       on_fail="running ECR-only mode.")
    # The board's columns are its interface -- roughly fourteen modules read them by name --
    # and this contract was declared in `hub.contracts` and applied to nothing at all. It
    # covers only what something downstream reads *unconditionally*; the optional columns
    # are deliberately absent so a degraded fetch still produces a usable board.
    # Sorted on `ecr` **and then `player`**. The tiebreaker cannot move a ranking -- it only
    # orders players who share an ECR, which was arbitrary before -- and it is what makes the
    # board reproducible: two identical `board_as_of` calls returned the same 1,103 players in
    # a different row order, the draft indexes the board by row, and every measurement drafting
    # from it wobbled by ~0.04 points a team-week. improvements.md #18, the other half.
    return DRAFT_BOARD.validate(board.sort(["ecr", "player"])), report


def board_as_of(season: int) -> tuple[pl.DataFrame, BuildReport]:
    """The board for `season`, built from the last consensus scrape before it opened.

    Lives here rather than in `hub.models.experiment`, which is where it started. Every line
    of it is draft-domain knowledge -- `season - 1`, `season_ahead`, the September cutoff --
    and putting it under `models/` inverted the tree's one consistent direction (six `draft/`
    modules import `models/`; nothing went the other way) to save two callers a lambda each.
    It also needed a function-local import to dodge the cycle that inversion created.

    The temporal rule itself is `build`'s, documented on `build`, and now stated next to it: a
    strategy scored against rankings published after the season is hindsight wearing a
    backtest's clothes.
    """
    return build(season=season - 1, season_ahead=season, as_of=f"{season}-09-01")


def recommend(board: pl.DataFrame, current_pick: int, *, rounds: int = 16,
              w: float = DEFAULT_ESPN_WEIGHT, top: int = 10,
              state: DraftState | None = None) -> tuple[str, pl.DataFrame]:
    """Rank the board for one specific pick, under the rule that pick's wait implies.

    Slot 3 of 12 alternates a 19-pick wait and a 5-pick wait, and that alternation should
    drive the pick more than any ranking does:

      scarcity -- a 19-pick wait follows. The question is not "who is best" but "who will
                  not survive", so rank by cost_of_waiting = VOR x P(gone by next turn).
      value    -- your next turn is 5 picks away. Almost anyone you like survives, so
                  availability is noise; take the highest VOR.

    Ranking by `edge` is deliberately not offered. The largest edges sit on players
    consensus does not rate, so an edge-sorted board drafts replacement level.
    """
    if state is not None:
        board = remaining(board, state)
    # ECR-only mode drops the column entirely rather than nulling it, and blended_adp
    # reads it unconditionally. Degrading to consensus-only must still produce a board.
    if "adp" not in board.columns:
        board = board.with_columns(pl.lit(None, pl.Float64).alias("adp"))
    picks = my_picks(rounds)
    now, nxt = next_two(picks, current_pick - 1)
    if now != current_pick:
        raise ValueError(f"{current_pick} is not one of your picks: {picks}")
    mode = draft_mode(now, rounds)
    if mode == "value":
        ranked = board.filter(pl.col("vor").is_not_null()).sort("vor", descending=True)
    else:
        ranked = pick_value(board, now, nxt, w=w)
    return mode, ranked.head(top)


def _emit(lines: list[str]) -> None:
    """Print what a renderer returned. The only place in this module that writes to stdout.

    Renderers return lines so they can be tested; this is the one adapter that turns them
    into output, which is what `live.py` has always done with its own `render()`.
    """
    for line in lines:
        print(line)


# The board's home in the store: a preseason artifact, so week 0.
BOARD_TABLE = "boards"
BOARD_WEEK = 0


def _archive(board: pl.DataFrame, *, season: int = SEASON_AHEAD,
             now: datetime | None = None, base: Path | None = None) -> Path | None:
    """Keep every build, immutably, through `hub.store`.

    improvements.md #8: `build()` writes one flat `draft_board.parquet` and overwrites it, so
    **every `make draft` destroyed the previous day's board** -- and with it that day's ADP,
    the one input `fit_espn_weight`, the opponent model and validating `edge` all need, and the
    one thing ESPN does not retain. The fetch layer had already solved this: `hub.store.write`
    is "immutable dated partitions; corrections write a new file, nothing is overwritten", and
    the odds fetcher has kept every snapshot since day one.

    Additive on purpose. `draft_board.parquet` stays exactly where it is, because `last_good`
    reads it, `adherence` copies it, `hub.inspect` special-cases it and `docs/draft-night.md`
    names it as the fallback -- and four days before a draft is not when the artifact everything
    falls back to should move. Migrating the readers is what remains of this item.

    Never allowed to break the build, for the same reason the ADP archive is not: this is an
    archival side effect and a board that will not print because an archive write failed is
    the operator-dependence CLAUDE.md warns about.
    """
    when = now or datetime.now(UTC)
    try:
        return store.write(board, BOARD_TABLE, "nfl", season, BOARD_WEEK,
                           name=f"board-{when:%Y%m%dT%H%M%S}", base=base)
    except Exception as e:                                  # pragma: no cover - disk
        print(f"  board archive skipped ({type(e).__name__}); the board itself is written")
        return None


def _persist(board: pl.DataFrame, *, out: Path | None = None,
             path: Path | None = None) -> None:
    """Write the board and the site's copy of it, creating both parents first.

    `data/processed/` gets created by `hub.store`, and the board does not go through
    `hub.store` -- that is improvements.md #8, filed as a tidiness issue. It is not one: on a
    machine where nothing has written to the store yet the directory does not exist, and
    `write_parquet` fails with a bare polars FileNotFoundError.

    Every developer machine already has that directory, which is exactly why `make draft`
    had never once worked on a fresh clone -- the first command in the README, on the repo
    that goes public on 2026-09-04.
    """
    # Resolved at call time, not bound as a default. A module-level path used as a default
    # argument is fixed when the function is defined, so monkeypatching the module attribute
    # does not reach it -- and a test that believes it redirected the output writes to the
    # real board instead. That happened.
    out = out if out is not None else OUT
    path = path if path is not None else BOARD_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    board.write_parquet(path)
    _archive(board)
    # Through `hub.jsonio`, not `json`, because a bare `NaN` is not JSON and the page that
    # reads this file is the draft-night fallback -- see that module's docstring.
    out.joinpath("draft_board.json").write_text(
        jsonio.dumps(board.head(300).to_dicts()))


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI. `argv` is threaded through so this is callable from a test.

    Every other CLI in the repo already took it; this one did not, which is why the module
    with the most churn in the repo also had the least covered entry point. All three bugs
    found in the draft-night rehearsal lived below this line.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--league-size", type=int, default=12)
    ap.add_argument("--scoring", default="ppr")
    ap.add_argument("--season", type=int, default=SEASON_COMPLETED)
    ap.add_argument("--pick", type=int, default=None,
                    help="rank the board for this pick of yours (e.g. 3, 22, 27)")
    ap.add_argument("--espn-weight", type=float, default=DEFAULT_ESPN_WEIGHT,
                    help="fraction of the room drafting off ESPN's board (0=all sharp)")
    ap.add_argument("--show-slots", action="store_true",
                    help="print league shape and your pick schedule, then exit")
    ap.add_argument("--taken", default=None,
                    help="comma-separated players just drafted; appends to draft state")
    ap.add_argument("--sync", action="store_true",
                    help="pull the live draft from ESPN instead of typing picks")
    ap.add_argument("--undo", type=int, default=0, help="remove the last N picks")
    ap.add_argument("--reset", action="store_true", help="clear the draft state")
    ap.add_argument("--sos", action="store_true",
                    help="show weeks 15-17 strength of schedule, softest and hardest")
    ap.add_argument("--fit-noise", action="store_true",
                    help="fit sigma(mu) from your league's past drafts, then exit")
    a = ap.parse_args(argv)

    if a.fit_noise:

        from dotenv import load_dotenv

        from hub.draft.availability import fit_pick_noise
        load_dotenv()
        from hub.fetch.espn import resolve_league_id
        fit_pick_noise(resolve_league_id(), [a.season - 2, a.season - 1])
        return 0

    st = state_mod.load()
    if a.reset:
        st = DraftState()
    if a.sync:
        st = state_mod.sync_from_espn()
    if a.undo:
        st = state_mod.undo(st, a.undo)
    if a.taken:
        st = state_mod.take(st, *[n.strip() for n in a.taken.split(",") if n.strip()])
    if a.reset or a.sync or a.undo or a.taken:
        state_mod.save(st)
        print(f"  draft state: {st.n_taken} picks recorded")


    if a.show_slots:
        picks = my_picks()
        waits = [b - x for x, b in itertools.pairwise(picks)]
        _emit(report_mod.slots(TEAMS, MY_SLOT, SLOTS, DRAFTED_POSITIONS, picks, waits,
                               draft_mode))
        return 0

    board, report, stale_h = build_or_last_good(a.league_size, a.season)
    if stale_h is None:
        _emit(report_mod.degraded(report.degraded()))

    # A mistyped pick is silent otherwise: the misspelt player stays on the board as
    # available and the next recommendation can hand back someone already drafted. ESPN
    # publishes nothing mid-draft for a practice room, so typing picks is the primary path
    # and this is its sharp edge. Checked here rather than where the picks are recorded,
    # because the board does not exist yet at that point.
    if a.taken:
        _emit(report_mod.mistyped(state_mod.suggest_unmatched(
            board, [n.strip() for n in a.taken.split(",") if n.strip()])))
    # Not when serving last-good: rewriting the file resets its mtime, so the age printed
    # above would immediately become a lie, and every later run would report a fresh board
    # that is actually as stale as the first failure.
    if stale_h is None:
        _persist(board)

        # Keep a dated copy of today's ADP before the next build overwrites it. ESPN does
        # not retain historical ADP, so this is the only chance to record it -- and its
        # absence is what blocks fit_espn_weight, the opponent model, and validating `edge`.
        #
        # Never allowed to break the build: this is an archival side effect, not the
        # product, and a board that will not print because an archive write failed is
        # exactly the operator-dependence CLAUDE.md warns about.
        try:
            if adp_history.snapshot(board) is not None:
                print(f"  adp archived: {len(adp_history.days())} days on file")
        except Exception as exc:
            print(f"  adp archive skipped: {type(exc).__name__}: {exc}")

    _emit(report_mod.header(board))
    _emit(report_mod.regression(board))
    _emit(report_mod.td_luck(board, report))
    _emit(report_mod.injuries(board, report))

    if a.sos:
        _emit(report_mod.sos(board, report))
        return 0

    _emit(report_mod.unmatched(state_mod.unmatched(board, st)))

    if a.pick is not None:
        mode, rec = recommend(board, a.pick, w=a.espn_weight, state=st)
        from hub.draft.optimize import the_pick
        _emit(report_mod.the_pick(the_pick(board, st, my_slot=MY_SLOT, teams=TEAMS)))
        _emit(report_mod.also_close(mode, rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
