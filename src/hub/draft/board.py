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
import argparse, collections, json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import polars as pl

from hub.config import RosterConfig, flex_positions, flex_share, starters
from hub.contracts import DRAFT_BOARD, ContractViolation
from hub.draft.availability import DEFAULT_ESPN_WEIGHT, pick_value
from hub.draft.picks import MY_SLOT, TEAMS, draft_mode, my_picks, next_two
from hub.draft import durability
from hub.draft import regression as td_regression
from hub.draft.playoff_sos import attach_sos, playoff_sos
from hub.draft.state import DraftState, _norm, remaining
from hub.draft import state as state_mod
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[3]
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


def expected_points(season: int = 2025) -> pl.DataFrame:
    """Season-total expected vs actual PPR points. The gap is the regression signal."""
    o = nfl.load_ff_opportunity(seasons=[season], stat_type="weekly")
    agg = o.group_by(["player_id", "full_name", "position"]).agg([
        pl.col("receptions_exp").sum().alias("rec_exp"),
        pl.col("receptions").sum().alias("rec_act"),
        pl.col("total_fantasy_points_exp").sum().alias("xfp"),
        pl.col("total_fantasy_points").sum().alias("fp"),
        pl.len().alias("games"),
    ])
    return agg.with_columns([
        (pl.col("fp") - pl.col("xfp")).alias("fp_over_expected"),
        (pl.col("xfp") / pl.col("games")).alias("xfp_per_game"),
    ])


# `load_ff_rankings("draft")` stacks 31 FantasyPros pages into one frame -- redraft,
# dynasty, best-ball, superflex, IDP -- each with its OWN ecr scale. Picking a player's
# row without filtering draws from 27 of them, which is how kickers ended up with ECRs of
# 11 and topped the edge list. This page is /nfl/rankings/ppr-cheatsheets.php: redraft,
# full PPR, no superflex, no IDP. It is the only one that matches this league.
CONSENSUS_PAGE = "redraft-overall"

# The season being drafted for. xFP describes the season just gone; the simulation has to
# be about the one ahead, or a strategy ranked on last year's numbers scores against last
# year's truth and reads as clairvoyant.
SEASON_AHEAD = 2026

# On the board because you draft them off it. K and DST are rostered but taken late off
# ESPN's own list, and their presence distorts every rank below the skill players.
DRAFTED_POSITIONS = ("QB", "RB", "WR", "TE")


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
    for pos in ("QB", "RB", "WR", "TE"):
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
    except Exception as e:  # noqa: BLE001
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
    norm = pl.col("player").map_elements(_norm, return_dtype=pl.Utf8).alias("_key")
    right = (xp.with_columns(pl.col("full_name").map_elements(_norm, return_dtype=pl.Utf8)
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


class SkipHistorical(Exception):
    """A stage that only exists for the current season, skipped on a historical board.

    Raised inside the same `try` that handles a stage failing, so the "did it run" bookkeeping
    stays in one place. It is not an error: `report.scoring_checked` staying False is the
    correct record either way, since the stage genuinely did not run.
    """


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


def build(league_size: int = 12, season: int = 2025, *,
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
    print(f"  replacement (xFP/gm): " + ", ".join(f"{k}={v:.1f}" for k, v in levels.items()))

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

    # Weeks 15-17 strength of schedule. A tiebreaker column, not a ranking: the fantasy
    # playoffs are three known games against known defences and nobody drafting off the
    # ESPN app prices them, but last season's defence is a noisy guide to this one.
    report = BuildReport()
    try:
        board = attach_sos(board, playoff_sos(season_ahead=season_ahead, dvp_season=season))
        report.sos = True
    except Exception as e:  # noqa: BLE001
        print(f"  weeks 15-17 SoS unavailable ({type(e).__name__}); board built without it.")

    # Touchdown luck from last season's actuals. Its own try: it reaches nflverse, and a
    # board that will not build because one advisory column is unavailable is the
    # operator-dependence CLAUDE.md warns about.
    try:
        board = td_regression.attach(board, td_regression.prior_season(season))
        report.td_luck = True
    except Exception as e:  # noqa: BLE001
        print(f"  touchdown luck unavailable ({type(e).__name__}); board built without it.")

    # The league owns the scoring weights, so check ours against them rather than assuming.
    # Fantasy points are an aggregate of real stats; if the commissioner moves to half-PPR,
    # every projection and every pick is silently mis-scored until someone notices.
    if not live:
        print("  scoring check skipped: ESPN publishes settings for the current season only.")
    try:
        from hub.fetch import espn as espn_fetch
        from hub.models.components import scoring_mismatch
        if not live:
            raise SkipHistorical("scoring settings")
        bad = scoring_mismatch(espn_fetch.scoring_settings())
        if bad:
            print("  SCORING MISMATCH -- the league scores differently from hub.models."
                  "components.SCORING:")
            for k, (theirs, ours) in sorted(bad.items()):
                print(f"    {k}: league {theirs}, this repo {ours}")
            print("  Every projection below is scored on the wrong weights until that is fixed.")
        report.scoring_checked = True
    except Exception as e:  # noqa: BLE001
        if live:
            print(f"  scoring check unavailable ({type(e).__name__}); assuming full PPR.")

    # The league owns the roster shape too, and for a long time nothing checked it: the
    # slots came back from `league_settings()` on every call and every caller discarded
    # them. A second flex would move replacement level at every position.
    if not live:
        print("  roster check skipped: ESPN publishes slots for the current season only.")
    try:
        from hub.config import roster_mismatch
        from hub.fetch import espn as espn_fetch
        if not live:
            raise SkipHistorical("roster slots")
        _, slots = espn_fetch.league_settings()
        bad_slots = roster_mismatch(slots or {})
        if bad_slots:
            print("  ROSTER MISMATCH -- the league starts a different lineup from "
                  "hub.config.RosterConfig:")
            for k, (theirs, ours) in sorted(bad_slots.items()):
                print(f"    {k}: league {theirs}, this repo {ours}")
            print("  Replacement level and every VOR below assume this repo's shape.")
        report.roster_checked = True
    except Exception as e:  # noqa: BLE001
        if live:
            print(f"  roster check unavailable ({type(e).__name__}); assuming {SLOTS}.")

    # Availability as a per-player trait. Its own try for the same reason as above.
    try:
        board = durability.attach(board, durability.prior_season(season))
        report.durability = True
    except Exception as e:  # noqa: BLE001
        print(f"  durability unavailable ({type(e).__name__}); board built without it.")

    adp = espn_adp(league_size, season_ahead) if live else None
    if adp is not None:
        report.adp = True
        board = _attach_edge(board, adp, league_size)
        # The forecast the optimiser plays on: the market's forward projection, nudged by
        # the xFP regression signal. Keeping VOR on xFP alone would have the greedy rank
        # players on last season while the simulation scores them on this one -- the two
        # must share a basis or the "edge" is just the gap between the two signals.
        board = board.with_columns(
            pl.coalesce(
                (pl.col("proj_ppg") + pl.col("xfp_per_game")) / 2.0,
                pl.col("proj_ppg"), pl.col("xfp_per_game"),
            ).alias("proj_blend"))
        # Mark quarterbacks down for last season's touchdown luck. ESPN's projection carries
        # the same bias the draft room does, but only at QB (-0.540 points per point of
        # luck, 99.5%; see docs/td-luck.md). This has to happen here rather than in the
        # display, because `hub.draft.optimize` scores seasons against proj_blend -- a bias
        # left in this column is a bias in every P(win) the optimiser reports.
        board = td_regression.correct_projection(board)
        # And for his own availability history, where the market leaves a residual: QB and
        # WR. Running backs are left alone -- the market already prices their durability.
        board = durability.correct_projection(board)
        # Replacement on the projection scale rather than the xFP one, so `vor_proj` and
        # `proj_blend` are the same currency.
        pb = replacement_levels(board["pos"], board["proj_blend"], board["games"],
                                league_size)
        board = board.with_columns(
            (pl.col("proj_blend")
             - pl.col("pos").replace_strict(pb, default=0.0)).alias("vor_proj"))
        n = int(board["edge"].is_not_null().sum())
        print(f"  ESPN ADP: {n} drafted players priced; edge on a common scale")

    # The board's columns are its interface -- roughly fourteen modules read them by name --
    # and this contract was declared in `hub.contracts` and applied to nothing at all. It
    # covers only what something downstream reads *unconditionally*; the optional columns
    # are deliberately absent so a degraded fetch still produces a usable board.
    return DRAFT_BOARD.validate(board.sort("ecr")), report


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


def _print_injuries(board: pl.DataFrame, report: BuildReport) -> None:
    """Players carrying a designation right now, and last season's fragile ones.

    Two different quantities kept visibly apart. Last season's missed games are priced into
    the projection where the market leaves a residual (QB, WR). Today's designation is not
    priced at all -- there is no history of preseason designations against outcomes to fit a
    coefficient on, so inventing one would be worse than showing the drafter the flag.
    """
    if not (report.durability or report.adp):
        return
    pool = board.filter(pl.col("adp").is_not_null() & (pl.col("adp") <= 120))
    if "injury_status" in pool.columns:
        hurt = pool.filter(
            pl.col("injury_status").map_elements(durability.is_flagworthy,
                                                 return_dtype=pl.Boolean)
        ).sort("adp")
        if hurt.height:
            print(f"\n  Carrying a designation today -- {hurt.height} inside ADP 120")
            print("  Out/Doubtful/IR are priced (-1.63 ppg, fitted on week-1 reports).")
            print("  QUESTIONABLE is not: 12.6% of the August board carries it against")
            print("  2.9% at week 1, so the fitted number is from a much sicker group.")
            for r in hurt.head(8).iter_rows(named=True):
                st = str(r["injury_status"]).upper()
                beta = durability.INJURY_BETA.get(st)
                note = f"{beta:+.2f} ppg" if beta else "not priced"
                print(f"    {r['player']:<24} {r['pos'] or '':<4} ADP {r['adp']:>5.1f}  "
                      f"{st:<15} {note}")

    if "missed" in pool.columns:
        frail = pool.filter(pl.col("missed").is_not_null()
                            & (pl.col("missed") >= 4)).sort("missed", descending=True)
        if frail.height:
            print(f"\n  Missed time last season -- priced for QB and WR only")
            print("  Running backs are left alone: the market already discounts them.")
            for r in frail.head(6).iter_rows(named=True):
                pos = r["pos"] or ""
                beta = durability.BETA.get(pos, 0.0)
                note = f"{beta * r['missed']:+.2f} ppg" if beta else "not priced"
                print(f"    {r['player']:<24} {pos:<4} ADP {r['adp']:>5.1f}  "
                      f"missed {int(r['missed']):>2}  {note}")


def _print_td_luck(board: pl.DataFrame, report: BuildReport) -> None:
    """Players priced on touchdowns their yardage does not support.

    A different cut from xFP-FP, and measurably so -- the two correlate at +0.16 on the live
    board, and they are signed in opposite directions, so a real overlap would show as a
    strong negative. See docs/td-luck.md, including how much of this is established
    (quarterbacks) and how much is only directional (running backs, receivers).
    """
    if not report.td_luck:
        return
    pool = board.filter(pl.col("td_luck").is_not_null()
                        & pl.col("adp").is_not_null() & (pl.col("adp") <= 120))
    if pool.height < 8:
        return
    print(f"\n  Touchdown luck, last season's actuals -- {pool.height} drafted players")
    print("  Points per game above the touchdowns their yardage supports. Touchdown rate")
    print("  has no year-over-year persistence, so this is the part least likely to repeat.")
    for label, frame in (("FADE", pool.sort("td_luck", descending=True).head(5)),
                         ("BUY ", pool.sort("td_luck").head(5))):
        for r in frame.iter_rows(named=True):
            print(f"    {label} {r['player']:<24} {r['pos'] or '':<4} "
                  f"ADP {r['adp']:>5.1f}  {r['td_luck']:>+6.2f}/gm")
    print("  Strongest for QB, where the room prices last year's touchdowns at 1.02 against")
    print("  volume and their true predictive weight is -0.05. Directional elsewhere.")


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI. `argv` is threaded through so this is callable from a test.

    Every other CLI in the repo already took it; this one did not, which is why the module
    with the most churn in the repo also had the least covered entry point. All three bugs
    found in the draft-night rehearsal lived below this line.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--league-size", type=int, default=12)
    ap.add_argument("--scoring", default="ppr")
    ap.add_argument("--season", type=int, default=2025)
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
        import os
        from dotenv import load_dotenv
        from hub.draft.availability import fit_pick_noise
        load_dotenv()
        fit_pick_noise(int(os.environ["ESPN_LEAGUE_ID"]), [a.season - 2, a.season - 1])
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
        waits = [b - x for x, b in zip(picks, picks[1:])]
        print(f"  teams {TEAMS} | slot {MY_SLOT} | starters "
              + " ".join(f"{k}{v}" for k, v in SLOTS.items()))
        print(f"  drafted positions: {'/'.join(DRAFTED_POSITIONS)}")
        print(f"  picks: {', '.join(map(str, picks[:8]))} ...")
        print(f"  waits: {', '.join(map(str, waits[:7]))} ...")
        for pk in picks[:6]:
            print(f"    pick {pk:>3}  ->  {draft_mode(pk)}")
        return 0

    board, report = build(a.league_size, a.season)
    if report.degraded():
        print(f"  built without: {', '.join(report.degraded())}")

    # A mistyped pick is silent otherwise: the misspelt player stays on the board as
    # available and the next recommendation can hand back someone already drafted. ESPN
    # publishes nothing mid-draft for a practice room, so typing picks is the primary path
    # and this is its sharp edge. Checked here rather than where the picks are recorded,
    # because the board does not exist yet at that point.
    if a.taken:
        for name, guess in state_mod.suggest_unmatched(
                board, [n.strip() for n in a.taken.split(",") if n.strip()]).items():
            if guess:
                print(f"\n  NOT ON THE BOARD: {name!r} -- did you mean {guess!r}?")
                print(f"    until that is fixed, {guess!r} is still shown as available")
            else:
                print(f"\n  not on the board: {name!r} (kicker or defence, most likely)")
    OUT.mkdir(parents=True, exist_ok=True)
    board.write_parquet(ROOT / "data" / "processed" / "draft_board.parquet")
    OUT.joinpath("draft_board.json").write_text(json.dumps(board.head(300).to_dicts(), default=str))

    # Summary only. Never print the frame -- that is the token rule in CLAUDE.md.
    print(f"\n  {board.height} players | {board['vor'].null_count()} missing xFP")
    top = board.filter(pl.col("fp_over_expected").is_not_null()).sort("fp_over_expected")
    print("\n  Biggest positive regression candidates (underperformed expectation):")
    for r in top.head(8).iter_rows(named=True):
        print(f"    {r['player']:<24} {r['pos'] or '':<4} ECR {r['ecr']:>5.1f}  "
              f"xFP-FP {-r['fp_over_expected']:>6.1f}")

    _print_td_luck(board, report)
    _print_injuries(board, report)

    if a.sos:
        pool = board.filter(pl.col("adp").is_not_null()
                            & pl.col("wk15_17_sos").is_not_null())
        print(f"\n  Weeks 15-17 strength of schedule -- {pool.height} drafted players")
        print("  1.00 = league-average defence for that position; higher is softer.\n")
        for label, frame in (("SOFTEST", pool.sort("wk15_17_sos", descending=True).head(8)),
                             ("HARDEST", pool.sort("wk15_17_sos").head(8))):
            print(f"  {label}:")
            for r in frame.iter_rows(named=True):
                print(f"    {r['player']:<24} {r['pos']:<3} {r['team'] or '':<4} "
                      f"ADP {r['adp']:>6.1f}  SoS {r['wk15_17_sos']:.3f}")
            print()
        # The actionable form: players the room prices the same, whose playoff slates
        # differ. Inside a tier this is a free upgrade; across tiers it is not.
        print("  Same-tier swaps (ADP within 8, SoS gap > 0.15):")
        rows = pool.sort("adp").iter_rows(named=True)
        rows = list(rows)
        shown = 0
        for i, x in enumerate(rows):
            for y in rows[i + 1:]:
                if y["adp"] - x["adp"] > 8:
                    break
                if x["pos"] == y["pos"] and abs(x["wk15_17_sos"] - y["wk15_17_sos"]) > 0.15:
                    hi, lo = (x, y) if x["wk15_17_sos"] > y["wk15_17_sos"] else (y, x)
                    print(f"    {hi['player']:<22} ({hi['wk15_17_sos']:.2f}) over "
                          f"{lo['player']:<22} ({lo['wk15_17_sos']:.2f})  "
                          f"{hi['pos']}, ADP {hi['adp']:.0f} vs {lo['adp']:.0f}")
                    shown += 1
                    break
            if shown >= 8:
                break
        return 0

    missing = state_mod.unmatched(board, st)
    if missing:
        print(f"\n  {len(missing)} recorded picks matched nobody on the board "
              f"(K/DST are excluded by design): {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else ""))

    if a.pick is not None:
        mode, rec = recommend(board, a.pick, w=a.espn_weight, state=st)
        # The market picks, and nothing else does.
        #
        # P0b measured championship equity against consensus-following on realised outcomes
        # across 2022-25: -19.66 points per team game, 95% CI [-23.16, -16.20] at n=80,
        # losing in all four seasons and winning 9 of 80 drafts. Per the rule fixed before
        # that run, equity leaves this output. See docs/adr/0009.
        from hub.draft.optimize import the_pick
        tp = the_pick(board, st, my_slot=MY_SLOT, teams=TEAMS)
        if tp:
            print(f"\n  THE PICK -- best available filling a need, by {tp.via}")
            print(f"    {tp.player}  {tp.pos or ''}  "
                  + (f"{'ADP' if 'ADP' in tp.via else 'ECR'} {tp.rank:.1f}"
                     if tp.rank is not None else "")
                  + (f"   [{'; '.join(tp.notes)}]" if tp.notes else ""))
            print("    Corrections shown are where our measurements say the market is wrong;")
            print("    they are not priced in. Yours to weigh.")
        else:
            print("\n  THE PICK unavailable -- the board carries neither ADP nor ECR, which "
                  "means it did not build. Serve site/data/draft_board.json instead.")

        rule = ("long wait ahead: take who will not survive it"
                if mode == "scarcity" else "short wait ahead: take the highest VOR")
        # Context, not a recommendation. VOR ordering was measured 5.06 pts/team-game worse
        # than the market (docs/market-value.md), so this is here to show what else is close
        # -- and how close -- rather than to be drafted off.
        print(f"\n  Also close, for context -- not a ranking to draft off ({mode}: {rule})")
        for r in rec.iter_rows(named=True):
            cw = r.get("cost_of_waiting")
            extra = f"cost_of_waiting {cw:>5.1f}" if cw is not None else ""
            v, sos = r["vor"], r.get("wk15_17_sos")
            tag = f"  SoS {sos:.2f}" if sos is not None else ""
            print(f"    {r['player']:<24} {r['pos'] or '':<4} "
                  f"VOR {0.0 if v is None else v:>5.1f}  {extra}{tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
