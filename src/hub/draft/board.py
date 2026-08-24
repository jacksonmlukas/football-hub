"""Full-PPR draft board for a 12-team ESPN redraft league.

Three signals, joined:
  1. FantasyPros ECR      -- consensus rank, with sd/best/worst as a free uncertainty band
  2. ffopportunity xFP    -- expected fantasy points, separating opportunity from efficiency
  3. ESPN ADP             -- what YOUR room will actually do (optional; needs cookies)

The output column that matters is `edge`: consensus rank minus ESPN ADP. Positive means
your leaguemates, drafting off ESPN's default board, will let this player fall.

Runs without cookies (ECR + xFP only). Add ESPN creds for the ADP diff.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl

from hub.contracts import ContractViolation
from hub.draft.availability import DEFAULT_ESPN_WEIGHT, pick_value
from hub.draft.picks import MY_SLOT, TEAMS, draft_mode, my_picks, next_two
from hub.draft import regression as td_regression
from hub.draft.playoff_sos import attach_sos, playoff_sos
from hub.draft.state import DraftState, _norm, remaining
from hub.draft import state as state_mod
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "site" / "data"

# Full PPR. Sourced from league settings when cookies are present; this is the ESPN default.
PPR = dict(pass_yd=0.04, pass_td=4, pass_int=-2, rush_yd=0.1, rush_td=6,
           rec=1.0, rec_yd=0.1, rec_td=6, fumble=-2)

# Starters per team. THREE WR slots plus a flex -- confirmed for this league, not the ESPN
# default. Overridden by league_settings() when cookies are present.
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = ("RB", "WR", "TE")


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
    keep = [c for c in ("player", "pos", "team", "ecr", "sd", "best", "worst") if c in r.columns]
    out = (r.filter(pl.col("page_type") == CONSENSUS_PAGE)
            .select(keep)
            .filter(pl.col("ecr").is_not_null())
            .filter(pl.col("pos").is_in(DRAFTED_POSITIONS))
            .unique(subset=["player"], keep="first")
            .sort("ecr"))
    if out.is_empty():
        raise ContractViolation(
            f"ff_rankings: page `{CONSENSUS_PAGE}` returned no {'/'.join(DRAFTED_POSITIONS)} "
            f"rows; FantasyPros likely renamed the page")
    return out


def consensus() -> pl.DataFrame:
    """FantasyPros ECR. `sd` is a crude prior on uncertainty until conformal lands."""
    return _select_consensus(nfl.load_ff_rankings("draft"))


# A per-game rate needs a real denominator. Without this, a player with one big game
# outranks every genuine starter and occupies a slot that defines replacement level --
# which is how WR replacement came out ABOVE RB (11.14 vs 11.10), contradicting the
# three-WR effect. At >= 10 games WR drops to 10.02 and the effect reappears; the sign
# is stable from 8 games up, so it is not fitted to the threshold.
MIN_GAMES = 10


def replacement_levels(df: pl.DataFrame, teams: int = 12,
                       min_games: int = MIN_GAMES) -> dict[str, float]:
    """Replacement = the xFP of the last startable player at each position.

    In full PPR with a 12-team flex, the flex is almost always a WR, which pushes WR
    replacement meaningfully deeper than a naive teams*slots calculation suggests.

    Only players with at least `min_games` are eligible to set the level: replacement is
    a statement about the quality freely available all season, and a one-game sample says
    nothing about that.
    """
    # Flex allocation. In a 2WR league the flex is overwhelmingly a WR. With THREE required
    # WR slots the top of the WR pool is already consumed by starters, so the flex tilts back
    # toward RB. Roughly even here, not WR-dominant.
    flex_share = {"RB": 0.45, "WR": 0.50, "TE": 0.05}
    levels = {}
    for pos in ("QB", "RB", "WR", "TE"):
        n = teams * SLOTS.get(pos, 0)
        n += round(teams * SLOTS.get("FLEX", 0) * flex_share.get(pos, 0)) if pos in FLEX_ELIGIBLE else 0
        pool = (df.filter((pl.col("position") == pos) & (pl.col("games") >= min_games))
                  .sort("xfp_per_game", descending=True))
        levels[pos] = float(pool["xfp_per_game"][min(n, pool.height) - 1]) if pool.height else 0.0
    return levels


def espn_adp(league_size: int = 12) -> pl.DataFrame | None:
    """ESPN's own ADP -- the thing your room is actually drafting off. Optional.

    Was built on espn_api's `posRank`, which is a *positional* rank (WR5 -> 5) parsed from
    a `positionalRanking` key that free-agent payloads omit entirely. It therefore returned
    empty lists, and subtracting a positional rank from an overall consensus rank would not
    have meant anything even if it had not. Real overall ADP now comes from the fetch layer.
    """
    try:
        from hub.fetch.espn import player_market
        adp = player_market(season=SEASON_AHEAD)
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
               .drop("full_name"))
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


def build(league_size: int = 12, season: int = 2025) -> pl.DataFrame:
    print("  loading ffopportunity ...")
    xp = expected_points(season)
    print("  loading consensus rankings ...")
    ecr = consensus()
    levels = replacement_levels(xp, league_size)
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
    try:
        board = attach_sos(board, playoff_sos(season_ahead=SEASON_AHEAD, dvp_season=season))
    except Exception as e:  # noqa: BLE001
        print(f"  weeks 15-17 SoS unavailable ({type(e).__name__}); board built without it.")

    # Touchdown luck from last season's actuals. Its own try: it reaches nflverse, and a
    # board that will not build because one advisory column is unavailable is the
    # operator-dependence CLAUDE.md warns about.
    try:
        board = td_regression.attach(board, td_regression.prior_season(season))
    except Exception as e:  # noqa: BLE001
        print(f"  touchdown luck unavailable ({type(e).__name__}); board built without it.")

    adp = espn_adp(league_size)
    if adp is not None:
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
        pb = replacement_levels(
            board.rename({"pos": "position_", "proj_blend": "xfp_per_game_"})
                 .with_columns(pl.col("position_").alias("position"),
                               pl.col("xfp_per_game_").alias("xfp_per_game")),
            league_size)
        board = board.with_columns(
            (pl.col("proj_blend")
             - pl.col("pos").replace_strict(pb, default=0.0)).alias("vor_proj"))
        n = int(board["edge"].is_not_null().sum())
        print(f"  ESPN ADP: {n} drafted players priced; edge on a common scale")
    return board.sort("ecr")


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


def _print_td_luck(board: pl.DataFrame) -> None:
    """Players priced on touchdowns their yardage does not support.

    A different cut from xFP-FP, and measurably so -- the two correlate at +0.16 on the live
    board, and they are signed in opposite directions, so a real overlap would show as a
    strong negative. See docs/td-luck.md, including how much of this is established
    (quarterbacks) and how much is only directional (running backs, receivers).
    """
    if "td_luck" not in board.columns:
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


def main():
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
    ap.add_argument("--win-prob", action="store_true",
                    help="rank candidates by P(win the league) instead of cost_of_waiting")
    ap.add_argument("--sims", type=int, default=150, help="season sims per draft sim")
    ap.add_argument("--sos", action="store_true",
                    help="show weeks 15-17 strength of schedule, softest and hardest")
    ap.add_argument("--fit-noise", action="store_true",
                    help="fit sigma(mu) from your league's past drafts, then exit")
    a = ap.parse_args()

    if a.fit_noise:
        import os
        from dotenv import load_dotenv
        from hub.draft.availability import fit_pick_noise
        load_dotenv()
        fit_pick_noise(int(os.environ["ESPN_LEAGUE_ID"]), [a.season - 2, a.season - 1])
        return

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
        return

    board = build(a.league_size, a.season)
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

    _print_td_luck(board)

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
        return

    missing = state_mod.unmatched(board, st)
    if missing:
        print(f"\n  {len(missing)} recorded picks matched nobody on the board "
              f"(K/DST are excluded by design): {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else ""))

    if a.pick is not None:
        mode, rec = recommend(board, a.pick, w=a.espn_weight, state=st)
        rule = ("19-pick wait ahead: take who will not survive it"
                if mode == "scarcity" else "5-pick wait ahead: take the highest VOR")
        print(f"\n  Pick {a.pick} -- {mode.upper()} ({rule})")
        for r in rec.iter_rows(named=True):
            cw = r.get("cost_of_waiting")
            extra = f"cost_of_waiting {cw:>5.1f}" if cw is not None else ""
            v, sos = r["vor"], r.get("wk15_17_sos")
            tag = f"  SoS {sos:.2f}" if sos is not None else ""
            print(f"    {r['player']:<24} {r['pos'] or '':<4} "
                  f"VOR {0.0 if v is None else v:>5.1f}  {extra}{tag}")

        if a.win_prob:
            from hub.draft.optimize import win_probability
            names = rec["player"].to_list()
            print(f"\n  Championship equity over {len(names)} candidates "
                  f"({a.sims} seasons x 6 drafts each) ...")
            wp = win_probability(board, st, names, my_slot=MY_SLOT, teams=TEAMS,
                                 n_season_sims=a.sims, w=a.espn_weight)
            for r in wp.iter_rows(named=True):
                lift, se = r["lift"] * 100, r["lift_se"] * 100
                sig = "*" if abs(lift) > 2 * se else " "
                print(f"    {r['player']:<24} P(win) {r['p_win']*100:>5.2f}%  "
                      f"lift {lift:+5.2f} +/-{se:4.2f} {sig}")
            print("    (* = lift exceeds 2 standard errors; anything else is noise)")
            print("    NOTE: read the lift, not the level. Absolute P(win) is inflated --")
            print("    the season is scored on the same projection the greedy ranks on.")
            print("    See the module docstring in hub/draft/optimize.py.")


if __name__ == "__main__":
    main()
