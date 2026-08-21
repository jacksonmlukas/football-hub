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
import polars as pl
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


def consensus() -> pl.DataFrame:
    """FantasyPros ECR. `sd` is a crude prior on uncertainty until conformal lands."""
    r = nfl.load_ff_rankings("draft")
    keep = [c for c in ("player", "pos", "team", "ecr", "sd", "best", "worst") if c in r.columns]
    return (r.select(keep)
             .filter(pl.col("ecr").is_not_null())
             .unique(subset=["player"], keep="first")
             .sort("ecr"))


def replacement_levels(df: pl.DataFrame, teams: int = 12) -> dict[str, float]:
    """Replacement = the xFP of the last startable player at each position.

    In full PPR with a 12-team flex, the flex is almost always a WR, which pushes WR
    replacement meaningfully deeper than a naive teams*slots calculation suggests.
    """
    # Flex allocation. In a 2WR league the flex is overwhelmingly a WR. With THREE required
    # WR slots the top of the WR pool is already consumed by starters, so the flex tilts back
    # toward RB. Roughly even here, not WR-dominant.
    flex_share = {"RB": 0.45, "WR": 0.50, "TE": 0.05}
    levels = {}
    for pos in ("QB", "RB", "WR", "TE"):
        n = teams * SLOTS.get(pos, 0)
        n += round(teams * SLOTS.get("FLEX", 0) * flex_share.get(pos, 0)) if pos in FLEX_ELIGIBLE else 0
        pool = df.filter(pl.col("position") == pos).sort("xfp_per_game", descending=True)
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
        from hub.fetch.espn import player_adp
        adp = player_adp()
    except Exception as e:  # noqa: BLE001
        print(f"  ESPN ADP unavailable ({type(e).__name__}); running ECR-only mode.")
        return None

    # Shape is not substance. The predecessor of this function returned a frame of 400 rows
    # whose every value was an empty list: not null, so it passed every null check, and
    # `edge` shipped as a column of empty lists with no crash and no warning. Keep asserting
    # on dtype and height here even though the fetch layer is now typed, because the failure
    # this catches is ESPN quietly changing a field, which is not a hypothetical.
    if adp.is_empty() or not adp["adp"].dtype.is_numeric():
        print(f"  ESPN ADP present but unusable "
              f"(dtype={adp['adp'].dtype}, rows={adp.height}); running ECR-only mode.")
        return None
    return adp


def _adp_saturation_cutoff(adp: pl.Series, teams: int) -> float | None:
    """The ADP value at which ESPN stops actually knowing anything.

    ESPN does not leave ADP null for players nobody drafts -- it parks them all on a
    shared value at the tail (~170 in a 12-team league, where 27 players share 169.97).
    A genuine ADP is near-unique; the sentinel is shared by dozens. So: the lowest value
    shared by at least `teams` players is the cutoff, and it and everything above is
    undrafted. Derived from the data rather than hardcoded, so it self-tunes if ESPN
    moves the cluster or the league changes size.
    """
    if adp.is_empty():
        return None
    counts = adp.value_counts().sort(adp.name)
    hits = counts.filter(pl.col("count") >= teams)
    return float(hits[adp.name][0]) if hits.height else None


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


def build(league_size: int = 12, season: int = 2025) -> pl.DataFrame:
    print("  loading ffopportunity ...")
    xp = expected_points(season)
    print("  loading consensus rankings ...")
    ecr = consensus()
    levels = replacement_levels(xp, league_size)
    print(f"  replacement (xFP/gm): " + ", ".join(f"{k}={v:.1f}" for k, v in levels.items()))

    vor = xp.with_columns(
        pl.struct(["position", "xfp_per_game"]).map_elements(
            lambda s: s["xfp_per_game"] - levels.get(s["position"], 0.0),
            return_dtype=pl.Float64).alias("vor")
    )
    board = (ecr.join(vor, left_on="player", right_on="full_name", how="left")
                .with_columns(pl.col("ecr").rank().alias("consensus_rank")))

    adp = espn_adp(league_size)
    if adp is not None:
        board = _attach_edge(board, adp, league_size)
        n = int(board["edge"].is_not_null().sum())
        print(f"  ESPN ADP: {n} drafted players priced; edge on a common scale")
    return board.sort("ecr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league-size", type=int, default=12)
    ap.add_argument("--scoring", default="ppr")
    ap.add_argument("--season", type=int, default=2025)
    a = ap.parse_args()

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


if __name__ == "__main__":
    main()
