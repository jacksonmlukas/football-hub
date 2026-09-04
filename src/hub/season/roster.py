"""Who is actually on the team, with a weekly mean and spread for each of them.

The one thing that could not exist before the draft. `docs/gaps.md` recorded
`data/processed/roster.parquet` as MISSING on 2026-08-25 with the note *"the roster is
knowable only after Sep 3"* -- `hub.season.lineup` has therefore never been runnable end to
end, because nothing in the repo wrote the file it reads. This writes it.

It is deliberately a **join, not a model**. Every number here comes from somewhere that was
already gated: `mu` and `sd` are `hub.models.predict.moments` over the board's `proj_blend`,
which is the same currency `hub.draft.optimize` scored seasons against. Adding a second way to
project a rostered player would be a third implementation of one idea, which
`docs/next.md` already names as how they drift.

**Identity is inferred, not configured.** The team is the one whose owners include the SWID
already in the environment for the ESPN cookie, so there is no `ESPN_TEAM_ID` to keep in sync
with a league you might leave. If the SWID matches no team the error says so rather than
silently picking team 1.

**Kickers and defences carry no projection and say so.** The board covers
`config.DRAFTED_POSITIONS` -- QB, RB, WR, TE -- because that was the scope decision, so K and
D/ST join to nothing. They come through with `projected = False` rather than `mu = 0.0`
presented as a forecast: a zero that means "we do not model this" must not render as a zero
that means "we expect nothing".
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import polars as pl

from hub.models.predict import moments
from hub.names import player_key
from hub.paths import ROSTER_PARQUET

# What `hub.season.lineup` requires, and therefore what this must always produce.
REQUIRED: tuple[str, ...] = ("player", "pos", "mu", "sd")

# ESPN's slot label for a benched player. Everything else is a starting slot of some kind.
BENCH = "BE"


def _swid() -> str:
    """The logged-in identity, normalised the way ESPN's team owners report it."""
    return (os.environ.get("ESPN_SWID") or "").strip("{}").upper()


def mine(league: Any, swid: str | None = None) -> Any:
    """The team owned by the SWID already in the environment.

    Raises rather than guessing. A wrong team here would be invisible downstream -- every
    number would compute cleanly against sixteen players who are not yours.
    """
    me = swid if swid is not None else _swid()
    if not me:
        raise LookupError("ESPN_SWID is not set, so no team can be identified as yours")
    for team in league.teams:
        owners = team.owners or []
        ids = {str(o.get("id", o) if isinstance(o, dict) else o).strip("{}").upper()
               for o in owners}
        if me in ids:
            return team
    raise LookupError(
        f"no team in this league is owned by the configured SWID "
        f"(checked {len(league.teams)} teams)")


def from_team(team: Any) -> pl.DataFrame:
    """One row per rostered player, straight off ESPN and not yet joined to anything."""
    rows = []
    for p in team.roster:
        name = str(getattr(p, "name", "") or "")
        rows.append({
            "player": name,
            "key": player_key(name),
            "pos": str(getattr(p, "position", "") or ""),
            "espn_id": int(getattr(p, "playerId", 0) or 0),
            "nfl_team": str(getattr(p, "proTeam", "") or ""),
            "slot": str(getattr(p, "lineupSlot", "") or ""),
            "injury_status": str(getattr(p, "injuryStatus", "") or ""),
            # ESPN's own two projections. Their *ratio* is the only season-level availability
            # signal in the payload -- see `espn_games` below.
            "espn_avg": float(getattr(p, "projected_avg_points", 0.0) or 0.0),
            "espn_total": float(getattr(p, "projected_total_points", 0.0) or 0.0),
        })
    return pl.DataFrame(rows, schema={
        "player": pl.Utf8, "key": pl.Utf8, "pos": pl.Utf8, "espn_id": pl.Int64,
        "nfl_team": pl.Utf8, "slot": pl.Utf8, "injury_status": pl.Utf8,
        "espn_avg": pl.Float64, "espn_total": pl.Float64})


def availability(espn: pl.DataFrame) -> pl.DataFrame:
    """How many games ESPN expects each player to play, and whether that is all of them.

    **`injury_status` does not carry suspensions.** Josh Jacobs was served a six-game ban and
    ESPN still reported him `DAY_TO_DAY`, so a roster trusting that field recommended starting
    him in week 1. The ban *is* in the payload, just not in the field with the obvious name:
    his `projected_total_points` divided by his `projected_avg_points` is 11.0 where every
    other player on the roster is 17.0.

    So availability is read off the ratio rather than the designation. `full` is the largest
    implied count on the roster rather than a hard-coded 17, because the number of games in a
    season is ESPN's to change and not ours to restate.

    This is a **season-level** signal and says nothing about *which* games are missed. It is
    therefore only safe in one direction: a player short of a full slate is not
    presumed available, and no claim is made about when a full-slate player plays.
    """
    if espn.is_empty():
        return espn.with_columns(pl.lit(None, dtype=pl.Float64).alias("espn_games"),
                                 pl.lit(0, dtype=pl.Int64).alias("missing_games"),
                                 pl.lit(True).alias("available"))
    games = pl.when(pl.col("espn_avg") > 0).then(
        pl.col("espn_total") / pl.col("espn_avg")).otherwise(None)
    out = espn.with_columns(games.alias("espn_games"))
    full = out["espn_games"].max()
    if full is None:                      # ESPN published no projections at all
        return out.with_columns(pl.lit(0, dtype=pl.Int64).alias("missing_games"),
                                pl.lit(True).alias("available"))
    missing = (pl.lit(float(cast(float, full))) - pl.col("espn_games")).round(0)
    return out.with_columns(
        missing.fill_null(0.0).cast(pl.Int64).alias("missing_games")
    ).with_columns((pl.col("missing_games") < 1).alias("available"))


def build(espn: pl.DataFrame, board: pl.DataFrame) -> pl.DataFrame:
    """Attach the board's projection to each rostered player and take its moments.

    Joined on `player_key` rather than on ESPN's player id, because the board is built from
    nflverse ids and the two id spaces do not meet. The key is the same normaliser the rest
    of the repo matches names with.
    """
    cols = [c for c in ("proj_blend", "proj_ppg", "xfp_per_game", "ecr", "adp", "vor")
            if c in board.columns]
    b = (board.with_columns(pl.col("player").map_elements(player_key, return_dtype=pl.Utf8)
                            .alias("key"))
              .select(["key", *cols])
              .unique(subset="key", keep="first"))
    joined = espn.join(b, on="key", how="left")

    # `projected` before `moments`, because moments fills a missing projection with 0.0 and
    # that zero is indistinguishable from a real one once it has been written.
    have = pl.any_horizontal(*[pl.col(c).is_not_null() for c in cols]) if cols else pl.lit(False)
    out = moments(availability(joined).with_columns(have.alias("projected")))
    return out.with_columns(
        pl.when(pl.col("projected")).then(pl.col("mu")).otherwise(None).alias("mu"),
        pl.when(pl.col("projected")).then(pl.col("sd")).otherwise(None).alias("sd"),
        (pl.col("slot") != BENCH).alias("starting"),
    ).sort(["projected", "mu"], descending=[True, True], nulls_last=True)


class Lock(NamedTuple):
    """The Sunday question: what is set, what should start, and what the difference is worth.

    Here rather than in `hub.publish` because it is a *decision*, not a rendering. It lived in
    the site writer for one evening, which meant the only way to ask it was to publish a
    website, and it could not be tested without a parquet on disk.
    """
    set_total: float | None
    best_total: float | None
    gain: float | None
    start: list[str]                 # who should start, availability respected
    bench: list[str]                 # set starters who should not
    withheld: list[str]              # excluded as unavailable, whatever they project


def lock(df: pl.DataFrame, *, include_unavailable: bool = False) -> Lock:
    """Compare the lineup as set against the best one that could be set.

    **Availability is applied before the comparison, not after.** The alternative -- rank on
    projection and caveat the result -- is what produced a recommendation to start a suspended
    player, and a caveat under a number does not stop the number being read.
    """
    from hub.season.lineup import NoLegalLineup, TooManyLineups, best_by_points

    proj = df.filter(pl.col("projected"))
    pool = proj if include_unavailable or "available" not in proj.columns else proj.filter(
        pl.col("available"))
    withheld = ([] if pool.height == proj.height
                else sorted(set(proj["player"]) - set(pool["player"])))
    if pool.is_empty():
        return Lock(None, None, None, [], [], withheld)

    try:
        best = best_by_points(pool)["starters"]
    except (NoLegalLineup, TooManyLineups):
        # Withholding can make a thin roster unfillable, and a lock that raises would take
        # the whole slate down with it -- CLAUDE.md's degradation rule. Report the players
        # withheld, which is the actionable half, and no comparison.
        return Lock(None, None, None, [], [], withheld)
    set_total = float(proj.filter(pl.col("starting"))["mu"].sum())
    best_total = float(best["mu"].sum())
    start, was = set(best["player"]), set(proj.filter(pl.col("starting"))["player"])
    return Lock(set_total, best_total, best_total - set_total,
                sorted(start - was), sorted(was - start), withheld)


def fetch(board: pl.DataFrame | None = None) -> pl.DataFrame:  # pragma: no cover - network
    """The current roster, projected. Reads ESPN and the board; neither is cached here."""
    from hub.draft.board import last_good
    from hub.fetch.espn import league_settings
    if board is None:
        board, _age = last_good()
    return build(from_team(mine(league_settings().league)), board)


def write(df: pl.DataFrame, path: Path | None = None) -> Path:
    """Persist, refusing to write a frame `hub.season.lineup` could not read."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"roster is missing {missing}; lineup.py requires {list(REQUIRED)}")
    p = path or ROSTER_PARQUET
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return p


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.season.roster",
        description="Write data/processed/roster.parquet from your ESPN team and the board.")
    ap.add_argument("--write", action="store_true", help="persist; otherwise just print")
    ap.add_argument("--out", default=None, help="override the output path")
    a = ap.parse_args(list(argv) if argv is not None else None)

    try:
        df = fetch()
    except Exception as e:                       # graceful degradation, per CLAUDE.md
        if ROSTER_PARQUET.exists():
            print(f"hub.season.roster: ESPN unreachable ({e}); "
                  f"serving last-good {ROSTER_PARQUET}", file=sys.stderr)
            df = pl.read_parquet(ROSTER_PARQUET)
        else:
            print(f"hub.season.roster: {e}", file=sys.stderr)
            return 1

    n_proj = int(df["projected"].sum())
    print(f"  {df.height} rostered, {n_proj} projected, "
          f"{df.height - n_proj} carrying no projection (K and D/ST are out of scope)")
    for r in df.iter_rows(named=True):
        mu, sd = r.get("mu"), r.get("sd")
        shown = f"{mu:>5.1f} +/- {sd:4.1f}" if mu is not None else "    not projected"
        flag = "S" if r["starting"] else " "
        inj = "" if r["injury_status"] in ("ACTIVE", "NORMAL", "") else f"  {r['injury_status']}"
        miss = ("" if r.get("available", True)
                else f"  MISSING {r['missing_games']} GAMES")
        print(f"  {flag} {r['pos']:<4} {r['player']:<24} {shown}{inj}{miss}")

    lk = lock(df)
    if lk.gain is not None:
        print(f"\n  set {lk.set_total:.1f}  ->  best {lk.best_total:.1f}  "
              f"({lk.gain:+.1f} a week)")
        for p in lk.start:
            print(f"    START  {p}")
        for p in lk.bench:
            print(f"    SIT    {p}")
    if lk.withheld:
        print(f"  withheld as unavailable: {', '.join(lk.withheld)}")
        print("  ESPN's injury field does not carry suspensions; its games projection does.")

    if a.write:
        print(f"  wrote {write(df, Path(a.out) if a.out else None)}")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
