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

**It does not read ESPN.** `hub.fetch.espn` owns that vocabulary and hands back rows --
`my_team` for the identity, `roster_rows` for the seven fields. This module used to duck-type
the vendor object itself, which meant ESPN's season projection had two readers in two packages
that shared no code, and neither could see the other disagree.

**Kickers and defences carry no projection and say so.** The board covers
`config.DRAFTED_POSITIONS` -- QB, RB, WR, TE -- because that was the scope decision, so K and
D/ST join to nothing. They come through with `projected = False` rather than `mu = 0.0`
presented as a forecast: a zero that means "we do not model this" must not render as a zero
that means "we expect nothing".
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import polars as pl

from hub.fetch.espn import my_team, roster_rows
from hub.models.predict import blend as predict_blend
from hub.models.predict import moments
from hub.names import player_key
from hub.paths import ROSTER_PARQUET

# What `hub.season.lineup` requires, and therefore what this must always produce.
REQUIRED: tuple[str, ...] = ("player", "pos", "mu", "sd")

# ESPN's two slots a player cannot start from. Only the bench was named here, and
# "everything else is a starting slot of some kind" was false: a player the manager has
# placed on injured reserve read as started, so his projection counted toward the lineup as
# set. The two are not the same kind of thing and the code below keeps them apart --
# `available` is a judgment about who will play, read off ESPN's own projection; injured
# reserve is mechanical, and the league will not let the slot start whatever anyone thinks.
BENCH = "BE"
INJURED_RESERVE = "IR"
NOT_STARTING = frozenset({BENCH, INJURED_RESERVE})


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


def market(joined: pl.DataFrame) -> pl.DataFrame:
    """Rebuild `proj_blend` against ESPN's *current* projection, not the board's copy of it.

    The board is a draft-day artifact. `proj_ppg` in it is ESPN's season projection frozen at
    build time, and in-season that number moves when something happens to a player.

    Measured 2026-09-04, five days after the draft, across fourteen projected players: the
    **median absolute drift was 0.04 points** and thirteen of fourteen were inside 0.6. The
    exception was MarShawn Lloyd at **+2.91** -- ESPN had repriced him 5.10 -> 8.01 because
    Josh Jacobs, the back ahead of him in Green Bay, had been suspended six games. So this is
    not a stale-data sweep; it is inert everywhere except the one player whose situation
    actually changed, which is exactly the property that makes it safe.

    **It is a refresh, not a new signal.** The quantity is the one the board already uses,
    read now instead of on Wednesday, so there is nothing here for a screen to ask "is this
    real?" about. Note what it buys for free: this repo screened depth-chart movement as a
    signal and rejected it (`docs/depth-chart-signal.md`), and it does not need one --
    succession is already priced by the projection, and the only bug was discarding the price.

    The board's own `proj_ppg` remains the fallback, so a player ESPN does not project keeps
    the draft-day number rather than losing his projection entirely.
    """
    if "espn_avg" not in joined.columns or "proj_ppg" not in joined.columns:
        return joined
    # Only players the board already scoped. ESPN projects kickers and defences too, and
    # taking its number for them would quietly overturn the decision that they carry none.
    scoped = pl.col("projected") if "projected" in joined.columns else pl.lit(True)
    live = pl.when(scoped & (pl.col("espn_avg") > 0)).then(pl.col("espn_avg")).otherwise(None)
    return (joined.with_columns(pl.coalesce(live, pl.col("proj_ppg")).alias("proj_ppg"))
                  .with_columns(predict_blend()))


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
    #
    # It is also settled before `market`, and the order is load-bearing: `projected` means
    # "the *board* carried a projection", which is the scope decision -- QB, RB, WR, TE. ESPN
    # projects kickers and defences perfectly happily, so refreshing first and asking after
    # let a D/ST in at 4.6 points and put it in the lineup.
    have = pl.any_horizontal(*[pl.col(c).is_not_null() for c in cols]) if cols else pl.lit(False)
    scoped = availability(joined).with_columns(have.alias("projected"))
    out = moments(market(scoped))
    return out.with_columns(
        pl.when(pl.col("projected")).then(pl.col("mu")).otherwise(None).alias("mu"),
        pl.when(pl.col("projected")).then(pl.col("sd")).otherwise(None).alias("sd"),
        (~pl.col("slot").is_in(NOT_STARTING)).alias("starting"),
        # Whether he may occupy a Slot at all, which is not whether he is in one and not
        # whether we expect him to play. Three neighbouring ideas with three names: see
        # CONTEXT.md, which keeps them apart on purpose.
        (pl.col("slot") != INJURED_RESERVE).alias("can_start"),
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

    **And it is applied to both sides of it.** The set total used to sum every projected
    starter while the best total was computed over available players only, so a suspended
    player set as a starter carried his full projection into one side and none of the other.
    The lock printed `set 118.8 -> best 106.8 (-12.0 a week)` directly above `SIT Josh
    Jacobs`: the headline argued for keeping him, which is the same defect this function
    exists to prevent, wearing the number instead of the caveat. A player who cannot play
    scores nothing, so the lineup as set is worth what the rest of it is worth.
    """
    from hub.season.lineup import NoLegalLineup, TooManyLineups, best_by_points

    proj = df.filter(pl.col("projected"))
    # Injured reserve first, and `include_unavailable` does not reach it: the override exists
    # to see a number you are declining, and there is no number to decline on a slot the
    # league will not start. Guarded on presence because `roster.parquet` on disk predates
    # the column and the Sunday panel reads it.
    eligible = proj.filter(pl.col("can_start")) if "can_start" in proj.columns else proj
    pool = eligible if include_unavailable or "available" not in eligible.columns \
        else eligible.filter(pl.col("available"))
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
    # Priced over `pool`, so both totals count the same players. A withheld starter drops out
    # of the set lineup's value rather than out of one side of a subtraction.
    set_total = float(pool.filter(pl.col("starting"))["mu"].sum())
    best_total = float(best["mu"].sum())
    start, was = set(best["player"]), set(proj.filter(pl.col("starting"))["player"])
    return Lock(set_total, best_total, best_total - set_total,
                sorted(start - was), sorted(was - start), withheld)


def fetch(board: pl.DataFrame | None = None) -> pl.DataFrame:  # pragma: no cover - network
    """The current roster, projected. Reads ESPN and the board; neither is cached here."""
    from hub.draft.board import readable
    from hub.fetch.espn import league_settings
    if board is None:
        board, source = readable()
        print(f"  board: {source}")
    return build(roster_rows(my_team(league_settings().league)), board)


def write(df: pl.DataFrame, path: Path | None = None) -> Path:
    """Persist, refusing to write a frame `hub.season.lineup` could not read.

    A frame with the right columns and no rows is refused for the same reason a frame with
    the wrong columns is: nothing downstream can use it. `lock` cannot price an empty pool,
    the CLI has no lineup to print, and `hub.publish.roster` publishes an artifact of nobody
    -- and every one of those reads as "you have no roster" when what happened is that one
    ESPN sync came back empty. The last roster on disk is the better answer, and refusing
    here is what leaves it there.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"roster is missing {missing}; lineup.py requires {list(REQUIRED)}")
    if df.is_empty():
        raise ValueError("roster has no players; refusing to overwrite the last good one "
                         "with an empty league -- an ESPN sync returning nothing is a "
                         "failed sync, not an empty team")
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
        # Per player, because the two reasons are different and the note below is only true
        # of one of them. Printing it over an IR-ed player would explain the wrong thing.
        why = {r["player"]: ("on injured reserve" if r["slot"] == INJURED_RESERVE
                             else f"ESPN projects {r['missing_games']} missed games")
               for r in df.iter_rows(named=True)}
        print("\n  withheld -- cannot be started this week:")
        for pl_name in lk.withheld:
            print(f"    {pl_name}  ({why.get(pl_name, 'unavailable')})")
        if any(why.get(n) != "on injured reserve" for n in lk.withheld):
            print("  ESPN's injury field does not carry suspensions; its games projection "
                  "does.")

    if a.write:
        try:
            print(f"  wrote {write(df, Path(a.out) if a.out else None)}")
        except ValueError as e:
            # The refusal above, reported rather than raised. An empty sync has to leave the
            # last-good parquet where it is, and a traceback on a Sunday is the
            # operator-dependence CLAUDE.md warns about -- the same reason `fetch` degrades.
            print(f"hub.season.roster: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
