"""Does the lineup optimiser beat sorting a column?

`hub.season.lineup` enumerates every legal lineup and picks the one maximising win probability
against an assumed opponent. It is 97% covered by unit tests, which is a claim about its
internals and not about whether it is *right* -- the draft board was also well covered while
recommending a fourth quarterback.

So it gets a gate, in the sense `CONTEXT.md` now defines: a **model** is tested against the
simplest thing that already works. Here that is **start your highest projections** -- fill each
required slot with your best projected player, flex the best remaining. One sort, no search.

**Pre-registered before the numbers**, and asymmetric on purpose. The optimiser is the
complicated thing and the burden sits on it:

    CI excludes zero favouring the optimiser -> trust it; it sets your Week 1 lineup
    CI contains zero                         -> start your projections; the search buys nothing
    CI excludes zero favouring projections   -> the optimiser is harmful; remove it

The middle branch is the likely one and it has an action rather than being a disappointment
to explain away -- the same discipline that made P0b's null usable.

**Why this can be trusted where the draft could not.** At the draft you compete against ADP:
a live, liquid, aggregated forecast of exactly the question you are asking, which is why five
attempts to beat it failed. Nobody prices *"start Chase or Nacua this week given my roster"*.
There is no market on the other side of a lineup decision, so a simulator has no competitor
here -- only a simpler rule, which is what this measures against.

    uv run python -m hub.season.lineup_gate --seasons 2022,2023,2024,2025
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.draft.board import board_as_of
from hub.league import REG_SEASON_WEEKS, starting_lineup
from hub.models.experiment import paired_report, summarise, walk_forward_inputs
from hub.names import player_key

# What the optimiser is assumed to be playing against each week. The league's own weekly team
# scores would be better and are not reconstructable for a simulated roster, so this is a
# stated assumption rather than a measurement -- and it is held identical across both arms,
# so it cannot favour either.
OPP_MU = 110.0
OPP_SD = 25.0


def weekly_grid(names: Sequence[str], realised: pl.DataFrame,
                weeks: int = REG_SEASON_WEEKS) -> np.ndarray:
    """(roster, weeks) of realised points. A player with no row scored nothing that week."""
    keys = [player_key(n) for n in names]
    grid = np.zeros((len(keys), weeks))
    idx = {k: i for i, k in enumerate(keys)}
    for row in realised.filter(pl.col("player").is_in(keys)).iter_rows(named=True):
        w = int(row["week"])
        if 1 <= w <= weeks:
            grid[idx[row["player"]], w - 1] = float(row["points"])
    return grid


def projection_lineup_points(grid: np.ndarray, pos: Sequence[str],
                             proj: Sequence[float]) -> float:
    """The baseline: start your highest projections, every week, and take what they scored.

    The lineup is chosen *once* from projections -- it does not change week to week, because
    a projection does not. That is the honest version of the simple rule: a manager following
    it sets the same lineup every week unless somebody is hurt.

    The selection is `hub.draft.season.starting_lineup`, which `hub.season.weekly_gate` also
    calls: it was the same eleven lines in both files, kept in agreement by a docstring.
    """
    starters = starting_lineup(pos, proj)
    if not starters:
        return 0.0
    return float(grid[starters, :].sum() / grid.shape[1])


def optimiser_lineup_points(grid: np.ndarray, names: Sequence[str], pos: Sequence[str],
                            mu: Sequence[float], sd: Sequence[float],
                            opp_mu: float = OPP_MU, opp_sd: float = OPP_SD) -> float:
    """What `hub.season.lineup.optimize` picks, scored on what those players actually did.

    **It sees only projections.** An earlier version of this arm chose each week's lineup from
    the realised scores, which is not a comparison at all -- it measured the value of perfect
    foresight (+31 points a game) and would have licensed trusting an optimiser that was never
    tested. A gate whose treatment arm has information the control arm lacks cannot fail.

    Both arms here see the same `mu`. The only thing the optimiser has that sorting does not
    is `sd`: it maximises P(beat the opponent) rather than expected points, so it will start a
    lower-projected, higher-variance player when the matchup needs upside. That difference is
    the entire hypothesis under test.
    """
    from hub.season.lineup import NoLegalLineup, TooManyLineups, optimize

    players = pl.DataFrame({"player": list(names), "pos": list(pos),
                            "mu": [float(x) for x in mu], "sd": [float(x) for x in sd]})
    try:
        got = optimize(players, opp_mu, opp_sd)
    except (NoLegalLineup, TooManyLineups):
        return 0.0
    idx = [list(names).index(n) for n in got["starters"]["player"].to_list()]
    return float(grid[idx, :].sum() / grid.shape[1])


Roster = list[tuple[str, str, float, float]]   # (player, position, mu, sd)


def compare(rosters: dict[int, list[Roster]], realised: dict[int, pl.DataFrame],
            weeks: int = REG_SEASON_WEEKS) -> pl.DataFrame:
    """Paired: one row per roster. `rosters` maps a season to the rosters drafted in it.

    Pure -- frames and lists in, a frame out, no network -- so the statistics are testable
    without hitting nflverse. Same reason `backtest.compare` is.
    """
    rows = []
    for season in sorted(rosters):
        real = realised[season]
        for k, roster in enumerate(rosters[season]):
            names = [n for n, _, _, _ in roster]
            pos = [p for _, p, _, _ in roster]
            mu = [m for _, _, m, _ in roster]
            sd = [v for _, _, _, v in roster]
            grid = weekly_grid(names, real, weeks)
            rows.append({
                "season": season, "roster": k,
                "projection": projection_lineup_points(grid, pos, mu),
                "optimiser": optimiser_lineup_points(grid, names, pos, mu, sd),
            })
    out = pl.DataFrame(rows)
    return out.with_columns((pl.col("optimiser") - pl.col("projection")).alias("diff"))


def verdict(summary: dict[str, float]) -> str:
    """The pre-registered action, read off the interval. Fixed before the numbers."""
    if summary["lo"] > 0:
        return ("TRUST: accounting for variance beats sorting on projection alone. The "
                "optimiser sets the lineup.")
    if summary["hi"] < 0:
        return ("REMOVE: the optimiser is worse than sorting on projection. Enumerating "
                "every legal lineup to maximise a win probability actively costs points.")
    return ("START YOUR PROJECTIONS: variance-awareness buys nothing detectable. Absence "
            "of evidence, not evidence of equivalence.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.lineup_gate",
        description="Does the lineup optimiser beat starting your highest projections?")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--drafts", type=int, default=20,
                    help="rosters per season, drafted by the market arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--parameter-uncertainty", action="store_true",
                    help="add sigma_pos/sqrt(games) to sd -- the quantity ADR-0012 never "
                         "measured; see docs/parameter-uncertainty.md")
    a = ap.parse_args(argv)

    from hub.config import RosterConfig
    from hub.draft.backtest import market_strategy, play

    cfg = RosterConfig()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    rosters: dict[int, list] = {}

    boards, realised = walk_forward_inputs(
        seasons, board_as_of,
        on_season=lambda yr: print(f"  building the {yr} board as of {yr}-09-01 ..."))
    for yr in seasons:
        board = boards[yr]
        # mu and sd from the same object the simulator uses, so the optimiser is fed exactly
        # what it is fed live -- the fitted square-root spread law, per position.
        from hub.models.predict import moments
        pred = moments(board)
        if a.parameter_uncertainty:
            # ADR-0012 closed this gate because sd = k*sqrt(mu) is a deterministic increasing
            # function of the mean, so the optimiser was handed no variance to read. It then
            # withdrew its re-run clause because per-player *volatility* beyond the positional
            # constant is +/-9.3% and not estimable. How well we know a player's mean is a
            # different quantity: it is sigma_pos/sqrt(games), it is +36% at one game against
            # twelve, and it is not a function of mu. This is that clause, tested rather than
            # assumed.
            sig = {"QB": 7.10, "RB": 5.21, "WR": 5.11, "TE": 3.76}
            games = pred["games"].fill_null(1).cast(pl.Float64).to_numpy()
            se = np.array([sig.get(str(p), 5.06) for p in pred["pos"].to_list()]) \
                / np.sqrt(np.clip(games, 1.0, None))
            pred = pred.with_columns(
                (pl.col("sd") ** 2 + pl.Series("se2", se ** 2)).sqrt().alias("sd"))
        who = pred["player"].to_list()
        proj_of = dict(zip(who, pred["mu"].fill_null(0.0).to_list(), strict=True))
        sd_of = dict(zip(who, pred["sd"].fill_null(0.0).to_list(), strict=True))
        made = []
        for k in range(a.drafts):
            names, pos = play(board, market_strategy(), my_slot=cfg.slot, teams=cfg.teams,
                              rounds=14, rng=np.random.default_rng(a.seed + 1000 * yr + k))
            made.append([(n, p, proj_of.get(n, 0.0), sd_of.get(n, 0.0))
                         for n, p in zip(names, pos, strict=True)])
        rosters[yr] = made

    paired = compare(rosters, realised)
    s = summarise(paired, seed=a.seed)
    for line in paired_report(s, arm_a="optimiser", arm_b="projections",
                              unit="points per game"):
        print(line)
    print(f"\n  {verdict(s)}")
    print("\n  Both arms see only projections. The optimiser's sole advantage is that it")
    print("  reads `sd` as well as `mu`, so it can start upside when the matchup wants it.")
    print("  Limitation: projections are static across the season, because weekly historical")
    print("  projections do not exist. This measures variance-awareness, not in-season news.")
    if a.out:
        paired.write_parquet(a.out)
        print(f"\n  wrote {paired.height} paired rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
