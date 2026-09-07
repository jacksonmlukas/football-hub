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

from hub.cli import unavailable
from hub.draft.backtest import stamped_for_publication
from hub.draft.board import board_as_of
from hub.league import REG_SEASON_WEEKS, starting_lineup
from hub.models.experiment import (
    Actions,
    gate,
    paired_report,
    per_season,
    summarise,
    walk_forward_inputs,
)
from hub.names import player_key

NOT_FITTED_BECAUSE = (
    "the walk-forward gate for the lineup optimiser. OPP_MU and OPP_SD describe the "
    "*opponent* a simulated week is played against -- a fixture for scoring two arms against "
    "each other, not an input any published prediction can reach. "
)

# What the optimiser is assumed to be playing against each week. The league's own weekly team
# scores would be better and are not reconstructable for a simulated roster, so this is a
# stated assumption rather than a measurement -- and it is held identical across both arms,
# so it cannot favour either.
OPP_MU = 110.0
OPP_SD = 25.0

# This gate's own unit, named once so the effect and the ceiling printed under it cannot end
# up quoted in two different ones.
UNIT = "points per game"

# What this gate's ceiling *is*, spelled where it is printed. The draft backtest's ceiling is
# a drafter who knew the season and the weekly gate's is a perfect weekly projection; this
# one is a perfect spread over a projection both arms already share. Three gates, three
# questions, three harnesses -- and `docs/gate-power.md` stage 2 compares each gate's MDE
# against its own ceiling and never against another's, so the line says which one it is
# rather than leaving a reader to take three numbers in three units for one quantity.
CEILING_ARM = "a perfect spread, not foresight"


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


def variance_oracle_points(grid: np.ndarray, names: Sequence[str], pos: Sequence[str],
                           mu: Sequence[float], opp_mu: float = OPP_MU,
                           opp_sd: float = OPP_SD) -> float:
    """The optimiser given the spread it is guessing at, and nothing else.

    **The ceiling for this gate, and deliberately not full foresight.** Both arms here already
    see the same `mu`; the optimiser's only advantage is that it also reads `sd`. So the
    largest effect this gate could show is what a *perfect* `sd` buys, holding the projection
    fixed -- and an arm that also knew `mu` would be bounding a different question, the one
    `optimiser_lineup_points` records an earlier version of this gate accidentally measuring
    at +31 points a game (#43).

    The true spread is the realised week-to-week standard deviation, which is what `sd` is an
    estimate of. `mu` is passed through untouched, so this arm and the treatment arm differ in
    exactly one input.

    A one-week grid has no spread to measure, so its sd is zero rather than undefined: with a
    single observation there is nothing for a variance-aware rule to know, which is the
    honest reading and not a degenerate one.
    """
    spread = grid.std(axis=1, ddof=0) if grid.shape[1] > 1 else np.zeros(grid.shape[0])
    return optimiser_lineup_points(grid, names, pos, mu,
                                   [float(x) for x in spread], opp_mu, opp_sd)


def foresight_lineup_points(grid: np.ndarray, pos: Sequence[str]) -> float:
    """The baseline rule handed a perfect projection: start the season's actual best.

    Not this gate's ceiling -- it is here so the two cannot be quietly interchanged. It knows
    `mu` as well as `sd`, so it bounds "what is a lineup worth" rather than "what is knowing
    the spread worth", and it is strictly the larger of the two. A ceiling that silently
    became this one would make the gate look powered when it was not.

    Chosen once from realised season means, the same shape as `projection_lineup_points`, so
    the only difference between them is the quality of the projection.
    """
    return projection_lineup_points(grid, pos, [float(x) for x in grid.mean(axis=1)])


Roster = list[tuple[str, str, float, float]]   # (player, position, mu, sd)


def compare(rosters: dict[int, list[Roster]], realised: dict[int, pl.DataFrame],
            weeks: int = REG_SEASON_WEEKS, *, ceiling: bool = False) -> pl.DataFrame:
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
            row = {
                "season": season, "roster": k,
                "projection": projection_lineup_points(grid, pos, mu),
                "optimiser": optimiser_lineup_points(grid, names, pos, mu, sd),
            }
            if ceiling:
                row["oracle"] = variance_oracle_points(grid, names, pos, mu)
            rows.append(row)
    out = pl.DataFrame(rows)
    out = out.with_columns((pl.col("optimiser") - pl.col("projection")).alias("diff"))
    if ceiling:
        # Its own column rather than its own frame, because unlike the draft backtest this
        # gate's paired frame is not pinned by a digest -- and the arms share a roster, so
        # splitting them would mean rebuilding the same grid twice.
        out = out.with_columns((pl.col("oracle") - pl.col("projection")).alias("ceiling_diff"))
    return out


def ceiling_report(summary: dict[str, float], paired: pl.DataFrame) -> list[str]:
    """The ceiling beside the effect, and a loud line when it does not bound it.

    Nothing at all when the frame carries no ceiling, which is how a run without `--ceiling`
    prints exactly what it printed before. Same shape `paired_report` uses for `mde`: a field
    nothing computed prints nothing, because a `nan` set against a unit reads as a
    measurement and silence does not.

    **A ceiling below the effect it is meant to bound is not a tight result, it is a broken
    one** -- either the oracle arm is not reading the realised spread or the arm under test is
    being scored on something else. Said loudly rather than published quietly, because the
    number's whole use is as an upper bound and a bound that does not bound reads exactly
    like a tight one. `backtest.with_ceiling` says the same thing for the draft gate; the two
    are separate because the sentence naming *which* ceiling this is differs, and #135's
    shared gate protocol is where they become one.

    Lines rather than prints, so the rule is reachable without the network `main` needs --
    the same reason `paired_report` returns lines and `publish_paired` below returns its own.
    """
    if "ceiling_diff" not in paired.columns:
        return []
    top = float(np.asarray(paired["ceiling_diff"].to_numpy()).mean())
    out = [f"  ceiling ({CEILING_ARM}) {top:+.2f} {UNIT}, measured on this gate's own "
           f"harness -- not comparable with another gate's"]
    if top < summary["mean"]:
        out.append(f"\n  CEILING BELOW THE EFFECT: {top:+.2f} < {summary['mean']:+.2f}. One "
                   f"of the two is measuring something the other is not; do not read the "
                   f"interval above as bounded.")
    return out


def publish_paired(paired: pl.DataFrame, path: str) -> str:
    """Write the paired rows carrying what produced them, and return the line a reader gets.

    Until this landed, this gate wrote a bare parquet -- no configuration digest, no data
    digest -- so two runs over different archives were indistinguishable after the fact.
    That is the silent case the pinning layer exists to remove (issue #71), left standing on
    the gate whose verdict ADR-0012 defers to.

    The rule is `backtest.stamped_for_publication` itself and not a second copy that agrees
    with it. This module already records what the other choice costs: `cohort` is imported
    rather than restated because the recipe had been written out in two places, and "a
    formula copied by hand into two places is one that eventually differs in one". It lives
    under `hub.draft` today because the draft gate needed it first; issue #135's shared gate
    protocol is where it stops being addressed through a draft module.

    A function rather than three lines under `if a.out:`, because reaching those needs a
    network, and a stamping rule only reachable behind a network is a stamping rule with no
    test. `stamped_for_publication` names the same reason for the same shape.
    """
    stamped, said = stamped_for_publication(paired)
    stamped.write_parquet(path)
    return f"\n  wrote {stamped.height} paired rows to {path}\n{said}"


# The pre-registered actions, fixed before the numbers and quoted in this module's own
# docstring above. The rule choosing between them is `experiment.gate` -- ADR-0019.
ACTIONS = Actions(
    adopt="TRUST: accounting for variance beats sorting on projection alone. The optimiser "
          "sets the lineup.",
    remove="REMOVE: the optimiser is worse than sorting on projection. Enumerating every "
           "legal lineup to maximise a win probability actively costs points.",
    show="START YOUR PROJECTIONS: variance-awareness buys nothing detectable.")


def verdict(summary: dict[str, float], seasons: pl.DataFrame) -> tuple[str, str]:
    """The pre-registered action, read off the interval *and* the seasons.

    The every-season half is new and does not move what ADR-0012 recorded: that interval is
    [-0.00, +0.00], which contains zero and is the middle branch under either form.
    """
    return gate(summary, seasons, ACTIONS)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.lineup_gate",
        description="Does the lineup optimiser beat starting your highest projections?")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--drafts", type=int, default=20,
                    help="rosters per season, drafted by the market arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ceiling", action="store_true",
                    help="also score the variance oracle -- the same optimiser handed the "
                         "realised spread, the projection untouched -- and report the "
                         "largest effect this gate could show. `docs/gate-power.md` stage 2")
    ap.add_argument("--parameter-uncertainty", action="store_true",
                    help="add sigma_pos/sqrt(games) to sd -- the quantity ADR-0012 never "
                         "measured; see docs/parameter-uncertainty.md")
    a = ap.parse_args(argv)

    from hub.draft.cohort import cohort

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    rosters: dict[int, list] = {}

    try:
        boards, realised = walk_forward_inputs(
            seasons, board_as_of,
            on_season=lambda yr: print(f"  building the {yr} board as of {yr}-09-01 ..."))
    except Exception as e:
        return unavailable("hub.season.lineup_gate", "the boards the rosters are drafted from", e)
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
        # The same Cohort the weekly gate scores, from the same seeded recipe. Both used to
        # write it out, and a formula copied by hand into two places is one that eventually
        # differs in one.
        drafted = cohort(board, yr, drafts=a.drafts, seed=a.seed)
        who_at = board["player"].to_list()
        made = [[(who_at[i], drafted.pos[i], proj_of.get(who_at[i], 0.0),
                  sd_of.get(who_at[i], 0.0)) for i in roster]
                for roster in drafted.rosters]
        rosters[yr] = made

    paired = compare(rosters, realised, ceiling=a.ceiling)
    s = summarise(paired, seed=a.seed)
    for line in [*paired_report(s, arm_a="optimiser", arm_b="projections", unit=UNIT),
                 *ceiling_report(s, paired)]:
        print(line)
    print(f"\n  {verdict(s, per_season(paired))[1]}")
    print("\n  Both arms see only projections. The optimiser's sole advantage is that it")
    print("  reads `sd` as well as `mu`, so it can start upside when the matchup wants it.")
    print("  Limitation: projections are static across the season, because weekly historical")
    print("  projections do not exist. This measures variance-awareness, not in-season news.")
    if a.out:
        print(publish_paired(paired, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
