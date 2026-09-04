"""Set the lineup against the matchup you have, not against the scoreboard in general.

`docs/foundation-plan.md` 5.2 asks for "the ceiling preference from
`docs/championship-leverage.md`". That doc's ceiling argument rests on a premise that is
false for this league, so this module implements the corrected version and says why.

The doc states *"12 teams, 8 make playoffs, 3 weeks (15-17), no byes"*, reasons that
*"dP(champ)/d(regular-season win) is close to zero"*, and concludes *"never sacrifice
ceiling for a marginal regular-season win."* The live league, read from raw
`mSettings.scheduleSettings` and recorded in `docs/decisions.md`, is `playoffTeamCount: 6`
with two byes. Half the league misses entirely and the top two seeds skip a
single-elimination round, so regular-season wins carry real championship equity and the
blanket rule does not follow.

Inverting it into a blanket floor preference would be the same mistake pointed the other
way. The doc's own point 3 has it right -- variance preference is state-dependent and
sign-flips -- so the objective here is simply **P(win this matchup)**, and ceiling-or-floor
falls out of whether you are favoured. A heavy underdog takes the volatile player because
the mean alone loses; a heavy favourite refuses him because he is already past the line.

Head-to-head is what licenses this ahead of any playoff argument: surplus points above the
opponent's score are worth nothing, so expected points is the wrong objective on its own.

Two assumptions, both worth knowing before reading small differences as real:

- **Player scores are independent.** They are not -- a QB and his WR1 move together, which
  is exactly the correlation layer `championship-leverage.md` calls L1 and nothing here
  builds. Independence understates the variance of a stacked lineup, which biases this
  module against stacks in underdog weeks, the very weeks stacks are for.
- **Totals are normal.** Weekly fantasy totals are right-skewed. The approximation is
  decent for an eight-man sum and is what makes the objective a closed form.

    uv run python -m hub.season.lineup --opp-mu 112
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Iterator, Mapping, Sequence

import polars as pl

from hub.league import FLEX_FROM, FLEX_SLOTS, STARTERS
from hub.models.predict import group_sd
from hub.paths import ROSTER_PARQUET

# Exhaustive enumeration is exact and, for a real 14-to-16 man roster, takes milliseconds
# -- a few thousand legal lineups. The cap exists so that if a caller ever hands over
# something enormous the answer is an error rather than a quietly approximate search.
MAX_LINEUPS = 200_000


class NoLegalLineup(Exception):
    """The roster cannot fill every starting slot."""


class TooManyLineups(Exception):
    """Too many legal lineups to enumerate exactly."""


def win_probability(mu: float, sd: float, opp_mu: float, opp_sd: float) -> float:
    """P(my total beats theirs), both normal and independent.

    The difference of two normals is normal, so this is one CDF evaluation. Note it
    depends on the spreads only through the sum of their squares: when the means are equal
    it is exactly 0.5 no matter how volatile either side is.
    """
    var = float(sd) ** 2 + float(opp_sd) ** 2
    if var <= 0.0:
        return 1.0 if mu > opp_mu else (0.5 if mu == opp_mu else 0.0)
    return 0.5 * (1.0 + math.erf((float(mu) - float(opp_mu)) / math.sqrt(2.0 * var)))


def _legal_lineups(pos: Sequence[str], slots: Mapping[str, int],
                   flex_from: Sequence[str], max_lineups: int,
                   flex_slots: int = FLEX_SLOTS) -> Iterator[tuple[int, ...]]:
    """Every legal assignment of roster indices to starting slots, as index tuples.

    Yields combinations rather than permutations: which player fills which of two RB slots
    does not change the total, and enumerating both orderings would multiply the work by
    2*3! for nothing.
    """
    by_pos = {p: [i for i, q in enumerate(pos) if q == p] for p in set(pos)}
    per_slot = [list(itertools.combinations(by_pos.get(p, []), n)) for p, n in slots.items()]
    if any(not c for c in per_slot):
        raise NoLegalLineup(
            "roster cannot fill every slot: needs "
            + ", ".join(f"{n} {p}" for p, n in slots.items())
            + "; has " + ", ".join(f"{len(by_pos.get(p, []))} {p}" for p in slots))

    flex_pool = [i for p in flex_from for i in by_pos.get(p, [])]
    total = math.prod(len(c) for c in per_slot) * (math.comb(len(flex_pool), flex_slots) or 1)
    if total > max_lineups:
        raise TooManyLineups(
            f"{total} legal lineups exceeds max_lineups={max_lineups}. Exhaustive search "
            "is exact; refusing rather than returning an approximate lineup.")

    found = False
    for combo in itertools.product(*per_slot):
        base = tuple(i for c in combo for i in c)
        used = set(base)
        # Combinations of `flex_slots`, not one player: with two flex slots both can come
        # from the same position, and picking one candidate at a time cannot express that.
        # At flex_slots=1 this yields exactly `(*base, f)` per eligible f, as it always did.
        for extra in itertools.combinations([f for f in flex_pool if f not in used],
                                            flex_slots):
            found = True
            yield (*base, *extra)
    if not found:
        raise NoLegalLineup(
            f"fewer than {flex_slots} players are left over to fill the flex")


def _evaluate(players: pl.DataFrame, slots: Mapping[str, int] | None,
              flex_from: Sequence[str] | None, max_lineups: int,
              score, flex_slots: int = FLEX_SLOTS) -> dict:
    slots = dict(slots if slots is not None else STARTERS)
    flex_from = tuple(flex_from if flex_from is not None else FLEX_FROM)

    pos = players["pos"].to_list()
    mu = players["mu"].to_numpy()
    sd = players["sd"].to_numpy()
    # A quarterback and his own pass catchers move together (+0.23), so a lineup holding
    # both is more volatile than summed variances say. Ignoring it makes an 80% interval
    # cover 72.9% of the time -- see docs/correlation.md.
    teams = (players["nfl_team"].to_list() if "nfl_team" in players.columns
             else [None] * players.height)

    best, best_score = None, -math.inf
    for idx in _legal_lineups(pos, slots, flex_from, max_lineups, flex_slots):
        i = list(idx)
        m = float(mu[i].sum())
        s = group_sd((sd[k], pos[k], teams[k]) for k in i)
        v = score(m, s)
        if v > best_score:
            best, best_score = (i, m, s), v
    if best is None:
        raise NoLegalLineup("no legal lineup")

    i, m, s = best
    return {"starters": players[i], "mu": m, "sd": s, "bench": players[
        [j for j in range(players.height) if j not in set(i)]]}


def optimize(players: pl.DataFrame, opp_mu: float, opp_sd: float = 25.0,
             slots: Mapping[str, int] | None = None,
             flex_from: Sequence[str] | None = None,
             max_lineups: int = MAX_LINEUPS) -> dict:
    """The lineup with the highest chance of winning this specific matchup.

    `players` needs columns player, pos, mu, sd. `opp_mu`/`opp_sd` describe the opponent's
    projected total -- see `opponent_moments`.
    """
    got = _evaluate(players, slots, flex_from, max_lineups,
                    lambda m, s: win_probability(m, s, opp_mu, opp_sd))
    got["win_prob"] = win_probability(got["mu"], got["sd"], opp_mu, opp_sd)
    return got


def best_by_points(players: pl.DataFrame, slots: Mapping[str, int] | None = None,
                   flex_from: Sequence[str] | None = None,
                   max_lineups: int = MAX_LINEUPS) -> dict:
    """The naive lineup: highest projected total, matchup ignored.

    Kept as a first-class function rather than a test fixture because it is the baseline
    the optimizer has to beat, and because it is the opponent model.
    """
    return _evaluate(players, slots, flex_from, max_lineups, lambda m, _s: m)


def opponent_moments(players: pl.DataFrame, slots: Mapping[str, int] | None = None,
                     flex_from: Sequence[str] | None = None) -> dict:
    """What the opponent is likely to score, assuming they start their best projections.

    `docs/championship-leverage.md` names this model and flags it as unvalidated: it should
    be checked against actual historical lineups via `espn-api`, which has not been done.
    Assuming they optimise against *you* would model an opponent nobody in this league is.
    """
    got = best_by_points(players, slots, flex_from)
    return {"mu": got["mu"], "sd": got["sd"], "starters": got["starters"]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.lineup",
        description="Pick the lineup with the best chance of winning this week's matchup.")
    ap.add_argument("--roster", default=str(ROSTER_PARQUET),
                    help="player, pos, mu, sd")
    ap.add_argument("--opp-mu", type=float, required=True,
                    help="opponent's projected total")
    ap.add_argument("--opp-sd", type=float, default=25.0)
    ap.add_argument("--include-unavailable", action="store_true",
                    help="optimise over players ESPN projects to miss games")
    a = ap.parse_args(argv)

    # Nothing in this repo writes a roster yet, so the default path is absent until after
    # the draft. It used to arrive as a bare polars FileNotFoundError traceback, which tells
    # an operator neither what the file is nor that it is optional.
    import pathlib
    path = pathlib.Path(a.roster)
    if not path.exists():
        print(f"hub.season.lineup: no roster at {path}.\n"
              "  It wants one row per rostered player, with columns: player, pos, mu, sd.\n"
              "  Nothing in this repo writes one yet -- see docs/gaps.md.\n"
              "  Not on any critical path: ADR-0012 measured this optimiser at +0.00 points\n"
              "  a game against simply starting your highest projections.", file=sys.stderr)
        return 1
    players = pl.read_parquet(path)
    missing = {"player", "pos", "mu", "sd"} - set(players.columns)
    if missing:
        print(f"hub.season.lineup: roster is missing {sorted(missing)}", file=sys.stderr)
        return 1
    # Injured reserve is not a lineup decision. The league will not start the slot whatever
    # ESPN still projects for him, so this is not behind `--include-unavailable`: that
    # override exists to see a number you are declining, and there is no number to decline
    # here. `hub.season.roster.lock` draws the same line for the same reason.
    if "can_start" in players.columns:
        out = players.filter(pl.col("can_start"))
        if out.height < players.height:
            gone = sorted(set(players["player"]) - set(out["player"]))
            print(f"  withholding {', '.join(gone)} -- on injured reserve")
        players = out
    # A player ESPN does not expect to play cannot be optimised into a lineup, whatever he
    # projects. Season-long projections are availability-blind, and this optimiser maximises
    # over the roster -- so an unavailable player with a high `mu` is exactly the row it
    # reaches for. See `hub.season.roster.availability`.
    if "available" in players.columns and not a.include_unavailable:
        out = players.filter(pl.col("available"))
        if out.height < players.height:
            gone = sorted(set(players["player"]) - set(out["player"]))
            print(f"  withholding {', '.join(gone)} -- ESPN projects them to miss games "
                  f"(--include-unavailable to override)")
        players = out
    try:
        got = optimize(players, opp_mu=a.opp_mu, opp_sd=a.opp_sd)
    except (NoLegalLineup, TooManyLineups) as e:
        print(f"hub.season.lineup: {e}", file=sys.stderr)
        return 1

    pts = best_by_points(players)
    print(f"  vs opponent projected {a.opp_mu:.1f} +/- {a.opp_sd:.1f}")
    for r in got["starters"].iter_rows(named=True):
        print(f"    {r['pos']:<3} {r['player']:<24} {r['mu']:>5.1f} +/- {r['sd']:.1f}")
    print(f"  projected {got['mu']:.1f} +/- {got['sd']:.1f}  "
          f"-> win {got['win_prob']:.1%}")

    swap = set(pts["starters"]["player"].to_list()) - set(got["starters"]["player"].to_list())
    if swap:
        base = win_probability(pts["mu"], pts["sd"], a.opp_mu, a.opp_sd)
        print(f"  max-points lineup would start {', '.join(sorted(swap))} for "
              f"{pts['mu']:.1f} projected and win {base:.1%} "
              f"({got['win_prob'] - base:+.1%})")
    else:
        print("  same as the max-points lineup this week -- the matchup does not change it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
