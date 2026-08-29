"""Does a lineup set off the **Weekly projection** beat one set off weekly consensus rank?

This is **the** gate for `hub.models.weekly`.
[ADR-0015](../../../docs/adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md)
records why it has to be a decision and not an accuracy test: six seasons of historical
weekly FantasyPros consensus ship `ecr` and no `r2p_pts`, so the only incumbent worth beating
exists as a *ranking* and there is nothing to take a paired error against.

`hub.season.lineup_gate` asks a different question on the same harness. That one varies the
**search** over identical projections and returned a structural zero, because `sd = k*sqrt(mu)`
makes spread a deterministic function of the mean and the optimiser was handed no variance to
read (ADR-0012). This one varies the **projection** under an identical search, which is the
axis still open. Its own docstring names the gap this fills: *"projections are static across
the season, because weekly historical projections do not exist."* They do now.

THE ARMS, and they see the same information.

    consensus  fill each slot with your highest-ranked rostered player by `weekly-op` ECR
    weekly     fill each slot by `hub.models.weekly`'s projection for that week

The search is fixed at *start your highest* in both, per ADR-0012. Both are restricted to the
same roster and score against the same realised grid.

THE RULES, pre-registered in `docs/weekly-projection-plan.md` before this ran:

  * **Weeks 1-14**, the fantasy regular season. 15-17 reported apart and never pooled.
  * **Paired by roster-week**, with a **cluster bootstrap by roster** -- a roster's fourteen
    weeks are not fourteen observations, and quoting the raw n would inflate the interval's
    precision by roughly the square root of fourteen.
  * **A rostered player missing from that week's consensus page is ranked last**, because the
    absence is the incumbent saying "do not start him" and it is real information it has. A
    missing *projection* is likewise a zero. Handing either arm information the other lacks is
    the defect that made the first `lineup_gate` unable to fail.
  * **Inactive weeks score zero**, not missing: starting a player who did not play is the most
    expensive weekly mistake there is and an honest lineup score has to eat it.

    uv run python -m hub.season.weekly_gate --run
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.draft.season import FLEX_FROM, FLEX_SLOTS, STARTERS
from hub.models.experiment import BOOTSTRAP

# The fantasy regular season. 15-17 is the playoffs, reported apart; 18 is meaningless.
GATE_WEEKS: tuple[int, ...] = tuple(range(1, 15))

# A player the consensus page does not list is ranked behind every player it does.
UNRANKED = -1e9

# Above this share of roster-weeks lost to a join failure, the run is VOID rather than
# reported. Pre-registered in docs/weekly-projection-plan.md, and tight because the error is
# *directional*: an unmatched player is ranked last, so a join failure does not add noise, it
# forces a bench on the incumbent's arm and biases the result toward us.
VOID_FLOOR = 0.02


def starters_by_score(pos: Sequence[str], score: np.ndarray) -> list[int]:
    """Indices of the lineup: fill each required slot with the best available, then the flex.

    The same rule as `lineup_gate.projection_lineup_points`, applied to **one week's** scores
    rather than to a season-long projection. That difference is the entire subject here.
    """
    order = np.argsort(-np.asarray(score, dtype=float))
    counts: dict[str, int] = {}
    starters: list[int] = []
    flex: list[int] = []
    for j in order:
        i = int(j)
        p = pos[i]
        if p in STARTERS and counts.get(p, 0) < STARTERS[p]:
            counts[p] = counts.get(p, 0) + 1
            starters.append(i)
        elif p in FLEX_FROM:
            flex.append(i)
    starters.extend(flex[:FLEX_SLOTS])
    return starters


WAIVER_LOOK = 15


def lineup_projection(roster: Sequence[int], pos: Sequence[str],
                      score: np.ndarray) -> float:
    """What this roster's best legal lineup *projects* to score, by this arm's own numbers."""
    idx = starters_by_score([pos[i] for i in roster], score[list(roster)])
    return float(sum(score[roster[j]] for j in idx))


def waiver_swap(roster: list[int], pool: list[int], pos: Sequence[str],
                score: np.ndarray, starters: int,
                *, look: int = WAIVER_LOOK) -> tuple[int, int] | None:
    """One add/drop for a week, chosen by **what it does to the starting lineup**.

    The obvious rule -- add the highest-scoring free agent, drop the lowest-scoring bench
    player -- is wrong, and wrong in a way that only bites one arm. Absolute weekly points are
    much larger at quarterback than anywhere else, so it adds a backup quarterback every week:
    the first run of this experiment picked up Russell Wilson at a projected 24.4 (he scored
    5.1), Marcus Mariota at 16.5 (-2.1), Jayden Daniels at 18.5 (2.7), for a mean add of 19.6
    projected against 13.7 realised. You start one quarterback. A second is a roster spot spent
    on somebody who will never play.

    Consensus rank does not make that mistake, because a weekly *ranking* already prices
    positional scarcity -- so the naive rule handed the incumbent a free win that had nothing
    to do with either arm's forecasting. Scoring the swap by the projected starting lineup
    removes it: a backup quarterback cannot improve a lineup whose quarterback slot is already
    filled by someone better.

    Both arms run this identically over an identical pool. Only `score` differs.
    """
    from hub.draft.season import STARTERS
    if not pool or len(roster) <= starters:
        return None
    held: dict[str, int] = {}
    for i in roster:
        held[pos[i]] = held.get(pos[i], 0) + 1
    droppable = [i for i in roster
                 if pos[i] not in STARTERS or held.get(pos[i], 0) > STARTERS[pos[i]]]
    if not droppable:
        return None

    base = lineup_projection(roster, pos, score)
    best, gain = None, 0.0
    for add in sorted(pool, key=lambda i: score[i], reverse=True)[:look]:
        for drop in droppable:
            trial = [i for i in roster if i != drop] + [add]
            lift = lineup_projection(trial, pos, score) - base
            if lift > gain:
                best, gain = (add, drop), lift
    return best


def season_points(realised: np.ndarray, pos: Sequence[str], score: np.ndarray,
                  roster: Sequence[int], pool: Sequence[int], weeks: Sequence[int],
                  *, churn: bool = False, addable: np.ndarray | None = None,
                  add_score: np.ndarray | None = None) -> dict[int, float]:
    """Points per week for one arm, optionally streaming a player a week.

    `realised`, `score` and `pos` are over the whole **universe** of board players, and a
    roster is a list of indices into it that evolves. With `churn=False` the roster never
    changes and this is the frozen-roster gate.

    `add_score` is what the **waiver decision** reads, where `score` is what the **lineup**
    reads. They differ when adds are ranked by a lower confidence bound: a waiver pick is the
    maximum over hundreds of candidates and so is biased upward, while a lineup is a choice
    among players you already hold and has no such selection. Defaults to `score`, which is
    the behaviour every earlier run had.

    `addable` masks the *pool* -- never the roster -- to the players both arms can score that
    week. Restricting it is what keeps this able to fail: consensus ranks only 35.8% of a
    935-player free-agent pool, so an unmasked pool would let the arm under test add six
    hundred players the incumbent cannot score at all. Masking the roster instead would bench
    a rostered player for being unranked, which is a different rule and the wrong one.
    """
    from hub.draft.season import STARTERS
    need = sum(STARTERS.values())
    cur, free = list(roster), list(pool)
    decide = score if add_score is None else add_score
    out: dict[int, float] = {}
    for w in weeks:
        col = score[:, w - 1]
        if churn:
            eligible = ([i for i in free if addable[i, w - 1]]
                        if addable is not None else free)
            swap = waiver_swap(cur, eligible, pos, decide[:, w - 1], need)
            if swap is not None:
                add, drop = swap
                cur = [i for i in cur if i != drop] + [add]
                free = [i for i in free if i != add] + [drop]
        idx = starters_by_score([pos[i] for i in cur], col[cur])
        chosen = [cur[j] for j in idx]
        out[w] = float(realised[chosen, w - 1].sum()) if chosen else 0.0
    return out


def compare_universe(rosters: dict[int, list[list[int]]], pos: dict[int, Sequence[str]],
                     realised: dict[int, np.ndarray], consensus: dict[int, np.ndarray],
                     weekly: dict[int, np.ndarray], pool: dict[int, list[list[int]]],
                     covered: set[tuple[int, int]],
                     addable: dict[int, np.ndarray] | None = None,
                     weeks: Sequence[int] = GATE_WEEKS,
                     *, churn: bool = False, z: float = 0.0,
                     se: dict[int, np.ndarray] | None = None) -> pl.DataFrame:
    """One row per roster-week, on the universe representation, with or without churn.

    `churn=False` reproduces `compare` exactly -- and it is checked against the published
    frozen result rather than assumed, because a churn number is only worth reading if the
    machinery underneath it gives the same answer as the machinery that produced the last one.
    """
    rows = []
    for season in sorted(rosters):
        wks = [w for w in weeks if (season, w) in covered]
        add = None if addable is None else addable[season]
        # Only the arm under test gets a lower confidence bound. Consensus is a *ranking* with
        # no uncertainty attached to subtract, so there is nothing to hand it -- and this is
        # our arm being more careful with its own estimate, not the incumbent being handicapped.
        lcb = (weekly[season] - z * se[season]) if (z and se is not None) else None
        for k, roster in enumerate(rosters[season]):
            # The pool is per draft: who is a free agent depends on what the other eleven
            # teams took in that room.
            free = pool[season][k]
            a = season_points(realised[season], pos[season], consensus[season], roster,
                              free, wks, churn=churn, addable=add)
            b = season_points(realised[season], pos[season], weekly[season], roster,
                              free, wks, churn=churn, addable=add, add_score=lcb)
            for w in wks:
                rows.append({"season": season, "roster": k, "week": w,
                             "consensus": a[w], "weekly": b[w]})
    out = pl.DataFrame(rows)
    if out.is_empty():
        return out
    return out.with_columns((pl.col("weekly") - pl.col("consensus")).alias("diff"))


def cluster_bootstrap(paired: pl.DataFrame, *, bootstrap: int = BOOTSTRAP,
                      seed: int = 0) -> dict[str, float]:
    """Mean paired difference and an interval, resampling **rosters** rather than rows.

    A roster's fourteen weeks share its players, its bye and its draft, so they are one
    observation with fourteen readings. Resampling rows would treat them as fourteen and
    report an interval about the square root of fourteen too narrow -- protocol item 3, which
    turned noise into an apparent 4-sigma result once already.
    """
    if paired.is_empty():
        return {"n": 0, "clusters": 0, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_better": float("nan")}
    keys = paired.select("season", "roster").unique().rows()
    by = {k: paired.filter((pl.col("season") == k[0]) & (pl.col("roster") == k[1]))["diff"]
                   .to_numpy().astype(float) for k in keys}
    means = np.array([v.mean() for v in by.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(means), size=(bootstrap, len(means)))
    draws = means[idx].mean(axis=1)
    return {"n": float(paired.height), "clusters": float(len(means)),
            "mean": float(means.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "p_better": float((draws > 0).mean())}


def per_season(paired: pl.DataFrame) -> pl.DataFrame:
    """Mean difference per held-out season -- the every-season half of the bar."""
    return (paired.group_by("season")
                  .agg(pl.col("diff").mean().alias("gain"), pl.len().alias("n"))
                  .sort("season"))


def verdict(summary: dict, seasons: pl.DataFrame,
            cover: dict[str, float] | None = None) -> tuple[str, str]:
    """The three branches, fixed in `docs/weekly-projection-plan.md` before this ran.

    Asymmetric on purpose: the Weekly projection is the complicated thing and the burden sits
    on it. The middle branch is the expected one and it carries an *action* rather than being
    a disappointment to explain away -- the same disposition the snap trend got in ADR-0013,
    and it was written down early precisely because "show it beside consensus" is a satisfying
    thing to decide after seeing a near-miss.
    """
    if cover is not None and cover["cells"] and cover["join_failure"] > VOID_FLOOR:
        return "VOID", (
            f"VOID: {cover['join_failure']:.1%} of roster-weeks are a join failure -- the "
            f"player was not on that week's consensus page and scored anyway -- against a "
            f"pre-registered floor of {VOID_FLOOR:.0%}.\n  An unmatched player is ranked "
            f"last, so this benches the incumbent's arm and biases the result toward us. "
            f"Fix the join before reading any number below.")
    if summary["clusters"] == 0:
        return "SHOW", "nothing measured -- no roster-week scored"
    won = int((seasons["gain"] > 0).sum())
    total = seasons.height
    if summary["lo"] > 0 and won == total:
        return "ADOPT", ("ADOPT: the Weekly projection sets lineups. It beat consensus rank in "
                         f"every held-out season ({won}/{total}) and the interval excludes zero.")
    if summary["hi"] < 0 and won == 0:
        return "REMOVE", ("REMOVE: worse than a free ranking in every season. Delete the "
                          "module rather than shipping it as an option.")
    zero = "the interval contains zero" if summary["lo"] <= 0 <= summary["hi"] else (
        "the interval excludes zero but the sign is not consistent across seasons")
    return "SHOW", ("SHOW, NEVER RANK ON: printed beside consensus, never sorted on. "
                    f"Won {won}/{total} seasons and {zero} -- absence of evidence, not "
                    "evidence of equivalence.")


def coverage(rosters: dict[int, list[list[int]]],
                      consensus: dict[int, np.ndarray], realised: dict[int, np.ndarray],
                      covered: set[tuple[int, int]],
                      weeks: Sequence[int] = GATE_WEEKS) -> dict[str, float]:
    """How much of the incumbent's arm is missing, and how much of that is a defect.

    Two different things, and conflating them would either void every run or none:

      * **unranked** -- the player is not on that week's page. Usually because he is out, and
        then the absence is the incumbent's *answer* and benching him is correct.
      * **join failure** -- unranked *and he scored*. Consensus would have ranked a player who
        played; the name did not match. This is the one that biases the comparison, and it is
        what `VOID_FLOOR` is measured against.
    """
    cells = unranked = failed = 0
    for season, made in rosters.items():
        for roster in made:
            for w in weeks:
                if (season, w) not in covered:
                    continue
                col = consensus[season][roster, w - 1]
                pts = realised[season][roster, w - 1]
                cells += col.size
                un = col == UNRANKED
                unranked += int(un.sum())
                failed += int((un & (pts > 0)).sum())
    if not cells:
        return {"cells": 0, "unranked": float("nan"), "join_failure": float("nan")}
    return {"cells": float(cells), "unranked": unranked / cells,
            "join_failure": failed / cells}


def main(argv: Sequence[str] | None = None) -> int:      # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.season.weekly_gate",
        description="Does the Weekly projection beat weekly consensus rank at setting lineups?")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--churn", action="store_true",
                    help="one waiver add/drop a week, both arms, from a pool both can score")
    ap.add_argument("--open-pool", action="store_true",
                    help=argparse.SUPPRESS)   # the asymmetric pool: NOT the gate
    ap.add_argument("--lcb", type=float, default=0.0, metavar="Z",
                    help="rank waiver adds by mu - Z*se; the pre-registered value is 1.0")
    ap.add_argument("--expected", action="store_true",
                    help="expected receptions and yardage in the priors, not realised")
    ap.add_argument("--shrink",
                    choices=("mae", "tail", "mae-market", "tail-market", "market-only"),
                    default=None,
                    help="shrink thin-sample projections toward the positional mean; "
                         "'mae' is the pre-registered fit, 'tail' the exploratory one")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--drafts", type=int, default=20, help="rosters per season")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    from hub.season.weekly_gate_data import assemble_universe
    ros, pos, realised, consensus, weekly, pool, addable, se, covered = assemble_universe(
        seasons, drafts=a.drafts, seed=a.seed, shrink=a.shrink, expected=a.expected)
    cover = coverage(ros, consensus, realised, covered)
    paired = compare_universe(ros, pos, realised, consensus, weekly, pool, covered,
                              None if a.open_pool else addable, churn=a.churn,
                              z=a.lcb, se=se)
    s = cluster_bootstrap(paired, seed=a.seed)
    seasons_tbl = per_season(paired)
    mode = ("one add/drop a week, pool both arms can score" if a.churn and not a.open_pool
            else "one add/drop a week, OPEN POOL -- not the gate" if a.churn
            else "frozen rosters")
    if a.shrink:
        mode += f", shrink={a.shrink}"
    if a.expected:
        mode += ", expected priors"
    if a.lcb:
        mode += f", waiver LCB z={a.lcb}"
    print(f"\n  {int(s['n'])} roster-weeks over {int(s['clusters'])} rosters, "
          f"on the {len(covered)} weeks consensus covers   [{mode}]")
    print(f"  unranked {cover['unranked']:.1%}, of which a join failure "
          f"{cover['join_failure']:.1%} (floor {VOID_FLOOR:.0%})")
    print(seasons_tbl)
    print(f"\n  weekly - consensus = {s['mean']:+.3f} points per team-week")
    print(f"  95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}]   "
          f"P(weekly better) {s['p_better'] * 100:.1f}%")
    print(f"\n  {verdict(s, seasons_tbl, cover)[1]}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
