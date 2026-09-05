"""The pool as a whole, not one entry in it.

`hub.season.survivor` answers "which teams should I pick". This answers "how long does the
contest last, and how many people are still in it" -- which is a different question and the
one a dollar figure needs. A buyback is worth what the pot will be times the chance of taking
it, and both of those depend on the field, not on us.

**Every game is drawn once per trial.** All entries holding that team share the result. This
is the whole point: a field playing chalk dies *together*, and the correlation is what decides
whether the pool reaches week 13 at all. Drawing per entry would make eliminations independent,
which would thin the field smoothly and stretch the contest far past anything real.

**Rivals are sampled, not deterministic**, and that is a correction rather than a refinement.
Twenty-one rivals following one deterministic rule against identical empty ledgers pick the
identical team every week and die in the same week: the surviving count is 21 until it is 0,
the ending week is a point mass, and the partially-thinned field a buyback is priced against
never occurs. So each rival samples among the teams absent from its own ledger, weighted by win
probability -- concentrated on chalk, but not identical to it.

What that buys is divergence. What it costs is a modelling choice with no measurement behind
it: nobody has observed this pool's rivals, and under Hidden Picks nobody can before the
deadline. The weighting is an assumption and is stated as one.

**Ledgers are reconstructed, not observed**, under the same sampling rule -- so the model is
exactly degenerate in week 1, where every ledger is empty and it knows nothing about the field
at all. That is where the season starts, so early figures carry more model risk than late ones.

**Ties are not modelled** because the grid cannot express one: `win_prob` comes from a
continuous margin model, which prices a tie at zero. The pool's rule that a tie eliminates is
therefore satisfied vacuously here rather than enforced.

Iteration order is sorted everywhere a set would otherwise decide it. `docs/weekly-blend-gate.md`
records an interval that moved between identical runs because `.unique()` order fed a bootstrap;
same class of bug, and a reproducible simulator cannot afford it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.config import PoolConfig

# Enough that the ending-week distribution is stable to about a percentage point, which is
# finer than any decision downstream reads it at. Callers wanting a tighter tail pass more.
DEFAULT_TRIALS = 2000


class PoolOutcome(NamedTuple):
    """What one simulation says about the contest, as distributions rather than points."""
    trials: int
    ending_week: dict[int, float]      # week -> P(the last live entry dies in it)
    co_survivors: dict[int, float]     # n -> P(exactly n entries outlast the final week)
    alive_by_week: dict[int, float]    # week -> mean entries still live after it


class _Week(NamedTuple):
    """One week's games and prices, in a shape the trial loop can use without re-querying."""
    games: tuple[tuple[str, str, float], ...]   # (team_a, team_b, P(team_a wins))
    teams: tuple[str, ...]                      # sorted, so iteration order is not a set's
    prob: dict[str, float]
    picks: int                                  # 1, or 2 in a double-pick week


def weeks_from_grid(grid: pl.DataFrame, weeks: Sequence[int],
                    pool: PoolConfig | None = None) -> list[_Week]:
    """Reshape the win-probability grid into one entry per week.

    Needs `game_id`: a week that takes two picks must know which rows are two sides of one
    fixture, both so the game is drawn once and so nobody is handed both sides. Raises rather
    than falling back to per-team draws, which would let a team and its opponent both win.
    """
    if "game_id" not in grid.columns:
        raise ValueError(
            "the grid must carry `game_id`; without it a game cannot be drawn once and a "
            "team and its opponent can both win in the same trial")
    cfg = pool or PoolConfig()
    out = []
    for w in weeks:
        wk = grid.filter(pl.col("week") == w)
        games = []
        for gid in sorted(wk["game_id"].unique().to_list()):
            side = wk.filter(pl.col("game_id") == gid).sort("team")
            if side.height != 2:
                continue        # a game only one side of which is priced is not a game
            a, b = side["team"][0], side["team"][1]
            games.append((str(a), str(b), float(side["win_prob"][0])))
        teams = tuple(sorted({t for g in games for t in (g[0], g[1])}))
        prob = {str(r["team"]): float(r["win_prob"]) for r in wk.iter_rows(named=True)}
        out.append(_Week(tuple(games), teams, prob,
                         2 if w in tuple(cfg.double_pick_weeks) else 1))
    return out


def _pick(rng: np.random.Generator, week: _Week, ledger: set[str], k: int) -> list[str] | None:
    """`k` teams this entry has not used, sampled toward the best available.

    None when the entry cannot field a legal pick -- it has spent too many teams to cover the
    week, which is elimination by the no-repeat rule rather than by losing.
    """
    avail = [t for t in week.teams if t not in ledger]
    if len(avail) < k:
        return None
    w = np.array([week.prob[t] for t in avail], dtype=float)
    total = w.sum()
    if total <= 0:
        return None
    return [str(t) for t in rng.choice(avail, size=k, replace=False, p=w / total)]


def simulate(grid: pl.DataFrame, weeks: Sequence[int], *, entries: int,
             pool: PoolConfig | None = None, ledgers: Sequence[set[str]] | None = None,
             trials: int = DEFAULT_TRIALS,
             rng: np.random.Generator | None = None) -> PoolOutcome:
    """Play the pool forward `trials` times and report what the field did.

    `ledgers` is what each entry has already spent, in entry order, for a mid-season run. Left
    out, every entry starts clean -- which is week 1 and is the case the model knows least
    about.
    """
    rng = rng or np.random.default_rng(0)
    wks = weeks_from_grid(grid, weeks, pool)
    ended: Counter[int] = Counter()
    co: Counter[int] = Counter()
    alive_tot = dict.fromkeys(weeks, 0)

    for _ in range(trials):
        led = [set(ledgers[i]) if ledgers is not None else set() for i in range(entries)]
        alive = [True] * entries
        for wk, w in zip(wks, weeks, strict=True):
            # One draw per game, shared by every entry holding either side.
            won: set[str] = set()
            for a, b, p_a in wk.games:
                won.add(a if rng.random() < p_a else b)
            for i in range(entries):
                if not alive[i]:
                    continue
                picks = _pick(rng, wk, led[i], wk.picks)
                if picks is None or not all(t in won for t in picks):
                    alive[i] = False
                    continue
                led[i].update(picks)
            n = sum(alive)
            alive_tot[w] += n
            if n == 0:
                ended[w] += 1
                break
        else:
            co[sum(alive)] += 1

    return PoolOutcome(
        trials=trials,
        ending_week={w: c / trials for w, c in sorted(ended.items())},
        co_survivors={n: c / trials for n, c in sorted(co.items())},
        alive_by_week={w: alive_tot[w] / trials for w in weeks},
    )
