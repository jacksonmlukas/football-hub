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


class EntryOutcome(NamedTuple):
    """What one entry's own position is worth, from here.

    `sole` is the ticket's question and `share` is the one a dollar figure asks: with a pot
    split among co-survivors, finishing level with two others is worth a third, not nothing.
    Both come off the same trials because a second pass over this simulator is not cheap, and
    computing them apart would let them disagree about the same run.
    """
    trials: int
    survives: float     # P(this entry outlasts the final week at all)
    sole: float         # P(it is the only one that does)
    share: float        # expected fraction of the pot under an even split


def _plural(n: int, word: str) -> str:
    """`1 team`, `2 teams`. A count printed beside a decision is read by a person."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class Buyback(NamedTuple):
    """A $20 decision, priced. Every field is reported rather than folded into a verdict.

    `net` is the answer and its sign is the recommendation: positive means the equity bought
    exceeds the fee paid. `breakeven` is the fee at which that flips, so the margin is visible
    instead of implied -- a net of +$0.40 and a net of +$40 are the same verdict and very
    different bets.
    """
    available: bool
    reason: str
    pot: float          # what the pot becomes once the expected buybacks are in
    field: int          # live entries once they are, ours included
    spent: int          # teams the re-entry inherits, which is why it is not a fresh entry
    share: float        # our expected fraction of that pot
    equity: float       # share * pot, in dollars
    fee: float
    net: float          # equity - fee. Positive means buy back
    breakeven: float    # the fee at which net is zero, which is the equity
    recommend: bool


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


def _play(rng: np.random.Generator, wks: Sequence[_Week], weeks: Sequence[int],
          led: list[set[str]], alive: list[bool]) -> tuple[int | None, list[int]]:
    """Play one trial out. Returns the week everyone died -- None if somebody lasted -- and
    the live count after each week.

    Every game is drawn once here and the result shared by each entry holding either side.
    Extracted rather than written twice: `simulate` and `entry_outcome` ask different
    questions of the same trial, and a second copy of this loop is a second answer about it.
    """
    counts = []
    for wk, w in zip(wks, weeks, strict=True):
        won: set[str] = set()
        for a, b, p_a in wk.games:
            won.add(a if rng.random() < p_a else b)
        for i in range(len(alive)):
            if not alive[i]:
                continue
            picks = _pick(rng, wk, led[i], wk.picks)
            if picks is None or not all(t in won for t in picks):
                alive[i] = False
                continue
            led[i].update(picks)
        counts.append(sum(alive))
        if counts[-1] == 0:
            return w, counts
    return None, counts


def entry_outcome(grid: pl.DataFrame, weeks: Sequence[int], *, entries: int,
                  ledger: Sequence[str] = (), pool: PoolConfig | None = None,
                  trials: int = DEFAULT_TRIALS,
                  rng: np.random.Generator | None = None) -> EntryOutcome:
    """What our own entry is worth from here, carrying the teams it has already spent.

    The quantity a buyback is priced against, and the reason it cannot be one-over-the-field:
    that number is blind to the ledger, so it is the same in week 2 and week 6 and a buyback
    figure built on it never moves with the teams already gone. Here a fuller ledger means
    fewer legal picks, which means more weeks the entry cannot cover.

    Our entry runs in the same trials as the field, so it shares game outcomes with every
    rival holding the same team -- which is why it cannot be computed on its own and then
    combined with a field number afterwards.
    """
    rng = rng or np.random.default_rng(0)
    wks = weeks_from_grid(grid, weeks, pool)
    ours = set(ledger)
    survived = sole = 0
    share = 0.0
    for _ in range(trials):
        led = [set(ours)] + [set() for _ in range(entries - 1)]
        alive = [True] * entries
        _play(rng, wks, weeks, led, alive)
        if not alive[0]:
            continue
        n = sum(alive)
        survived += 1
        sole += int(n == 1)
        share += 1.0 / n
    return EntryOutcome(trials=trials, survives=survived / trials,
                        sole=sole / trials, share=share / trials)


def buyback(grid: pl.DataFrame, weeks: Sequence[int], *, week: int,
            ledger: Sequence[str], live_entries: int, pot: float,
            rival_buybacks: int = 0, pool: PoolConfig | None = None,
            trials: int = DEFAULT_TRIALS,
            rng: np.random.Generator | None = None) -> Buyback:
    """Whether paying the fee to re-enter is worth it, and the fee at which that changes.

    Not a survival question. Re-entering buys a share of a pot that the buybacks themselves
    enlarge, against a field those same buybacks refill -- so a rival re-entry moves the
    numerator and the denominator together, and modelling only the first would make every
    buyback look better than it is.

    **The re-entry is not a fresh entry.** It carries the teams already spent, which the
    commissioner confirmed, so the same $20 buys less in week 6 than in week 2. That is why
    the equity comes from `entry_outcome` on the real ledger rather than from one over the
    field, which is ledger-blind and identical in both.

    `rival_buybacks` is an argument, not a model. Nobody has observed this pool's rivals and
    Hidden Picks means nobody can before a deadline, so a propensity fitted here would be an
    invention wearing a number's clothes. The caller states an assumption and the figure moves
    with it.

    Future re-entries are not priced. A buyback that is itself later lost could be bought back
    again while the rule still allows it, and that option has value this ignores -- so the
    figure is a floor rather than a point.

    **The equity assumes the pot splits evenly among co-survivors**, which is what
    `entry_outcome.share` measures. `PoolConfig.co_survivor_rule` carries that as its default
    and it is the one rule nobody has confirmed; under a rollover or a tiebreak the same share
    is worth something else, and this figure would need the rule applied rather than assumed.
    """
    cfg = pool or PoolConfig()
    if cfg.buyback_cap <= 0:
        return Buyback(False, "no buybacks: the cap is zero", pot, live_entries,
                       len(set(ledger)), 0.0, 0.0, cfg.buyback_fee, 0.0, 0.0, False)
    if week > cfg.buyback_cutoff_week:
        return Buyback(False,
                       f"no buyback: week {week} is past week "
                       f"{cfg.buyback_cutoff_week}, the last one that allows it",
                       pot, live_entries, len(set(ledger)), 0.0, 0.0, cfg.buyback_fee,
                       0.0, 0.0, False)

    # Ours plus theirs, bounded by the cap. Both sides of the ledger move: the fees enlarge
    # the pot and the entries refill the field.
    rivals = max(0, min(int(rival_buybacks), cfg.buyback_cap))
    field = live_entries + 1 + rivals
    grown = pot + (1 + rivals) * cfg.buyback_fee

    out = entry_outcome(grid, weeks, entries=field, ledger=ledger, pool=cfg,
                        trials=trials, rng=rng)
    equity = out.share * grown
    net = equity - cfg.buyback_fee
    yes = net > 0
    return Buyback(
        available=True,
        reason=(f"{'BUY BACK' if yes else 'DO NOT BUY BACK'}: "
                f"${equity:.2f} of equity against a ${cfg.buyback_fee:.2f} fee, "
                f"re-entering with {_plural(len(set(ledger)), 'team')} already spent"),
        pot=grown, field=field, spent=len(set(ledger)), share=out.share,
        equity=equity, fee=cfg.buyback_fee, net=net, breakeven=equity, recommend=yes)


def report(b: Buyback, *, places: int = 2) -> list[str]:
    """The decision as lines rather than prints, so it can be composed and asserted on.

    `hub.models.experiment.paired_report` exists for the same reason: a block that prints
    cannot be capped, composed, or tested.
    """
    if not b.available:
        return [f"\n  {b.reason}"]
    return [
        f"\n  {b.reason}",
        f"  pot ${b.pot:.{places}f} across {b.field} entries   "
        f"share {b.share * 100:.1f}%   net ${b.net:+.{places}f}",
        f"  breakeven fee ${b.breakeven:.{places}f}",
    ]


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
        died, counts = _play(rng, wks, weeks, led, alive)
        for w, n in zip(weeks, counts, strict=False):
            alive_tot[w] += n
        if died is not None:
            ended[died] += 1
        else:
            co[sum(alive)] += 1

    return PoolOutcome(
        trials=trials,
        ending_week={w: c / trials for w, c in sorted(ended.items())},
        co_survivors={n: c / trials for n, c in sorted(co.items())},
        alive_by_week={w: alive_tot[w] / trials for w in weeks},
    )
