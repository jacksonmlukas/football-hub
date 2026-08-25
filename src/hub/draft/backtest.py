"""Does championship equity beat the draft market? Measured on realised outcomes.

This is P0 built as code, per ADR-0007 (a measurement that steers the product must be
committed code).
The original P0 ran in an afternoon and committed nothing: both its commits touch
`docs/next.md` and nothing else, so the +0.04 [-3.64, +3.58] that demoted championship equity
to a tiebreaker cannot be reproduced. Two departures from its own pre-registered design went
unrecorded as a result -- the shortlist was top-8 by VOR rather than `recommend()`'s, and n
was 36 rather than the 60 the design fixed.

**The design, pre-registered.** Two arms on the same room and the same seed, so the comparison
is paired:

  * **A, the market**: best available by that season's consensus that fills an unfilled
    starting slot, lexicographically -- `optimize.market_pick(by="ecr")`.
  * **B, the optimizer**: the top of `win_probability` over `recommend()`'s shortlist, ties
    broken by consensus.

**Outcome**: realised weekly points. Best legal lineup each week from what the players
actually scored, summed and divided by team games. Never a projection -- escaping the
circularity is the entire point, since one object both ranks candidates and scores seasons.

**Why arm A ranks on consensus rather than ADP.** ESPN publishes ADP for the current season
only, and the FantasyPros archive carries no historical ADP. Consensus is a real market and
genuinely contemporaneous; the cost is that arm A is a *consensus*-follower while the shipped
THE PICK is a *draft-market*-follower. That is the first of four named gaps between this
harness and the tool it audits, all listed under LIMITATIONS below.

**Decision rule, fixed before the numbers.** Asymmetric on purpose: the market already leads,
and the burden is on the complicated thing.

    CI excludes zero favouring B  -> promote equity back to the headline; P1 fires
    CI contains zero              -> nothing changes
    CI excludes zero favouring A  -> remove equity from the draft-night output entirely

    uv run python -m hub.draft.backtest --seasons 2022,2023,2024,2025 --drafts 20
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from hub.config import DraftConfig, HubConfig, RosterConfig, config_digest
from hub.draft.optimize import (
    DEFAULT_ROUNDS,
    market_pick,
    rank_tiers,
    simulate_remaining_draft,
    win_probability,
)
from hub.draft.season import REG_SEASON_WEEKS, lineup_points
from hub.draft.state import DraftState, _norm
from hub.models.measure import BOOTSTRAP, realised_ppg, summarise  # noqa: F401

# Gaps between this harness and the tool it audits. Written here rather than in the result,
# because a limitation discovered after the numbers is a rationalisation.
LIMITATIONS = (
    "arm A follows consensus (ECR); the shipped THE PICK follows the draft market (ADP), "
    "which ESPN publishes for the current season only",
    "arm B scores seasons on prior-season xFP; the live board scores on proj_blend, which "
    "blends in an ESPN projection that does not exist for past seasons",
    "arm B breaks ties by consensus; the shipped tool refuses to break them and asks you",
    "the room is simulated -- consensus plus fitted pick noise, lexicographic need -- not "
    "the eleven people who were actually in those drafts",
    "arm B is the POST-FIX optimizer: win_probability now seeds every seat with the roster "
    "it already holds. P0 measured the pre-fix one, which was blind to your own roster, so "
    "any movement from P0's +0.04 cannot be read as 'the shortlist tipped it'",
)

def score_roster(names: Sequence[str], pos: Sequence[str], realised: pl.DataFrame,
                 weeks: int = REG_SEASON_WEEKS) -> float:
    """Points per team game from the best legal lineup each week.

    Weekly rather than on season totals. Totals would silently reward even production and
    erase bye weeks and mid-season injuries -- the things a roster is built to survive -- and
    would score the backtest on a different objective than the one being tested.

    A player with no realised row scored nothing, which is correct: he was hurt, cut, or
    never played. Zero, not null, because the lineup rule has to be able to bench him.
    """
    if not names:
        return 0.0
    keys = [_norm(n) for n in names]
    grid = np.zeros((len(keys), weeks))
    idx = {k: i for i, k in enumerate(keys)}
    for row in realised.filter(pl.col("player").is_in(keys)).iter_rows(named=True):
        w = int(row["week"])
        if 1 <= w <= weeks:
            grid[idx[row["player"]], w - 1] = float(row["points"])
    # (sims=1, weeks, roster) -- the same rule the season simulator uses.
    total = lineup_points(grid.T[None, :, :], np.asarray(pos, dtype=object))
    return float(total.sum() / weeks)


def _pool_index(pool: pl.DataFrame, name: str) -> int:
    return pool["player"].to_list().index(name)


def market_strategy(by: str = "ecr"):
    """Arm A. Best available in `by` that fills an unfilled starting slot."""
    def pick(pool, live, counts, taken):
        avail = pool[[int(i) for i in live]]
        name = market_pick(avail, counts, by=by)
        if name is None:
            return int(live[0])
        return _pool_index(pool, name)
    return pick


def optimizer_strategy(board: pl.DataFrame, *, my_slot: int, teams: int, rounds: int,
                       n_draft_sims: int, n_season_sims: int, seed: int,
                       tiebreak: str = "ecr"):
    """Arm B. Top of `win_probability` over `recommend()`'s shortlist, ties broken by `by`.

    The tie-break is not a tidy default: `rank_tiers` exists because the top two candidates
    routinely sit inside two standard errors of each other, and its own docstring notes that
    re-running names a different one. The shipped tool hands that tie to a human. A harness
    has no human, so it needs a rule the product does not have -- and taking the consensus
    among co-leaders makes arm B exactly "equity leads, market breaks ties", the mirror of
    arm A. The lead is then the only difference between the two arms, which is the question.
    """
    from hub.draft.board import recommend

    def pick(pool, live, counts, taken):
        state = DraftState(taken=list(taken))
        overall = len(taken) + 1
        try:
            _, rec = recommend(board, overall, rounds=rounds, state=state)
        except ValueError:
            rec = pool[[int(i) for i in live]].head(10)
        names = [n for n in rec["player"].to_list()
                 if n in set(pool["player"][[int(i) for i in live]].to_list())]
        if not names:
            return int(live[0])
        if len(names) == 1:
            return _pool_index(pool, names[0])
        wp = win_probability(board, state, names, my_slot=my_slot, teams=teams,
                             rounds=rounds, n_draft_sims=n_draft_sims,
                             n_season_sims=n_season_sims, seed=seed)
        leaders = rank_tiers(wp).filter(pl.col("co_leader"))["player"].to_list()
        ranked = (board.filter(pl.col("player").is_in(leaders))
                       .sort(tiebreak, nulls_last=True))
        return _pool_index(pool, ranked["player"][0] if ranked.height else leaders[0])
    return pick


def play(board: pl.DataFrame, strategy, *, my_slot: int, teams: int, rounds: int,
         rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Play one draft with `strategy` in my seat. Returns (my player names, my positions)."""
    rosters = simulate_remaining_draft(board, DraftState(taken=[]), my_slot=my_slot,
                                       teams=teams, rounds=rounds, rng=rng,
                                       my_pick=strategy)
    mine = rosters[my_slot - 1]
    names = [board["player"][int(i)] for i in mine]
    pos = [board["pos"][int(i)] or "NA" for i in mine]
    return names, pos


def compare(boards: dict[int, pl.DataFrame], realised: dict[int, pl.DataFrame], *,
            n_drafts: int = 20, seed: int = 0, my_slot: int | None = None,
            teams: int | None = None, rounds: int = DEFAULT_ROUNDS,
            n_draft_sims: int = 12, n_season_sims: int = 250) -> pl.DataFrame:
    """Paired arm A against arm B, one row per (season, draft).

    Pure: takes frames, returns a frame, touches no network. That is what makes the
    statistics testable, and a backtest whose statistics can only be exercised by hitting
    ESPN is one nobody re-runs.

    `n_draft_sims` and `n_season_sims` are pinned at the shipped 12 x 250. The first P0 run
    used 6 x 120 -- a quarter of the live budget -- and produced -5.79 [-9.17, -2.45], an
    artifact that vanished at adequate power. Lower them and you are measuring a different
    optimizer.
    """
    cfg = RosterConfig()
    my_slot = cfg.slot if my_slot is None else my_slot
    teams = cfg.teams if teams is None else teams

    rows = []
    for season in sorted(boards):
        board, real = boards[season], realised[season]
        arm_a = market_strategy()
        for k in range(n_drafts):
            # Common random numbers: the same room, twice. The only thing that differs
            # between the arms is who sits in my seat.
            room = seed + 1000 * season + k
            a_names, a_pos = play(board, arm_a, my_slot=my_slot, teams=teams,
                                  rounds=rounds, rng=np.random.default_rng(room))
            arm_b = optimizer_strategy(board, my_slot=my_slot, teams=teams, rounds=rounds,
                                       n_draft_sims=n_draft_sims,
                                       n_season_sims=n_season_sims, seed=room)
            b_names, b_pos = play(board, arm_b, my_slot=my_slot, teams=teams,
                                  rounds=rounds, rng=np.random.default_rng(room))
            rows.append({
                "season": season, "draft": k,
                "market": score_roster(a_names, a_pos, real),
                "optimizer": score_roster(b_names, b_pos, real),
            })
    out = pl.DataFrame(rows)
    return out.with_columns((pl.col("optimizer") - pl.col("market")).alias("diff"))


# Your first six turns from slot 3 of 12. Fixed rather than read from the live draft state,
# because `--diagnose` is run twice at two different commits and anything state-dependent
# would not be comparable between them.
#
# The spread matters: at pick 3 you hold nothing, so seeding a roster is a no-op and the
# objective is unaffected by it. By pick 70 you hold five players, and any blindness to them
# has had five chances to show.
DIAGNOSE_PICKS = (3, 22, 27, 46, 51, 70)


def diagnose(board: pl.DataFrame, *, picks: Sequence[int] = DIAGNOSE_PICKS,
             my_slot: int | None = None, teams: int | None = None,
             rounds: int = DEFAULT_ROUNDS, n_draft_sims: int = 12,
             n_season_sims: int = 250, seed: int = 0) -> pl.DataFrame:
    """What championship equity recommends at each of your first turns.

    Run once before a change to the objective and once after, and diff. The draft is advanced
    by the *market* rather than by equity, so the path through the draft is identical in both
    runs and the only thing that can differ is what equity says about the same situation.

    Returns one row per pick: what you hold, who equity names, and by how much.
    """
    cfg = RosterConfig()
    my_slot = cfg.slot if my_slot is None else my_slot
    teams = cfg.teams if teams is None else teams
    want = set(picks)
    rows: list[dict] = []

    from hub.draft.board import recommend

    def pick(pool, live, counts, taken):
        overall = len(taken) + 1
        avail = pool[[int(i) for i in live]]
        if overall in want:
            state = DraftState(taken=list(taken))
            try:
                _, rec = recommend(board, overall, rounds=rounds, state=state)
                names = [n for n in rec["player"].to_list()
                         if n in set(avail["player"].to_list())]
            except ValueError:
                names = []
            if len(names) >= 2:
                wp = rank_tiers(win_probability(
                    board, state, names, my_slot=my_slot, teams=teams, rounds=rounds,
                    n_draft_sims=n_draft_sims, n_season_sims=n_season_sims, seed=seed))
                top = wp.row(0, named=True)
                pos_of = dict(zip(board["player"].to_list(), board["pos"].to_list(), strict=True))
                # Does any co-leader fill a slot you cannot currently start? The tripwire
                # needs this: a need-filling candidate the simulation cannot separate from
                # the leader means the objective has not *rejected* need, it has declined
                # to distinguish -- which is a tie, not a defect.
                from hub.config import required_starters
                req = required_starters(cfg)
                short = [p for p, n in req.items() if counts.get(p, 0) < n]
                co = wp.filter(pl.col("co_leader"))["player"].to_list()
                need_co_led = any(pos_of.get(c) in short for c in co)
                rows.append({
                    "need_co_led": bool(need_co_led),
                    "pick": overall,
                    "held": ", ".join(f"{k}{v}" for k, v in sorted(counts.items())) or "-",
                    "leader": top["player"],
                    "leader_pos": pos_of.get(top["player"]),
                    "lift": float(top["lift"]),
                    "co_leaders": int(wp["co_leader"].sum()),
                    "candidates": len(names),
                    # Typed rather than parsed back out of `held`, which is for reading.
                    **{f"held_{p.lower()}": int(counts.get(p, 0))
                       for p in ("QB", "RB", "WR", "TE")},
                })
        # Advance by the market so the path is identical across runs.
        name = market_pick(avail, counts, by="adp" if "adp" in avail.columns else "ecr")
        return _pool_index(pool, name) if name else int(live[0])

    simulate_remaining_draft(board, DraftState(taken=[]), my_slot=my_slot, teams=teams,
                             rounds=rounds, rng=np.random.default_rng(seed), my_pick=pick)
    return pl.DataFrame(rows)


def tripwire(board: pl.DataFrame, diagnosed: pl.DataFrame) -> list[str]:
    """Whether an objective is fit to pick with: does it ever name a player at a required
    position you have already filled, ahead of one filling a slot you have not?

    **This is a quality check on the objective, not a regression gate on a code change**, and
    conflating those two jobs is how it got talked past. On 2026-08-24 it fired at picks 46
    and 70 for naming a running back while WR and QB sat empty. Read as a regression gate --
    "did my refactor break something?" -- that looked like a false positive, because the
    need-filling alternatives were co-leaders and a tie is not a rejection. So a co-leader
    clause was added and the gate went quiet.

    Read as a quality check, it was correct and the clause was wrong. P0b then measured that
    same running-back-over-need preference at **-19.66 points per team game** against
    consensus-following, across four seasons, losing in all of them. The clause is reverted:
    a need-filling co-leader means the objective cannot tell the difference between filling a
    hole and not filling one, which is exactly the thing worth knowing.

    Deliberately still not a threshold on how much a recommendation moved. A correctness fix
    that changes recommendations is doing its job.
    """
    from hub.config import required_starters
    required = required_starters(RosterConfig())
    bad = []
    for r in diagnosed.iter_rows(named=True):
        held = {p: int(r.get(f"held_{p.lower()}", 0)) for p in required}
        pos = r["leader_pos"]
        if pos in required and held.get(pos, 0) >= required[pos]:
            unfilled = [p for p, n in required.items() if held.get(p, 0) < n]
            if unfilled:
                co = " (a co-leader fills one, which is not a defence)" if r.get(
                    "need_co_led", False) else ""
                bad.append(f"pick {r['pick']}: named {r['leader']} ({pos}), but {pos} is "
                           f"full and {'/'.join(unfilled)} is not{co}")
    return bad


def correction_report(board: pl.DataFrame) -> pl.DataFrame:
    """Which players corrected ADP moves, and by how much. Sorted by size of move.

    The diagnostic for ADR-0011, and the shape is deliberately the same as the one that
    gated the roster-seeding fix: run it, read it, and check the tripwire below before
    shipping a change to what the draft ranks on.
    """
    need = {"player", "pos", "adp", "adp_corrected", "proj_correction"}
    if not need <= set(board.columns):
        return pl.DataFrame(schema={"player": pl.Utf8, "pos": pl.Utf8, "adp": pl.Float64,
                                    "adp_corrected": pl.Float64, "move": pl.Float64,
                                    "proj_correction": pl.Float64})
    return (board.select("player", "pos", "adp", "adp_corrected", "proj_correction")
                 .drop_nulls("adp")
                 .with_columns((pl.col("adp_corrected") - pl.col("adp")).alias("move"))
                 .filter(pl.col("move").abs() > 1e-9)
                 .sort(pl.col("move").abs(), descending=True))


def correction_tripwire(board: pl.DataFrame, clamp_frac: float | None = None) -> list[str]:
    """Fixed before the numbers. Fires only on things that cannot happen if the code is right.

    Deliberately NOT "did the recommendation change" -- it is supposed to change, and gating
    on that would reject the change for working. That was the mistake made with the
    seeding-fix tripwire on the morning of 2026-08-24 and corrected the same day.

    Two impossibilities:
      * a player carrying no fitted correction moves at all -- the shift is a function of the
        correction, so zero in must give zero out;
      * any move exceeds the clamp -- the clamp is applied unconditionally.

    Either means the exchange rate or the clamp is wrong, not that the corrections disagree
    with the market.
    """
    from hub.config import DraftConfig
    if clamp_frac is None:
        clamp_frac = DraftConfig().correction_clamp_frac
    rep = correction_report(board)
    bad: list[str] = []
    for r in rep.iter_rows(named=True):
        if abs(r["proj_correction"] or 0.0) < 1e-12:
            bad.append(f"{r['player']} has no correction but moved "
                       f"{r['move']:+.2f} picks")
        if abs(r["move"]) > clamp_frac * abs(r["adp"]) + 1e-6:
            bad.append(f"{r['player']} moved {r['move']:+.2f} picks, past the "
                       f"{clamp_frac:.0%} clamp of {clamp_frac * abs(r['adp']):.2f}")
    return bad


def verdict(summary: dict[str, float]) -> str:
    """The pre-registered action, read off the interval. Fixed before the numbers."""
    if summary["lo"] > 0:
        return ("PROMOTE: championship equity returns to the headline, and P1 (break the "
                "circularity) fires.")
    if summary["hi"] < 0:
        return ("REMOVE: championship equity leaves the draft-night output. A tiebreaker "
                "measurably worse than the market steers close calls the wrong way.")
    return ("NO CHANGE: the market leads and equity stays a tiebreaker. Absence of "
            "evidence, not evidence of equivalence.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.backtest",
        description="Championship equity against the market, on realised outcomes.")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--drafts", type=int, default=20, help="drafts per season")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--draft-sims", type=int, default=12)
    ap.add_argument("--season-sims", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the paired rows to this parquet path")
    ap.add_argument("--board", default=None,
                    help="parquet snapshot of the board. Written if absent, reused if "
                         "present, so two --diagnose runs at two commits compare the same "
                         "board rather than two live ADP fetches.")
    ap.add_argument("--diagnose-corrections", action="store_true",
                    help="which players corrected ADP moves and by how much, on the live "
                         "board, plus the pre-registered tripwire. Gates ADR-0011.")
    ap.add_argument("--diagnose", action="store_true",
                    help="what equity recommends at each of your first turns, on the live "
                         "board. Run before and after a change to the objective and diff.")
    a = ap.parse_args(argv)

    from hub.draft.board import build
    from hub.fetch import nflverse

    if a.diagnose_corrections:
        print("  building the live board ...")
        board, _ = build()
        rep = correction_report(board)
        print(f"\n  Corrected ADP moves {rep.height} of {board.height} players.")
        print(f"  Clamp: {DraftConfig().correction_clamp_frac:.0%} of each player's own ADP.\n")
        print(f"  {'player':<24} {'pos':<4} {'ADP':>6} {'->':>2} {'corrected':>9} "
              f"{'move':>7} {'ppg corr':>9}")
        for r in rep.head(20).iter_rows(named=True):
            print(f"  {str(r['player'])[:24]:<24} {r['pos'] or ''!s:<4} "
                  f"{r['adp']:>6.1f} {'->':>2} {r['adp_corrected']:>9.1f} "
                  f"{r['move']:>+7.1f} {r['proj_correction']:>+9.2f}")
        if rep.height > 20:
            print(f"  ... and {rep.height - 20} more")
        bad = correction_tripwire(board)
        print()
        if bad:
            print("  TRIPWIRE TRIPPED -- these are impossible if the code is right:")
            for line in bad:
                print(f"    {line}")
            return 1
        print("  tripwire clear: every move is a function of a real correction, "
              "and none exceeds the clamp.")
        if a.out:
            rep.write_parquet(a.out)
            print(f"  wrote {rep.height} rows to {a.out}")
        return 0

    if a.diagnose:
        # Pin the board. `--diagnose` is run twice at two commits to gate a change, and
        # `build()` refetches live ESPN ADP every time -- ADP moves, so two runs minutes
        # apart are not the same experiment. First run writes the snapshot, later runs
        # reuse it, so the only thing that differs between them is the code.
        snap = Path(a.board) if a.board else None
        if snap and snap.exists():
            board = pl.read_parquet(snap)
            print(f"  board pinned from {snap}")
        else:
            print("  building the live board ...")
            board, _ = build()
            if snap:
                board.write_parquet(snap)
                print(f"  board snapshot written to {snap}")
        got = diagnose(board, rounds=a.rounds, n_draft_sims=a.draft_sims,
                       n_season_sims=a.season_sims, seed=a.seed)
        if got.is_empty():
            print("  no pick produced a rankable shortlist; nothing to compare.")
            return 1
        print(f"\n  Championship equity at your first {got.height} turns."
              f"  {a.draft_sims} x {a.season_sims} sims.")
        print("  The draft is advanced by the market, so the path is identical across runs\n"
              "  and the only thing that can differ is what equity says.\n")
        print(f"  {'pick':>4}  {'held':<16} {'leader':<24} {'pos':<4} "
              f"{'lift':>7}  {'co-led':>6}  {'cands':>5}")
        for r in got.iter_rows(named=True):
            print(f"  {r['pick']:>4}  {r['held']:<16} {str(r['leader'])[:24]:<24} "
                  f"{r['leader_pos'] or ''!s:<4} {r['lift']*100:>+6.2f}%  "
                  f"{r['co_leaders']:>6}  {r['candidates']:>5}")
        bad = tripwire(board, got)
        print()
        if bad:
            print("  TRIPWIRE TRIPPED -- equity named a filled position over an empty one:")
            for line in bad:
                print(f"    {line}")
        else:
            print("  tripwire clear: no pick named a filled required position "
                  "over an unfilled one.")
        if a.out:
            got.write_parquet(a.out)
            print(f"  wrote {got.height} rows to {a.out}")
        return 0

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    boards, realised = {}, {}
    for yr in seasons:
        print(f"  building the {yr} board as of {yr}-09-01 ...")
        boards[yr], _ = build(season=yr - 1, season_ahead=yr, as_of=f"{yr}-09-01")
        # `position` and `season` are in the PLAYER_STATS contract's required set, so they
        # are narrowed *in* rather than out -- the contract is validated after the narrowing.
        stats = nflverse.load("player_stats", [yr],
                              cols=["player_id", "player_display_name", "position",
                                    "season", "week", "fantasy_points_ppr"])
        realised[yr] = realised_ppg(stats)

    print(f"  playing {a.drafts} drafts x {len(seasons)} seasons, "
          f"{a.draft_sims} x {a.season_sims} sims per optimizer call ...")
    paired = compare(boards, realised, n_drafts=a.drafts, seed=a.seed, rounds=a.rounds,
                     n_draft_sims=a.draft_sims, n_season_sims=a.season_sims)
    s = summarise(paired, seed=a.seed)

    print(f"\n  n={int(s['n'])}  optimizer - market = {s['mean']:+.2f} points per team game")
    print(f"  95% CI [{s['lo']:+.2f}, {s['hi']:+.2f}]   "
          f"P(optimizer better) {s['p_better']*100:.1f}%")
    print(f"\n  {verdict(s)}")
    print("\n  Limitations, fixed before the run:")
    for line in LIMITATIONS:
        print(f"    - {line}")

    if a.out:
        stamped = paired.with_columns(
            pl.lit(config_digest(HubConfig())).alias("cfg_digest"))
        stamped.write_parquet(a.out)
        print(f"\n  wrote {paired.height} paired rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
