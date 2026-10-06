"""#388: the gate's null size and power over cluster count and within-season rows.

Pre-registered in `docs/gate-power.md`, section *Gate horizon (#388)* -- read that first; the
design is written there before any number this script prints, and this docstring only says
where each piece of it lives.

**What is simulated.** The combined rule that ships -- the t-interval half, the every-season
half, the tie mechanism -- driven through the shipped `experiment.gate`, which calls
`summarise` and `per_season` itself. Nothing is reimplemented; a trial hands `gate` a paired
frame and reads the verdict it returns. The ceiling is held non-binding (`rule16_combined_power`
precedent): this study is about the interval and tie-aware halves, not about stage 2.

**The generating process** (every constant below is estimated from published history, by
`estimates()`, and `--estimates` prints the arithmetic):

    season mean   ~ N(delta, tau^2)            between-season spread, disattenuated
    cluster mean  ~ N(season mean, v / r)       one value per within-season cluster
                                                (roster / room), `r` rows averaged into it

so the weekly path's within-season rows are `m` rosters x `r` weeks and the draft path's are
`m` rooms x 1. A cluster's rows are collapsed to their mean before they reach `gate`: for a
balanced frame that is exactly what `summarise` (cluster mean over the season) and `per_season`
(roster mean, then the bootstrap) compute first, so the verdict is identical to the one the
row-level frame gives -- `tests/unit/test_gate_horizon.py` holds that equality.

    uv run python scripts/gate_horizon.py --estimates
    uv run python scripts/gate_horizon.py --controls --out $TMPDIR/388-controls.json
    uv run python scripts/gate_horizon.py --table --trials 40000 --workers 6 \
        --out $TMPDIR/388-table.json
    uv run python scripts/gate_horizon.py --sensitivity --trials 20000 --out $TMPDIR/388-sens.json
    uv run python scripts/gate_horizon.py --render $TMPDIR/388-table.json \
        --with-sensitivity $TMPDIR/388-sens.json
"""
from __future__ import annotations

import os

# One polars thread per worker: the pool is the parallelism, and N workers x all cores each
# is how a heavy run turns into load 60 (memory: bound lanes by cores).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context

import numpy as np
import polars as pl

from hub.models import experiment
from hub.models.experiment import SEASON_CLUSTER, Actions, Ceiling, gate

BOOTSTRAP = experiment.BOOTSTRAP        # the shipped bootstrap, not the precedent's 200
CHUNK = 2_000                            # trials per task: the unit of seeding and of load-balance
BASE_SEED = 388
_ACTIONS = Actions(adopt="ADOPT", remove="REMOVE", show="SHOW")
_TOP = Ceiling("non-binding", np.array([float("inf")]))

# --- the history the constants are estimated from -------------------------------------------
#
# Weekly: #382's re-run of the #378 recipe -- the only run that recorded per-season SEs
# (`state/gate-width.json`, entry of 2026-09-21T19:39:54Z): m = 20 rosters a season, scored
# weeks 13 / 13 / 11 / 13 (rows 260 / 260 / 220 / 260 -- the gain's denominators say so).
WEEKLY_GAINS = (-1.119769230769231, -1.4479230769230762, -0.2587272727272733, -1.7613846153846153)
WEEKLY_SES = (0.4088479112937116, 0.44738202810661887, 0.5428066491511976, 0.5456916529523149)
WEEKLY_WEEKS = (13, 13, 11, 13)
WEEKLY_M = 20
# Within a roster the week-to-week diffs carry the variance, not the roster: from the frozen
# frame `data/processed/gate/weekly_post248.parquet` (40 rosters x 11-13 weeks, read through
# code, summaries only), the between-roster variance component, disattenuated by the within-
# roster week variance over the weeks, came out -1.98, 0.45, 0.61, -0.11 across the four
# seasons (mean -0.26) -- zero within its own noise -- and the lag-1 autocorrelation of a
# roster's weekly diffs -0.08 / -0.04 / -0.09 / -0.03. So the model is row-iid with no roster
# component: a cluster mean over r weeks has variance row_var / r.
WEEKLY_ROSTER_COMPONENT = 0.0

# Draft: the gains are #376's hold-out run (ADR-0019, the published gate run); the within-room
# SDs are the four seasons of the frozen frame `p245_shipped_seed0.parquet` (20 rooms each),
# the only draft frame with per-room rows -- the ledger's draft entries carry no SE.
DRAFT_GAINS = (-15.35, -8.41, -5.99, -19.86)
DRAFT_ROOM_SDS = (13.671, 14.062, 13.005, 9.558)
DRAFT_M = 20

WEEKLY_DELTAS = (0.3, 0.5)
DRAFT_DELTA = 2.0
FULL_SEASON_WEEKS = 14                   # `hub.config.REG_SEASON_WEEKS`


@dataclass(frozen=True)
class Process:
    """What generates a trial's paired frame, apart from the cell it is run at."""

    tau: float                           # between-season SD of the true season effect
    row_var: float                       # variance of one row (a roster-week, a room)


def _tau2(gains: Sequence[float], se2: Sequence[float]) -> float:
    """Method-of-moments between-season variance: the variance of the observed season gains
    less the mean sampling variance of a season gain, floored at zero. The observed variance
    already contains each season's own noise; fitting it raw would fit that too (method.md,
    *Noted twice*: a dispersion fitted on observed variance absorbs the sampling noise of the
    thing it is fitted on)."""
    return max(0.0, float(np.var(gains, ddof=1)) - float(np.mean(se2)))


def estimates() -> dict[str, Process]:
    """The two processes, with the arithmetic that produced them (printed by `--estimates`).

    A bootstrap SE over `m` cluster means is `s * sqrt((m-1)/m) / sqrt(m)` for the cluster SD
    `s`, so `s = se * m / sqrt(m-1)` and the true sampling variance of a season gain is
    `s^2 / m = se^2 * m / (m-1)`.
    """
    m = WEEKLY_M
    unit_sd = [se * m / math.sqrt(m - 1) for se in WEEKLY_SES]
    # cluster variance = row_var / weeks, so row_var = unit_sd^2 * weeks, pooled by the mean
    row_var = float(np.mean([s * s * w for s, w in zip(unit_sd, WEEKLY_WEEKS)]))
    wk_tau2 = _tau2(WEEKLY_GAINS, [s * s / m for s in unit_sd])
    dm = DRAFT_M
    room_var = float(np.mean([s * s for s in DRAFT_ROOM_SDS]))
    dr_tau2 = _tau2(DRAFT_GAINS, [room_var / dm] * len(DRAFT_GAINS))
    return {"weekly": Process(tau=math.sqrt(wk_tau2), row_var=row_var),
            "draft": Process(tau=math.sqrt(dr_tau2), row_var=room_var)}


# --- one trial -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One simulated design: the path, each season's (clusters, rows-per-cluster), and delta."""

    path: str
    seasons: tuple[tuple[int, int], ...]
    delta: float
    tau_scale: float = 1.0               # the between-season SD as a multiple of its estimate

    @property
    def k(self) -> int:
        return len(self.seasons)


def _frame(rng: np.random.Generator, proc: Process, seasons: Sequence[tuple[int, int]],
           delta: float, tau_scale: float = 1.0) -> pl.DataFrame:
    """One paired frame: one row per within-season cluster, carrying that cluster's mean."""
    mus = rng.normal(delta, proc.tau * tau_scale, len(seasons))
    yr, unit, diff = [], [], []
    for i, (m, r) in enumerate(seasons):
        diff.append(rng.normal(mus[i], math.sqrt(proc.row_var / r), m))
        yr.append(np.full(m, 2022 + i))
        unit.append(np.arange(m))
    return pl.DataFrame({"season": np.concatenate(yr), "unit": np.concatenate(unit),
                         "diff": np.concatenate(diff)})


def _read(paired: pl.DataFrame, seed: int) -> dict[str, int | float]:
    """Drive `gate` on the frame and read the shipped verdict, the pre-#335 verdict, and the
    interval half alone, off the one summary and one seasons frame `gate` produced."""
    run = gate(paired, cluster=SEASON_CLUSTER, within=("unit",), ceiling=_TOP,
               actions=_ACTIONS, bootstrap=BOOTSTRAP, seed=seed)
    ship = run.verdict[0]
    # The pre-#335 reading: a seasons frame with no `se`/`m` is read by sign alone
    # (`_seasons_won_tied_lost`'s documented backward compatibility), the rest unchanged.
    sign = experiment._verdict(run.summary, run.seasons.select("season", "gain", "n"),
                               _ACTIONS)[0]
    recs = experiment._season_records(run.seasons)
    gains = run.seasons["gain"].to_numpy()
    return {"ship": ship, "sign": sign,
            "ties": sum(r["disposition"] == "tie" for r in recs),
            "int_adopt": int(run.summary["t_lo"] > 0), "int_remove": int(run.summary["t_hi"] < 0),
            "all_pos": int(bool((gains > 0).all())), "all_neg": int(bool((gains < 0).all()))}


def run_chunk(cell: Cell, proc: Process, n: int, seed_seq: np.random.SeedSequence) -> dict:
    """`n` trials of one cell, seeded from `seed_seq` alone: the answer does not depend on how
    many workers split the cells."""
    rng = np.random.default_rng(seed_seq)
    ship, sign = Counter(), Counter()
    tot = Counter()
    violations = 0
    for _ in range(n):
        out = _read(_frame(rng, proc, cell.seasons, cell.delta, cell.tau_scale), int(rng.integers(2**31)))
        ship[out["ship"]] += 1
        sign[out["sign"]] += 1
        tot["ties"] += out["ties"]
        tot["any_tie"] += out["ties"] > 0
        for key in ("int_adopt", "int_remove", "all_pos", "all_neg"):
            tot[key] += out[key]
        # #335's prediction as a per-trial inclusion, not only a rate: whatever the tie-aware
        # rule adopts, the sign-alone rule adopts on the same frame (and mirrored for REMOVE).
        violations += (out["ship"] == "ADOPT" and out["sign"] != "ADOPT") \
            + (out["ship"] == "REMOVE" and out["sign"] != "REMOVE")
        # ...and the discordant frames the tie mechanism is answerable for: the paired count
        # that makes the difference between the two columns a difference rather than a rate.
        tot["sign_not_ship"] += out["sign"] == "ADOPT" and out["ship"] != "ADOPT"
    return {"n": n, "ship": dict(ship), "sign": dict(sign), "tot": dict(tot),
            "violations": violations}


def _merge(parts: Sequence[dict]) -> dict:
    out: dict = {"n": 0, "ship": Counter(), "sign": Counter(), "tot": Counter(), "violations": 0}
    for p in parts:
        out["n"] += p["n"]
        out["violations"] += p["violations"]
        for key in ("ship", "sign", "tot"):
            out[key].update(p[key])
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in out.items()}


def run_cells(cells: Sequence[Cell], procs: dict[str, Process], *, trials: int, workers: int,
              label: str = "") -> list[dict]:
    """Every cell, `trials` each, split into `CHUNK`-trial tasks over `workers` processes."""
    jobs = []
    for ci, cell in enumerate(cells):
        cell_seq = np.random.SeedSequence([BASE_SEED, ci])
        n_chunks = max(1, math.ceil(trials / CHUNK))
        for chunk, seq in enumerate(cell_seq.spawn(n_chunks)):
            n = min(CHUNK, trials - chunk * CHUNK)
            jobs.append((ci, cell, procs[cell.path], n, seq))
    t0 = time.time()
    done = [[] for _ in cells]
    if workers <= 1:
        for ci, cell, proc, n, seq in jobs:
            done[ci].append(run_chunk(cell, proc, n, seq))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            futs = [(ci, ex.submit(run_chunk, cell, proc, n, seq)) for ci, cell, proc, n, seq in jobs]
            for i, (ci, fut) in enumerate(futs):
                done[ci].append(fut.result())
                if (i + 1) % max(1, len(futs) // 20) == 0:
                    print(f"  {label} {i + 1}/{len(futs)} tasks, {time.time() - t0:.0f}s", flush=True)
    return [{"cell": asdict(cell), **_merge(parts)} for cell, parts in zip(cells, done)]


# --- the grids ---------------------------------------------------------------------------------

K_GRID = (4, 5, 6, 7, 8)
WEEKLY_ROWS_GRID = (11, 13, FULL_SEASON_WEEKS)     # scored weeks per roster: observed low, high, full
DRAFT_ROOMS_GRID = (20, 40, 80)                    # rooms a season: observed, then more
SENSITIVITY_K = (4, 6, 8)
SENSITIVITY_TAU = (0.0, 2.0)                       # multiples of the estimated tau
SENSITIVITY_TRIALS = 20_000
TARGET_POWER = 0.80                                # `experiment.POWER`


def weekly_cell(k: int, weeks: int, delta: float, tau_scale: float = 1.0) -> Cell:
    return Cell("weekly", ((WEEKLY_M, weeks),) * k, delta, tau_scale)


def draft_cell(k: int, rooms: int, delta: float, tau_scale: float = 1.0) -> Cell:
    return Cell("draft", ((rooms, 1),) * k, delta, tau_scale)


def table_cells() -> list[Cell]:
    cells: list[Cell] = []
    for k in K_GRID:
        for w in WEEKLY_ROWS_GRID:
            cells += [weekly_cell(k, w, d) for d in (0.0, *WEEKLY_DELTAS)]
    # the 2026 question: a fifth season scored through only 4 weeks, or through all 14
    for w26 in (4, FULL_SEASON_WEEKS):
        seasons = ((WEEKLY_M, 13),) * 4 + ((WEEKLY_M, w26),)
        cells += [Cell("weekly", seasons, d) for d in (0.0, *WEEKLY_DELTAS)]
    for k in K_GRID:
        for rooms in DRAFT_ROOMS_GRID:
            cells += [draft_cell(k, rooms, d) for d in (0.0, DRAFT_DELTA)]
    return cells


def sensitivity_cells() -> list[Cell]:
    """The between-season SD is estimated from four seasons and is the least certain input:
    its sensitivity, at the observed rows, at the two ends of the range it could plausibly be."""
    cells: list[Cell] = []
    for ts in SENSITIVITY_TAU:
        for k in SENSITIVITY_K:
            cells += [weekly_cell(k, 13, d, ts) for d in (0.0, 0.3)]
            cells += [draft_cell(k, 20, d, ts) for d in (0.0, DRAFT_DELTA)]
    return cells


# --- the controls (rule 18) ---------------------------------------------------------------------

PLANTED_DELTA = {"weekly": 20.0, "draft": 60.0}    # ~40 and ~20 season SEs: far above any range


def control_cells() -> list[Cell]:
    return [weekly_cell(8, 13, PLANTED_DELTA["weekly"]), weekly_cell(4, 13, PLANTED_DELTA["weekly"]),
            draft_cell(8, 20, PLANTED_DELTA["draft"]),
            *(weekly_cell(k, 13, 0.0) for k in K_GRID), *(draft_cell(k, 20, 0.0) for k in K_GRID),
            weekly_cell(4, 13, 0.3), weekly_cell(4, 13, 0.5), draft_cell(4, 20, DRAFT_DELTA)]


def se_of(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def rate(res: dict, key: str, which: str = "ship") -> float:
    return res[which].get(key, 0) / res["n"]


def tot(res: dict, key: str) -> float:
    return res["tot"].get(key, 0) / res["n"]


def _index(results: Sequence[dict]) -> dict:
    return {(r["cell"]["path"], tuple(map(tuple, r["cell"]["seasons"])), r["cell"]["delta"],
             r["cell"]["tau_scale"]): r for r in results}


def _modal(res: dict) -> str:
    top = max(("ADOPT", "REMOVE", "SHOW", "NOT-RUNNABLE"), key=lambda v: res["ship"].get(v, 0))
    return f"{top} ({rate(res, top):.2f})"


def render(results: Sequence[dict], sensitivity: Sequence[dict] = ()) -> str:
    """The deliverable's tables, as markdown, from a `--table` result file (and, for the
    last one, a `--sensitivity` result file)."""
    by = _index(results)

    def get(path: str, seasons: tuple, delta: float) -> dict:
        return by[(path, seasons, delta, 1.0)]

    out = ["**Weekly path** (δ in points per roster-week); rows are rosters × scored weeks.", "",
           "| k (seasons) | within-season rows | tie rate | null ADOPT | power δ=0.3 | power δ=0.5 "
           "| modal verdict (δ=0.3) |", "|---|---|---|---|---|---|---|"]
    for k in K_GRID:
        for w in WEEKLY_ROWS_GRID:
            s = ((WEEKLY_M, w),) * k
            n0, a, b = get("weekly", s, 0.0), get("weekly", s, 0.3), get("weekly", s, 0.5)
            out.append(f"| {k} | {WEEKLY_M * w} ({WEEKLY_M}×{w}) | {tot(a, 'ties') / k:.3f} | "
                       f"{rate(n0, 'ADOPT'):.4f} | {rate(a, 'ADOPT'):.4f} | {rate(b, 'ADOPT'):.4f} "
                       f"| {_modal(a)} |")
    out += ["", f"**Draft path** (δ = {DRAFT_DELTA} points per team-game); rows are rooms.", "",
            f"| k (seasons) | within-season rows | tie rate | null ADOPT | power δ={DRAFT_DELTA} "
            f"| modal verdict (δ={DRAFT_DELTA}) |", "|---|---|---|---|---|---|"]
    for k in K_GRID:
        for rooms in DRAFT_ROOMS_GRID:
            s = ((rooms, 1),) * k
            n0, a = get("draft", s, 0.0), get("draft", s, DRAFT_DELTA)
            out.append(f"| {k} | {rooms} | {tot(a, 'ties') / k:.3f} | {rate(n0, 'ADOPT'):.4f} | "
                       f"{rate(a, 'ADOPT'):.4f} | {_modal(a)} |")
    out += ["", "**The fifth season, through 4 weeks or all 14** (k = 5; seasons 1-4 at 13 weeks).", "",
            "| 2026 weeks scored | tie rate | P(≥1 tie) δ=0.3 | null ADOPT | power δ=0.3 "
            "| power δ=0.5 | modal verdict (δ=0.3) |", "|---|---|---|---|---|---|---|"]
    for w26 in (4, FULL_SEASON_WEEKS):
        s = ((WEEKLY_M, 13),) * 4 + ((WEEKLY_M, w26),)
        n0, a, b = get("weekly", s, 0.0), get("weekly", s, 0.3), get("weekly", s, 0.5)
        out.append(f"| {w26} | {tot(a, 'ties') / 5:.3f} | {tot(a, 'any_tie'):.3f} | "
                   f"{rate(n0, 'ADOPT'):.4f} | {rate(a, 'ADOPT'):.4f} | {rate(b, 'ADOPT'):.4f} "
                   f"| {_modal(a)} |")
    out += ["", apportionment(results), ]
    if sensitivity:
        out += ["", sensitivity_table(sensitivity)]
    return "\n".join(out)


def apportionment(results: Sequence[dict]) -> str:
    """Three rules on the same frames -- interval half alone, interval + unanimity of sign
    (the pre-#335 rule), and the shipped rule -- and the gap to `TARGET_POWER` split by them."""
    by = _index(results)
    head = ("| k | rows | P(≥1 tie) | P(interval alone) | P(pre-#335) | P(shipped) "
            "| gap to 0.80 | tie mechanism owns (pre-#335 → alone) | k owns |\n"
            "|---|---|---|---|---|---|---|---|---|")

    def row(label: str, k: int, rows: str, a: dict) -> str:
        p_int, p_sign, p_ship = rate(a, "int_adopt", "tot"), rate(a, "ADOPT", "sign"), rate(a, "ADOPT")
        gap = TARGET_POWER - p_ship
        lo, hi = p_sign - p_ship, p_int - p_ship
        k_owned = max(TARGET_POWER - p_int, 0.0)
        share = f"{lo / gap:.0%} – {hi / gap:.0%}" if gap > 0 else "n/a"
        own_k = f"{k_owned / gap:.0%}" if gap > 0 else "n/a"
        return (f"| {k} | {rows} | {tot(a, 'any_tie'):.3f} | {p_int:.4f} | {p_sign:.4f} | "
                f"{p_ship:.4f} | {gap:.3f} | {share} | {own_k} |")

    out = [f"**Where the gap to 80% power sits.** Weekly δ=0.3, then δ=0.5, then draft δ={DRAFT_DELTA}.",
           "", head]
    for delta in WEEKLY_DELTAS:
        out.append(f"| *weekly δ={delta}* | | | | | | | | |")
        for k in K_GRID:
            for w in WEEKLY_ROWS_GRID:
                a = by[("weekly", ((WEEKLY_M, w),) * k, delta, 1.0)]
                out.append(row("", k, f"{WEEKLY_M * w}", a))
    out.append(f"| *draft δ={DRAFT_DELTA}* | | | | | | | | |")
    for k in K_GRID:
        for rooms in DRAFT_ROOMS_GRID:
            a = by[("draft", ((rooms, 1),) * k, DRAFT_DELTA, 1.0)]
            out.append(row("", k, f"{rooms}", a))
    return "\n".join(out)


def sensitivity_table(results: Sequence[dict]) -> str:
    by = _index(results)
    out = ["**Sensitivity to the between-season SD** (observed rows; τ as a multiple of its estimate).",
           "", "| path | k | τ × | null ADOPT | power | P(≥1 tie) |", "|---|---|---|---|---|---|"]
    for path, rows, delta in (("weekly", WEEKLY_ROWS_GRID[1], 0.3), ("draft", DRAFT_ROOMS_GRID[0], DRAFT_DELTA)):
        for k in SENSITIVITY_K:
            for ts in SENSITIVITY_TAU:
                seasons = ((WEEKLY_M, rows),) * k if path == "weekly" else ((rows, 1),) * k
                n0, a = by[(path, seasons, 0.0, ts)], by[(path, seasons, delta, ts)]
                out.append(f"| {path} δ={delta} | {k} | {ts:g} | {rate(n0, 'ADOPT'):.4f} | "
                           f"{rate(a, 'ADOPT'):.4f} | {tot(a, 'any_tie'):.3f} |")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gate_horizon")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimates", action="store_true")
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--table", action="store_true")
    mode.add_argument("--sensitivity", action="store_true")
    mode.add_argument("--render", metavar="TABLE_JSON")
    ap.add_argument("--with-sensitivity", metavar="JSON", default=None)
    ap.add_argument("--trials", type=int, default=40_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    procs = estimates()
    if a.estimates:
        for name, p in procs.items():
            print(f"{name}: tau={p.tau:.4f} row_var={p.row_var:.4f} row_sd={math.sqrt(p.row_var):.4f}")
        return 0
    if a.render:
        with open(a.render) as fh:
            table = json.load(fh)
        sens = []
        if a.with_sensitivity:
            with open(a.with_sensitivity) as fh:
                sens = json.load(fh)
        print(render(table, sens))
        return 0
    name = ("controls" if a.controls else "sensitivity" if a.sensitivity else "table")
    cells = control_cells() if a.controls else sensitivity_cells() if a.sensitivity else table_cells()
    workers = min(a.workers, 6)           # the lane's own cap
    t0 = time.time()
    results = run_cells(cells, procs, trials=a.trials, workers=workers, label=name)
    print(f"{len(cells)} cells x {a.trials} trials in {time.time() - t0:.0f}s on {workers} workers")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
