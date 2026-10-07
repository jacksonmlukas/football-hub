"""#432: the 2026 forward weekly gate's null size and power, over the weeks it will have.

Pre-registered in `docs/weekly-forward.md` -- read that first; the design is written there before
any number this prints, and this docstring only says where each piece lives.

**What is simulated.** The forward gate's own call: `experiment.gate` on a paired frame of one
season (2026) with the **week** as the cluster of the interval half and the **roster** as the
within-season unit of the season's Disposition, the ceiling held non-binding (`gate_horizon`'s
precedent: this is about the interval and Disposition halves, not stage 2). The baseline process
is #388's (`gate_horizon.estimates()`, #418's finding): the weekly row variance fitted from the
published per-season SEs. One constant is added to it and it is the only one fitted for this
study: the **week component**, the share of a roster-week's diff that is common to every roster
in that week. #388 estimated the roster component at zero and treated rows as iid; that is right
for a roster cluster and wrong for a week cluster, which is the cluster a one-season gate has to
use. Estimated from the frozen 2022-25 frame (spent seasons; no 2026 row exists in it):

    diff[roster, week] = delta + a[week] + e[roster, week]
    a ~ N(0, WEEK_VAR)                          common to the cohort in that week
    e ~ N(0, row_var - WEEK_VAR)                independent across rosters and weeks

    uv run python scripts/weekly_forward_power.py --controls
    uv run python scripts/weekly_forward_power.py --table --trials 6000 --workers 2 \
        --out $TMPDIR/432-table.json
    uv run python scripts/weekly_forward_power.py --render $TMPDIR/432-table.json
"""
from __future__ import annotations

import os

# One polars thread per worker, as `gate_horizon` does: the pool is the parallelism.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from hub.models import experiment
from hub.models.experiment import Actions, Ceiling, gate

_HERE = Path(__file__).resolve().parent
BASE_SEED = 432
CHUNK = 500
ROSTERS = 20                       # `gate_horizon.WEEKLY_M`: the cohort the spent gate drew
# The week component, fitted on 2022-25 (`--estimates` prints the arithmetic from the numbers
# below, each a summary of the frozen frame read through code): the variance across weeks of
# the cohort-mean diff, 1.8814^2 = 3.540 at 40 rosters, less the within-week variance over the
# rosters, 61.5 / 40 = 1.54. Disattenuated, as `gate_horizon._tau2` is, so the sampling noise
# of the weekly cohort mean is not fitted as if it were signal.
WEEKS_SD_OF_MEANS = 1.8814
WITHIN_WEEK_VAR = 61.5
FROZEN_ROSTERS = 40
WEEK_VAR = WEEKS_SD_OF_MEANS ** 2 - WITHIN_WEEK_VAR / FROZEN_ROSTERS      # ~2.00
WEEKS_GRID = (6, 8, 9, 10)         # admitted weeks: weeks 5-14 is 10; 6-14 is 9; and two shorter
ALL_WEEKS = 14                     # weeks 1-14: what counting weeks 1-4 would have given
DELTAS = (-1.5, -0.825, -0.3, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0)
SPENT_MEAN = -0.825                # the mean of the four spent season gains, as the ledger holds
_ACTIONS = Actions(adopt="ADOPT", remove="REMOVE", show="SHOW")
_TOP = Ceiling("non-binding", np.array([float("inf")]))


def _load_horizon_harness():
    spec = importlib.util.spec_from_file_location("gate_horizon", _HERE / "gate_horizon.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_horizon"] = mod
    spec.loader.exec_module(mod)
    return mod


def baseline_row_var() -> float:
    """#388's weekly roster-week variance (#418's baseline), not re-derived here."""
    return _load_horizon_harness().estimates()["weekly"].row_var


@dataclass(frozen=True)
class Cell:
    weeks: int
    delta: float
    week_var: float = WEEK_VAR
    rosters: int = ROSTERS
    bootstrap: int = experiment.BOOTSTRAP


def frame(rng: np.random.Generator, cell: Cell, row_var: float) -> pl.DataFrame:
    """One simulated 2026: `rosters` x `weeks` diffs, with a week effect common to the rows."""
    a = rng.normal(0.0, math.sqrt(cell.week_var), cell.weeks)
    e = rng.normal(0.0, math.sqrt(max(row_var - cell.week_var, 0.0)), (cell.rosters, cell.weeks))
    d = cell.delta + a[None, :] + e
    return pl.DataFrame({
        "season": np.full(d.size, 2026), "roster": np.repeat(np.arange(cell.rosters), cell.weeks),
        "week": np.tile(np.arange(cell.weeks), cell.rosters), "diff": d.ravel()})


def read(paired: pl.DataFrame, seed: int, bootstrap: int) -> str:
    """The forward gate's call, as `hub.season.weekly_forward` makes it: the week is the cluster
    of the interval half, the roster the within-season unit of the season's Disposition."""
    run = gate(paired, cluster=("week",), within=("roster",), ceiling=_TOP,
               actions=_ACTIONS, bootstrap=bootstrap, seed=seed)
    return run.verdict[0]


def run_chunk(cell: Cell, row_var: float, n: int, seed_seq: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(seed_seq)
    out: Counter = Counter()
    for _ in range(n):
        out[read(frame(rng, cell, row_var), int(rng.integers(2**31)), cell.bootstrap)] += 1
    return {"n": n, "verdict": dict(out)}


def run_cells(cells: Sequence[Cell], *, trials: int, workers: int) -> list[dict]:
    row_var = baseline_row_var()
    jobs = []
    for ci, cell in enumerate(cells):
        for seq in np.random.SeedSequence([BASE_SEED, ci]).spawn(max(1, math.ceil(trials / CHUNK))):
            jobs.append((ci, cell, min(CHUNK, trials), seq))
    done: list[list[dict]] = [[] for _ in cells]
    t0 = time.time()
    if workers <= 1:
        for ci, cell, n, seq in jobs:
            done[ci].append(run_chunk(cell, row_var, n, seq))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            futs = [(ci, ex.submit(run_chunk, cell, row_var, n, seq)) for ci, cell, n, seq in jobs]
            for i, (ci, fut) in enumerate(futs):
                done[ci].append(fut.result())
                if (i + 1) % max(1, len(futs) // 10) == 0:
                    print(f"  {i + 1}/{len(futs)} tasks, {time.time() - t0:.0f}s", flush=True)
    results = []
    for cell, parts in zip(cells, done, strict=True):
        tally: Counter = Counter()
        n = 0
        for p in parts:
            tally.update(p["verdict"])
            n += p["n"]
        results.append({"cell": asdict(cell), "n": n, "verdict": dict(tally)})
    return results


def table_cells(weeks: Sequence[int] = WEEKS_GRID) -> list[Cell]:
    return [Cell(w, d) for w in weeks for d in DELTAS]


def sensitivity_cells() -> list[Cell]:
    """The week component is the one fitted constant: at the two ends of what it could be."""
    return [Cell(10, d, wv) for wv in (0.0, 2 * WEEK_VAR) for d in (-0.825, 0.0, 0.5, 1.5)]


def control_cells() -> list[Cell]:
    """Rule 18's plants: an effect a hundred standard errors wide must ADOPT, its negative must
    REMOVE, and a null must neither almost always."""
    return [Cell(10, 100.0), Cell(10, -100.0), Cell(10, 0.0)]


def rate(res: dict, verdict: str) -> float:
    return res["verdict"].get(verdict, 0) / res["n"]


def se_of(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def render(results: Sequence[dict]) -> str:
    by = {(r["cell"]["weeks"], r["cell"]["delta"], r["cell"]["week_var"]): r for r in results}
    out = ["| admitted weeks | clusters | delta | P(ADOPT) | P(REMOVE) | P(SHOW) | SE |",
           "|---|---|---|---|---|---|---|"]
    for (w, d, wv), r in sorted(by.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        if wv != WEEK_VAR and wv not in (0.0, 2 * WEEK_VAR):
            continue
        tag = "" if wv == WEEK_VAR else f" (week var {wv:.2f})"
        out.append(f"| {w}{tag} | {w} | {d:+.3f} | {rate(r, 'ADOPT'):.4f} | "
                   f"{rate(r, 'REMOVE'):.4f} | {rate(r, 'SHOW'):.4f} | "
                   f"{se_of(max(rate(r, 'ADOPT'), rate(r, 'REMOVE')), r['n']):.4f} |")
    return "\n".join(out)


def estimates() -> str:
    rv = baseline_row_var()
    return (f"baseline row_var = {rv:.4f} (sd {math.sqrt(rv):.4f}), #388's weekly process\n"
            f"week variance = {WEEKS_SD_OF_MEANS}^2 - {WITHIN_WEEK_VAR}/{FROZEN_ROSTERS} = "
            f"{WEEK_VAR:.4f} (sd {math.sqrt(WEEK_VAR):.4f}); cohort-mean week sd at "
            f"{ROSTERS} rosters = {math.sqrt(WEEK_VAR + (rv - WEEK_VAR) / ROSTERS):.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimates", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--render", nargs="+")
    ap.add_argument("--weeks", type=int, nargs="+", help="restrict --table to these week counts")
    ap.add_argument("--trials", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.estimates:
        print(estimates())
        return 0
    if a.render:
        print(render([r for f in a.render for r in json.loads(Path(f).read_text())]))
        return 0
    cells = (control_cells() if a.controls else sensitivity_cells() if a.sensitivity
             else table_cells(a.weeks or WEEKS_GRID) if a.table else [])
    if not cells:
        ap.print_usage()
        return 2
    res = run_cells(cells, trials=min(a.trials, 2000) if a.controls else a.trials,
                    workers=a.workers)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
