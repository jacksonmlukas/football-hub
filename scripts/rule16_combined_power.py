"""Rule 16 (docs/method.md): the combined rule's null size and power, computed *before* the
ADR-0019 #335 amendment was written -- the pre-registered condition on #357's own ticket.

"The combined rule" is #357's t-interval half **and** #335's tie-aware every-season half,
both read by the one `experiment.gate` a real gate run calls. Simulated at the repo's own
published season-clustered figures, from #357's (S1) table: draft `k=4, s=7.34`, weekly blend
`k=4, s=0.382`; the within-season cluster counts `m=20` and `m=40` are this ticket's own
pre-registration, not measured. Compared against **unanimity-alone power** -- the fixed
interval-half rule's own power at the same delta, from that same table: 0.136 (draft),
0.378 (weekly).

**Within-season noise, an assumption stated rather than measured.** Neither gate's real
within-season standard deviation is published; this script sets it equal to the gate's own
between-season `s`, a conservative order-of-magnitude stand-in rather than a fitted quantity.
The point of rule 16 is whether the combined rule's power survives a plausible within-season
noise level, not a precise re-derivation of one this repo has not measured -- if a real
within-season sd is measured later, re-run this script rather than editing its numbers by
hand (`docs/method.md` rule 13).

Calls the shipped `experiment.gate` directly (#386: the paired-frame-in interface, which
materialises `summarise`/`per_season` itself now) rather than reimplementing the rule, so what
this measures is the rule the repo actually runs.

    uv run python scripts/rule16_combined_power.py
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from hub.models.experiment import SEASON_CLUSTER, Actions, Ceiling, gate

TRIALS = 10_000
BOOTSTRAP = 200
SIZE_SEED = 0
POWER_SEED = 1

_ACTIONS = Actions(adopt="ADOPT", remove="REMOVE", show="SHOW")

# (name, k seasons, between-season s, a realistic delta, m within-season clusters,
#  unanimity-alone power at that delta -- #357's own table)
CELLS = (
    ("draft", 4, 7.34, 2.0, 20, 0.136),
    ("weekly blend", 4, 0.382, 0.3, 40, 0.378),
)


def _trial(rng: np.random.Generator, *, k: int, s: float, m: int, delta: float) -> pl.DataFrame:
    """One synthetic paired frame: `k` seasons, each `m` within-season cluster means drawn
    around that season's own mean, which is itself drawn around `delta` with spread `s` --
    and the within-season draw uses the same `s` (the stated assumption above)."""
    season_means = rng.normal(delta, s, k)
    rows = []
    for yr, mu in enumerate(season_means):
        cluster_means = rng.normal(mu, s, m)
        for c, cm in enumerate(cluster_means):
            rows.append((yr, c, cm))
    return pl.DataFrame(rows, schema=["season", "unit", "diff"], orient="row")


# The simulation is about the interval half and the tie-aware every-season half -- the two
# terms rule 17's incident was about. It is not about S6's stage-2 precondition (#363: a gate
# that measured no ceiling is NOT-RUNNABLE in both directions), which is a separate branch
# ahead of both. So every trial hands `gate` a ceiling that cannot bind, by construction, and
# says so here rather than leaving it off: with it off, #363 makes every trial NOT-RUNNABLE
# and the ADOPT rate reads 0 whatever the inputs -- a check whose outcome cannot vary with the
# thing it is about (`docs/method.md`, the note beside rule 17). That is how the table this
# script produced for ADR-0019 stopped reproducing the day #363 landed behind it.
NON_BINDING_CEILING = float("inf")


def _adopt_rate(*, k: int, s: float, m: int, delta: float, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    adopts = 0
    for _ in range(trials):
        df = _trial(rng, k=k, s=s, m=m, delta=delta)
        # #386: `gate` takes the paired frame directly now and materialises `summarise`/
        # `per_season` itself -- one call where this used to be two plus the old summary-dict
        # `gate`. `NON_BINDING_CEILING`'s mean, not the scalar itself, is what `gate` reads,
        # so a one-row `Ceiling` carrying it is the frame-in equivalent of the old keyword.
        top = Ceiling("non-binding", np.array([NON_BINDING_CEILING]))
        status, _ = gate(df, cluster=SEASON_CLUSTER, within=("unit",), ceiling=top,
                         actions=_ACTIONS, bootstrap=BOOTSTRAP, seed=seed).verdict
        adopts += status == "ADOPT"
    return adopts / trials


def main() -> None:
    print(f"trials={TRIALS} bootstrap={BOOTSTRAP}\n")
    for name, k, s, delta, m, unanimity_power in CELLS:
        t0 = time.time()
        size = _adopt_rate(k=k, s=s, m=m, delta=0.0, trials=TRIALS, seed=SIZE_SEED)
        power = _adopt_rate(k=k, s=s, m=m, delta=delta, trials=TRIALS, seed=POWER_SEED)
        dt = time.time() - t0
        print(f"{name}: k={k} s={s} m={m} delta={delta}")
        print(f"  combined-rule null size:  {size:.4f}")
        print(f"  combined-rule power:      {power:.4f}")
        print(f"  unanimity-alone power:    {unanimity_power:.3f}  (from #357's table)")
        print(f"  ({dt:.1f}s)\n")


if __name__ == "__main__":
    main()
