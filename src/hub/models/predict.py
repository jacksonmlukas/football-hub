"""One prediction object: what a player does in a week.

The pieces of this existed but as three implementations of one idea -- `weekly_moments`
handed back two moments, the skewed correlated draw lived inside `simulate_weeks`, and
`hub/season/lineup.py` computed its own group spread. Three copies of one idea is how they
drift apart, and drift is silent.

The split is by subject, not convenience: **this module is what a player does; `hub.draft.season`
is how a league works.** Dispersion laws, skew, talent and teammate correlation live here.
Rosters, schedules, brackets and lineups live there.

Everything here was fitted rather than assumed, and each constant carries its own provenance
below: `TALENT_CV` in `docs/talent-cv.md`, the square-root spread law in
`docs/weekly-spread.md`, the skew in `docs/component-projection.md`, teammate correlation in
`docs/correlation.md`.

**One constraint the unification had to respect.** Component-derived spread was measured
*worse* than the fitted square-root law at predicting a player's actual weekly sd -- mean
error 1.365 against 1.140, P(better) 0.0%. So this object keeps the fitted laws for its
moments and exposes components alongside them rather than deriving one from the other.
Unifying must not quietly swap a validated number for a tidier one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# How wrong a preseason projection typically is about a player's season, as a fraction of
# his projected per-game points. This is the single most important number in the model:
# at 0 the projection is truth and drafting on it is clairvoyance; the larger it gets, the
# more a draft is a lottery and the flatter every candidate's championship equity becomes.
#
# FITTED 2026-08-23 against this league's own past drafts -- `hub.draft.calibrate`, written
# up in `docs/talent-cv.md`. 460 drafted skill players over 2023-25: 0.411, 95% CI
# [0.380, 0.434]. The previous value of 0.35 was a guess and sat 4.6 se low. Refitted the
# same day once `weekly_moments` moved to the square-root law, since the fit subtracts
# weekly sampling and therefore depends on it -- the two constants are coupled.
#
# A draft pick is market opinion recorded before week 1, so E[realized | pick, position] is
# the market's projection and cannot have been revised after the fact. Availability is
# inside the number on purpose: scoring is measured per team game, and the simulator benches
# a low-talent player the same way you bench an injured one.
#
# A single scalar is a compromise. RB fits above it and TE below; see the doc.
#
# **REFITTED 2026-09-11 net of absence, issue #235: 0.42 -> 0.32.** The dispersion the
# fit measures did not move -- 0.408, CI [0.370, 0.453], on the same 460 player-seasons.
# What moved is the inversion behind `nominal`: since #183 the simulator draws missed games
# for itself (`season._absence_factor`), and `calibrate.nominal_for` was still inverting
# through a full season for every player, so the constant carried the absence variance a
# second time. Inverted through the real games-played distribution (sd 2.9-4.3 games by
# position) the nominal is 0.322, with the dispersion interval's ends inverting to
# [0.267, 0.385]. The double count, measured through the simulator at a mean-14 projection
# with two missed games prior: season-total spread was overstated 18% (RB) to 46% (QB).
# Restated in docs/talent-cv.md. Availability is no longer "inside the number on purpose":
# it is drawn, and the number is talent net of it.
TALENT_CV = 0.32

# Per position, from the same fit, shrunk toward the pool in proportion to each position's
# own standard error rather than taken raw -- four positions holding 51 to 200 player-seasons
# do not support four independent numbers.
#
# REFITTED 2026-09-07 under the corrected bootstrap, issue #172. RB 0.50 -> 0.48 and
# TE 0.32 -> 0.33; QB and WR do not move. **No raw estimate changed** -- QB 0.407, RB 0.471,
# WR 0.387, TE 0.282 are what they were. What changed is the standard errors those raw
# numbers carry, and `calibrate._shrink` reads them to decide how much of the spread between
# positions is real: understated errors shrink too little and so overstate the differences.
# The corrected errors are larger by 1.05x (TE) to 2.09x (QB), so every position is pulled
# further toward the pool.
#
# **The reading is weaker than it was.** Only TE now sits beyond two standard errors of the
# pool (-3.8 se). RB was +2.6 se and is +1.7 se, which by this repo's own two-se bar is no
# longer distinguishable from the pool -- so "an early running back is more of a lottery than
# his projection suggests" is now a direction the fit leans rather than a difference it
# establishes. The number is still the best estimate available and is still shrunk toward the
# pool; what it is not any more is significant. See docs/talent-cv.md.
#
# REFITTED 2026-09-11 net of absence (#235), each position inverted through its own games
# distribution: QB 0.42 -> 0.20, RB 0.48 -> 0.38, WR 0.42 -> 0.31, TE 0.33 -> 0.18. The
# shrunk raw values are unchanged (QB 0.408, RB 0.451, WR 0.390, TE 0.315); quarterbacks
# and tight ends move most because their seasons vary most in games played (sd 4.3 and
# 2.9 of ~14) relative to their spread. Which positions differ from the pool is a statement
# about the raw values and is unchanged.
TALENT_CV_BY_POS = {"QB": 0.20, "RB": 0.38, "WR": 0.31, "TE": 0.18}


def talent_cv_for(pos: np.ndarray) -> np.ndarray:
    """Per-player talent dispersion. Anything unfitted (K, DST) falls back to the pool."""
    return np.array([TALENT_CV_BY_POS.get(str(p), TALENT_CV) for p in pos], dtype=float)


# Weekly spread follows sqrt(mean), not the mean. `sd = 0.55 * mu` assumed proportional;
# fitted against 1,174 player-seasons of nflverse weekly scoring the exponent is
# 0.498 +/- 0.012, which is the Poisson value and about 42 se from 1.
#
# This is derived rather than fitted-for-its-own-sake. Weekly points are a sum of
# count-driven components -- receptions, carries, touchdowns -- whose variance grows with
# their mean, so the spread of the total grows with its square root. Touchdowns are the
# lumpy part: 54% of a quarterback's weekly variance and 21% of a receiver's, against 32%
# and 17% of their points. See docs/weekly-spread.md.
#
# Predicting a player's weekly sd this way cuts RMSE about a third versus the old constant.
# Note how little is left between positions once the law is right: most of what looked like
# a position effect in weekly CV was position differences in mean points.
WEEKLY_K = {"QB": 1.88, "RB": 2.07, "WR": 2.13, "TE": 1.99}
WEEKLY_K_POOLED = 2.04

# Weekly scoring is right-skewed, measured within player-season across 2022-25. A normal
# says 0.00, and drawing normals made the simulator believe the typical week was the
# projection -- the observed median week is about 0.90 of the mean, because the mean is
# carried by touchdown spikes. That flatters every floor-based decision.
#
# Quarterbacks are nearly symmetric because passing yardage is high volume and steady, so
# the lumpy touchdown term is a smaller share of their total. See
# docs/component-projection.md, where this falls out of sampling the components rather than
# being imposed here.
WEEKLY_SKEW = {"QB": 0.15, "RB": 0.67, "WR": 0.66, "TE": 0.72}
WEEKLY_SKEW_POOLED = 0.60
# Beyond this the gamma is indistinguishable from a normal and the shift gets numerically
# silly, so fall back rather than push it.
MIN_SKEW = 0.05


def weekly_skew_for(pos: np.ndarray) -> np.ndarray:
    """Per-player weekly skew; anything unfitted falls back to the pooled value."""
    return np.array([WEEKLY_SKEW.get(str(p), WEEKLY_SKEW_POOLED) for p in pos], dtype=float)


def skewed(mean, sd, skew, z):
    """Map standard normal draws to a distribution with this mean, spread and skew.

    Cornish-Fisher rather than a gamma, so that correlation can be applied to `z` before the
    transform: a Gaussian latent is trivially correlatable and a gamma is not. The quadratic
    term supplies the skew, and dividing by sqrt(1 + 2a^2) restores the variance the term
    adds, so mean and spread come out exactly as asked.

    Clipped at zero -- the transform has support below it and nobody scores negative points
    often enough to matter.
    """
    a = np.maximum(skew, MIN_SKEW) / 6.0
    y = (z + a * (z ** 2 - 1.0)) / np.sqrt(1.0 + 2.0 * a ** 2)
    return np.clip(mean + sd * y, 0.0, None)


# Above this share of a run's correlated blocks failing to factor, the draw refuses rather
# than handing back a "correlated" simulation that is mostly not one.
#
# Private and lower-cased in intent, for the reason `board._TD_LUCK_NOTE` is: this is a
# pre-registered guard rather than a fitted quantity, so it must not move the model version.
# See `hub.config.FITTED_MODULES`. Its opposite number is `weekly_gate.VOID_FLOOR`, which
# voids a gate run above a join-failure rate, and it is tight for the same reason that one is
# -- the error is *directional*. A block that will not factor is not drawn with noise added,
# it is drawn under the exact model this structure exists to replace: for a quarterback and
# his own pass catchers, independence turns a nominal 80% interval into 72.9% coverage
# against the correlated 80.4% (docs/correlation.md).
#
# Set from those two published figures rather than from any board: a share s of blocks
# falling back costs roughly s x 7.5 points of coverage on the lineups those blocks price, so
# holding that under half a point wants s below about 0.07. A twentieth is that, rounded
# toward the tighter side, and it is stated here BEFORE it was measured against a real board
# -- which matters, because the first board it met fails it (see docs/correlation.md,
# 2026-09-07) and a floor chosen after the fact would have been chosen to clear it.
_INDEPENDENT_FLOOR = 0.05


# A repaired block whose smallest eigenvalue lands exactly on zero is on the boundary of the
# PSD cone, and a Cholesky of it fails about as often as it succeeds. This lifts the floor off
# the boundary by an amount far below the third decimal any correlation here is quoted to.
_EIG_FLOOR = 1e-8


def nearest_correlation(r: np.ndarray, *, iterations: int = 100,
                        tol: float = 1e-9) -> np.ndarray:
    """The nearest valid correlation matrix to `r`, in Frobenius norm.

    Higham's alternating projections with Dykstra's correction (2002): project onto the PSD
    cone, project onto unit diagonal, repeat. Both sets are convex, so this converges to the
    nearest point of their intersection rather than to whichever one a single clip happens to
    land on -- a bare eigenvalue clip is *not* the nearest correlation matrix, because
    rescaling its diagonal back to one moves it again by an amount nobody measures.

    **This is the repair, and the repair exists so that the fit does not have to be
    constrained.** Estimating within-team correlations under a PSD constraint would mean
    constraining the estimate to be representable, which chooses the answer for the
    solver's convenience and -- worse -- makes `CorrelationReport.independent` permanently
    zero, since a fit that cannot produce an invalid block can never fail to factor. Fit
    freely, repair second, record what the repair cost: then the constraint's price is a
    number in `CorrelationReport.repairs` that somebody can look at, rather than a bias
    nobody can see. Issue #187.

    The final clip is not part of Higham: the algorithm converges *to* the boundary of the
    PSD cone, where the smallest eigenvalue is zero to within rounding and `np.linalg.cholesky`
    is a coin toss. Lifting it to `_EIG_FLOOR` and rescaling the diagonal (a congruence, so
    it preserves PSD) buys a factorisation that always succeeds for 1e-8 of movement.
    """
    y = np.array(r, dtype=float)
    ds = np.zeros_like(y)
    for _ in range(iterations):
        rk = y - ds
        w, v = np.linalg.eigh(rk)
        x = (v * np.maximum(w, 0.0)) @ v.T
        ds = x - rk
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        if np.linalg.norm(y - x, "fro") <= tol:
            break
    w, v = np.linalg.eigh(y)
    if w.min() < _EIG_FLOOR:
        y = (v * np.maximum(w, _EIG_FLOOR)) @ v.T
        d = np.sqrt(np.diag(y))
        y = y / np.outer(d, d)
    np.fill_diagonal(y, 1.0)
    return y


@dataclass(frozen=True)
class BlockRepair:
    """What repairing one team's block cost, in the units the block is quoted in.

    Recorded per team rather than per factorisation: the same team is factored thousands of
    times in a run and the repair is identical every time, so a count here would measure the
    simulation's size and not the model's. `CorrelationReport.repaired` carries the count.

    `moved` is the Frobenius norm of the whole change and `largest_shift` the biggest move
    on any single pairing. Both are published, because they answer different questions: a
    block can move a lot in total while no one correlation moves much, and that is a
    materially different repair from one that halves a single edge.
    """
    team: str
    size: int
    min_eig_before: float
    min_eig_after: float
    moved: float
    largest_shift: float

    def line(self) -> str:
        """One team's repair, for the run's output."""
        return (f"    {self.team}: {self.size}x{self.size}, smallest eigenvalue "
                f"{self.min_eig_before:+.4f} -> {self.min_eig_after:+.4f}, moved "
                f"{self.moved:.4f} (largest single pairing {self.largest_shift:+.4f})")


class CorrelationVoid(RuntimeError):
    """Too much of a correlated draw was drawn independently for it to be one.

    A refusal rather than a degradation, and deliberately not the repo's usual graceful
    fallback: CLAUDE.md's rule is that a failed *fetch* serves last-good state instead of
    erroring, because hours-old ADP beats a stack trace. There is no last-good draw. What the
    fallback would serve here is a simulation reported as correlated and performed
    independently, which is worse than no answer because nothing downstream can tell.
    """


@dataclass
class CorrelationReport:
    """How much of a correlated draw was actually correlated.

    The shape is `hub.draft.board.BuildReport`'s and the reason is the same one: a run that
    quietly did less than it claims used to be invisible, because the only record of the
    degradation was a `continue` statement. There the consumers *sniffed* for the evidence
    and drifted apart; here there was no evidence to sniff for -- a block simulated
    independently leaves a `z` of exactly the shape and dtype a correlated one leaves.

    Owned by the caller and mutated by the draw, which is `board._stage(board, report, ...)`
    rather than a return value, because one run makes thousands of draws (`win_probability`
    is candidates x draft sims) and the count that matters is the run's, not one call's.

    `blocks` counts the blocks that carried a correlation to lose. A team with one skill
    player, or with no quarterback, has an identity block and is not counted either way:
    nothing about it is degraded by being drawn independently, and folding those in would
    make the share read low exactly when the affected teams were few.

    The unit is one factorisation and not one team, because one run factors the same team
    once per draw -- so a run's `blocks` counts into the thousands. `share` is the figure to
    read and is the same either way; the raw counts are kept because a share with no
    denominator cannot say whether it came from one block or ten thousand.

    `repaired` counts factorisations that needed a repair to factor, and `repairs` holds one
    `BlockRepair` per *team* that needed one -- see `BlockRepair` for why the two units
    differ. A repaired block is not a degraded one: it was drawn correlated, under a matrix
    a stated distance from the fitted one. That distance is the thing to look at, and it is
    published rather than counted, because "four blocks were repaired" says nothing about
    whether the repair mattered.
    """
    blocks: int = 0
    independent: int = 0
    repaired: int = 0
    repairs: dict[str, BlockRepair] = field(default_factory=dict)
    floor: float = _INDEPENDENT_FLOOR

    @property
    def share(self) -> float:
        """The share of correlated blocks that fell back to independence."""
        return self.independent / self.blocks if self.blocks else 0.0

    @property
    def repaired_share(self) -> float:
        """The share of correlated blocks that needed repair to factor."""
        return self.repaired / self.blocks if self.blocks else 0.0

    def record(self, repair: BlockRepair) -> None:
        """Count one repaired factorisation, keeping one record per team."""
        self.repaired += 1
        self.repairs.setdefault(repair.team, repair)

    def repair_lines(self) -> list[str]:
        """The per-team repair record, published whenever anything was repaired.

        Pre-registered in #187: *if repair changes a block materially, the size of the change
        is published per team*. Published whether or not it looks material, for the reason
        `note` is said on every run -- a figure that appears only when somebody judged it
        large is a figure whose absence means either "small" or "nobody looked".
        """
        if not self.repairs:
            return []
        return [f"  repaired to the nearest valid correlation matrix "
                f"({len(self.repairs)} team{'s' if len(self.repairs) > 1 else ''}, "
                f"{self.repaired_share:.1%} of factorisations):"] + \
               [self.repairs[t].line() for t in sorted(self.repairs)]

    def degraded(self) -> bool:
        """Whether any block failed. What makes a degraded run distinguishable from a clean
        one -- the question a `continue` in a loop cannot answer afterwards."""
        return self.independent > 0

    def note(self) -> str:
        """One line for the run's output. Said on every run, not only degraded ones: a line
        that appears only when something broke is a line whose absence means either "nothing
        broke" or "nobody looked"."""
        if not self.blocks:
            return "correlation: no team block carried a correlation to apply."
        if not self.independent:
            if self.repaired:
                return (f"correlation: all {self.blocks} team blocks factored, "
                        f"{self.repaired} of them ({self.repaired_share:.1%}) after repair "
                        f"to the nearest valid matrix.")
            return f"correlation: all {self.blocks} team blocks factored."
        return (f"correlation: {self.independent} of {self.blocks} team blocks would not "
                f"factor ({self.share:.1%}); those teams' players were simulated "
                f"independently.")

    def check(self) -> None:
        """Refuse a run that dropped more than the floor. Raises `CorrelationVoid`."""
        if self.blocks and self.share > self.floor:
            raise CorrelationVoid(
                f"{self.independent} of {self.blocks} team correlation blocks would not "
                f"factor ({self.share:.1%}), against a floor of {self.floor:.0%}. Those "
                f"teams' players would be simulated independently, which is the model the "
                f"correlation structure exists to replace -- so this is not a correlated "
                f"simulation and is not reported as one. Check `TEAMMATE_RHO` and the "
                f"positions on the board that produced these blocks.")


# Factored blocks, keyed on the block itself. One run factors the same handful of blocks
# thousands of times -- `win_probability` is candidates x draft-sims x seasons -- and the
# repair below is an eigendecomposition per iteration, which is affordable once per distinct
# block and not once per draw.
#
# **Keyed on the matrix and not on the positions that built it.** A positions key would be
# stale the moment `TEAMMATE_RHO` changed under it, which is exactly what a refit does and
# exactly what a test that monkeypatches the table does -- so the cache would quietly serve a
# factor of the old numbers and the test would pass against a model nobody is running.
#
# The cached value carries no team name. Two teams with the same positions build the *same*
# block and so share a key, and a cached `BlockRepair` would report the second team's repair
# under the first team's name -- which is the one thing the per-team record exists to get
# right. The measured quantities are cached; the name is attached per call.
_FACTORS: dict[bytes, tuple[np.ndarray | None, tuple[int, float, float, float, float] | None]] = {}
_FACTOR_CACHE_MAX = 4096


def _factor(r: np.ndarray, team: str) -> tuple[np.ndarray | None, BlockRepair | None]:
    """Cholesky factor for one team's block, repairing it first if it will not factor.

    Returns `(chol, repair)`. `chol` is None only if the *repaired* block will not factor
    either, which is what leaves `CorrelationReport.independent` able to fire at all: repair
    is not assumed to work, it is attempted and checked.
    """
    key = r.tobytes()
    got = _FACTORS.get(key)
    if got is None:
        got = _measure(r)
        if len(_FACTORS) < _FACTOR_CACHE_MAX:
            _FACTORS[key] = got
    chol, facts = got
    if facts is None:
        return chol, None
    size, before, after, moved, largest = facts
    return chol, BlockRepair(team=team, size=size, min_eig_before=before,
                             min_eig_after=after, moved=moved, largest_shift=largest)


def _measure(r: np.ndarray):
    """Factor one block, repairing if needed. The team-independent half of `_factor`."""
    if not np.isfinite(r).all():
        # **LAPACK is not a finiteness check, and on a NaN block it is not reliably a
        # positive-definiteness check either.** Every pivot comparison against NaN is false, so
        # `dpotrf` never finds the non-positive leading minor it refuses on: some builds report
        # success and hand back a NaN-filled factor. This repo hit exactly that split -- the
        # void fired on macOS Accelerate and did not on Linux OpenBLAS, so the same NaN in
        # `TEAMMATE_RHO` refused on one machine and produced silent NaN draws on the other.
        # Refuse here, where the answer does not depend on which BLAS the machine was built
        # against. A non-finite block is unrepairable by construction: `nearest_correlation`
        # eigendecomposes, and there is no nearest correlation matrix to a matrix with no
        # finite entries to be near.
        return None, None
    try:
        return np.linalg.cholesky(r), None
    except np.linalg.LinAlgError:
        pass
    # The repair is attempted, not assumed. `nearest_correlation` throws rather than returns
    # on a block that is not merely non-PSD -- a NaN out of a bad refit is the case -- and an
    # unrepairable block has to leave the run the way it always did, drawn independently and
    # counted, rather than taking the whole simulation down with it.
    try:
        fixed = nearest_correlation(r)
        chol = np.linalg.cholesky(fixed)
    except np.linalg.LinAlgError:
        return None, None
    delta = fixed - r
    return chol, (int(r.shape[0]),
                  float(np.linalg.eigvalsh(r).min()),
                  float(np.linalg.eigvalsh(fixed).min()),
                  float(np.linalg.norm(delta, "fro")),
                  float(delta.flat[np.abs(delta).argmax()]))


def correlated_normal(rng, size, pos, nfl_team, *,
                      report: CorrelationReport | None = None):
    """Standard normals correlated between teammates, independent otherwise.

    Only the quarterback's edges carry anything -- QB-WR +0.232, QB-TE +0.225, QB-RB +0.054,
    everything else within a few points of zero (docs/correlation.md). Applied by Cholesky
    on each NFL team's own small block, which is exact and costs nothing at 32 teams.

    **A block that will not factor is counted and, past a floor, refused.** It used to be
    caught and skipped, so that team's players were drawn independently with no counter, no
    warning and nothing recorded. `report` is the caller's own `CorrelationReport` when it
    wants the count to survive the call; without one the accounting still happens locally,
    so the refusal holds for every caller rather than only the instrumented ones.
    """
    z = rng.standard_normal(size)
    if nfl_team is None:
        return z
    report = CorrelationReport() if report is None else report
    teams = np.asarray(nfl_team, dtype=object)
    for team in {t for t in teams.tolist() if t is not None and t not in NOT_A_TEAM}:
        idx = np.flatnonzero(teams == team)
        if idx.size < 2:
            continue
        r = np.eye(idx.size)
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                r[i, j] = r[j, i] = teammate_rho(str(pos[idx[i]]), str(pos[idx[j]]))
        if not np.any(r - np.eye(idx.size)):
            continue
        # GUARD a-block-that-will-not-factor-is-counted [unit/test_predict.py]: deleting
        # either line puts this back to a bare `continue`, which simulates that team
        # independently and leaves nothing anywhere saying so.
        report.blocks += 1
        chol, repair = _factor(r, str(team))
        if chol is None:
            report.independent += 1
            continue
        # /GUARD
        # GUARD a-block-that-will-not-factor-is-repaired [unit/test_predict.py]: dropping
        # the record leaves the run silently drawing a team under a matrix it did not fit.
        if repair is not None:
            report.record(repair)
        # /GUARD
        z[..., idx] = z[..., idx] @ chol.T
    report.check()
    return z


# Values in a board's team column that are not a team. Free agents share the label and share
# no quarterback, so correlating them is correlating strangers -- on the 2024 board that is a
# single 43-player "team" whose block is the worst-conditioned on the board by an order of
# magnitude (smallest eigenvalue -0.914, against -0.335 for the worst real one).
#
# **It has to be excluded here rather than repaired.** Before #187 that block failed to
# factor and fell back to independence, which is the right answer for a free agent and was
# reached by accident. Repair would have turned an accidentally-correct independent draw into
# a confidently-wrong correlated one -- so the repair made this worse until the exclusion
# landed beside it. `playoff_sos.canon_team` has held the same rule since it was written.
NOT_A_TEAM = frozenset({"FA", ""})


# Within-game correlation between teammates, measured on standardised weekly points,
# 2022-25. Only the quarterback edges carry anything: a quarterback and a receiving
# teammate move together (+0.23), and everything else is inside +/-0.06 of zero --
# including two receivers on the same team, at +0.014, which is not what the folklore says.
#
# The gate in docs/correlation.md is what makes this worth carrying: for a lineup holding a
# quarterback and his own pass catchers, treating them as independent gives an 80% interval
# that covers 72.9% of the time. Adding these puts it at 80.4%. For a lineup with no
# quarterback, independence is already calibrated and this changes nothing.
TEAMMATE_RHO: dict[tuple[str, str], float] = {
    ("QB", "WR"): 0.232, ("QB", "TE"): 0.225, ("QB", "RB"): 0.054,
}


def teammate_rho(a: str, b: str) -> float:
    """Correlation between two teammates by position. Zero for anything unmeasured."""
    return TEAMMATE_RHO.get((a, b), TEAMMATE_RHO.get((b, a), 0.0))


def group_sd(mu_sd_pos_team) -> float:
    """Spread of a group's combined score, counting teammate covariance.

    `mu_sd_pos_team` is an iterable of (sd, position, nfl_team). Players on different NFL
    teams contribute no covariance; a summed variance that ignores the ones who do is the
    overconfidence the L1 gate measured.
    """
    rows = list(mu_sd_pos_team)
    var = sum(float(sd) ** 2 for sd, _, _ in rows)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            sd_i, pos_i, team_i = rows[i]
            sd_j, pos_j, team_j = rows[j]
            if team_i is not None and team_i == team_j:
                var += 2.0 * teammate_rho(str(pos_i), str(pos_j)) * float(sd_i) * float(sd_j)
    return float(max(var, 0.0) ** 0.5)


def blend() -> pl.Expr:
    """The forecast everything plays on: the market's forward projection and the xFP rate.

    One expression rather than two copies. `hub.draft.board` builds it at draft time and
    `hub.season.roster` rebuilds it in-season against a refreshed market half, and those two
    must not be able to disagree about what `proj_blend` means -- `docs/next.md` names a
    second implementation of one idea as how they drift.

    The coalesce is the whole design: average the two when both exist, otherwise take
    whichever does. A player with no prior season has no xFP of his own, so the board
    interpolates one from consensus rank (`board._impute_xfp`); that value is a rank
    transform rather than an observation, which matters when the market half is fresher than
    the rank -- see `hub.season.roster.market`.
    """
    return pl.coalesce(
        (pl.col("proj_ppg") + pl.col("xfp_per_game")) / 2.0,
        pl.col("proj_ppg"), pl.col("xfp_per_game"),
    ).alias("proj_blend")


def moments(xp: pl.DataFrame, floor_sd: float = 0.0) -> pl.DataFrame:
    """Per-player weekly mean, spread and skew.

    mu comes from expected points rather than realised: xFP already strips the week-to-week
    luck being re-added, so using realised points would double-count variance and make every
    roster look more volatile than it is.

    `floor_sd` defaults to zero. The old floor of 2.0 gave a player projected at nothing a
    real weekly spread, which the best-lineup rule -- a max over the roster -- turned into
    free points off the end of the bench. sqrt(0) is 0, which is what an unprojected player
    should carry.
    """
    cols = [pl.col(c) for c in ("proj_blend", "proj_ppg", "xfp_per_game")
            if c in xp.columns]
    if not cols:
        raise ValueError(
            "moments needs one of proj_blend, proj_ppg or xfp_per_game; "
            f"got {sorted(xp.columns)}")
    mu = pl.coalesce(*cols).fill_null(0.0)
    pos_col = ("position" if "position" in xp.columns
               else ("pos" if "pos" in xp.columns else None))
    if pos_col is None:
        k: pl.Expr = pl.lit(WEEKLY_K_POOLED)
        sk: pl.Expr = pl.lit(WEEKLY_SKEW_POOLED)
    else:
        # cast + fill_null: an all-null position column comes through as dtype Null, which
        # replace_strict refuses outright rather than defaulting.
        base = pl.col(pos_col).cast(pl.Utf8).fill_null("")
        k = base.replace_strict(WEEKLY_K, default=WEEKLY_K_POOLED,
                                return_dtype=pl.Float64)
        sk = base.replace_strict(WEEKLY_SKEW, default=WEEKLY_SKEW_POOLED,
                                 return_dtype=pl.Float64)
    return xp.with_columns(mu.alias("mu")).with_columns(
        pl.max_horizontal(k * pl.col("mu").clip(0.0).sqrt(), pl.lit(floor_sd)).alias("sd"),
        sk.alias("skew"))


def components(pick: float, position: str, proj_ppg: float) -> dict[str, float]:
    """The component line behind a points projection.

    Here rather than only in `hub.models.volume` so a consumer wanting stats instead of
    points -- the props audit, a future volume model -- reads them off the same object that
    produced the moments.
    """
    from hub.models.volume import decompose
    return decompose(pick, position, proj_ppg)
