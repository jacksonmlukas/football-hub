"""Simulate a season to a champion.

`cost_of_waiting` answers "who will I regret not taking". It does not answer "which pick
most improves my chance of winning", and those come apart in ways that matter: a fourth
good WR has real VOR and almost no effect on a championship, because you can only start
three. Roster construction, positional saturation and week-to-week variance only show up
if you play the season out.

The model is deliberately shallow where shallowness is cheap and honest where it is not:

  * weekly points ~ Normal(mu, sd) truncated at zero, per player, independent
  * mu is expected points per game (xFP), which is less noisy than realised points
  * sd is that player's own week-to-week dispersion
  * games played are drawn per season from the player's own durability history, and the
    weeks he misses are zeros
  * lineups are set optimally in hindsight each week

The last one is the biggest simplification: real managers start the wrong guy. It inflates
everyone's scores roughly equally, so it distorts P(win) far less than it distorts points.

The absence line is new (#183) and replaces a worse simplification rather than adding a
better one. Until 2026-09-10 there was no absence at all: a bust was a talent draw landing
near zero, which -- because weekly spread follows realised talent -- made a player who missed
the season a low-mean **low-variance** player. That is the opposite of an injury, and it is
the one place where being shallow was not honest. `_absence_factor` below says what replaced
it, what it deliberately does not model, and which constant is now carrying absence twice.
"""
from __future__ import annotations

import numpy as np

# The league's shape and the one rule for filling it now live in `hub.league`, a leaf. They
# were declared here, which put them inside the draft package and made four of the six
# `hub.season` modules import a draft simulator to learn how many receivers they start.
#
# Re-exported rather than merely moved: `optimize`, `evaluate`, `leverage`, `backtest` and a
# dozen tests import them from here, and a refactor that breaks its callers to be tidier is
# not an improvement -- the same reason the `hub.models.predict` names below are re-exported.
from hub.league import (  # noqa: F401
    FLEX_CAPACITY,
    FLEX_FROM,
    FLEX_SLOTS,
    PLAYOFF_ROUNDS,
    PLAYOFF_TEAMS,
    REG_SEASON_WEEKS,
    ROSTER,
    STARTERS,
    starting_lineup,
)

# Player-level prediction lives in `hub.models.predict`: what a player does in a week, as
# opposed to how a league works, which is this module. These constants are re-exported
# because calibrate, leverage and the tests import them from here, and a refactor that
# breaks its callers to be tidier is not an improvement.
#
# `moments` is deliberately NOT re-exported. It was, under the old name `weekly_moments`,
# and that alias outlived the function it named: `moments` returns three moments, so the
# name promised a shape the object no longer had. Callers wanting a player-level
# prediction import it from `hub.models.predict`, which is where it lives.
from hub.models.predict import (  # noqa: F401
    MIN_SKEW,
    TALENT_CV,
    TALENT_CV_BY_POS,
    WEEKLY_K,
    WEEKLY_K_POOLED,
    WEEKLY_SKEW,
    WEEKLY_SKEW_POOLED,
    CorrelationReport,
    CorrelationVoid,
    correlated_normal,
    skewed,
    talent_cv_for,
    weekly_skew_for,
)

_skewed = skewed
_correlated_normal = correlated_normal



def lineup_points(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Best legal lineup, vectorised over (sims, weeks).

    scores: (sims, weeks, roster) -- one draw per player per week
    pos:    (roster,) -- position string per player

    Public because `hub.draft.backtest` scores realised seasons with it. A backtest using
    its own lineup rule would answer a slightly different question than the simulator it is
    auditing, and the difference would be invisible -- both would read as "best legal
    lineup". One rule, one owner.
    """
    total = np.zeros(scores.shape[:2])
    bench = []
    for p, n in STARTERS.items():
        idx = np.flatnonzero(pos == p)
        if idx.size == 0:
            continue
        block = -np.sort(-scores[:, :, idx], axis=2)      # descending
        total += block[:, :, :n].sum(axis=2)
        if p in FLEX_FROM and block.shape[2] > n:
            # Up to FLEX_SLOTS of them, not one: with two flex slots the best pair can come
            # from the same position, and taking one candidate per position cannot see that.
            bench.append(block[:, :, n:n + FLEX_SLOTS])
    if bench and FLEX_SLOTS:
        pool = np.concatenate(bench, axis=2)
        total += -np.sort(-pool, axis=2)[:, :, :FLEX_SLOTS].sum(axis=2)
    return total


def _absence_factor(missed, n_sims: int, weeks: int, rng: np.random.Generator) -> np.ndarray:
    """Per (sim, week, player): 0.0 for a week missed, and above 1.0 for a week played.

    Returns a multiplier for the weekly draw, shaped `(n_sims, weeks, len(missed))`.

    **What it does.** `durability.next_season_absence` turns each player's prior-season
    missed games into the mean and spread of what he will miss this season, using the
    published +0.407 persistence. One draw per player per *season* -- not per week -- sets
    how many weeks he misses, and which weeks those are is chosen uniformly. So a season-long
    absence is a real outcome with real probability, and the weeks it costs are zeros rather
    than a shrunken mean. That is #183's first acceptance criterion, and the second follows
    from it: a player with injury history now has strictly more season variance than an
    identical player without one, because a Bernoulli-like factor with a non-degenerate mean
    adds `((1-p)/p) * E[week]^2` to it and takes nothing away.

    **Why the played weeks are scaled up, which looks like a fudge and is not.** `mu` reaching
    this simulator is points per *team* game: `xfp_per_game` is a season total over the team's
    schedule, and `durability.correct_projection` has already marked QB and WR down for
    expected absence on top. Multiplying that by a zero-one mask would price absence a second
    time -- every player quietly discounted by his own miss rate, every downstream constant
    re-levelled, and none of it asked for by this ticket. Points per game *played* is
    `mu / (fraction of games played)`, so dividing the played weeks by that fraction is the
    definitional conversion and not a correction. The consequence is exactly the one wanted:
    the expected season total is unchanged, and only its spread moves.

    **What it does not model, deliberately.** The weeks missed are drawn as a count and
    scattered, not as a block, so the total is right and the run structure is not. Nothing in
    this simulator reads a run -- the schedule is a round robin and the lineup is set weekly
    -- so the difference is invisible to every number computed from here. Byes are absent for
    a different reason: they are known in advance rather than drawn, `hub/draft` has no
    schedule source, and they are #226.

    **The constant that is now carrying absence twice, stated rather than hidden.**
    `TALENT_CV` was fitted on points per team game, so missed time already sits inside it as
    a population average -- `durability`'s own docstring says so, and `calibrate.nominal_for`
    feeds a full season to every player *because* of it. Drawing absence explicitly here does
    not remove it from there, so season-level spread is now overstated by the absence
    variance inside `TALENT_CV`. #183's fifth criterion allows a constant to be refitted or
    explicitly carried with the choice stated: this is the carry. The refit is
    `hub.draft.calibrate` run against the real games-played distribution instead of `full`,
    which needs an archive this change does not touch, and it will move `TALENT_CV` **down**.
    The direction of the error is therefore known: too much season variance, not too little,
    which is the safe side for a decision that already prefers certainty.
    """
    from hub.draft.durability import TEAM_GAMES, next_season_absence

    mu_missed, sd_missed = next_season_absence(missed)
    n = mu_missed.size
    # Draw the season's missed games per player, on the 17-game scale the history is measured
    # on, then convert to the weeks actually simulated. Normal-and-clip rather than a
    # distribution with the right support, because the conditional moments are all the
    # persistence figure determines -- choosing a shape would be choosing a second parameter,
    # which is the thing this ticket's disposition refused.
    drawn = rng.normal(mu_missed[None, :], np.maximum(sd_missed, 0.0)[None, :],
                       size=(n_sims, n))
    np.clip(drawn, 0.0, float(TEAM_GAMES), out=drawn)
    weeks_out = np.rint(drawn * weeks / TEAM_GAMES).astype(np.int64)
    np.clip(weeks_out, 0, weeks, out=weeks_out)
    # Which weeks, uniformly: rank a uniform key per (sim, week, player) and miss the lowest
    # `weeks_out` of them. `argsort` twice is the vectorised "rank within axis".
    keys = rng.random((n_sims, weeks, n))
    rank = keys.argsort(axis=1).argsort(axis=1)
    played = rank >= weeks_out[:, None, :]
    # The mean-preserving scale, from the *expected* miss rate rather than the drawn one, so
    # a player's expected season total does not depend on his own draw. Floored at one game
    # in seventeen: a player expected to miss everything would otherwise divide by zero, and
    # "he plays one game at seventeen times the rate" is a bounded answer to an input the
    # board should never produce.
    play_frac = np.maximum(1.0 - mu_missed / TEAM_GAMES, 1.0 / TEAM_GAMES)
    return played / play_frac[None, None, :]


def _schedule_for(season: int):
    """The season's regular-season schedule through the validated loader -- one seam, so a
    test can stand a frame in without reaching the network."""
    from hub.fetch.nflverse import load
    return load("schedules", seasons=[season],
                cols=["game_id", "season", "week", "home_team", "away_team", "game_type"])


def bye_weeks(season: int) -> dict[str, int]:
    """Each team's bye: the one regular-season week it has no game -- issue #226.

    Derived rather than listed, from the same nflverse schedule `playoff_sos` reads for
    weeks 15-17, through the validated loader rather than inline. A bye is the week a team
    is absent from every fixture; it is not a column anywhere, and the derivation is what
    keeps the placement in step with the schedule the season is actually played on.

    **One per team, or refuse.** The NFL plays 17 games in 18 weeks, so every team sits
    exactly once. A team that sits twice or never is a schedule this league has not modelled
    -- a shortened season, a cancelled game, a partial pull -- and silently taking the first
    bye, or none, would read as a normal season with one week wrong. Raising names the team.

    **Every bye has to fall inside the fantasy season.** `REG_SEASON_WEEKS` is 14 and byes
    run weeks 5-14, so today they all do; a bye in week 15 would be one the simulated season
    cannot see, which is `simulate_weeks`' refusal below and the ticket's fourth criterion --
    the two constants agreeing is a fact the test holds rather than an assumption.

    **What it moves, measured 2026-09-11 on the served 457-player board** (435 placed, 22
    free agents null; byes weeks 5-14, 27 to 79 players a week). `win_probability` over
    `recommend`'s shortlist at picks 3, 22, 27, 46, 51 and 70 from slot 3, common random
    numbers, bye on against bye off, 12 rollouts x 250 seasons: the mean |change in lift|
    is 0.002-0.004 and the largest 0.009, against a per-candidate `lift_se` of 0.0035. The
    leader's name changed at three of the six picks -- and at every one of them, under two
    seeds, each arm's leader sits inside the other arm's co-leader tier (`rank_tiers`), with
    the tiers sharing 3-8 of 7-8 members. So the bye reorders inside ties and not across
    them: it does not move the board ordering at the resolution the board has, which is
    recorded here rather than read as a reason to leave the week unmodelled. It is kept
    because it is the season's shape, not because it changes a pick.
    """
    import polars as pl

    sched = _schedule_for(season)
    reg = sched.filter(pl.col("game_type") == "REG") if "game_type" in sched.columns else sched
    played = pl.concat([reg.select("week", pl.col("home_team").alias("team")),
                        reg.select("week", pl.col("away_team").alias("team"))])
    weeks = sorted(int(w) for w in reg["week"].unique().to_list())
    out: dict[str, int] = {}
    bad: dict[str, list[int]] = {}
    for team in sorted(played["team"].unique().to_list()):
        seen = set(played.filter(pl.col("team") == team)["week"].to_list())
        byes = [w for w in weeks if w not in seen]
        if len(byes) == 1:
            out[str(team)] = byes[0]
        else:
            bad[str(team)] = byes
    if bad:
        raise ValueError(
            f"a team should sit exactly once in a regular season; these do not: {bad}. A "
            f"shortened season, a cancelled game or a partial schedule pull -- refused rather "
            f"than read as a normal season with one week wrong.")
    return out


def attach_bye(board, byes: dict[str, int]):
    """Add `bye_week` to the board: the week each player's team sits, null where the team
    cannot be placed -- a free agent, a team the canon does not know. Never a default: the
    simulator reads 0 as "no bye", and a placement failure is not that. Same shape as
    `playoff_sos.attach_sos`, and the same reason."""
    import polars as pl

    from hub.draft.playoff_sos import canon_team
    lookup = pl.DataFrame({"_team": list(byes), "bye_week": list(byes.values())},
                          schema={"_team": pl.Utf8, "bye_week": pl.Int64})
    keyed = board.with_columns(
        pl.col("team").map_elements(canon_team, return_dtype=pl.Utf8).alias("_team"))
    return keyed.join(lookup, on="_team", how="left").drop("_team")


def simulate_weeks(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                   pos: np.ndarray, n_sims: int, weeks: int = REG_SEASON_WEEKS,
                   rng: np.random.Generator | None = None,
                   talent_cv: float | np.ndarray | None = None,
                   nfl_team: np.ndarray | None = None,
                   skew: np.ndarray | None = None,
                   missed: np.ndarray | None = None,
                   report: CorrelationReport | None = None,
                   bye_week: np.ndarray | None = None) -> np.ndarray:
    """Weekly points for every team. Returns (sims, weeks, teams).

    Realised talent is drawn once per season, then weekly points are drawn around it.
    Without that first draw the projection IS the truth, and any strategy that ranks on
    the projection drafts with perfect foresight while an ADP-following opponent does
    not -- which is not an edge, it is a leak.

    `missed` is each player's prior-season missed games, and it is what makes absence
    absence (#183) -- see `_absence_factor`. **`None` means "not modelled", not "nobody gets
    hurt"**, and it is the honest answer for a caller that has no durability history: the
    board's `missed` column comes from an advisory stage that `build` absorbs on failure, and
    `hub.draft.leverage` sweeps one parameter at a time over a synthetic pool that has no
    history to sweep. On that path the absence branch is not entered at all, so no random
    numbers are consumed and every seeded result predating this change is reproduced byte for
    byte -- which is why the sweeps and the correlation tests did not have to be re-baselined.
    `test_no_history_means_not_modelled_and_never_reaches_the_absence_path` holds the branch
    rather than the agreement, because a default value agrees with itself either way.

    `report` is the caller's `CorrelationReport`, carried through so that a run making many
    simulations counts its unfactorable blocks once for the run rather than losing the
    count with each call's stack frame.
    """
    rng = rng or np.random.default_rng(0)
    # None means per position; a scalar is still accepted, which is what the sweeps in
    # hub.draft.leverage need in order to vary one thing at a time.
    if talent_cv is None:
        talent_cv = talent_cv_for(pos)
    true_mu = mu[None, :] * (1.0 + rng.normal(0.0, talent_cv, size=(n_sims, mu.size)))
    np.clip(true_mu, 0.0, None, out=true_mu)
    # Weekly spread follows *realised* talent, not the projection. Keying it to the
    # projection made busts impossible: a player projected at 15 whose talent went to zero
    # was drawn from N(0, 8.25) and clipped, which is a half-normal averaging 3.3 points a
    # game -- 22% of his projection, manufactured entirely by the clip. Scaling by the
    # realised ratio leaves an average player untouched and lets a collapsed one score
    # nothing, which is what a bust is.
    # Spread follows *realised* talent. Under sd = k*sqrt(mu) that means scaling by
    # sqrt(realised/projected), not by the ratio -- a player who realises a quarter of his
    # projection keeps half his weekly spread, not a quarter of it.
    #
    # **This line is correct for talent and was wrong for absence, which is #183.** A talent
    # collapse genuinely is low-mean and low-spread: a back who loses his job scores a little,
    # every week, predictably. An injury is not that, and until absence was drawn separately
    # below this was the only way the model could express one -- so a missed season arrived
    # here as a bust and came out with *less* variance than a healthy player, which is
    # backwards. Absence no longer routes through the talent draw, so what this scaling means
    # is now what its comment always said it meant.
    ratio = np.divide(true_mu, mu[None, :], out=np.zeros_like(true_mu),
                      where=mu[None, :] > 0)
    sd_eff = (sd[None, :] * np.sqrt(ratio))[:, None, :]
    z = _correlated_normal(rng, (n_sims, weeks, mu.size), pos, nfl_team, report=report)
    # Skew comes from the caller when it has it. `hub.models.predict.moments` already
    # returns a skew column alongside mu and sd; recomputing it here from `pos` agreed only
    # because both routes read the same table, and would go on agreeing right up until one
    # of them stopped being a function of position alone.
    if skew is None:
        skew = weekly_skew_for(pos)
    draws = _skewed(true_mu[:, None, :], sd_eff, skew[None, None, :], z)
    # After the weekly draw and not before it, so the two are independent and so that the
    # `missed is None` path consumes no random numbers at all -- see the docstring.
    if missed is not None:
        draws = draws * _absence_factor(missed, n_sims, weeks, rng)
    # A bye is a scheduled zero, not a draw: the same week in every sim, and no other week
    # touched (#226). It is deliberately *not* mean-preserving, unlike the absence factor
    # above -- `mu` is points per team game and a bye is a week the team does not play, so
    # fourteen fantasy weeks with one bye is thirteen games. And it is orthogonal to that
    # factor: `missed` counts games missed out of the seventeen a team plays, which never
    # includes the bye. Applied after every draw so it consumes no random numbers.
    if bye_week is not None:
        bye = np.asarray(bye_week, dtype=int)
        if bye.shape != (mu.size,):
            raise ValueError(f"bye_week has shape {bye.shape}; expected ({mu.size},)")
        beyond = bye[(bye < 0) | (bye > weeks)]
        if beyond.size:
            raise ValueError(
                f"bye week(s) {sorted(set(beyond.tolist()))} fall outside the {weeks}-week "
                f"simulated season. REG_SEASON_WEEKS and the schedule's bye placement have "
                f"drifted apart -- a bye the fantasy season cannot see is not 'no bye'.")
        on_bye = bye > 0
        if on_bye.any():
            # Row = week index, column = player; True where that player's team sits.
            mask = (np.arange(1, weeks + 1)[:, None] == bye[None, :]) & on_bye[None, :]
            draws = draws * (~mask)[None, :, :]
    out = np.empty((n_sims, weeks, len(rosters)))
    for t, r in enumerate(rosters):
        out[:, :, t] = lineup_points(draws[:, :, r], pos[r]) if r.size else 0.0
    return out


def _round_robin(teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method schedule, repeated until the season is full."""
    rot = list(range(teams))
    base = []
    for _ in range(teams - 1):
        base.append([(rot[i], rot[teams - 1 - i]) for i in range(teams // 2)])
        rot = [rot[0], rot[-1], *rot[1:-1]]
    return [base[w % len(base)] for w in range(weeks)]


def seed_table(pts: np.ndarray, *, reg_weeks: int = REG_SEASON_WEEKS,
               playoff_teams: int = PLAYOFF_TEAMS) -> tuple[np.ndarray, np.ndarray]:
    """Regular-season wins and the playoff field, from weekly points.

    Seeds on wins, ties broken on total points -- ESPN's `playoffSeedingRule
    TOTAL_POINTS_SCORED`. **Regular-season points only**: `pts` now carries playoff weeks
    too, and summing those into the tiebreak would let a team's semi-final performance
    decide the seed it entered the playoffs with.

    Shared rather than duplicated. `leverage.py` re-implemented this loop line for line
    because `champion_probability` gave it no way to ask for the pieces.
    """
    n_sims, _, teams = pts.shape
    wins = np.zeros((n_sims, teams))
    for w, pairs in enumerate(_round_robin(teams, reg_weeks)):
        for a, b in pairs:
            a_won = pts[:, w, a] > pts[:, w, b]
            wins[:, a] += a_won
            wins[:, b] += ~a_won
    key = wins * 10_000.0 + pts[:, :reg_weeks, :].sum(axis=1)
    return wins, np.argsort(-key, axis=1)[:, :playoff_teams]


def champion(pts: np.ndarray, seeds: np.ndarray, sim: int, *,
             reg_weeks: int = REG_SEASON_WEEKS) -> int:
    """Winner of one bracket: seeds 1-2 bye, 3v6 and 4v5, then semis, then the final.

    Playoff weeks index `pts` directly. They used to be taken modulo the array width, so a
    14-week simulation replayed weeks 1-3 as the three playoff rounds -- meaning the games
    that decided the title were the same draws that had already set the seeding.
    """
    f = seeds[sim]
    w = reg_weeks
    alive = [f[0], f[1]]
    for a, b in ((f[2], f[5]), (f[3], f[4])):
        alive.append(a if pts[sim, w, a] > pts[sim, w, b] else b)
    semi = [alive[0] if pts[sim, w + 1, alive[0]] > pts[sim, w + 1, alive[3]] else alive[3],
            alive[1] if pts[sim, w + 1, alive[1]] > pts[sim, w + 1, alive[2]] else alive[2]]
    return int(semi[0] if pts[sim, w + 2, semi[0]] > pts[sim, w + 2, semi[1]] else semi[1])


def champion_probability(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                         pos: np.ndarray, n_sims: int = 400,
                         rng: np.random.Generator | None = None,
                         talent_cv: float | np.ndarray | None = None,
                         nfl_team: np.ndarray | None = None,
                         skew: np.ndarray | None = None,
                         missed: np.ndarray | None = None,
                         report: CorrelationReport | None = None,
                         bye_week: np.ndarray | None = None) -> np.ndarray:
    """P(each team wins the league). Returns (teams,) summing to 1.

    14-week H2H regular season, top 6 seeds, two byes, then single elimination on one-week
    matchups -- read off the live league, not assumed. Three further weeks are simulated so
    the bracket has draws of its own.

    The two byes here are playoff seeding; `bye_week` is the NFL bye #226 is about, one
    per player, handed straight to `simulate_weeks` like `missed`, which is the absence
    model -- see `_absence_factor`. `leverage.py` uses the word in the seeding sense.
    """
    rng = rng or np.random.default_rng(0)
    teams = len(rosters)
    pts = simulate_weeks(rosters, mu, sd, pos, n_sims,
                         REG_SEASON_WEEKS + PLAYOFF_ROUNDS, rng, talent_cv, nfl_team, skew,
                         missed, report, bye_week=bye_week)
    _, seeds = seed_table(pts)
    champs = np.array([champion(pts, seeds, s) for s in range(n_sims)])
    return np.bincount(champs, minlength=teams) / n_sims


