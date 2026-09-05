"""Data contracts. The Week 7 failure mode is not a crash -- it is ESPN silently
renaming a field and your projections going quietly wrong for three weeks.

Every fetch function asserts its contract at the boundary. Violations raise loudly
and the pipeline serves last-good state rather than propagating bad data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import polars as pl


class ContractViolation(Exception):
    pass


def _family(dt: Any) -> str:
    """The dtype family a column belongs to.

    Families, not exact dtypes, because the exact one is not the invariant anybody holds:
    nflverse ships a count as `Int32` one season and `Int64` the next and nothing downstream
    cares, while a count arriving as `Utf8` breaks every arithmetic expression that touches
    it. Comparing families catches the second and ignores the first.

    `Null` is its own family and matches nothing. That is the case this check exists for: a
    source that returns no rows, or a column of all nulls, comes back typed `Null`, passes
    every column-presence check, and then silently produces nulls wherever it is used.
    `fetch/espn.py:161` hand-wrote a `schema=` to work around exactly that.
    """
    if dt == pl.Null:
        return "null"
    if dt == pl.Boolean:                       # before numeric: a flag is not a count
        return "bool"
    if dt.is_temporal():
        return "temporal"
    if dt.is_numeric():
        return "numeric"
    if dt == pl.Utf8:
        return "string"
    return str(dt)


# The sentence a violation carries when the declaration it broke has never met a response.
# Both CFBD contracts were read off the documentation on a machine with no key, so their
# first red build has two suspects -- the guess and the source -- and the reader should be
# told which to open first.
#
# Two sentences, because there are two ways not to have met a response and they send the
# reader to different places. "Written from documentation" is a claim about how a contract
# was authored, true of the CFBD pair and not known of anything else: five of the fourteen
# are validated against no frozen payload at all, so nobody has measured whether they have
# met live data. Saying nothing about those was the defect in #66 -- the flag defaulted to
# verified, so their violations read as "the source broke" on no evidence either way.
# GUARD unverified-note [contracts/test_every_contract_is_applied.py]: names the suspect
_UNVERIFIED_NOTE = (
    ". NOTE: this contract was written from documentation and has never been "
    "checked against a live response -- suspect the declaration before the source")
_UNMEASURED_NOTE = (
    ". NOTE: nothing in this repo records whether this contract has ever been checked "
    "against a real response -- the declaration is as much a suspect as the source")
_PROVENANCE_NOTES: dict[bool | None, str] = {False: _UNVERIFIED_NOTE, None: _UNMEASURED_NOTE}
# /GUARD


@dataclass(frozen=True)
class Contract:
    name: str
    required: dict[str, type]                      # column -> polars dtype family
    non_null: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    min_rows: int = 1
    # Whether this declaration has ever been checked against a real response. Two were
    # written from documentation and never run, so their first failure is as likely to mean
    # "the guess was wrong" as "the source broke" -- and a red build should say which is the
    # likelier suspect rather than leaving the reader to work it out.
    #
    # Three states, and `None` -- unmeasured -- is the default. It used to default to `True`,
    # which made silence a claim: six of the eleven contracts were validated against no frozen
    # payload at all, and every one of them said it had met live data on no evidence either
    # way. A contract added from documentation, exactly like the CFBD pair below, inherited
    # that claim and nothing went red (#66). Five are still in that position and nobody has
    # measured them, so `None` says that and no more; guessing an answer for them would be
    # worse than saying nothing. The sixth was `ESPN_SCOREBOARD`, and what moved it was
    # routing its capture through a validation -- not anyone deciding it had met live data.
    #
    # Not a free-text claim. `tests/contracts/test_every_contract_is_applied.py` resolves
    # which frozen payload each contract is validated against and requires this flag to agree
    # with that payload's provenance, which `tests/golden/fixtures/README.md` records in the
    # filename: a capture requires `True`, a hand-built shape requires `False`, and no
    # payload at all requires `None`. Flipping either CFBD contract to `True` fails there, by
    # name -- until this was written, flipping both left all twenty tests in that file green.
    # The way off `None` is to validate the contract against a frozen payload, not to edit a
    # list somewhere.
    verified_against_live: bool | None = None

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        problems = []
        if df.height < self.min_rows:
            problems.append(f"{df.height} rows < min {self.min_rows}")
        missing = set(self.required) - set(df.columns)
        if missing:
            problems.append(f"missing columns: {sorted(missing)}")
        # The dtypes in `required` were declared from the start and read by nothing -- the
        # mapping was used as `set(self.required)` and never for its values, so a retyped or
        # all-null column passed every contract in the repo. The module docstring names
        # renaming as the failure mode; retyping is the same failure with a quieter symptom.
        # GUARD dtypes-are-checked [unit/test_fetch_nflverse.py]: a retype is caught
        for col, declared in self.required.items():
            if col not in df.columns:
                continue                        # already reported as missing
            got, want = _family(df.schema[col]), _family(declared)
            if got != want:
                problems.append(f"{col} is {df.schema[col]} ({got}), declared {want}")
        # /GUARD
        for c in self.non_null:
            if c in df.columns and df[c].null_count():
                problems.append(f"{c} has {df[c].null_count()} nulls")
        for c in self.unique:
            if c in df.columns and df[c].n_unique() != df.height:
                problems.append(f"{c} not unique ({df[c].n_unique()}/{df.height})")
        for c, (lo, hi) in self.ranges.items():
            if c in df.columns:
                mn, mx = df[c].min(), df[c].max()
                if mn is not None and (cast(float, mn) < lo or cast(float, mx) > hi):
                    problems.append(f"{c} range [{mn}, {mx}] outside [{lo}, {hi}]")
        if problems:
            # `.get`, so only the two states that have something to say add a sentence --
            # `True` is the contract that has met a real response and needs no caveat.
            note = _PROVENANCE_NOTES.get(self.verified_against_live, "")
            raise ContractViolation(f"{self.name}: " + "; ".join(problems) + note)
        return df


# The board frame is the widest interface in the repo: ~14 modules read columns off it by
# name, and for a long time this contract declared three of them and was applied to none.
# The columns below are the ones something downstream reads *unconditionally* -- anything
# optional (edge, proj_blend, td_luck, sos) stays out, because those legitimately go missing
# when a fetch degrades and `build` is written to keep going.
#
# `ecr_sd` is here under that name on purpose. FantasyPros calls it `sd`, which is also what
# `hub.models.predict.moments` calls a weekly points spread; both landed on frames derived
# from this one.
DRAFT_BOARD = Contract(
    name="draft_board",
    required={"player": pl.Utf8, "pos": pl.Utf8, "ecr": pl.Float64,
              "xfp_per_game": pl.Float64, "games": pl.UInt32, "vor": pl.Float64,
              "consensus_rank": pl.Float64},
    non_null=("player", "ecr", "pos"),
    unique=("player",),
    ranges={"ecr": (1, 1000)},
    min_rows=300,
)

FF_OPPORTUNITY = Contract(
    name="ff_opportunity",
    required={"player_id": pl.Utf8, "position": pl.Utf8, "total_fantasy_points_exp": pl.Float64},
    non_null=("player_id",),
    ranges={"total_fantasy_points_exp": (-10, 80)},
    min_rows=1000,
    # Checked against `nflverse_ff_opportunity.json`, a real 2025 capture. Stated because
    # the default is now `None` -- see `verified_against_live`.
    verified_against_live=True,
)

PBP = Contract(
    name="pbp",
    required={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32},
    non_null=("game_id", "season", "week"),
    # Ranges from three observed seasons (2023-25, 147,928 plays) widened ~20%, per the
    # rule in the data-contracts skill: set them from history, not theory. Observed epa
    # ran -12.69..8.88 and yards_gained -34..98.
    #
    # Deliberately NOT non_null: posteam. It is null on 8,080 of those plays -- kickoffs,
    # timeouts, end-of-quarter rows -- so requiring it would fail every honest refresh.
    ranges={"week": (1, 22), "epa": (-16, 12), "wp": (0, 1), "yards_gained": (-45, 120)},
    min_rows=1000,
    # Checked against `nflverse_pbp.json`, a real 2025 capture. Stated because the default
    # is now `None` -- see `verified_against_live`.
    verified_against_live=True,
)

# Play-level personnel and alignment. The scheme layer's foundation: who was on the field,
# what shape they were in, and what the defence showed.
#
# `offense_formation` is null on ~20% of plays (9,237 of 45,919 in 2024) -- special teams and
# plays the charter did not resolve -- so it is required but explicitly NOT non_null. Nor is
# there a `season` column: rows key on `nflverse_game_id` and `play_id`, and the season is a
# load-time argument. Declaring one would fail every honest refresh.
PARTICIPATION = Contract(
    name="nflverse_participation",
    required={"nflverse_game_id": pl.Utf8, "play_id": pl.Float64,
              "offense_personnel": pl.Utf8, "defense_personnel": pl.Utf8,
              "defenders_in_box": pl.Int32, "offense_formation": pl.Utf8},
    non_null=("nflverse_game_id", "play_id"),
    ranges={"defenders_in_box": (0, 12)},
    min_rows=1000,
)

# FTN's charting: motion, play action, screens, blitzers, box count. Complements
# PARTICIPATION rather than duplicating it -- that one is personnel, this one is intent.
FTN_CHARTING = Contract(
    name="nflverse_ftn_charting",
    required={"nflverse_game_id": pl.Utf8, "nflverse_play_id": pl.Int32,
              "season": pl.Int32, "week": pl.Int32,
              "is_play_action": pl.Boolean, "is_motion": pl.Boolean,
              "n_defense_box": pl.Int32},
    non_null=("nflverse_game_id", "nflverse_play_id", "season", "week"),
    ranges={"week": (1, 22), "n_defense_box": (0, 12), "n_blitzers": (0, 11)},
    min_rows=1000,
)

PLAYER_STATS = Contract(
    name="nflverse_player_stats",
    required={"player_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32,
              "week": pl.Int32, "fantasy_points_ppr": pl.Float64},
    non_null=("player_id", "season", "week"),
    # A single full-PPR week has run past 60 points and, with fumbles and interceptions,
    # can go negative. 100 leaves room for an outlier without admitting a units change.
    ranges={"week": (1, 22), "fantasy_points_ppr": (-30, 100)},
    min_rows=1,
)

SCHEDULES = Contract(
    name="nflverse_schedules",
    required={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
              "home_team": pl.Utf8, "away_team": pl.Utf8},
    non_null=("game_id", "season", "week", "home_team", "away_team"),
    unique=("game_id",),
    # spread_line is the home side, positive when the home team is favoured. The widest
    # NFL closing spreads on record sit around 26.5; 40 leaves room without admitting a
    # sign flip, which would show as a plausible number on the wrong team.
    ranges={"week": (1, 22), "spread_line": (-40, 40), "total_line": (20, 80)},
    min_rows=1,
    # Checked against `nflverse_schedules.json`, a real 2025 capture. Stated because the
    # default is now `None` -- see `verified_against_live`.
    verified_against_live=True,
)

CFBD_GAMES = Contract(
    name="cfbd_games",
    required={"id": pl.Int64, "season": pl.Int64, "week": pl.Int64,
              "homeTeam": pl.Utf8, "awayTeam": pl.Utf8},
    non_null=("id", "season", "week", "homeTeam", "awayTeam"),
    unique=("id",),
    # CFB runs longer than the NFL: 15 regular-season weeks plus postseason.
    ranges={"week": (1, 20), "homePoints": (0, 120), "awayPoints": (0, 120)},
    min_rows=1,
    # No CFBD key on the machine this was written on, so the shape below is read off
    # the documentation rather than off a response. See `verified_against_live`.
    verified_against_live=False,
)

CFBD_LINES = Contract(
    name="cfbd_lines",
    required={"id": pl.Int64, "season": pl.Int64, "week": pl.Int64,
              "homeTeam": pl.Utf8, "awayTeam": pl.Utf8},
    non_null=("id", "homeTeam", "awayTeam"),
    unique=("id",),
    # College spreads reach much further than NFL ones -- 50+ happens in September.
    ranges={"week": (1, 20)},
    min_rows=1,
    # No CFBD key on the machine this was written on, so the shape below is read off
    # the documentation rather than off a response. See `verified_against_live`.
    verified_against_live=False,
)

ODDS_SNAPSHOT = Contract(
    name="odds_snapshot",
    required={"game_id": pl.Utf8, "close_spread": pl.Float64,
              "captured_at": pl.Datetime},
    non_null=("game_id", "close_spread", "captured_at"),
    # Deliberately NOT unique on game_id: several snapshots per game is the entire point,
    # and it is what makes AS_OF_LINES more than a normal join.
    ranges={"close_spread": (-40, 40)},
    min_rows=1,
)

ESPN_SCOREBOARD = Contract(
    name="espn_scoreboard",
    required={"id": pl.Utf8, "state": pl.Utf8, "home": pl.Utf8, "away": pl.Utf8},
    non_null=("id", "state", "home", "away"),
    unique=("id",),
    # Zero games is a fact about the day, not a broken scoreboard -- there is no NFL slate in
    # February and the deploy runs all year. `min_rows=1` here asserted that a game is always
    # on, which is the declaration being wrong rather than the source; found by applying it.
    min_rows=0,
    # Checked against `espn_scoreboard.json`, a real 2026-09-05 capture of the public
    # college scoreboard (`groups=80`) -- the NFL board carried no in-progress game that day,
    # and the live path is the one worth freezing. Stated because the default is now `None`
    # -- see `verified_against_live`.
    #
    # It said `None` until 2026-09-05 while that capture sat in the tree, because the only
    # test reading the file asserted four fields by hand rather than putting a frame through
    # this contract, and the resolver has nothing to see in a hand-written assert. The flag
    # moved when the validation was wired; declaring it would have been the invented answer
    # the resolver exists to refuse.
    verified_against_live=True,
)


# FantasyPros' consensus archive, as DynastyProcess republishes it. Not season-partitioned:
# one row per (scrape, ranking page, player), 1.83M rows from 2019-12-27 to 2026-09-04
# measured on 2026-09-05, which is why `hub.fetch.nflverse` reaches it by page type and
# as-of rather than by a season list.
#
# The columns are the ones a reader takes. `page_type` because 47 ranking pages are stacked
# in this one frame on their own ECR scales and `hub.draft.board._select_consensus` refuses
# without it; `sd`, `best` and `worst` because the same function selects them; `scrape_date`
# because `APPEND_ONLY` bounds the archive on it, and a row that cannot be placed in time
# cannot be pinned.
#
# Two columns are deliberately NOT non_null, both measured over the whole archive on
# 2026-09-05: `player` is null on 34 rows and `ecr` on 104. Declaring either would fail
# every honest refresh, and `_select_consensus` already drops null-ECR rows itself.
#
# `id` is left out entirely rather than declared: the `draft` page types it `Int64` and the
# `all` archive types it `Utf8`, so one declaration would be wrong for one of the two pages
# this contract covers. Nothing in the repo reads it.
#
# Ranges from the whole archive (ecr 1.0-999.5, sd 0-449, best 0-1000, worst 1-1000),
# widened past the observed edge in the direction a FantasyPros page could plausibly grow.
FF_RANKINGS = Contract(
    name="nflverse_ff_rankings",
    required={"page_type": pl.Utf8, "player": pl.Utf8, "pos": pl.Utf8, "team": pl.Utf8,
              "ecr": pl.Float64, "sd": pl.Float64, "best": pl.Float64,
              "worst": pl.Float64, "scrape_date": pl.Utf8},
    non_null=("page_type", "scrape_date"),
    ranges={"ecr": (0, 1200), "sd": (0, 600), "best": (0, 1200), "worst": (0, 1200)},
    min_rows=1,
    # Checked against `nflverse_ff_rankings.json`, a real capture of the `all` archive.
    verified_against_live=True,
)

# The weekly injury report. `report_status` is the designation `hub.models.injury` fits its
# retention table on and `hub.models.panel.injury_severity` reads for the screen's ordinal.
#
# It is required and explicitly NOT non_null: it is null on 21,490 of 40,204 rows over
# 2019-25, because a player on the report with no game designation is the ordinary case, not
# a break. `practice_status` is null on 45 of those and is out of `non_null` for the same
# reason. What must be there is the row's identity -- who, which week, which team.
#
# `season` and `week` are declared `Int32`, which is what a single-season load returns; a
# multi-season load comes back `Float64` because nflreadpy concatenates seasons with
# `diagonal_relaxed`. Both are the numeric family, which is the level `_family` compares at
# and the reason it compares there.
INJURIES = Contract(
    name="nflverse_injuries",
    required={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8,
              "game_type": pl.Utf8, "gsis_id": pl.Utf8, "position": pl.Utf8,
              "full_name": pl.Utf8, "report_status": pl.Utf8,
              "practice_status": pl.Utf8},
    non_null=("season", "week", "team", "game_type", "gsis_id", "full_name"),
    # Deliberately NOT unique on gsis_id: a player appears once per week, and two rows for
    # one player-week exist in the archive besides (2 of 11,814 over 2023-24).
    ranges={"week": (1, 22)},
    min_rows=1,
    # Checked against `nflverse_injuries.json`, a real 2024 capture.
    verified_against_live=True,
)

# Pro Football Reference's snap counts. `offense_pct` is the one column the repo reads --
# `hub.models.panel.snap_share` and `hub.models.spread` both take it -- and it is the reason
# the percentage ranges are here: PFR publishes these as fractions, and the failure worth
# catching is the day they arrive as whole percents, which would multiply every snap share
# by a hundred and read as a plausible number in a column nobody prints.
#
# `1.05`, not `1.0`. Measured over 2019-25, `st_pct` reaches 1.01 -- PFR's rounding, not a
# units change -- so a ceiling at 1 would fail an honest refresh. `defense_pct` and `st_pct`
# ride along unread because a units change would hit all three at once and they cost nothing.
#
# Not unique on anything a single column can express: the key is (game_id, pfr_player_id),
# which the 181,477 rows over 2019-25 do respect and this contract cannot say.
SNAP_COUNTS = Contract(
    name="nflverse_snap_counts",
    required={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
              "game_type": pl.Utf8, "player": pl.Utf8, "pfr_player_id": pl.Utf8,
              "position": pl.Utf8, "team": pl.Utf8, "opponent": pl.Utf8,
              "offense_snaps": pl.Float64, "offense_pct": pl.Float64,
              "defense_pct": pl.Float64, "st_pct": pl.Float64},
    non_null=("game_id", "season", "week", "game_type", "player", "pfr_player_id", "team"),
    ranges={"week": (1, 22), "offense_snaps": (0, 130), "offense_pct": (0, 1.05),
            "defense_pct": (0, 1.05), "st_pct": (0, 1.05)},
    min_rows=1,
    # Checked against `nflverse_snap_counts.json`, a real 2024 capture.
    verified_against_live=True,
)
