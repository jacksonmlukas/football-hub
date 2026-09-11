"""Data contracts. The Week 7 failure mode is not a crash -- it is ESPN silently
renaming a field and your projections going quietly wrong for three weeks.

Every fetch function asserts its contract at the boundary. Violations raise loudly
and the pipeline serves last-good state rather than propagating bad data.

**A contract has two verbs, and for a long time it had one.** `validate` refuses a frame
that breaks the declaration. `conform` hands a consumer the columns it names, in the shape
this file declares them -- because refusal is not an answer a module that has to keep going
can use, and the one that could not use it grew a private copy of the schema instead. That
copy is what issue #132 was: `SNAP_COUNTS` bounded `offense_pct` and named the day PFR ships
whole percents as the failure it exists to catch, while `hub.models.spread.snap_usage` --
a module this file names as its own consumer -- detected the same case forty lines away and
divided by a hundred, with nothing pointing at the other. A declaration nobody can read is a
declaration somebody will restate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import polars as pl

NOT_FITTED_BECAUSE = (
    "declared plausibility bounds, not fitted inputs. A range here says what a source may "
    "plausibly return -- SNAP_COUNTS bounds offense_pct at [0, 1.05], measured over 2019-25 "
    "where PFR's own rounding reaches 1.01 -- so moving one changes what the fetch layer "
    "refuses. The one number here a consumer's arithmetic can reach is a Normalisation's, and "
    "it is the same kind of number read from the other side: 0.01 is what a percent *is*, and "
    "1.5 is a height no honest fraction reaches. Stated as a module rather than constant by "
    "constant because every float in this file is that kind of number by construction, and "
    "because #71 established that adding a data source must not move a model version: the "
    "eleven contracts before #33 held only integer bounds and this scan had never met one of "
    "these. "
)


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
_VERIFICATION_NOTES: dict[bool | None, str] = {False: _UNVERIFIED_NOTE, None: _UNMEASURED_NOTE}
# /GUARD


def _announce(name: str, rescaled: dict[str, str]) -> None:
    """Say, on the terminal, that a declared repair fired.

    A repair that *succeeds* was silent until #140, and the silence became load-bearing the
    day the repaired frame started reaching the cache: a whole-percent refresh is rescaled,
    written, pinned and served from that cache forever after, with nothing anywhere saying
    the units moved upstream. Before a contract could repair, the same response was refused
    loudly and somebody went and looked. Noticing that a source changed shape is the entire
    reason this module exists, so the second verb has to pay for the noticing it removed.

    One line per column, naming the column, the rule's own stated reason and the contract --
    which is the source -- so a reader can go and check upstream without opening this file.

    It sits in `validate` rather than at each of the four boundaries because a boundary that
    has to remember is a boundary that will not: `conform` reports on the same terms, and no
    call site can be the one that forgot.

    **Reporting may not break the live path.** This runs at a fetch boundary CLAUDE.md
    requires to degrade rather than raise, and `hub.publish.live` reads one of them every
    five minutes through a game window with its stdout wherever the runner put it. A `print`
    that fails -- a closed stream, a full one -- must cost the announcement and not the
    frame, so the failure is swallowed here. It is the one place in this module where
    swallowing is right: nothing reads what this writes, and the refusals that everything
    downstream *does* read are untouched by it.
    """
    try:
        for col in sorted(rescaled):
            print(f"  {name}: {col} was rescaled on ingest -- {rescaled[col]}")
    except Exception:                          # the announcement is not worth the frame
        pass


def _reasons(fired: list[tuple[Normalisation, list[str]]]) -> dict[str, str]:
    """Each rescaled column, against the reason the rule moving it gives.

    One statement of that mapping, read by the frame the repair produces, by the refusal a
    still-broken bound raises, and by the boundary that records what it stored.
    """
    return {c: rule.because for rule, present in fired for c in present}


def _reads_as_percents(s: pl.Series, above: float) -> bool:
    """Whether this column can only be the percent form of the quantity it declares.

    The two halves of a `Normalisation`'s trigger, asked of one column and answered
    together, because either half alone is satisfied by a frame the other refuses:

      * **implausible as fractions** -- no value in the column is one a fraction reaches.
        Not *some* value, which is all the trigger used to ask: a fractional frame with one
        corrupt reading in it satisfies "some value is too tall" exactly as a percent frame
        does, and the two are then indistinguishable to it.
      * **plausible as percents** -- every value in it is one a percent reaches. A column of
        fractions carrying one bad number says, read as percents, that everybody else took
        under one and a half percent of the snaps, which is not a thing a snap frame says.

    Both halves are the same comparison seen from the two sides, so they are one line: the
    column's *minimum*, not its maximum. A units change is a whole-column fact -- PFR does
    not publish half a column as percents -- so the bottom of the column is where a units
    change shows and where a single corrupt reading does not.

    Zero is skipped and cannot carry the column on its own. It is the one value that means
    the same thing in both units, so it is evidence for neither, and it is ordinary: a
    receiver's `defense_pct` is 0.0 in every honest refresh. Counting it would make every
    all-zero column say "percents" and rescale the frame around it.

    Nothing here bounds the column from above. A value too tall to *be* a percent -- 500,
    which a hundredth of leaves at 5 -- is a question about validity and not about units,
    and `validate` asks it afterwards against the declared range, where the refusal can say
    the numbers it quotes were rescaled. Answering it here instead would take that sentence
    away from the reader and put nothing in its place.
    """
    seen = s.drop_nulls()
    seen = seen.filter(seen != 0)
    if seen.is_empty():
        return False
    low = seen.min()
    return low is not None and float(cast(float, low)) > above


@dataclass(frozen=True)
class Normalisation:
    """A known upstream variation a contract answers by restating a column rather than
    refusing it.

    The second thing a declaration can say, and the reason it can be read at all. A contract
    whose whole vocabulary is refusal leaves a consumer that must cope with a recurring
    variation nowhere to say so *inside* the contract, so it says it privately: `SNAP_COUNTS`
    refused the whole-percent form of `offense_pct` while `hub.models.spread.snap_usage`
    repaired the identical case forty lines away, two answers to one variation with nothing
    reconciling them. One of them was going to move without the other.

    **Narrow on purpose, and it widens the vocabulary rather than softening the guard.** A
    units change is a whole-column fact -- PFR does not publish half a column as percents --
    so the trigger is a whole column of the declared quantity reading as percents and the
    repair is one multiplication. `above` is the height that separates the two readings: a
    column every one of whose non-zero values clears it cannot be fractions and can be
    percents, and `_reads_as_percents` argues why both halves of that have to be asked.
    `validate` checks the declared range *after* this has run, so a value the factor cannot
    bring back inside the bound is refused exactly as it was before, and anything needing a
    row-level decision is not a normalisation and never becomes one.

    `because` is carried into the refusal a rescaled column still earns, which is the one
    message where a reader needs to know the frame was not the frame the source sent.
    """

    columns: tuple[str, ...]
    above: float
    scale: float
    because: str


@dataclass(frozen=True)
class Contract:
    name: str
    required: dict[str, type]                      # column -> polars dtype family
    non_null: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # The repairs this declaration owns, applied before any bound is checked. Empty for
    # thirteen of the fourteen contracts, because a source that has never varied has nothing
    # to declare here and an unexercised repair is worse than none.
    normalisations: tuple[Normalisation, ...] = ()
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
        """Every check here is a refusal, and every one of them is declared.

        Six checks append to one list and one `raise` turns the list into a
        `ContractViolation`. Only the dtype check carried a `# GUARD` until #62, so five
        refusals applied to fourteen contracts at every fetch boundary in the repo were
        proved by nothing -- and the unmarked five sat either side of the marked one, in the
        same function, appending to the same list. That is the decay this marker habit exists
        to stop, caught in the one place it is easiest to see.

        `tests/contracts/test_guards_are_load_bearing.py` reads this function rather than
        trusting the sweep: every `problems.append(` and every `raise ContractViolation(` in
        this module must sit inside a `# GUARD` or a `# UNPROVED` block, so a seventh check
        cannot land unmarked. The `# GUARD` blocks are then proved the usual way -- deleted
        one at a time, with `tests/contracts/test_contracts.py` required to go red.

        The declared normalisations run first and the frame they produce is what every check
        below sees and what a caller gets back, which is what makes a repair a thing this
        file states once rather than a thing each consumer works out. A caller that discards
        the return still gets the refusal; `conform` is how a consumer asks for the frame.

        A repair that fires says so on the terminal before any check below runs. `_announce`
        argues why that belongs here rather than at each boundary, and why a repair that
        *succeeded* is the case worth reporting.
        """
        df, rescaled = self._normalised(df)
        # GUARD repair-is-announced: a repair that succeeds is reported, not only one that
        # leaves a bound broken afterwards
        _announce(self.name, rescaled)
        # /GUARD
        problems = []
        # GUARD too-few-rows-refused: a truncated response is refused rather than served
        if df.height < self.min_rows:
            problems.append(f"{df.height} rows < min {self.min_rows}")
        # /GUARD
        # GUARD missing-column-refused: a renamed or dropped column is caught
        missing = set(self.required) - set(df.columns)
        if missing:
            problems.append(f"missing columns: {sorted(missing)}")
        # /GUARD
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
        # GUARD nulls-in-a-required-column-refused: an all-null or partly-null key is caught
        for c in self.non_null:
            if c in df.columns and df[c].null_count():
                problems.append(f"{c} has {df[c].null_count()} nulls")
        # /GUARD
        # GUARD duplicate-keys-refused: a doubled row is caught before it doubles a join
        for c in self.unique:
            if c in df.columns and df[c].n_unique() != df.height:
                problems.append(f"{c} not unique ({df[c].n_unique()}/{df.height})")
        # /GUARD
        # GUARD out-of-range-refused: a units change inside a plausible column is caught
        for c, (lo, hi) in self.ranges.items():
            # Numeric, because `<` between a `str` and an `int` is a `TypeError` and not a
            # refusal: a bounded column that arrived as text used to come out of here as a
            # stack trace, past the dtype problem already sitting in `problems` that says
            # exactly what happened. Found by declaring that `_normalised` may skip a
            # column of the wrong kind and leave the dtype check to answer for it.
            if c in df.columns and df.schema[c].is_numeric():
                mn, mx = df[c].min(), df[c].max()
                if mn is not None and (cast(float, mn) < lo or cast(float, mx) > hi):
                    # A column that was rescaled and *still* does not fit says so. Without
                    # it the reader sees a range refusal quoting numbers no source sent,
                    # which is the one way a normalisation could mislead rather than help.
                    tail = f" after rescaling: {rescaled[c]}" if c in rescaled else ""
                    problems.append(f"{c} range [{mn}, {mx}] outside [{lo}, {hi}]{tail}")
        # /GUARD
        # The one exit, marked for the same reason as the six checks above it: a seventh
        # check that raised here directly rather than appending would be a refusal the
        # `problems.append(` scan cannot see, so the scan reads this shape too and this block
        # is what covers it. Excising it makes `validate` return every frame it is given.
        # GUARD problems-are-raised-not-returned: an accumulated problem is refused, not returned
        if problems:
            # `.get`, so only the two states that have something to say add a sentence --
            # `True` is the contract that has met a real response and needs no caveat.
            note = _VERIFICATION_NOTES.get(self.verified_against_live, "")
            raise ContractViolation(f"{self.name}: " + "; ".join(problems) + note)
        # /GUARD
        return df

    def _triggered(self, df: pl.DataFrame) -> list[tuple[Normalisation, list[str]]]:
        """Each declared repair this frame trips, with the columns it will rescale.

        *Whether* a repair fires is decided here and nowhere else. `_normalised` applies
        exactly what this names and decides nothing of its own, and `repairs` reports it
        without touching the frame -- so a boundary that has to record a repair writes down
        the answer the boundary applied rather than re-deriving the trigger beside it. Two
        derivations of one declaration is the shape this package keeps being bitten by.

        A column that is missing, or that has arrived as the wrong kind of thing entirely,
        is skipped rather than rescaled -- multiplying a `Utf8` column would raise something
        that is not a `ContractViolation`, and the dtype refusal downstream is the answer
        the reader wants for that frame anyway.

        **What fires it is a whole column, and it used to be a single value.** A one-sided
        trigger -- some value in the rule is taller than `above` -- cannot tell a units
        change from one bad reading, because both look exactly like that from the top. One
        corrupt number inside an otherwise-correct fractional frame therefore bought a
        hundred-fold rescale of every column in the rule, and the rescaled frame then sat
        comfortably inside the very bound that would have refused it (#149). That mattered
        more once `validate` began returning the repaired frame and the fetch boundaries
        began persisting the return (#140/#141): the corrupted units are what is written,
        pinned and served from cache afterwards, where before the damage ended with the
        read. `_reads_as_percents` is the two-sided question, per column.
        """
        out: list[tuple[Normalisation, list[str]]] = []
        for rule in self.normalisations:
            # One decision for the whole rule, not one per column. A units change is a fact
            # about the response -- the declaration says so itself, "PFR does not publish
            # half a column as percents" -- and deciding column by column lets a frame
            # through half-repaired: `offense_pct` at 85.0 trips its own trigger and comes
            # back a fraction, while `st_pct` at 0.8, which is the percent form of 0.008,
            # trips nothing and sits inside its bound. Nothing refuses that frame, because
            # every column in it is individually plausible, and a reader gets eighty percent
            # where the source said eight tenths of one.
            #
            # `any`, and it stays one decision for the rule: the columns of one quantity do
            # not all carry the evidence. A whole-percent refresh leaves `st_pct` full of
            # ordinary specialists' shares -- 0.14, 0.22 -- which read as percents are tiny
            # and read as fractions are ordinary, so that column says nothing either way.
            # One column that can only be percents is the response saying its units moved,
            # and the rule then moves everything it declares, which is the paragraph above.
            present = [c for c in rule.columns
                       if c in df.columns and df.schema[c].is_numeric()]
            if not any(_reads_as_percents(df[c], rule.above) for c in present):
                continue
            out.append((rule, present))
        return out

    def repairs(self, df: pl.DataFrame) -> dict[str, str]:
        """Which declared repairs this frame trips, and the reason each one gives.

        The reporting verb, for a boundary that has to write the answer down rather than
        watch it go past. `hub.fetch.nflverse.load` asks it so the `Pin` beside a cache entry
        can record that the archive under it was rescaled on ingest; `Pin.rescaled` argues
        why a pin has to say so and why the terminal line alone does not.

        **Ask it of the frame that arrived, before validating, and not after.** It reads what
        it is given, so a frame that has already been repaired trips nothing and answers
        `{}` -- correct, and exactly the wrong thing to record. The one caller does ask
        first, and a test holds it there.
        """
        return _reasons(self._triggered(df))

    def _normalised(self, df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
        """The frame with every declared repair applied, and why each one fired.

        No refusal lives here on purpose. A normalisation that does not fire leaves the
        column alone and every check in `validate` runs on the frame the source sent; one
        that does fire hands those same checks a repaired column and is named in the
        returned mapping, so a bound that is still broken can say the numbers it is quoting
        are not the ones that arrived -- and so `validate` can say the repair happened at
        all.
        """
        fired = self._triggered(df)
        for rule, present in fired:
            for c in present:
                df = df.with_columns(pl.col(c) * rule.scale)
        return df, _reasons(fired)

    def conform(self, df: pl.DataFrame, *columns: str) -> pl.DataFrame:
        """The columns a consumer names, repaired and checked as this contract declares them.

        The reading verb. `validate` answers "is this whole response servable", which is the
        fetch boundary's question and no use to a module holding a slice of one; this answers
        "may I read these columns, and are they in the units you say", which is what a
        consumer was hand-rolling a second schema to find out. Everything it enforces is the
        same declaration `validate` enforces, narrowed -- there is one statement of what the
        source may return, and asking it a smaller question does not create a second.

        Narrowed to the columns named and no further. `hub.models.spread.snap_usage` reads
        four of `SNAP_COUNTS`' thirteen, so demanding all thirteen would refuse a legitimate
        slice; the frozen capture `tests/panelarchive.py` drives the Panel from holds exactly
        the five columns `hub.models.panel.injury_severity` reads, and no more. `min_rows`
        goes the same way: how big a response has to be is a fact about a refresh, and a
        consumer is handed whatever the boundary has already vouched for.

        Naming a column this contract does not declare is itself a refusal, and it is the
        one that catches the drift this method exists to end. A consumer that starts reading
        a column the source never promised has left the declaration behind, and it should
        hear about it here rather than three joins later.
        """
        # GUARD undeclared-column-refused: a consumer reading past the declaration is caught
        unknown = sorted(set(columns) - set(self.required))
        if unknown:
            raise ContractViolation(
                f"{self.name}: asked for {unknown}, which this contract does not declare. "
                f"It declares {sorted(self.required)} -- add the column here if the source "
                f"really returns it, rather than reading it on trust.")
        # /GUARD
        keep = set(columns)
        return replace(
            self,
            required={c: dt for c, dt in self.required.items() if c in keep},
            non_null=tuple(c for c in self.non_null if c in keep),
            unique=tuple(c for c in self.unique if c in keep),
            ranges={c: r for c, r in self.ranges.items() if c in keep},
            # Narrowed to the columns named -- except the repairs, which are not narrowed.
            # A `Normalisation` is a fact about the response and is decided across all the
            # columns it names; splitting it per caller puts back the half-repaired frame
            # `_normalised` exists to prevent, and `conform` hands back the whole frame, so
            # those other columns leave here in a reader's hands either way. What narrows is
            # what is *checked*: a consumer answers for the columns it reads.
            normalisations=self.normalisations,
            min_rows=0,
        ).validate(df)


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

# The Big Ten availability archive's index, since #215: one row per document per capture.
# This is a contract on what `hub.fetch.bigten` *stores*, not on what the conference
# publishes -- the reports are archived as the bytes the conference served, and nothing here
# parses them yet, so there is no designation column to declare. What the index has to hold
# is the provenance a later study needs: which deadline (`deadline`) a row was captured for,
# when the conference last updated the report (`report_updated_at`, the source's own stamp
# and not the fetch time), and the hash that says whether the bytes moved between two
# captures (`new_content`). `kind` is `article` for the page's own content record, `file`
# for a linked document, `page` for the raw page kept when parsing failed, and `lines` for
# the odds snapshot taken in the same run.
#
# `label` and `report_updated_at` are required and NOT non_null: a `lines` row has neither,
# and a linked document whose anchor carries no text has no label. Nothing is unique --
# the same document is legitimately captured at every deadline, unchanged, and a row saying
# so is how a missed deadline is told apart from an unchanged report.
BIGTEN_CAPTURES = Contract(
    name="bigten_captures",
    required={"deadline": pl.Utf8, "deadline_name": pl.Utf8, "captured_at": pl.Utf8,
              "season": pl.Int64, "kind": pl.Utf8, "url": pl.Utf8, "label": pl.Utf8,
              "report_updated_at": pl.Utf8, "sha256": pl.Utf8, "bytes": pl.Int64,
              "path": pl.Utf8, "new_content": pl.Boolean},
    non_null=("deadline", "deadline_name", "captured_at", "season", "kind", "url", "sha256",
              "bytes", "path", "new_content"),
    # The regime began in 2026; a season before it is a row nothing here wrote. An empty
    # document is a real capture (a 404 body is not stored, but a zero-byte file is a fact
    # about the source), so bytes has a floor of zero; the ceiling is far above the 1.4MB
    # the 2024 weekly PDF measured at on 2026-09-11.
    ranges={"season": (2026, 2100), "bytes": (0, 100_000_000)},
    min_rows=1,
    # Checked against `bigten_captures.synthetic.json`, a hand-built index: the 2026 page
    # held no report on the day this was written, so no capture exists to freeze. See
    # `verified_against_live`.
    verified_against_live=False,
)

# Two markets and their prices, since #211. The four number columns are two pairs and the
# pairing is the declaration: a point without the price beside it says -7 at -120 and -7 at
# -105 are the same fact, and they are not the same price.
#
# **`close_spread` is the only priced column that may not be null, and the asymmetry is
# deliberate.** A row exists because the betting market posted a spread -- that is the rule
# `hub.fetch.odds` has always applied and every consumer of `lines` reads. The other three
# are nullable because each is separately absent in the wild: a book posts a spread and no
# total on a game it has not hung yet, and a price can be unreadable while the point beside
# it is fine. A null there is the honest answer, and it is the answer a snapshot written
# before this contract gained the columns gives too -- the store reads its partitions with
# `union_by_name`, so a three-column parquet comes back with nulls in the three it never
# had rather than refusing the glob. Absent, never fabricated: nothing derives a total from
# a spread, which is the one number a totals hypothesis must not be fitted on.
#
# Ranges are plausibility bounds and are wider than the observed market on purpose, because
# a refusal here fires *after* the credit is spent and takes the whole snapshot with it
# (`SnapshotIncomplete`). Totals: NFL games are hung between about 30 and 63, so [20, 100]
# catches a units change or a spread written into the total column -- 8.25 is below 20 --
# without refusing a slate on a cold December weather line. Prices: main-market spread and
# total juice lives inside [-130, +110] and [-2000, 2000] leaves room for an extreme honest
# quote while still refusing a moneyline price that reached the wrong column. The one units
# change the bound cannot catch is decimal odds, whose values are far inside it, so that is
# refused upstream where it can be recognised: `hub.fetch.odds._american` drops anything
# between -100 and +100 because American odds have a hole there and decimal odds do not.
#
# **Two columns say how long the spread quote has stood still, since #210.** `polls_unmoved`
# is the count of consecutive polls, this one included, that returned this quote, and
# `unmoved_since` is when that run began. Both are non-null: `hub.fetch.odds.staleness`
# derives them from the archive for every row it writes, and a first poll is a run of one
# rather than an unknown. They are required because a snapshot without them is exactly the
# labelling error #210 names -- a dated capture that ranks above the schedule's own field on
# provenance while carrying nothing that says whether the number has moved in a fortnight.
# The range on `polls_unmoved` is a floor of one, since a row is its own first poll, and a
# ceiling no honest archive reaches: at a poll an hour, a season is under ten thousand.
#
# Partitions written before #210 have neither column and stay readable for the reason the
# pre-#211 ones do: the store unions by name and the reader derives the two on the way out.
# Nothing rewrites a dated partition to add them.
ODDS_SNAPSHOT = Contract(
    name="odds_snapshot",
    required={"game_id": pl.Utf8, "close_spread": pl.Float64,
              "spread_price": pl.Float64, "close_total": pl.Float64,
              "total_price": pl.Float64, "captured_at": pl.Datetime,
              "polls_unmoved": pl.Int64, "unmoved_since": pl.Datetime},
    non_null=("game_id", "close_spread", "captured_at", "polls_unmoved", "unmoved_since"),
    # Deliberately NOT unique on game_id: several snapshots per game is the entire point,
    # and it is what makes AS_OF_LINES more than a normal join.
    ranges={"close_spread": (-40, 40), "close_total": (20, 100),
            "spread_price": (-2000, 2000), "total_price": (-2000, 2000),
            "polls_unmoved": (1, 10000)},
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
    # Checked against two real 2026-09-05 captures of the public scoreboard:
    # `espn_scoreboard_cfb.json` (`groups=80`, all three states, the live path) and
    # `espn_scoreboard_nfl.json` (pre-only -- the season had not kicked off, so no
    # in-progress NFL game existed to capture; #75 froze what there was and left that path
    # to the nightly canary). Stated because the default is now `None` -- see
    # `verified_against_live`.
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
# And because a units change hits all three at once, the repair names all three. It is one
# declaration and not one per column for the same reason the bound above is three copies of
# one number: PFR does not switch units on a single column, and a repair that mended
# `offense_pct` alone would leave the other two refusing the very frame it had just accepted.
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
    normalisations=(
        Normalisation(
            columns=("offense_pct", "defense_pct", "st_pct"), above=1.5, scale=0.01,
            because="PFR publishes these as fractions and nflverse has shipped them as whole "
                    "percents. A whole column of shares above 1.5 can only be the percent "
                    "form -- the honest ceiling is 1.01, PFR's own rounding -- so the "
                    "hundred comes back out before the bound is checked. A column that only "
                    "reaches 1.5 in places is a fractional column with a bad reading in it, "
                    "and the bound is what answers that"),
    ),
    min_rows=1,
    # Checked against `nflverse_snap_counts.json`, a real 2024 capture.
    verified_against_live=True,
)
