"""Contract tests run against FROZEN fixtures, not the live API.

Live-API tests are flaky and useless in CI. These prove our parsing logic is correct.
A separate nightly job (tests/golden) hits the real API and diffs against the fixture,
which is what actually catches ESPN renaming a field.
"""
import polars as pl
import pytest

from hub.contracts import (
    DRAFT_BOARD,
    SNAP_COUNTS,
    Contract,
    ContractViolation,
    Normalisation,
)


def _board(n=300, **override):
    """A frame carrying every column `DRAFT_BOARD` requires.

    Built from the contract itself rather than a hand-written column list, so adding a
    required column cannot leave this fixture quietly behind the thing it is testing.
    """
    fill = {pl.Utf8: [f"v{i}" for i in range(n)],
            pl.Float64: [float(i + 1) for i in range(n)],
            pl.UInt32: pl.Series([16] * n, dtype=pl.UInt32)}
    df = pl.DataFrame({c: fill[dt] for c, dt in DRAFT_BOARD.required.items()})
    return df.with_columns(**override) if override else df


def test_contract_passes_on_valid_frame():
    assert DRAFT_BOARD.validate(_board()).height == 300


def test_the_board_contract_names_the_columns_downstream_reads():
    """It declared three columns and was applied to nothing, while ~14 modules read the
    board by name. These are the ones read unconditionally."""
    assert {"player", "pos", "ecr", "xfp_per_game", "vor"} <= set(DRAFT_BOARD.required)


def test_the_consensus_spread_is_contracted_under_its_disambiguated_name():
    """`sd` meant a rank spread here and a points spread in `hub.models.predict`."""
    assert "sd" not in DRAFT_BOARD.required


def test_optional_columns_are_not_contracted():
    """`build` degrades on purpose when a fetch fails -- contracting `edge` or `td_luck`
    would turn a documented graceful degradation into a hard failure on draft night."""
    for optional in ("edge", "proj_blend", "td_luck", "wk15_17_sos", "vor_proj"):
        assert optional not in DRAFT_BOARD.required


def test_contract_catches_a_truncated_response():
    """The check that had nothing watching it. `min_rows` is 300 on the board and 1,000 on
    four nflverse frames, so it is the line between a real refresh and a source that answered
    with a header and three rows -- the quiet half of the Week 7 failure mode, where nothing
    crashes and the projections are built off almost nothing."""
    df = pl.DataFrame({"player": ["A"]})
    with pytest.raises(ContractViolation, match="rows < min"):
        Contract("t", required={"player": pl.Utf8}, min_rows=5).validate(df)


def test_contract_catches_nulls_in_a_column_declared_non_null():
    """The other unwatched one. A key column arriving all-null passes every presence check
    and then silently drops every row it is joined on -- `_family` says the same thing about
    the `Null` dtype, and this is the row-level half of it."""
    df = pl.DataFrame({"player": ["A", None]})
    with pytest.raises(ContractViolation, match="nulls"):
        Contract("t", required={"player": pl.Utf8}, non_null=("player",)).validate(df)


def test_contract_catches_missing_column():
    df = pl.DataFrame({"player": ["A"], "ecr": [1.0]})
    with pytest.raises(ContractViolation, match="missing columns"):
        Contract("t", required={"player": pl.Utf8, "pos": pl.Utf8}).validate(df)


def test_contract_catches_duplicate_players():
    df = pl.DataFrame({"player": ["A", "A"], "pos": ["WR", "WR"], "ecr": [1.0, 2.0]})
    with pytest.raises(ContractViolation, match="not unique"):
        Contract("t", required={"player": pl.Utf8}, unique=("player",)).validate(df)


def test_contract_catches_out_of_range():
    """A scoring-rule change that doubles projections should fail loudly, not silently."""
    df = pl.DataFrame({"xfp": [500.0]})
    with pytest.raises(ContractViolation, match="range"):
        Contract("t", required={"xfp": pl.Float64}, ranges={"xfp": (-10, 80)}).validate(df)


# --- the second verb (issue #132) -----------------------------------------

_REPAIRED = Contract(
    name="t_repaired",
    required={"key": pl.Utf8, "pct": pl.Float64, "other": pl.Float64},
    non_null=("key",),
    ranges={"pct": (0, 1.05)},
    normalisations=(Normalisation(columns=("pct",), above=1.5, scale=0.01,
                                  because="the upstream ships whole percents"),),
    min_rows=1,
)
"""`SNAP_COUNTS` in miniature: a bounded share, and the one upstream variation it expects.

A toy rather than the real contract, because these are tests of what a declaration *can*
say. What the shipped one actually says is pinned separately, at the foot of this file,
against the range it protects.
"""


_WHOLE_RULE = Contract(
    name="t_whole_rule",
    required={"key": pl.Utf8, "pct": pl.Float64, "other": pl.Float64},
    non_null=("key",),
    ranges={"pct": (0, 1.05), "other": (0, 1.05)},
    normalisations=(Normalisation(columns=("pct", "other"), above=1.5, scale=0.01,
                                  because="the upstream ships whole percents"),),
    min_rows=1,
)
"""The shape `SNAP_COUNTS` actually has: one repair naming *several* columns of the same
quantity. `_REPAIRED` above names one, which cannot express the case where the columns
disagree about whether the trigger was tripped -- and that case is the whole reason a
`Normalisation` takes a tuple of columns rather than one."""


def _repairable(**override):
    df = pl.DataFrame({"key": ["a"], "pct": [0.5], "other": [1.0]})
    return df.with_columns(**override) if override else df


def test_a_declared_repair_runs_before_the_bound_that_would_have_refused_it():
    """The whole ticket, in one frame. `pct` at 50 is outside `[0, 1.05]` and is not a
    broken response -- it is the units change the bound was written to notice, and a
    contract that can only refuse leaves the consumer to answer it privately."""
    assert _REPAIRED.validate(_repairable(pct=50.0))["pct"][0] == pytest.approx(0.5)


def test_a_frame_the_repair_does_not_fire_on_arrives_exactly_as_it_was_sent():
    """A repair that ran unconditionally would divide every honest fraction by a hundred,
    which is a far quieter failure than the one it exists to fix."""
    assert _REPAIRED.validate(_repairable())["pct"][0] == 0.5


def test_a_value_no_rescaling_can_bring_inside_the_bound_is_still_refused():
    """This widens the vocabulary; it does not soften the guard. 500 rescales to 5, which
    is not a snap share however it was published, and the range still says so."""
    with pytest.raises(ContractViolation, match="pct range"):
        _REPAIRED.validate(_repairable(pct=500.0))


def test_a_refusal_that_follows_a_repair_says_the_numbers_were_rescaled():
    """Otherwise the message quotes numbers no source ever sent -- `[5.0, 5.0]` against a
    response carrying 500 -- and the reader goes looking for a column that does not exist.
    It is the one way a normalisation could mislead rather than help."""
    with pytest.raises(ContractViolation, match="after rescaling: the upstream ships whole"):
        _REPAIRED.validate(_repairable(pct=500.0))


def test_a_column_that_arrived_as_the_wrong_kind_is_refused_rather_than_rescaled():
    """Multiplying a `Utf8` column raises something that is not a `ContractViolation`, so
    the repair steps aside and the dtype refusal answers for the frame.

    It answers alone, too. Comparing that same text column against its bound is a
    `TypeError` -- which is how this frame left `validate` before the repair was declared,
    with a perfectly good dtype problem already sitting unreported in `problems`.
    """
    with pytest.raises(ContractViolation, match="pct is String"):
        _REPAIRED.validate(_repairable(pct=pl.lit("50")))


def test_conform_enforces_only_the_columns_a_consumer_names():
    """A consumer holding a slice is not the fetch boundary. `hub.models.spread.snap_usage`
    reads four of `SNAP_COUNTS`' thirteen columns, so demanding all thirteen would refuse a
    legitimate frame -- which is how it came to carry its own column list instead."""
    df = pl.DataFrame({"key": ["a"], "pct": [0.5]})
    assert _REPAIRED.conform(df, "key", "pct").height == 1
    with pytest.raises(ContractViolation, match="missing columns"):
        _REPAIRED.validate(df)


def test_conform_refuses_a_column_the_contract_does_not_declare():
    """The drift this seam exists to end, caught coming the other way. A consumer that
    starts reading a column the source never promised has left the declaration behind, and
    reading it on trust is exactly how the second copy of a schema begins."""
    df = pl.DataFrame({"key": ["a"], "pct": [0.5], "share": [0.5]})
    with pytest.raises(ContractViolation, match=r"asked for \['share'\], which this "):
        _REPAIRED.conform(df, "key", "share")


def test_conform_hands_back_the_repaired_column_and_not_only_a_verdict():
    """The half `validate` could not give anybody: the frame, in the units declared."""
    df = pl.DataFrame({"key": ["a"], "pct": [50.0]})
    assert _REPAIRED.conform(df, "key", "pct")["pct"][0] == pytest.approx(0.5)


def test_the_repair_is_one_decision_for_every_column_it_declares():
    """A units change is a fact about the source, not about a column.

    `Normalisation` argues its own trigger this way -- "PFR does not publish half a column
    as percents" -- and the frozen capture puts three columns of the same quantity side by
    side. Deciding per column instead lets a frame through half-repaired: `offense_pct` at
    85.0 trips a per-column trigger and comes back 0.85, while `st_pct` at 0.8 -- eight
    tenths of one percent, the percent form of 0.008 -- trips nothing, sits inside
    `[0, 1.05]`, and is read by anything downstream as eighty percent. Nothing refuses it,
    because on its own it is a plausible share. One hundred-fold error, silent, in the
    column `hub.models.panel` reads.
    """
    df = pl.DataFrame({"key": ["a"], "pct": [85.0], "other": [0.8]})
    out = _WHOLE_RULE.conform(df, "key", "pct", "other")
    assert out["pct"][0] == pytest.approx(0.85)
    assert out["other"][0] == pytest.approx(0.008), (
        "one column tripped the trigger and the other was left in the units the first one "
        "proved the frame was not in"
    )


def test_a_frame_under_the_trigger_everywhere_is_left_entirely_alone():
    """The other half of one decision: no column trips it, so no column moves."""
    df = pl.DataFrame({"key": ["a"], "pct": [0.85], "other": [0.008]})
    out = _WHOLE_RULE.conform(df, "key", "pct", "other")
    assert out["pct"][0] == pytest.approx(0.85)
    assert out["other"][0] == pytest.approx(0.008)


def test_conform_narrows_what_is_checked_and_not_what_is_repaired():
    """What a consumer answers for is the columns it reads; what the units are is not its
    call.

    This asserted the opposite until the review of #132: that naming one column repaired
    only that one, so a return value never depended on a declaration the caller had not
    named. The reasoning does not survive `conform` handing back the *whole* frame. A
    consumer that names `offense_pct` still receives `st_pct`, and narrowing the repair
    left that column in units the frame had already proved it was not in -- the
    half-repaired frame under a different door. Narrowing the checks is the part that was
    always right: an empty `ranges` here, and no refusal for a column nobody read.
    """
    df = pl.DataFrame({"key": ["a"], "pct": [85.0], "other": [0.8]})
    out = _WHOLE_RULE.conform(df, "key")
    assert out["pct"][0] == pytest.approx(0.85), "the repair is the frame's, not the caller's"
    assert out["other"][0] == pytest.approx(0.008)

    # ...and the narrowing that does happen. 1.2 is above the bound and below the trigger,
    # so no repair reaches it and the only question left is whether anybody read it.
    unread = pl.DataFrame({"key": ["a"], "pct": [1.2], "other": [0.5]})
    assert _WHOLE_RULE.conform(unread, "key")["pct"][0] == pytest.approx(1.2)
    with pytest.raises(ContractViolation, match="pct range"):
        _WHOLE_RULE.conform(unread, "key", "pct")


def test_conform_does_not_apply_the_volume_floor():
    """How big a response has to be is a fact about a refresh, and the boundary has already
    asked it. A consumer handed an empty slice gets an empty slice, not a refusal."""
    empty = pl.DataFrame(schema={"key": pl.Utf8, "pct": pl.Float64})
    assert _REPAIRED.conform(empty, "key", "pct").height == 0
    with pytest.raises(ContractViolation, match="rows < min"):
        _REPAIRED.validate(empty.with_columns(pl.lit(1.0).alias("other")))


# --- a repair that succeeds says so (issue #140) ------------------------------
#
# The mapping of rescaled column to reason was used in exactly one place: the refusal a
# column earns by being rescaled and *still* sitting outside its bound. So the case the
# declaration was written for -- the repair that works -- was the one case nobody was told
# about, and it stopped being harmless the day the repaired frame reached the cache. A
# whole-percent refresh is now rescaled, written, pinned and served from that cache
# afterwards; before the second verb existed the same response was refused loudly and
# somebody went and looked.


def test_a_repair_that_fires_says_so_where_a_person_will_see_it(capsys):
    """The whole ticket. A repair that succeeded left no trace anywhere at all."""
    _REPAIRED.validate(_repairable(pct=50.0))
    said = capsys.readouterr().out
    assert "pct" in said, f"the rescaled column is not named: {said!r}"
    assert "the upstream ships whole percents" in said, (
        f"the rule's own stated reason is not carried, so the reader is told a number moved "
        f"and not why anyone thinks it should have: {said!r}")
    assert "t_repaired" in said, (
        f"the source is not named, so a reader cannot go and check upstream without first "
        f"working out which contract spoke: {said!r}")


def test_a_frame_that_trips_no_repair_says_nothing(capsys):
    """Every honest refresh goes through here. A line on each of them is a line nobody reads,
    and the announcement is worth having only while it means something happened."""
    _REPAIRED.validate(_repairable())
    assert capsys.readouterr().out == ""


def test_every_column_a_repair_moves_is_named_and_not_just_the_first(capsys):
    """One decision moves several columns, and a reader checking upstream needs all of them
    -- `SNAP_COUNTS` rescales three at once."""
    _WHOLE_RULE.validate(pl.DataFrame({"key": ["a"], "pct": [85.0], "other": [0.8]}))
    said = capsys.readouterr().out
    assert "pct" in said and "other" in said, said


def test_a_refusal_after_a_repair_reports_the_repair_as_well(capsys):
    """The two messages answer different questions and the refusal does not replace the
    report: one says the units moved upstream, the other says the move was not enough."""
    with pytest.raises(ContractViolation):
        _REPAIRED.validate(_repairable(pct=500.0))
    assert "rescaled on ingest" in capsys.readouterr().out


def test_a_terminal_that_cannot_be_written_to_does_not_take_down_the_fetch(monkeypatch):
    """CLAUDE.md's rule, at the boundary it was written for. `hub.publish.live` validates
    every five minutes through a game window with its stdout wherever the runner put it, and
    a report that raises would convert a repaired frame into a dead overlay -- turning the
    fix for a silent success into a louder failure than the silence."""
    def broken(*_a, **_k):
        raise OSError("stdout is closed")

    monkeypatch.setattr("builtins.print", broken)
    out = _REPAIRED.validate(_repairable(pct=50.0))
    assert out["pct"][0] == pytest.approx(0.5), (
        "the announcement failed and took the repaired frame with it")


def test_what_a_frame_reports_is_what_the_repair_actually_moved():
    """`repairs` is what a boundary writes down, so it has to be the same answer the frame
    was given rather than a second derivation of the trigger beside the one that fired."""
    df = pl.DataFrame({"key": ["a"], "pct": [85.0], "other": [0.8]})
    said = _WHOLE_RULE.repairs(df)
    assert set(said) == {"pct", "other"}
    assert set(said.values()) == {"the upstream ships whole percents"}

    out = _WHOLE_RULE.validate(df)
    moved = {c for c in ("pct", "other") if out[c][0] != df[c][0]}
    assert moved == set(said), f"reported {sorted(said)}, moved {sorted(moved)}"


def test_a_frame_that_has_already_been_repaired_reports_nothing():
    """The ordering a caller has to get right, written down where it can go red. `repairs`
    reads the frame it is handed, so asking it after `validate` answers "nothing was
    rescaled" about a frame that was -- which is the wrong thing to pin."""
    df = pl.DataFrame({"key": ["a"], "pct": [85.0], "other": [0.8]})
    assert _WHOLE_RULE.repairs(_WHOLE_RULE.validate(df)) == {}


def test_the_percent_repair_covers_every_column_the_percent_bound_covers():
    """The shipped declaration, held to the reason it was written as one rule for three
    columns: a units change hits `offense_pct`, `defense_pct` and `st_pct` at once, so a
    repair that mended one would leave the other two refusing the frame it just accepted."""
    (rule,) = SNAP_COUNTS.normalisations
    bounded = {c for c, (_, hi) in SNAP_COUNTS.ranges.items() if hi == 1.05}
    assert set(rule.columns) == bounded == {"offense_pct", "defense_pct", "st_pct"}
