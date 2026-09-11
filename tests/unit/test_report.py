"""The draft-night output, tested as lines.

These blocks lived inline in `board.main`, where they were 124 statements no test could
reach. The ECR-only crash of 2026-08-25 -- two sections reading `adp` while guarding on a
different flag -- shipped inside them and was found by running the CLI. Everything below is
an assertion that could not be written until the renderers returned lines instead of
printing them.

All offline.
"""
import polars as pl

from hub.draft import report
from hub.draft.optimize import ThePick


def _board(**cols):
    n = len(next(iter(cols.values())))
    base = {"player": [f"P{i}" for i in range(n)], "pos": ["RB"] * n}
    return pl.DataFrame({**base, **cols})


def _rep(**flags):
    """A BuildReport with the named stages marked as having run."""
    from hub.draft.board import BuildReport
    return BuildReport(**flags)


def _ecr_only():
    """What `build` returns with no ESPN key: everything but ADP."""
    return pl.DataFrame({"player": ["A", "B"], "pos": ["RB", "WR"],
                         "td_luck": [1.0, -1.0], "missed": [5, 6],
                         "injury_status": ["QUESTIONABLE", None]})


# --- the ECR-only path, which is a documented fallback and used to crash ----

def test_the_injury_report_is_skipped_without_adp():
    """Both halves are scoped to `inside ADP 120` and both print an ADP, so there is nothing
    to show -- and the filter used to raise ColumnNotFoundError."""
    assert report.injuries(_ecr_only(), _rep(adp=False)) == []


def test_the_touchdown_luck_report_is_skipped_without_adp():
    """It reaches nflverse, so it can succeed while ESPN ADP fails."""
    assert report.td_luck(_ecr_only(), _rep(td_luck=True, adp=False)) == []


def test_those_reports_still_run_when_adp_is_present():
    """The guard must not have turned them off altogether."""
    df = _ecr_only().with_columns(pl.Series("adp", [10.0, 20.0]))
    assert report.injuries(df, _rep(adp=True)), "the designation report should have content"


# --- issue #146: both halves of the injury section ask the report ----------

def _disagreeing_board(**cols):
    """A board inside ADP 120 carrying whichever stage columns the caller names."""
    return _board(adp=[10.0, 20.0], **cols)


def test_the_injury_section_holds_the_report_against_a_frame_that_disagrees():
    """Issue #146, on frames whose columns and the report disagree in both directions.

    A rule test rather than one built from a reachable board, for #164's reason: on every
    board `build` can emit, `injury_status` and `missed` are present exactly when their
    stage's flag is set, so a reachable fixture passes against the sniff this replaces and
    proves nothing. Only a frame that contradicts its own report can tell the two apart.

    **Direction one -- the frame says yes, the report says no.** `missed` is on the board
    with values that would print, and `report.durability` is false. The durability stage did
    not run, so last season's missed time is not shown: a column of numbers surviving from
    somewhere else is not evidence that the stage that owns it happened. The designation
    half still renders, so this is the report being read rather than the section being off.

    **Direction two -- the frame says no, the report says yes.** The report says the
    draft-market stage ran and `injury_status` is not there. `injury_status` is one of
    `board.STAGE_COLUMNS["adp"]`, so the report has already answered for it, and the read
    goes ahead and raises. Skipping the half silently is exactly what the sniff did, and it
    is indistinguishable from the stage having run and found nobody hurt.
    """
    import polars as _pl
    import pytest as _pytest

    frame_says_yes = _disagreeing_board(injury_status=["QUESTIONABLE", None],
                                        missed=[8, 9])
    without = "\n".join(report.injuries(frame_says_yes, _rep(adp=True, durability=False)))
    assert "Carrying a designation today" in without, (
        "the designation half is gated on `report.adp`, which is set here")
    assert "Missed time last season" not in without, (
        "`missed` is on the frame but its stage did not run -- the report is the answer")

    with_it = "\n".join(report.injuries(frame_says_yes, _rep(adp=True, durability=True)))
    assert "Missed time last season" in with_it, (
        "and the half is not simply off: the same frame renders it when the report says the "
        "durability stage ran")

    frame_says_no = _disagreeing_board(missed=[8, 9])
    with _pytest.raises(_pl.exceptions.ColumnNotFoundError, match="injury_status"):
        report.injuries(frame_says_no, _rep(adp=True, durability=True))


# --- corrections depend on the route that chose the pick --------------------

def _tp(via, notes=("missed 1 last season",), rank=1.5, label="ECR"):
    return ThePick(player="A", pos="WR", via=via, rank=rank, rank_label=label,
                   notes=list(notes))


def test_the_corrections_note_is_route_aware():
    """"bounded at 20% of ADP" on a board with no ADP is the kind of sentence that gets
    believed at 9pm. Corrections move a player relative to ADP; the ECR route has none."""
    corrected = "\n".join(report.corrections_note(_tp("draft market, corrected")))
    assert "bounded at 20% of ADP" in corrected
    ecr = "\n".join(report.corrections_note(_tp("consensus (ECR) -- no ADP today")))
    assert "bounded at 20% of ADP" not in ecr and "NOT folded into this" in ecr


def test_the_uncorrected_adp_route_does_not_claim_corrections_either():
    """`by draft market (ADP)` means the board predates corrected ADP -- nothing was folded
    in there either, and saying otherwise would misdescribe the ranking."""
    out = "\n".join(report.corrections_note(_tp("draft market (ADP)", ["td luck +2.0/gm"])))
    assert "bounded at 20% of ADP" not in out


def test_a_pick_with_no_notes_says_nothing():
    assert report.corrections_note(_tp("consensus (ECR) -- no ADP today", ())) == []


# --- THE PICK ---------------------------------------------------------------

def test_the_pick_leads_with_the_route_that_chose_it():
    out = report.the_pick(_tp("draft market, corrected", rank=1.4, label="ADP"))
    assert "THE PICK" in out[0] and "draft market, corrected" in out[0]
    assert "A" in out[1] and "ADP 1.4" in out[1]


def test_an_absent_pick_says_what_to_serve_instead():
    """The board carries neither ADP nor ECR, so it did not build. At 9pm the useful output
    is the fallback path, not a blank."""
    out = report.the_pick(None)
    assert len(out) == 1 and "draft_board.json" in out[0]


def test_a_pick_with_no_rank_still_renders():
    """The ECR route can carry a null rank; formatting it must not raise."""
    out = report.the_pick(_tp("consensus (ECR) -- no ADP today", rank=None))
    assert "A" in out[1]


# --- also close -------------------------------------------------------------

def _rec(**cols):
    n = len(next(iter(cols.values()))) if cols else 2
    return pl.DataFrame({"player": ["X", "Y"][:n], "pos": ["RB", "WR"][:n], **cols})


def test_also_close_names_the_mode_and_its_rule():
    out = report.also_close("scarcity", _rec(vor=[3.0, 2.0]))
    assert "scarcity" in out[0] and "will not survive" in out[0]
    assert "not a ranking to draft off" in out[0]


def test_the_value_mode_gives_the_other_rule():
    out = report.also_close("value", _rec(vor=[3.0, 2.0]))
    assert "highest VOR" in out[0]


def test_cost_of_waiting_is_shown_only_when_present():
    """It exists in scarcity mode and not in value mode; a missing column must not raise."""
    with_cw = report.also_close("scarcity", _rec(vor=[3.0, 2.0],
                                                 cost_of_waiting=[9.0, 8.0]))
    without = report.also_close("value", _rec(vor=[3.0, 2.0]))
    assert "cost_of_waiting" in "\n".join(with_cw)
    assert "cost_of_waiting" not in "\n".join(without)


def test_a_null_vor_renders_as_zero_rather_than_raising():
    out = report.also_close("value", _rec(vor=[None, 2.0]))
    assert "VOR   0.0" in out[1]


# --- the smaller blocks -----------------------------------------------------

def test_the_header_reports_size_and_missing_xfp():
    out = report.header(_board(vor=[1.0, None, 3.0]))
    assert "3 players" in out[0] and "1 missing xFP" in out[0]


def test_regression_lists_the_biggest_underperformers_first():
    df = _board(ecr=[1.0, 2.0], fp_over_expected=[-40.0, -10.0])
    out = report.regression(df)
    assert out[1].index("P0") and "P0" in out[1], "most negative first"


def test_an_unmatched_pick_is_reported_with_a_cap():
    assert report.unmatched([]) == []
    out = report.unmatched([f"n{i}" for i in range(7)])
    assert "7 recorded picks" in out[0] and out[0].endswith("...")


def test_a_mistyped_pick_names_the_suggestion():
    out = report.mistyped({"Jamar Chase": "Ja'Marr Chase"})
    assert "NOT ON THE BOARD" in out[0] and "Ja'Marr Chase" in out[0]
    assert "still shown as available" in out[1], "the consequence is the point"


def test_a_pick_with_no_suggestion_is_probably_a_kicker():
    out = report.mistyped({"Some Kicker": None})
    assert "kicker or defence" in out[0]


def test_degraded_is_silent_when_nothing_degraded():
    assert report.degraded(()) == []
    assert "built without" in report.degraded(("adp",))[0]


# --- built or served, which is the same question asked once ------------------

def test_a_built_board_reports_what_it_was_built_without():
    """The stages that did not make it, and nothing about the board being served."""
    out = report.built_or_served(_rep(sos=True), None)
    assert "built without" in out[0] and "SERVED" not in out[0]
    assert not any("CORRECTED ADP" in line for line in out), (
        "no ADP means no corrected ranking to be short a term")


def test_a_whole_board_still_names_the_disputed_coefficient_it_applied():
    """#48's third criterion. A board with every stage was silent here until #48, and that
    silence was the ticket: the durability correction applies two numbers the refit
    disagrees with, and an operator ranking on them could not see which.

    It is not a degradation note -- nothing failed, and `built without` stays empty.
    """
    whole = _rep(sos=True, td_luck=True, durability=True, bye=True, adp=True,
                 scoring_checked=True, roster_checked=True)
    out = "\n".join(report.built_or_served(whole, None))
    assert "built without" not in out and "CORRECTED ADP is missing" not in out
    assert "CORRECTIONS APPLIED" in out and "durability" in out
    assert "UNREPRODUCED" in out and "docs/fitted-corrections.md" in out


def test_the_disposition_is_not_printed_for_a_correction_that_did_not_run():
    """A flag beside a ranking the term is not in would be worse than no flag: it says the
    board applied a disputed number when the board applied nothing."""
    absorbed = _rep(sos=True, td_luck=True, durability=False, adp=True)
    out = "\n".join(report.built_or_served(absorbed, None))
    assert "CORRECTIONS APPLIED" not in out
    assert "CORRECTED ADP is missing durability" in out


def test_no_corrected_ranking_means_no_disposition_either():
    """An ECR-only board ranks on raw consensus, so no correction reached it to be disputed
    -- the same reason `corrections_missing` returns nothing there."""
    out = "\n".join(report.built_or_served(_rep(durability=True, adp=False), None))
    assert "CORRECTIONS APPLIED" not in out


def test_touchdown_luck_is_never_named_as_a_disputed_correction_that_applied():
    """#48 zeroed it rather than flagging it, which is the ticket's own split: a
    sign-reversed coefficient is a bug and does not ship behind a flag."""
    whole = _rep(sos=True, td_luck=True, durability=True, bye=True, adp=True)
    out = "\n".join(report.built_or_served(whole, None))
    assert "touchdown luck" not in out


def _served(**flags):
    from hub.draft.board import SERVED, BuildReport
    return BuildReport(source=SERVED, **flags)


def test_a_served_board_names_itself_its_age_and_what_it_holds():
    """Readable without knowing the fallback exists, which is the whole requirement: a
    reader who has never seen a build fail still learns that this board is not tonight's."""
    out = "\n".join(report.built_or_served(_served(adp=True, td_luck=True), 3.5))
    assert "SERVED BOARD" in out and "3.5h ago" in out and "not rebuilt just now" in out
    assert "it carries" in out and "td_luck" in out and "adp" in out
    assert "built without" not in out, \
        "a board off disk was not built by this run, and must not claim to have been"


def test_a_served_board_that_carries_nothing_says_that_rather_than_nothing():
    out = "\n".join(report.built_or_served(_served(), 1.0))
    assert "no optional signal at all" in out


def test_a_served_board_carrying_everything_still_says_it_is_served():
    """The distinguishing line cannot be a side effect of something being missing."""
    out = "\n".join(report.built_or_served(
        _served(sos=True, td_luck=True, durability=True, bye=True, adp=True,
                scoring_checked=True, roster_checked=True), 2.0))
    assert "SERVED BOARD" in out and "every optional signal" in out


def test_an_unknown_age_is_said_rather_than_formatted():
    """`build_or_last_good` always has an age here. A renderer that raises on a missing one
    would take the draft-night output down for a number that is decoration."""
    assert "age unknown" in "\n".join(report.built_or_served(_served(adp=True), None))


def test_sos_reports_both_ends_and_the_swaps():
    df = _board(pos=["RB"] * 4, team=["A", "B", "C", "D"],
                adp=[10.0, 12.0, 60.0, 90.0],
                wk15_17_sos=[1.30, 0.90, 1.00, 1.00])
    out = "\n".join(report.sos(df, _rep(adp=True, sos=True)))
    assert "SOFTEST" in out and "HARDEST" in out
    assert "Same-tier swaps" in out
    assert "P0" in out.split("Same-tier swaps")[1], "ADP 10 vs 12, SoS gap 0.40 -> a swap"


def test_a_swap_needs_both_a_close_adp_and_a_real_sos_gap():
    """Two players a round apart are not a swap, however different their slates."""
    df = _board(pos=["RB", "RB"], team=["A", "B"], adp=[10.0, 90.0],
                wk15_17_sos=[1.40, 0.80])
    out = "\n".join(report.sos(df, _rep(adp=True, sos=True))).split("Same-tier swaps")[1]
    assert "over" not in out


def test_nothing_here_prints():
    """The whole point. A renderer that prints cannot be composed or capped, which is why
    `live.py` returns lines and this module now does too."""
    import inspect
    src = inspect.getsource(report)
    assert "print(" not in src


def test_the_renderers_take_the_report_not_loose_booleans():
    """Destructuring `BuildReport` at the call site is what let two consumers pick different
    flag combinations, which is how the ECR-only crash shipped. A sixth build stage must not
    change these signatures."""
    import inspect
    for fn in (report.td_luck, report.injuries):
        params = list(inspect.signature(fn).parameters)
        assert "report" in params, f"{fn.__name__} should take the report"
        assert not any(p.startswith("has_") for p in params), \
            f"{fn.__name__} still destructures the report"


def test_the_sos_report_is_skipped_without_adp():
    """It was the one renderer with no gate, and it failed exactly as the other two did:
    `make draft --sos` with no ESPN key raised ColumnNotFoundError before printing anything.
    Every line below is scoped to a drafted player and prints an ADP."""
    df = _board(pos=["RB"], team=["A"], wk15_17_sos=[1.1])
    assert report.sos(df, _rep(adp=False, sos=True)) == []


def test_the_sos_report_is_skipped_when_the_stage_did_not_run():
    """`wk15_17_sos` exists only when the SoS stage succeeded."""
    df = _board(pos=["RB"], team=["A"], adp=[10.0])
    assert report.sos(df, _rep(adp=True, sos=False)) == []


def test_every_renderer_reading_an_optional_column_takes_the_report():
    """The property, not the instance. `adp`, `wk15_17_sos`, `td_luck`, `missed` and
    `injury_status` are all absent from a degraded board and none is in DRAFT_BOARD.required,
    so any renderer touching one must be handed the flags rather than trusting the frame."""
    import inspect
    optional = ("adp", "wk15_17_sos", "td_luck", "missed", "injury_status")
    for name in ("sos", "td_luck", "injuries"):
        fn = getattr(report, name)
        src = inspect.getsource(fn)
        if any(f'"{c}"' in src for c in optional):
            assert "report" in inspect.signature(fn).parameters, \
                f"{name} reads an optional column without being handed the report"
