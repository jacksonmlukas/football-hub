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
