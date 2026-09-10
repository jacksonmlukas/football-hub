"""JSON a browser can parse.

Python's `json` emits bare `NaN` and reads it back happily; JSON has no such literal and
neither does JavaScript, so `JSON.parse` throws on the whole document. `site/index.html`
catches that and renders the panel as absent -- which is how `draft_board.json`, present and
fresh at 199KB with 135 `NaN`s in it, showed as "No draft board. Run `make draft`".

That file is what `docs/draft-night.md` names as the last-resort fallback for draft night.
"""
import json
import math

import pytest

from hub import jsonio


def test_nan_and_infinity_become_null():
    got = jsonio.dumps({"a": float("nan"), "b": float("inf"), "c": float("-inf")})
    assert json.loads(got) == {"a": None, "b": None, "c": None}


def test_finite_numbers_are_untouched():
    payload = {"a": 1.5, "b": 0.0, "c": -3, "d": 1e300}
    assert json.loads(jsonio.dumps(payload)) == payload


def test_it_reaches_any_depth():
    payload = {"rows": [{"vor": float("nan"), "nested": {"x": [1.0, float("inf")]}}]}
    assert json.loads(jsonio.dumps(payload)) == {
        "rows": [{"vor": None, "nested": {"x": [1.0, None]}}]}


def test_the_output_is_valid_json_by_a_strict_parser():
    """`json.loads` accepts `NaN` by default, so the round-trip above is not the whole test --
    a strict parser is what the browser is."""
    got = jsonio.dumps([{"v": float("nan")}])
    assert "NaN" not in got and "Infinity" not in got
    json.loads(got, parse_constant=_reject)


def _reject(name):
    raise AssertionError(f"a non-finite literal survived: {name}")


def test_non_serialisable_values_still_fall_back_to_str():
    import datetime as dt
    got = json.loads(jsonio.dumps({"when": dt.date(2026, 9, 3)}))
    assert got["when"] == "2026-09-03"


def test_indent_is_available_for_the_artifacts_that_want_it():
    assert "\n" in jsonio.dumps({"a": 1}, indent=2)
    assert "\n" not in jsonio.dumps({"a": 1})


def test_jsonio_is_a_leaf():
    """Both writers need it -- `hub.publish` and `hub.draft.board` -- and neither should have
    to import the other to get it."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "jsonio.py"
    mods = {n.module for n in ast.walk(ast.parse(src.read_text()))
            if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.startswith("hub") for m in mods)


def test_both_writers_use_it():
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    for rel in ("publish.py", "draft/board.py"):
        tree = ast.parse((root / rel).read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                 for a in n.names}
        assert "jsonio" in names, f"{rel} writes JSON without the NaN-safe dumper"
    assert math.isnan(float("nan")), "sanity"


# --- the envelope cannot be redefined by a caller ---------------------------

# `name`, `source` and `rows` are positional parameters, so Python refuses them before the
# check ever runs -- a different exception for the same reason. Only these two can reach it.
@pytest.mark.parametrize("field", ["generated_at", "n"])
def test_extra_cannot_displace_an_envelope_key(field):
    """`artifact`'s whole reason for existing is that two writers of one file cannot
    disagree about its shape -- `site/data/live.json` had a site writer producing the
    envelope every panel reads and a poller producing `{ts, games, detail}`, and the page
    could only read the first.

    A caller passing one of these in `extra` would be redefining that shape from the
    outside, which is the disagreement rather than the fix. Found untested by the excision
    harness: deleting the check left this whole module green.
    """
    with pytest.raises(ValueError, match=field):
        jsonio.artifact("x", "src", [], **{field: "hijacked"})


@pytest.mark.parametrize("field", ["name", "source", "rows"])
def test_an_envelope_key_that_is_a_parameter_is_refused_by_python(field):
    """Same refusal, enforced one layer down. Worth pinning so a future signature change
    that turned one of these into a keyword would surface here rather than silently widen
    what a caller can redefine."""
    with pytest.raises(TypeError):
        jsonio.artifact("x", "src", [], **{field: "hijacked"})


def test_an_extra_field_the_envelope_does_not_own_is_fine():
    """The refusal is scoped. `extra` is for what one artifact has and another does not --
    the league a scoreboard is for, the season a slate belongs to."""
    got = jsonio.artifact("survivor", "hub.season.survivor", [], season=2026, survival=0.007)
    assert got["season"] == 2026 and got["survival"] == 0.007
    assert got["n"] == 0 and got["name"] == "survivor"


# --- an artifact declares which shape it is (#227) --------------------------


def test_a_row_shaped_artifact_declares_itself_row_shaped():
    """The declaration `tests/contracts/test_published_envelopes.py` reads.

    Before #227 the difference between a row-shaped artifact and a summary lived in
    `NOT_ROW_SHAPED`, a set of filenames in that contract, while the decision was made here.
    Nothing connected the two, so `site/data/cfbd.json` -- correct, and simply not on the
    list -- broke CI on a push ten commits after the one that published it.
    """
    got = jsonio.artifact("preds_2026_wk01", "preds", [{"a": 1}])
    assert got["shape"] == jsonio.ROWS
    assert got["n"] == 1 and got["rows"] == [{"a": 1}]


def test_a_summary_declares_itself_a_summary_and_carries_no_rows():
    """The other shape: an artifact that reports rather than carries."""
    got = jsonio.summary("cfbd", "hub.fetch.cfbd", rows_by_endpoint={"games": 12})
    assert got["shape"] == jsonio.SUMMARY
    assert "rows" not in got and "n" not in got
    assert got["name"] == "cfbd" and got["generated_at"]
    assert got["rows_by_endpoint"] == {"games": 12}


def test_the_two_shapes_are_the_only_ones_and_they_differ():
    """`SHAPES` is what the contract reads its vocabulary from, so an empty or duplicated
    tuple here would make "the shapes are these" say nothing."""
    assert set(jsonio.SHAPES) == {jsonio.ROWS, jsonio.SUMMARY}
    assert jsonio.ROWS != jsonio.SUMMARY
    assert len(jsonio.SHAPES) == 2


@pytest.mark.parametrize("field", ["generated_at", "shape", "n", "rows"])
def test_a_summary_cannot_restate_the_envelope_or_claim_rows(field):
    """`shape` is the envelope's now, so a caller cannot set it -- an artifact that could
    name its own shape from the outside is back to two writers disagreeing.

    `n` and `rows` are refused for the stronger reason: a document carrying rows *is*
    row-shaped, and one carrying them under `shape: "summary"` would be a declaration that
    contradicts its own document, which is worse than the allowlist this replaced.
    """
    with pytest.raises(ValueError, match=field):
        jsonio.summary("x", "src", **{field: "hijacked"})


def test_shape_cannot_be_displaced_on_a_row_shaped_artifact_either():
    """The same refusal on `artifact`, which is where the guard already lived; `shape`
    joined the set it protects."""
    with pytest.raises(ValueError, match="shape"):
        jsonio.artifact("x", "src", [], shape="summary")


def test_a_summary_stamps_the_file_it_read_when_it_is_given_one(tmp_path):
    """`as_of` works the same on both writers. A summary derived from a file is as fresh as
    the file, not as the run -- the issue #122 rule, and the reason it is not spelled
    `generated_at`."""
    src = tmp_path / "x.parquet"
    src.write_text("")
    got = jsonio.summary("track_record", "preds+results", jsonio.file_stamp(src), n_scored=0)
    assert got["generated_at"] == jsonio.file_stamp(src)
    assert got["shape"] == jsonio.SUMMARY
