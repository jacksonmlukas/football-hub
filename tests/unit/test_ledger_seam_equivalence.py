"""#385's equivalence control, authored against `main` before the Ledger seam moved.

**Ordering rule 1**, the same one `test_gate_seam_equivalence.py` states for #386: this file
was written and proven -- green on unchanged code -- *before* `review_width` was replaced by
`hub.ledger.Ledger`. It was committed on its own, ahead of the seam commit, so the control that
holds the refactor to byte-identical output was not itself fitted to the refactor.

**What was captured**, in `tests/unit/fixtures/ledger_seam_equivalence_control.json`, by running
`experiment.review_width` once per scenario (on the pre-#385 tree) and freezing the returned
lines (and, for the `seasons` case, the entry's own `seasons` field) as the golden: a first run
with nothing to compare, a narrower second run (REQUIRES REVIEW), a third run wider than the
most recent one (not the oldest), a run at a different config digest (named, not compared), an
unreadable ledger (refused, not replaced), the pre-#362 dict shape (read, named, not compared),
and a `seasons` frame's round trip through the entry.

**Now that the seam has moved**, this file drives the identical scenarios through
`hub.ledger.Ledger` -- the interface `review_width`'s callers moved to -- and asserts the same
golden lines, byte for byte. `review_width` itself is gone (`run_gate` calls `Ledger.record`
directly), so there is nothing left to call on the old side; this file is the record that the
new side says what the old one said.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hub.ledger import Ledger, WidthEntry

GOLDEN = json.loads((Path(__file__).parent / "fixtures"
                     / "ledger_seam_equivalence_control.json").read_text())


def test_the_ledger_reproduces_review_width_s_lines_scenario_by_scenario(tmp_path):
    """Same scenarios, same order, same golden -- `experiment.review_width` no longer exists
    to call, so this is `Ledger.record` alone, held against the lines it captured."""
    path = tmp_path / "gate-width.json"
    ledger = Ledger(path)

    def record(**kw: Any):
        base: dict[str, Any] = {"name": "draft", "config_digest": "cfg", "data_digest": "dat",
                                "clusters": 4.0, "verdict": "SHOW"}
        entry = WidthEntry(**(base | kw))
        return ledger.record(entry).lines

    assert record(width=4.0, lo=-2.0, hi=2.0) == GOLDEN["first_run"]
    assert record(width=1.0, lo=-0.5, hi=0.5, verdict="REMOVE") == GOLDEN["second_run_narrower"]
    assert record(width=3.0, lo=-1.5, hi=1.5) == GOLDEN["third_run_wider_than_most_recent"]
    assert record(width=0.5, lo=-0.25, hi=0.25, config_digest="other-cfg") == \
        GOLDEN["different_config_digest_not_compared"]

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{ not json")
    solo = Ledger(unreadable)
    got = solo.record(WidthEntry(name="draft", config_digest="cfg", data_digest="dat",
                                 width=2.0, clusters=4.0, lo=-1.0, hi=1.0, verdict="SHOW"))
    assert got.lines == GOLDEN["unreadable_ledger"]
    assert unreadable.read_text() == "{ not json", "the unreadable ledger was replaced"

    pre362 = tmp_path / "pre362.json"
    pre362.write_text(json.dumps({"draft": {"width": 4.0, "clusters": 4.0, "lo": -2.0,
                                            "hi": 2.0, "requires_review": False}}))
    got = Ledger(pre362).record(WidthEntry(name="draft", config_digest="cfg",
                                           data_digest="dat", width=2.0, clusters=4.0,
                                           lo=-1.0, hi=1.0, verdict="SHOW"))
    assert got.lines == GOLDEN["pre_362_dict_shape"]

    seasons_ledger = Ledger(tmp_path / "seasons.json")
    got = seasons_ledger.record(WidthEntry(
        name="weekly", config_digest="cfg", data_digest="dat", width=1.1, clusters=4.0,
        lo=-1.6, hi=-0.5, verdict="SHOW", seasons=GOLDEN["seasons_entry_seasons_field"]))
    assert got.lines == GOLDEN["seasons_first_run"]
    assert got.entry.seasons == GOLDEN["seasons_entry_seasons_field"]
