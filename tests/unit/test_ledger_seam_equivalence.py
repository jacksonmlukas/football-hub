"""#385's equivalence control, authored against `main` before the Ledger seam moves.

**Ordering rule 1**, the same one `test_gate_seam_equivalence.py` states for #386: this file is
written and proven -- green on unchanged code -- *before* `review_width` is replaced by
`hub.ledger.Ledger`. It is committed on its own, ahead of the seam commit, so the control that
holds the refactor to byte-identical output is not itself fitted to the refactor.

**What is captured**, in `tests/unit/fixtures/ledger_seam_equivalence_control.json`, captured by
running today's `experiment.review_width` once per scenario and freezing the returned lines (and,
for the `seasons` case, the entry's own `seasons` field) as the golden: a first run with nothing
to compare, a narrower second run (REQUIRES REVIEW), a third run wider than the most recent one
(not the oldest), a run at a different config digest (named, not compared), an unreadable ledger
(refused, not replaced), the pre-#362 dict shape (read, named, not compared), and a `seasons`
frame's round trip through the entry.

**After the seam moves**, this file's second half exercises the same scenarios through
`hub.ledger.Ledger` -- the interface `review_width`'s callers move to -- and asserts the same
golden lines. Until that commit lands, only the first half (today's interface) runs; the
second half is added in the commit that adds `hub.ledger`.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from hub.models import experiment

GOLDEN = json.loads((Path(__file__).parent / "fixtures"
                     / "ledger_seam_equivalence_control.json").read_text())

_SEASONS = pl.DataFrame({
    "season": [2022, 2023, 2024, 2025],
    "gain": [-1.120, -1.448, -0.259, -1.761],
    "n": [40, 40, 40, 40],
    "se": [0.30, 0.30, 0.30, 0.30],
    "m": [40, 40, 40, 40],
})


def test_the_review_width_scenarios_match_the_golden_captured_against_main(tmp_path):
    """Today's interface, today's behaviour, frozen. `hub.ledger.Ledger` inherits every one of
    these lines byte-for-byte -- see `test_the_same_scenarios_through_the_ledger_match` below,
    added in the seam commit."""
    path = tmp_path / "gate-width.json"

    def call(name, summary, **kw):
        return experiment.review_width(name, summary, path=path, **kw)

    assert call("draft", {"lo": -2.0, "hi": 2.0, "clusters": 4.0}, verdict="SHOW",
               config_digest="cfg", data_digest="dat") == GOLDEN["first_run"]
    assert call("draft", {"lo": -0.5, "hi": 0.5, "clusters": 4.0}, verdict="REMOVE",
               config_digest="cfg", data_digest="dat") == GOLDEN["second_run_narrower"]
    assert call("draft", {"lo": -1.5, "hi": 1.5, "clusters": 4.0}, verdict="SHOW",
               config_digest="cfg", data_digest="dat") == GOLDEN["third_run_wider_than_most_recent"]
    assert call("draft", {"lo": -0.25, "hi": 0.25, "clusters": 4.0}, verdict="SHOW",
               config_digest="other-cfg", data_digest="dat") == \
        GOLDEN["different_config_digest_not_compared"]

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{ not json")
    assert experiment.review_width(
        "draft", {"lo": -1.0, "hi": 1.0, "clusters": 4.0}, verdict="SHOW", config_digest="cfg",
        data_digest="dat", path=unreadable) == GOLDEN["unreadable_ledger"]

    pre362 = tmp_path / "pre362.json"
    pre362.write_text(json.dumps({"draft": {"width": 4.0, "clusters": 4.0, "lo": -2.0,
                                            "hi": 2.0, "requires_review": False}}))
    assert experiment.review_width(
        "draft", {"lo": -1.0, "hi": 1.0, "clusters": 4.0}, verdict="SHOW", config_digest="cfg",
        data_digest="dat", path=pre362) == GOLDEN["pre_362_dict_shape"]

    seasons_path = tmp_path / "seasons.json"
    assert experiment.review_width(
        "weekly", {"lo": -1.6, "hi": -0.5, "clusters": 4.0}, verdict="SHOW", config_digest="cfg",
        data_digest="dat", path=seasons_path, seasons=_SEASONS) == GOLDEN["seasons_first_run"]
    entries = json.loads(seasons_path.read_text())["entries"]
    assert entries[-1]["seasons"] == GOLDEN["seasons_entry_seasons_field"]
