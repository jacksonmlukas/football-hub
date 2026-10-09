"""`hub.ledger.Ledger`: `record(entry: WidthEntry) -> Comparison` (#385).

`experiment.review_width` (86 lines) read `state/gate-width.json`, picked the comparable entry,
built the narrowing sentence, built the entry and wrote the file back -- one function whose
interface was the file, and nine tests here used to `json.loads` it to assert. This file's tests
were most of those nine; they now assert on the `Comparison` `Ledger.record` returns, never on
the file. `path=None` (the in-memory adapter) is what several of them use instead of `tmp_path`,
sharing one `Ledger` across two `record` calls so a second run can compare against the first
without a file existing at all.

All offline; nothing here reaches the network.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from hub.ledger import Ledger, WidthEntry

_BASE: dict[str, Any] = {"name": "draft", "config_digest": "cfg", "data_digest": "dat",
                         "clusters": 4.0, "verdict": "SHOW"}


def _entry(**kw: Any) -> WidthEntry:
    return WidthEntry(**(_BASE | kw))


def test_a_first_run_has_nothing_to_compare_and_says_nothing():
    got = Ledger(path=None).record(_entry(width=4.0, lo=-2.0, hi=2.0))
    assert got.lines == [] and got.previous is None and not got.requires_review


def test_the_width_is_recorded_for_the_next_run_and_carries_the_review_flag():
    """Two runs of the same gate: the first has nothing to compare against, the second
    compares against the first and its own record carries `requires_review`."""
    ledger = Ledger(path=None)
    first = ledger.record(_entry(width=4.0, lo=-2.0, hi=2.0))
    assert first.lines == []
    assert first.entry.width == pytest.approx(4.0)
    assert isinstance(first.entry.timestamp, str) and first.entry.timestamp

    second = ledger.record(_entry(width=1.0, lo=-0.5, hi=0.5, verdict="REMOVE"))
    assert "REQUIRES REVIEW" in "\n".join(second.lines)
    assert second.requires_review is True
    assert second.previous is not None and second.previous.width == pytest.approx(4.0)
    assert second.entry.verdict == "REMOVE" and second.entry.width == pytest.approx(1.0)


def test_one_gate_s_history_is_not_another_s():
    """Keyed by name (among the other keys #362 adds), so three gates do not overwrite each
    other and read a narrowing that is really a different gate's interval."""
    ledger = Ledger(path=None)
    ledger.record(_entry(width=4.0, lo=-2.0, hi=2.0))
    got = ledger.record(_entry(name="weekly", width=1.0, lo=-0.5, hi=0.5))
    assert got.lines == [] and got.previous is None and got.elsewhere == 0, \
        "a different gate's own entry is not 'elsewhere' either -- it is simply not this gate's"


def test_a_third_run_of_the_same_gate_compares_against_the_most_recent_one():
    """Not the first entry ever written for a gate -- the last one -- so a gate run three
    times reads its own second run's width, not its first."""
    ledger = Ledger(path=None)
    ledger.record(_entry(width=4.0, lo=-2.0, hi=2.0))    # first
    ledger.record(_entry(width=2.0, lo=-1.0, hi=1.0))    # second
    third = ledger.record(_entry(width=3.0, lo=-1.5, hi=1.5))
    assert third.previous is not None and third.previous.width == pytest.approx(2.0)
    assert "wider" in "\n".join(third.lines), "3 against the most recent 2, not the oldest 4"


def test_the_pre_362_dict_shape_still_reads_but_is_not_compared(tmp_path):
    """A file this module has not yet rewritten -- the one-record-per-gate shape #362
    replaces -- does not read as empty, so an old file's history is not silently dropped.
    But an old entry carries no `recipe` key at all, so it is of *unknown* recipe and is not
    compared against: read into the ledger and named, not read as a narrowing."""
    path = tmp_path / "gate-width.json"
    path.write_text(json.dumps({"draft": {"width": 4.0, "clusters": 4.0, "lo": -2.0, "hi": 2.0,
                                          "requires_review": False}}))
    got = Ledger(path).record(_entry(width=1.0, lo=-1.0, hi=1.0))
    text = "\n".join(got.lines)
    assert "REQUIRES REVIEW" not in text
    assert "1 earlier run(s) of this gate at another config or data digest" in text
    assert got.previous is None and got.elsewhere == 1
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 2 and entries[0]["width"] == pytest.approx(4.0), "the old row is kept"


def test_two_runs_at_different_digests_are_not_compared():
    """Rule 18's positive control for the digest condition: the same gate, the same width
    halving, at a different config digest -- the case that produced a meaningless REQUIRES
    REVIEW on 2026-09-21 (a --holdout draft run against a non-holdout one). It must not
    compare, and it must say why."""
    ledger = Ledger(path=None)
    ledger.record(_entry(config_digest="holdout-2022", width=4.0, lo=-2.0, hi=2.0))
    got = ledger.record(_entry(config_digest="shipped", width=1.0, lo=-0.5, hi=0.5))
    text = "\n".join(got.lines)
    assert "REQUIRES REVIEW" not in text
    assert got.previous is None and got.elsewhere == 1
    assert "1 earlier run(s) of this gate at another config or data digest" in text
    assert got.requires_review is False


def test_the_same_digest_still_compares_and_a_data_digest_alone_is_enough_to_block():
    """Both halves of the condition: identical digests compare (the flag can still fire), and
    a changed *data* digest alone -- same constants, a different board -- blocks it too."""
    ledger = Ledger(path=None)
    ledger.record(_entry(width=4.0, lo=-2.0, hi=2.0))
    same = ledger.record(_entry(width=1.0, lo=-0.5, hi=0.5))
    assert "REQUIRES REVIEW" in "\n".join(same.lines)
    other = ledger.record(_entry(data_digest="another-board", width=0.5, lo=-0.25, hi=0.25))
    assert "REQUIRES REVIEW" not in "\n".join(other.lines)


def test_same_digests_different_recipes_do_not_compare():
    """#384's own control, required here rather than there: two entries at one digest pair
    but a different `recipe` -- the run's arm -- do not compare, the same way a different
    digest does not. `key` is `(name, recipe, config_digest, data_digest)`; unequal on any one
    field is unequal, and `recipe` is one of the four."""
    ledger = Ledger(path=None)
    ledger.record(_entry(recipe="a", width=4.0, lo=-2.0, hi=2.0))
    got = ledger.record(_entry(recipe="b", width=1.0, lo=-0.5, hi=0.5))
    assert got.previous is None and got.elsewhere == 1
    assert "REQUIRES REVIEW" not in "\n".join(got.lines)
    # And the boundary: the *same* recipe at the same digest pair does compare.
    same = ledger.record(_entry(recipe="a", width=0.5, lo=-0.25, hi=0.25))
    assert same.previous is not None and same.previous.width == pytest.approx(4.0)


def test_an_unreadable_history_costs_a_line_and_not_the_run_and_is_not_replaced(tmp_path):
    """CLAUDE.md's degradation rule. A gate that cannot read its own history still has a
    verdict; a harness that dies because a JSON file is half-written does not. **And the
    history is not destroyed**: before 2026-09-21 an unreadable ledger read as no history, the
    run appended its one row to nothing and wrote the file back -- an append-only ledger
    replaced by a one-entry file that then looked valid. The positive control is the bytes:
    plant an unparsable file, record with the default `write=True`, and the bytes are what
    they were."""
    path = tmp_path / "gate-width.json"
    planted = "{ this is not json"
    path.write_text(planted)
    got = Ledger(path).record(_entry(width=1.0, lo=-1.0, hi=1.0))
    assert path.read_text() == planted, "the unreadable ledger was replaced"
    assert len(got.lines) == 1 and "does not parse" in got.lines[0]
    assert "not replaced" in got.lines[0]
    assert "REQUIRES REVIEW" not in got.lines[0]
    assert got.previous is None


def test_a_missing_ledger_is_created_and_a_present_one_is_appended_atomically(tmp_path):
    """The other side of the same fix: absent is not unreadable -- a first run creates the
    file, a second appends, and no scratch file is left beside it."""
    path = tmp_path / "gate-width.json"
    ledger = Ledger(path)
    ledger.record(_entry(width=4.0, lo=-2.0, hi=2.0))
    ledger.record(_entry(width=2.0, lo=-1.0, hi=1.0))
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 2
    assert sorted(p.name for p in path.parent.iterdir()) == ["gate-width.json"]


def test_a_width_state_that_cannot_be_written_does_not_take_the_gate_down(tmp_path):
    """The narrowing record is a convenience, and the gate's verdict is not.

    A `Ledger` remembers the previous interval so criterion 5 can report a run that narrowed.
    If that file cannot be written -- a read-only checkout, a runner with no state directory --
    `record` must still return a `Comparison`. What it loses is the comparison on the *next*
    run, which is the right thing to lose.
    """
    d = tmp_path / "ro"
    d.mkdir()
    path = d / "gate-width.json"
    d.chmod(0o500)                                  # writable no longer
    try:
        got = Ledger(path).record(_entry(width=1.0, lo=-2.0, hi=-1.0))
        assert got is not None, "a gate lost its verdict to a state file it could not write"
        assert not path.exists(), "the fixture did not actually make the write fail"
    finally:
        d.chmod(0o700)


# --- #382: the ledger's `seasons` field ---------------------------------------------------

_SEASONS_RECORDS = [
    {"season": 2022, "gain": -1.120, "se": 0.30, "m": 40, "disposition": "loss"},
    {"season": 2023, "gain": -1.448, "se": 0.30, "m": 40, "disposition": "loss"},
    {"season": 2024, "gain": -0.259, "se": 0.30, "m": 40, "disposition": "tie"},
    {"season": 2025, "gain": -1.761, "se": 0.30, "m": 40, "disposition": "loss"},
]


def test_a_ledger_entry_round_trips_its_seasons_field(tmp_path):
    """A caller's per-season records land in the recorded entry exactly as handed in -- a
    later reader gets the per-season `gain`/`se`/`m`/`disposition` back off the `Comparison`
    (and, on disk, off `state/gate-width.json`) rather than only off a run's stdout (#382,
    split from #381)."""
    path = tmp_path / "gate-width.json"
    got = Ledger(path).record(_entry(name="weekly", width=1.1, lo=-1.6, hi=-0.5,
                                     seasons=_SEASONS_RECORDS))
    assert got.entry.seasons == _SEASONS_RECORDS
    entries = json.loads(path.read_text())["entries"]
    assert entries[-1]["seasons"] == _SEASONS_RECORDS
    assert [r["season"] for r in entries[-1]["seasons"]] == [2022, 2023, 2024, 2025]


def test_a_ledger_entry_with_no_seasons_frame_carries_no_seasons_field(tmp_path):
    """The field is additive: an entry constructed with no `seasons` -- every entry before
    #382, and every call in this file above -- writes exactly the pre-#382 shape, so the
    pre-#362 dict-shape read (which has no `seasons` key either) stays unaffected."""
    path = tmp_path / "gate-width.json"
    Ledger(path).record(_entry(width=1.0, lo=-1.0, hi=1.0))
    entries = json.loads(path.read_text())["entries"]
    assert "seasons" not in entries[-1]


def test_a_ledger_entry_round_trips_the_abstention_count(tmp_path):
    """#381 (C): a verdict read over fewer seasons than were run says so in the record, as
    numbers beside the verdict -- `resolved` and `abstained` survive the write and the read."""
    path = tmp_path / "gate-width.json"
    ledger = Ledger(path)
    got = ledger.record(_entry(name="weekly", width=1.1, lo=-1.6, hi=-0.5, seasons=_SEASONS_RECORDS,
                               resolved=3, abstained=1))
    assert (got.entry.resolved, got.entry.abstained) == (3, 1)
    on_disk = json.loads(path.read_text())["entries"][-1]
    assert (on_disk["resolved"], on_disk["abstained"]) == (3, 1)
    # and a second write rereads the first without dropping or inventing either number
    ledger.record(_entry(name="weekly", width=1.2, lo=-1.7, hi=-0.5))
    first, second = json.loads(path.read_text())["entries"]
    assert (first["resolved"], first["abstained"]) == (3, 1)
    assert "resolved" not in second and "abstained" not in second


def test_a_ledger_entry_round_trips_the_inputs_it_read(tmp_path):
    """#429: the per-source reads survive the write and the read, an entry without them carries
    no key, and a malformed value on disk reads as unrecorded rather than raising."""
    path = tmp_path / "gate-width.json"
    reads = [{"source": "ff_rankings", "as_of": "2025-08-31", "digest": "abc"},
             {"source": "player_stats", "as_of": None, "digest": "unpinned"}]
    ledger = Ledger(path)
    got = ledger.record(_entry(width=1.0, lo=-1.0, hi=1.0, inputs=reads))
    assert got.entry.inputs == reads
    ledger.record(_entry(width=1.1, lo=-1.0, hi=1.0))
    first, second = json.loads(path.read_text())["entries"]
    # #434: the row names the reads by digest; the full set is stored once beside the ledger.
    assert "inputs" not in first and "inputs_digest" in first, "the row carries a digest only"
    assert "inputs" not in second and "inputs_digest" not in second
    entries = Ledger(path)._read()
    assert entries is not None
    assert sorted(ledger.inputs_of(entries[0]) or [], key=lambda r: r["source"]) == reads
    assert ledger.inputs_of(entries[1]) is None
    first["inputs"] = "not a list"
    del first["inputs_digest"]
    path.write_text(json.dumps({"entries": [first, second]}))
    entries = Ledger(path)._read()
    assert entries is not None and all(e.inputs is None for e in entries)


def test_a_row_with_no_abstention_count_reads_as_unrecorded_not_zero(tmp_path):
    """Every entry before #381 carries neither key, and `0 abstained` would be a claim about
    them. `None` stays `None` through a rewrite; a non-integer on disk reads as `None` too."""
    path = tmp_path / "gate-width.json"
    path.write_text(json.dumps({"entries": [
        {"gate": "weekly", "config_digest": "c", "data_digest": "d", "width": 1.0, "clusters": 4,
         "lo": -1.0, "hi": 0.0, "verdict": "SHOW", "recipe": None},
        {"gate": "weekly", "config_digest": "c", "data_digest": "d", "width": 1.0, "clusters": 4,
         "lo": -1.0, "hi": 0.0, "verdict": "SHOW", "recipe": None, "resolved": "3",
         "abstained": True}]}))
    entries = Ledger(path)._read()
    assert entries is not None
    assert all(e.resolved is None and e.abstained is None for e in entries)


# --- the in-memory adapter is a real adapter, not a stub -----------------------------------


def test_an_in_memory_ledger_behaves_like_a_file_backed_one_across_calls():
    """`path=None` keeps entries in the object rather than on disk, but every other rule --
    the most-recent-comparable lookup, `elsewhere`, the narrowing sentence -- is unchanged.
    This is the adapter `run_gate`'s callers inject in place of a tmp file (#385)."""
    ledger = Ledger(path=None)
    a = ledger.record(_entry(name="p", width=4.0, lo=-2.0, hi=2.0))
    assert a.lines == []
    b = ledger.record(_entry(name="p", width=1.0, lo=-0.5, hi=0.5))
    assert "REQUIRES REVIEW" in "\n".join(b.lines)
    # A second, independent in-memory ledger shares nothing with the first.
    fresh = Ledger(path=None).record(_entry(name="p", width=1.0, lo=-0.5, hi=0.5))
    assert fresh.lines == [] and fresh.previous is None


def test_write_false_compares_but_does_not_persist(tmp_path):
    """`hub.draft.backtest`'s noise sweep reads this way: it must not crowd a gate's own
    history with a sensitivity sweep's rows, but a `write=False` ledger still reads and
    compares against whatever is already there."""
    path = tmp_path / "gate-width.json"
    Ledger(path).record(_entry(width=4.0, lo=-2.0, hi=2.0))
    sweep = Ledger(path, write=False)
    got = sweep.record(_entry(width=1.0, lo=-0.5, hi=0.5))
    assert "REQUIRES REVIEW" in "\n".join(got.lines), "still compares against the real history"
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 1, "the sweep's own row was never appended"


def test_a_planted_recipe_regression_would_be_caught():
    """Rule 18, restated: mutate `WidthEntry.comparable` to drop the recipe term and confirm
    this file's own `test_same_digests_different_recipes_do_not_compare` is what catches it.
    Not runnable as a mutation from here (that test IS the plant-and-confirm); this test pins
    the boundary case the mutation would touch, at the type level, so a future edit to `.key`
    or `.comparable` that silently forgets `recipe` fails immediately rather than only on a
    directly-targeted mutation run."""
    a = _entry(recipe="a", width=4.0, lo=-2.0, hi=2.0)
    b = _entry(recipe="b", width=4.0, lo=-2.0, hi=2.0)
    assert a.key != b.key
    assert not a.comparable(b)


def test_an_entry_with_null_bounds_reads_and_does_not_take_the_gate_down(tmp_path):
    """Rule 18 for the read side's own degradation promise: a row whose `lo`, `hi` or
    `clusters` is `null` (hand-edited, or written by a future field's absence) is read as
    missing numbers, not raised on -- `record` never raises, whatever it finds on disk."""
    path = tmp_path / "gate-width.json"
    path.write_text(json.dumps({"entries": [
        {"gate": "draft", "recipe": None, "code_digest": None, "config_digest": "cfg",
         "data_digest": "dat", "width": 4.0, "clusters": None, "lo": None, "hi": None,
         "verdict": "SHOW"}]}))
    got = Ledger(path).record(_entry(width=1.0, lo=-0.5, hi=0.5))
    assert got.previous is not None and got.previous.width == pytest.approx(4.0)


def test_a_recipe_less_row_at_the_same_digests_is_named_for_its_recipe_not_a_digest(tmp_path):
    """Every row `review_width` wrote after #362 carries both digests and no `recipe` key.
    Such a row is skipped only because its recipe is unknown, so the line says that -- the
    digest line would name a mismatch that is not there."""
    path = tmp_path / "gate-width.json"
    path.write_text(json.dumps({"entries": [
        {"gate": "draft", "config_digest": "cfg", "data_digest": "dat", "width": 4.0,
         "clusters": 4.0, "lo": -2.0, "hi": 2.0, "verdict": "SHOW"}]}))
    got = Ledger(path).record(_entry(width=1.0, lo=-0.5, hi=0.5))
    text = "\n".join(got.lines)
    assert got.previous is None and got.elsewhere == 1
    assert "1 earlier run(s) of this gate of unknown recipe" in text
    assert "another config or data digest" not in text


def test_a_row_of_unknown_recipe_stays_unknown_through_a_rewrite_of_the_file(tmp_path):
    """#427, rule 18. An entry written before #385 has no `recipe` key and reads `known=False`.
    `record` rewrites the whole file, and `_as_dict` used to emit `"recipe": null` for it, which
    reads back `known=True` -- "no arm declared" -- so the first run after an old row existed
    changed what the record says about that row, and dropped its `requires_review`. Write an
    old-shape row, record two new entries (two rewrites), and the old row must still carry no
    `recipe` key, still carry its flag, and still read as unknown."""
    path = tmp_path / "gate-width.json"
    old = {"gate": "draft", "config_digest": "c1", "data_digest": "d1", "width": 4.0,
           "clusters": 4.0, "lo": -2.0, "hi": 2.0, "verdict": "REMOVE",
           "requires_review": True, "timestamp": "2026-09-16"}
    path.write_text(json.dumps({"entries": [old]}))
    Ledger(path).record(_entry(name="draft", config_digest="c1", data_digest="d1", width=1.0, lo=-1.0, hi=1.0))
    got = Ledger(path).record(_entry(name="draft", config_digest="c1", data_digest="d1", width=1.0, lo=-1.0, hi=1.0))
    first = json.loads(path.read_text())["entries"][0]
    assert "recipe" not in first, "unknown was rewritten as 'no arm declared'"
    assert first["requires_review"] is True and first["timestamp"] == "2026-09-16"
    assert [e["recipe"] for e in json.loads(path.read_text())["entries"][1:]] == [None, None]
    # and it is still not compared: the third run names it as of unknown recipe
    assert any("unknown recipe" in ln for ln in got.lines) or got.previous is not None
    assert not WidthEntry._from_dict(first).known


# --- the shared import-closure helpers (#456) -----------------------------------------------------

def test_the_closure_helpers_refuse_a_name_that_is_not_a_source_module():
    """A name that does not resolve raises, rather than hashing or walking nothing: a pin over a
    module that is not there would pass trivially."""
    from hub.closure import first_party_imports, module_digests

    with pytest.raises(ValueError, match="not a module with a source file"):
        module_digests(["hub.nothing_here_456"])
    with pytest.raises(ValueError, match="not a module with a source file"):
        first_party_imports("hub.nothing_here_456")


def test_module_digests_are_per_module_and_an_exempt_module_is_not_walked_in():
    from hub.closure import import_closure, module_digests

    got = module_digests(["hub.names", "hub.league"])
    assert set(got) == {"hub.names", "hub.league"} and all(len(v) == 12 for v in got.values())
    assert module_digests(["hub.league"])["hub.league"] == got["hub.league"]
    assert "hub.paths" not in import_closure(["hub.league", "hub.names"])
