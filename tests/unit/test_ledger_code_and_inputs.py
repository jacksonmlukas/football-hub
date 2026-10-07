"""`hub.ledger`: the key carries the code the arms run (#435), and a run's inputs are stored once
beside the ledger rather than inside every row (#434).

Split from `test_ledger.py`, which holds the key's other terms, so each ticket's controls read
in one place. All offline.
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hub.ledger import Ledger, WidthEntry, code_digest

_BASE: dict[str, Any] = {"name": "draft", "config_digest": "cfg", "data_digest": "dat",
                         "clusters": 4.0, "verdict": "SHOW"}


def _entry(**kw: Any) -> WidthEntry:
    return WidthEntry(**(_BASE | kw))


def _planted_module(tmp_path: Path, monkeypatch, body: str, name: str = "planted435_arm") -> str:
    """A throwaway importable module under `tmp_path`, rewritten in place by the test."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / f"{name}.py").write_text(body)
    importlib.invalidate_caches()
    sys.modules.pop(name, None)
    return name


# --- #435 ------------------------------------------------------------------------------------

def test_equal_config_and_data_but_changed_arm_source_do_not_compare(tmp_path, monkeypatch):
    """#435, rule 18. The #361 case: the injury retention baseline changed (a lookahead became
    a strictly-prior mean), config and data did not, and the width review compared the new
    interval to the old. Planted: one arm module, hashed, edited, hashed again, with the two
    runs identical in every other key term. They must not compare, the run must say why, and
    a later run on the *unchanged* source must still compare -- against the run it matches,
    not the one after the edit.

    The plant is shown to flip: with the code term removed the two entries have equal keys, so
    a `comparable` that forgot `code_digest` is exactly what this test fails on."""
    mod = _planted_module(tmp_path, monkeypatch, "BASELINE = 'within-season'\n")
    before = code_digest([mod])
    (tmp_path / f"{mod}.py").write_text("BASELINE = 'strictly prior'\n")
    after = code_digest([mod])
    assert before != after, "an edited arm module must move the digest"

    ledger = Ledger(path=None)
    ledger.record(_entry(width=4.0, lo=-2.0, hi=2.0, recipe="r", code_digest=before))
    changed = ledger.record(_entry(width=1.0, lo=-0.5, hi=0.5, recipe="r", code_digest=after))
    assert changed.previous is None and changed.elsewhere == 1
    assert not changed.requires_review, "a narrowing across a code change is not a narrowing"
    text = "\n".join(changed.lines)
    assert "other arm source" in text and "another config or data digest" not in text

    # the flip: without the code term the pair would have compared
    first = _entry(recipe="r", code_digest=before, width=4.0, lo=-2.0, hi=2.0)
    assert replace(changed.entry, code_digest=None).key == replace(first, code_digest=None).key
    assert changed.entry.key != first.key

    same = ledger.record(_entry(width=3.0, lo=-1.5, hi=1.5, recipe="r", code_digest=before))
    assert same.previous is not None and same.previous.code_digest == before
    assert same.previous.width == pytest.approx(4.0), "compared to its own source's run"


def test_a_row_with_no_code_digest_reads_as_unknown_code_named_and_never_compared(tmp_path):
    """#435: every row written before the key carried code has a recipe and no `code_digest`
    key. It is not compared (not even against a run that declares no modules, whose
    `code_digest` is a known `None`), it is named for its code and not for a digest, and a
    rewrite of the file does not turn it into 'no code declared'."""
    path = tmp_path / "gate-width.json"
    old = {"gate": "draft", "recipe": "r", "config_digest": "cfg", "data_digest": "dat",
           "width": 4.0, "clusters": 4.0, "lo": -2.0, "hi": 2.0, "verdict": "SHOW"}
    path.write_text(json.dumps({"entries": [old]}))
    got = Ledger(path).record(_entry(recipe="r", width=1.0, lo=-0.5, hi=0.5))
    text = "\n".join(got.lines)
    assert got.previous is None and got.elsewhere == 1
    assert "1 earlier run(s) of this gate of unknown code" in text
    assert "another config or data digest" not in text and "unknown recipe" not in text
    rows = json.loads(path.read_text())["entries"]
    assert "code_digest" not in rows[0], "unknown was rewritten as 'no code declared'"
    assert rows[1]["code_digest"] is None
    assert not WidthEntry._from_dict(rows[0]).code_known
    assert WidthEntry._from_dict(rows[1]).code_known
    # and the run recorded just now compares with the next one
    nxt = Ledger(path).record(_entry(recipe="r", width=1.0, lo=-0.5, hi=0.5))
    assert nxt.previous is not None


def test_code_digest_is_order_free_and_refuses_what_it_cannot_hash(tmp_path, monkeypatch):
    a = _planted_module(tmp_path, monkeypatch, "A = 1\n", "planted435_a")
    b = _planted_module(tmp_path, monkeypatch, "B = 2\n", "planted435_b")
    assert code_digest([a, b]) == code_digest([b, a, a])
    assert code_digest([a]) != code_digest([a, b])
    with pytest.raises(ValueError, match="not a module with a source file"):
        code_digest(["planted435_nothing_here"])
    with pytest.raises(ValueError, match="package"):
        code_digest(["hub.models"])
