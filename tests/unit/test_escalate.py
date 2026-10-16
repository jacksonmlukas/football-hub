"""`hub.escalate` surfaces a planted `escalate: true` and stays silent on false (#498)."""
from __future__ import annotations

import json

from hub import escalate


def _put(d, name, doc):
    (d / name).write_text(doc if isinstance(doc, str) else json.dumps(doc))


def test_planted_escalation_is_found_and_reported(tmp_path, capsys):
    _put(tmp_path, "cfbd.json", {"escalate": True, "unfetched_since": "2026-10-06",
                                 "generated_at": "2026-10-14T15:41:51+00:00"})
    _put(tmp_path, "survivor.json", {"escalate": False})
    report = tmp_path / "out.md"
    assert escalate.main(["--site-data", str(tmp_path), "--report", str(report)]) == 0
    assert capsys.readouterr().out.strip() == "escalated=1"
    text = report.read_text()
    assert escalate.MARKER in text and "cfbd.json" in text and "2026-10-06" in text
    assert "survivor.json" not in text


def test_no_escalation_prints_zero_and_writes_nothing(tmp_path, capsys):
    _put(tmp_path, "cfbd.json", {"escalate": False})
    _put(tmp_path, "other.json", {"x": 1})
    report = tmp_path / "out.md"
    assert escalate.main(["--site-data", str(tmp_path), "--report", str(report)]) == 0
    assert capsys.readouterr().out.strip() == "escalated=0"
    assert not report.exists()


def test_only_the_boolean_true_at_top_level_counts(tmp_path):
    _put(tmp_path, "a.json", {"escalate": "true"})
    _put(tmp_path, "b.json", {"nested": {"escalate": True}})
    _put(tmp_path, "c.json", [{"escalate": True}])
    _put(tmp_path, "d.json", "{not json")
    assert escalate.escalated(tmp_path) == []


def test_the_cfbd_record_as_committed_on_2026_10_14_is_caught(tmp_path):
    _put(tmp_path, "cfbd.json", {"name": "cfbd", "fetched": False, "stale": True,
                                 "unfetched_since": "2026-10-06", "escalate": True})
    assert [i["artifact"] for i in escalate.escalated(tmp_path)] == ["cfbd.json"]


def test_a_missing_directory_is_a_refusal_not_zero(tmp_path, capsys):
    assert escalate.main(["--site-data", str(tmp_path / "nope")]) == 2
    out = capsys.readouterr()
    assert "escalated=" not in out.out and "not a directory" in out.err
