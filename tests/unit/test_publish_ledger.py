"""`hub.publish --ledger` writes `ledger.json`, and the manifest lists it (#483)."""
import json

from hub import publish


def test_ledger_is_written_with_its_own_generated_at(tmp_path):
    got = publish._ledger(tmp_path)
    assert isinstance(got, dict)
    assert got["n"] == len(got["rows"]) >= 10
    written = json.loads((tmp_path / "ledger.json").read_text())
    assert written["generated_at"] and written["shape"] == "rows"
    assert {r["verdict"] for r in written["rows"]} >= {"Removed", "Inconclusive", "Not established"}


def test_publish_all_puts_the_ledger_in_the_manifest(tmp_path):
    # The declared artifact itself, not `publish_all`, which would also reach the store.
    art = next(a for a in publish.artifacts(2026, 1, out=tmp_path) if a.name == "ledger")
    entry = art.record(tmp_path)
    assert entry["present"] and not entry["stale"] and entry["generated_at"]


def test_a_missing_docs_tree_keeps_last_good(tmp_path):
    from hub import publish_ledger

    publish._ledger(tmp_path)
    before = (tmp_path / "ledger.json").read_text()
    got = publish_ledger.ledger(out=tmp_path, root=tmp_path / "no-docs-here")
    assert isinstance(got, publish.Kept)
    assert (tmp_path / "ledger.json").read_text() == before


def test_cli_ledger_flag(tmp_path, capsys):
    assert publish.main(["--ledger", "--week", "1", "--out", str(tmp_path)]) == 0
    assert "verdicts" in capsys.readouterr().out
    assert (tmp_path / "ledger.json").exists()
