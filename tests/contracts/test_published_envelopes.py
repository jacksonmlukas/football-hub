"""Every published artifact's envelope names the file it is in.

`jsonio.artifact` exists because one file had two writers producing two documents:
`site/data/live.json` carried the envelope every panel reads when the site writer wrote it and
`{ts, games, detail}` when the poller did, and the page could only read the first. The envelope
is the shape neither writer may contradict, and `name` is its self-description.

It went stale the moment the weekly artifacts gained their season. Issue #23 renamed the two
committed files with `git mv` and never regenerated them, so each carried the name of a file
that no longer existed. Nothing broke -- the page loads by the name the manifest gives -- but an
envelope that exists so two writers cannot disagree was disagreeing with itself, and the next
scheduled slate silently corrected only the week it happened to publish. The archived season's
artifact stayed wrong for another two days.

That is the shape this checks: not that a writer is correct today, but that a *renamed* or
hand-edited artifact cannot sit in the tree describing itself as something else.
"""
import json
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[2] / "site" / "data"

# `manifest.json` is the index rather than an artifact -- it has no `name` of its own, and the
# names it carries are the other files'. `draft_board.json` is a bare list of rows, written by
# the board builder rather than through the envelope, which `publish._board` records.
NOT_ENVELOPES = {"manifest.json", "draft_board.json"}

# Carries the envelope's `name` and `generated_at` but not its row shape. The track record is
# a calibration summary rather than a collection of rows -- it counts in `n_scored` and has no
# `rows` at all -- so `publish.track_record` builds it by hand and passes that count key
# explicitly where the shared writer expects `n`. Named here rather than silently skipped,
# because the difference is a real one about what the artifact *is*.
NOT_ROW_SHAPED = {"track_record.json"}


def _published() -> list[Path]:
    return sorted(p for p in SITE.glob("*.json") if p.name not in NOT_ENVELOPES)


def test_there_are_artifacts_to_check():
    """The premise. A glob that matched nothing would make the assertion below vacuously
    true, which is how a guard like this dies quietly."""
    found = _published()
    assert len(found) >= 4, f"only found {[p.name for p in found]}"
    assert any(p.name.startswith("preds_") for p in found), (
        "no weekly artifact found; the naming this exists to check may have moved")


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.stem)
def test_the_envelope_names_the_file_it_is_in(path):
    got = json.loads(path.read_text())
    assert got.get("name") == path.stem, (
        f"{path.name} describes itself as {got.get('name')!r}. The envelope is what stops two "
        f"writers of one file disagreeing about its shape; one that names a different file is "
        f"disagreeing with itself. A `git mv` of a published artifact has to rewrite the name "
        f"inside it, or the writer has to republish it.")


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.stem)
def test_every_envelope_carries_a_freshness_stamp(path):
    """`generated_at` is what the page ages a panel by and what the watchdog reads as a
    heartbeat. An artifact without one reads as fresh forever."""
    got = json.loads(path.read_text())
    assert got.get("generated_at"), f"{path.name} carries no freshness stamp"


@pytest.mark.parametrize("path", [p for p in _published() if p.name not in NOT_ROW_SHAPED],
                         ids=lambda p: p.stem)
def test_a_row_shaped_envelope_counts_the_rows_it_carries(path):
    """`n` is what the manifest reports without opening `rows`, so the two disagreeing means
    a reader is told one thing and shown another."""
    got = json.loads(path.read_text())
    assert isinstance(got.get("n"), int), f"{path.name} carries no row count"
    assert isinstance(got.get("rows"), list), f"{path.name} carries no rows"
    assert got["n"] == len(got["rows"]), (
        f"{path.name} says {got['n']} rows and carries {len(got['rows'])}")
