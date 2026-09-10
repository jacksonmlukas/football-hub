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

**Which assertions apply is the artifact's declaration, not this file's list (#227).** Until
then a hand-maintained `NOT_ROW_SHAPED` set of filenames here decided which artifacts had to
carry `n` and `rows`, while the writers that decide it live in `hub.jsonio`, `hub.publish` and
`hub.fetch.cfbd`. Nothing connected them, so the two could only agree by someone remembering.
On 2026-09-09 nobody did: the scheduled slate published `site/data/cfbd.json` for the first
time -- a quota and freshness envelope, deliberately no rows -- and this contract failed on the
next human push, ten commits and eleven hours later, for an artifact that was correct.

So the envelope carries `shape`, written by `jsonio.artifact` and `jsonio.summary` at the point
the artifact is built, and the row assertions branch on what the file declares. **A missing or
unknown declaration is a failure**, which is the whole difference: a new artifact type is
caught for not declaring itself, which is a true statement about it, rather than for not being
in a list, which is not.
"""
import ast
import json
from pathlib import Path

import pytest

from hub import jsonio

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"
SRC = ROOT / "src" / "hub"

# `manifest.json` is the index rather than an artifact -- it has no `name` of its own, and the
# names it carries are the other files'. It is the only one left: `draft_board.json` was here
# too, as a bare list of rows, and #107 put it in the envelope with the other three
# accommodations its shape required.
NOT_ENVELOPES = {"manifest.json"}

# The vocabulary comes from the writer, so a shape the writer can emit is a shape this
# accepts, and one it cannot is unknown here too. Spelling the two strings out again would
# rebuild the disconnect #227 removed, one word shorter.
KNOWN_SHAPES = frozenset(jsonio.SHAPES)

DECLARE = (
    "Every published envelope declares `shape`, and it is written by the envelope writer, "
    f"not by hand: `jsonio.artifact(...)` stamps {jsonio.ROWS!r} for an artifact that carries `rows` "
    f"and the `n` counting them, and `jsonio.summary(...)` stamps {jsonio.SUMMARY!r} for one that "
    "reports in fields of its own -- the track record's `n_scored`, the CFBD envelope's "
    "`rows_by_endpoint`. Build this artifact through one of them. If it is already published "
    "and you are adding the key to a file the writer no longer emits, add only `shape` and "
    "leave the rows and the stamp untouched."
)


def _published() -> list[Path]:
    return sorted(p for p in SITE.glob("*.json") if p.name not in NOT_ENVELOPES)


def _shape(path: Path) -> str | None:
    got = json.loads(path.read_text())
    declared = got.get("shape")
    return declared if isinstance(declared, str) else None


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
        f"disagreeing with itself. Republish it through the writer; correcting the name by "
        f"hand is the fallback for an archived week the writer no longer emits, and it "
        f"has to leave the rows and the stamp untouched.")


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.stem)
def test_every_envelope_carries_a_freshness_stamp(path):
    """`generated_at` is what the page ages a panel by and what the watchdog reads as a
    heartbeat. An artifact without one reads as fresh forever."""
    got = json.loads(path.read_text())
    assert got.get("generated_at"), f"{path.name} carries no freshness stamp"


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.stem)
def test_every_envelope_declares_a_shape_the_writer_can_write(path):
    """The declaration itself, and the assertion that replaced the allowlist.

    An artifact with no `shape`, or one carrying a word no writer emits, fails here -- which
    is a true statement about the file, unlike "it is not in `NOT_ROW_SHAPED`". This is also
    what keeps the branch below honest: without it, an artifact could dodge the row-count
    assertions simply by declaring nothing.
    """
    declared = _shape(path)
    assert declared in KNOWN_SHAPES, (
        f"{path.name} declares shape {declared!r}; the shapes are "
        f"{sorted(KNOWN_SHAPES)}. {DECLARE}")


def test_both_declared_shapes_are_actually_published():
    """Non-vacuity for the branch. If every artifact declared `summary`, the row-count
    assertions would apply to nothing and pass -- the way a guard like this dies quietly,
    which `test_guards_are_load_bearing.py` exists about. The site publishes both kinds, so
    both arms are required to be exercised."""
    by_shape: dict[str | None, list[str]] = {}
    for path in _published():
        by_shape.setdefault(_shape(path), []).append(path.name)
    for shape in sorted(KNOWN_SHAPES):
        assert by_shape.get(shape), (
            f"nothing published declares shape {shape!r}, so the assertions for it are "
            f"vacuous: {by_shape}")


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.stem)
def test_a_row_shaped_envelope_counts_the_rows_it_carries(path):
    """`n` is what the manifest reports without opening `rows`, so the two disagreeing means
    a reader is told one thing and shown another.

    Applies to exactly the artifacts that declare themselves row-shaped. A summary is held to
    the other half of the same claim -- that it carries no rows -- so the declaration is
    load-bearing in both directions rather than being a way out of one of them.
    """
    got = json.loads(path.read_text())
    if got.get("shape") == jsonio.SUMMARY:
        assert "rows" not in got and "n" not in got, (
            f"{path.name} declares itself a summary and carries "
            f"{sorted({'n', 'rows'} & set(got))}. An artifact carrying rows is row-shaped; "
            f"republish it through `jsonio.artifact`.")
        return
    assert isinstance(got.get("n"), int), f"{path.name} carries no row count"
    assert isinstance(got.get("rows"), list), f"{path.name} carries no rows"
    assert got["n"] == len(got["rows"]), (
        f"{path.name} says {got['n']} rows and carries {len(got['rows'])}")


# --- the declaration is written by a writer, not by a person ----------------

# An envelope built as a dict literal names all three of these; nothing else in `src/` does.
# The manifest's per-artifact row comes closest and carries `name` without `source`, because
# it is a row *about* an artifact rather than an artifact.
ENVELOPE_KEYS = {"name", "source", "generated_at"}

# `hub/jsonio.py` is where the envelope is defined; it is the one file allowed to write one
# out by hand, and every guard in this file reads what it produces.
WRITES_THE_ENVELOPE = "jsonio.py"


def _hand_built_envelopes() -> list[str]:
    """Every dict literal in `src/hub/` that builds an envelope without the writer."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == WRITES_THE_ENVELOPE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if ENVELOPE_KEYS <= keys:
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_the_scan_for_hand_built_envelopes_can_see_one():
    """Non-vacuity, and the shape of the thing it looks for.

    A scan that matched nothing would report every module clean whether or not one had gone
    back to building envelopes by hand -- the failure `test_guards_are_load_bearing.py`
    exists about, and the one this whole ticket is a case of.
    """
    handmade = ast.parse(
        'got = {"name": "cfbd", "source": "hub.fetch.cfbd", '
        '"generated_at": jsonio.stamp(), "rows_by_endpoint": counts}')
    node = next(n for n in ast.walk(handmade) if isinstance(n, ast.Dict))
    keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
    assert ENVELOPE_KEYS <= keys, (
        "this is `hub.fetch.cfbd`'s envelope as it stood on 2026-09-09, before #227; the "
        "scan must recognise it")


def test_no_module_builds_an_envelope_without_the_writer():
    """The acceptance this ticket turns on: the declaration is *written by the writer*.

    `hub.fetch.cfbd` and `publish.track_record` each built their envelope as a dict literal,
    which is how both came to have no shape to declare and why this contract had to keep a
    list of their filenames instead. A module that writes `name`/`source`/`generated_at` by
    hand can no more be relied on to write `shape` than those two were, and the artifact it
    publishes would fail the declaration check above -- on whichever push follows the
    publish, which is #228's complaint and the reason this is checked in `src/`, where it is
    visible before anything is written.
    """
    handmade = _hand_built_envelopes()
    assert not handmade, (
        f"an envelope is built by hand at {handmade}. Build it through "
        f"`jsonio.artifact(...)` if it carries rows or `jsonio.summary(...)` if it reports "
        f"in fields of its own; both stamp `shape`, which is what "
        f"`test_every_envelope_declares_a_shape_the_writer_can_write` reads.")
