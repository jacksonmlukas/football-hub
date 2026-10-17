"""`ledger.json` is generated from the documents that record a verdict, and cannot drop one (#483).

The README's conclusions table, the ADR index and the Ledger page would otherwise be three
hand-written copies of the same six-plus verdicts. This contract holds the generated copy to
the documents:

* every ADR whose index row says *accepted* and whose decision names a verdict yields a row
  (an ADR without a front-matter block is the failure the issue names);
* every row cites a document that exists;
* the six verdicts the README table carries today each appear, with the same verdict word.

**Rule 18 -- the controls.** A check that cannot go red proves nothing, so each half has a
planted failure over a throwaway tree: an ADR whose verdict the parser would drop, and a row
citing a document that is not there. They run the same `problems` the real tree is held to.
"""
import shutil
from pathlib import Path

from hub import publish_ledger as pl

ROOT = Path(__file__).resolve().parents[2]

# The README table's six, by the word it prints in bold. Written out here rather than parsed
# from the README, which the issue keeps out of the pipeline: this is the acceptance list.
README_SIX = {
    "Championship equity": "Removed",
    "VOR ordering": "Demoted",
    "edge (consensus vs ADP)": "Shown never ranked",
    "Lineup optimiser": "Inert",
    "Weekly injury cost": "Not established",
}


def _tree(tmp_path: Path) -> Path:
    """A copy of the documents the ledger reads -- nothing else -- to plant failures in."""
    shutil.copytree(ROOT / "docs", tmp_path / "docs",
                    ignore=shutil.ignore_patterns("*.png", "plans", "audits", "brainstorms"))
    return tmp_path


def test_the_real_tree_has_no_problems():
    assert pl.problems(ROOT) == []


def test_the_readmes_verdicts_each_appear_with_the_same_word():
    rows, _ = pl.build(ROOT)
    got = {r["name"]: r["verdict"] for r in rows}
    for name, verdict in README_SIX.items():
        assert got.get(name) == verdict, f"{name!r}: ledger says {got.get(name)!r}"
    # The sixth README row, the five preseason signals: each is a Null row of its own.
    nulls = [r for r in rows if r["verdict"] == "Null"]
    assert len(nulls) == 5 and all(r["adr"] is None for r in nulls)


def test_every_row_has_the_shape_the_page_reads():
    rows, _ = pl.build(ROOT)
    assert rows
    for r in rows:
        assert set(r) >= {"name", "verdict", "figure", "adr", "issue", "as_of"}
        assert r["verdict"] in pl.VERDICTS
        assert r["as_of"][:2] == "20" and len(r["as_of"]) == 10
        assert r["adr"] is None or r["adr"].startswith("docs/adr/")


def test_a_restatement_is_one_front_matter_edit(tmp_path):
    """The acceptance criterion: change one block and the next build carries it."""
    root = _tree(tmp_path)
    adr = next((root / "docs" / "adr").glob("0010-*.md"))
    adr.write_text(adr.read_text().replace("as_of: 2026-09-21", "as_of: 2026-12-01"))
    rows, _ = pl.build(root)
    assert next(r for r in rows if r["adr"] == f"docs/adr/{adr.name}")["as_of"] == "2026-12-01"


# --- controls: a planted failure must turn the check red ---------------------------------

def test_control_an_adr_with_a_verdict_the_parser_drops_is_red(tmp_path):
    root = _tree(tmp_path)
    adr = next((root / "docs" / "adr").glob("0010-*.md"))
    text = adr.read_text()
    adr.write_text(text[text.index("# `edge`"):])  # the block removed: a verdict nobody reads
    bad = pl.problems(root)
    assert any(adr.name in line and "yields no ledger row" in line for line in bad), bad


def test_control_a_malformed_block_is_red_not_a_crash(tmp_path):
    root = _tree(tmp_path)
    adr = next((root / "docs" / "adr").glob("0012-*.md"))
    adr.write_text(adr.read_text().replace("verdict: Inert", "verdict: Inertish"))
    bad = pl.problems(root)
    assert any("Inertish" in line for line in bad), bad
    rows, _ = pl.build(root)  # still builds, minus the bad row
    assert all(r["name"] != "Lineup optimiser" for r in rows)


def test_control_a_row_citing_a_nonexistent_anchor_is_red(tmp_path):
    root = _tree(tmp_path)
    rows, _ = pl.build(root)
    assert pl.anchor_problems(rows, root) == []  # clean first
    planted = [{**rows[0], "adr": "docs/adr/9999-never-written.md"}]
    assert pl.anchor_problems(planted, root), "a row citing a missing ADR passed"
    shot = [{**rows[0], "source": "docs/gone.md"}]
    assert pl.anchor_problems(shot, root)


def test_control_a_restatement_newer_than_as_of_is_red(tmp_path):
    root = _tree(tmp_path)
    adr = next((root / "docs" / "adr").glob("0016-*.md"))
    adr.write_text(adr.read_text() + "\n> **Restated 2026-12-24.** A new figure.\n")
    assert any("2026-12-24" in line for line in pl.problems(root))


def test_no_row_is_older_than_its_sources_latest_dated_revision():
    """Derived from each row's own document, not from a list of expected dates: a revision
    block newer than `as_of` is a verdict restated and a ledger row left behind (#483 review:
    ADR-0016's Amendment and ADR-0009's Superseded figures were both missed this way)."""
    rows, _ = pl.build(ROOT)
    for r in rows:
        latest = pl.latest_revision((ROOT / r["source"]).read_text())
        assert latest is None or r["as_of"] >= latest, (
            f"{r['name']}: as_of {r['as_of']} is older than {r['source']}'s revision {latest}")


def test_the_revision_forms_the_docs_use_are_all_recognised():
    for form in ("Restated 2026-01-02", "Re-scored 2026-01-02", "Amendment, 2026-01-02",
                 "Amended 2026-01-02", "Superseded figures, 2026-01-02",
                 "Attributed 2026-01-02", "Resolved, 2026-01-02: SHOW"):
        assert pl.latest_revision(f"# t\n\n> **{form}.** x\n") == "2026-01-02", form


def _plant(tmp_path, glob, block):
    root = _tree(tmp_path)
    adr = next((root / "docs" / "adr").glob(glob))
    adr.write_text(adr.read_text() + block)
    return root


def test_control_an_amendment_block_newer_than_as_of_is_red(tmp_path):
    root = _plant(tmp_path, "0010-*.md", "\n> **Amendment, 2026-12-24 (#1).** A new verdict.\n")
    assert any("2026-12-24" in line for line in pl.problems(root))


def test_control_a_superseded_figures_block_newer_than_as_of_is_red(tmp_path):
    root = _plant(tmp_path, "0009-*.md", "\n> **Superseded figures, 2026-12-24.** New.\n")
    assert any("2026-12-24" in line for line in pl.problems(root))


def test_front_matter_parser_is_defensive():
    assert pl.front_matter("# no block\n") is None
    assert pl.front_matter("---\nverdict: Null\n") is None  # never closed
    assert pl.front_matter('---\nname: a b\nnot a pair\nfigure: "x: y"\n---\n') == {
        "name": "a b", "figure": "x: y"}
