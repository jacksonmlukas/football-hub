"""The scheduled slate runs the contracts that read what it publishes, on its own run.

The slate commit `4520276` ("slate: week 1 refresh") wrote seven files under `site/data/` and
`state/`. Only `pages.yml` runs on that commit -- a push made with `GITHUB_TOKEN` cannot
trigger another workflow, which is GitHub's guard against a job that commits and re-triggers
itself forever, and it is why `pages.yml` reaches the slate through `workflow_run` instead. So
the first process to read the new `site/data/cfbd.json` was the contract suite on the next
*human* push: `f6510f2`, ten commits and eleven hours later.

The failure was correct. The attribution was not: a red run appeared on a push that changed
nothing about publishing, and the first hypothesis was that the push had broken it. This repo
has paid that specific cost before -- a CI red that had been red for four hours before the push
it was blamed on.

**Graceful degradation makes it worse rather than better.** `CLAUDE.md` requires the slate to
run with zero attention. A publish that can silently emit an artifact no contract has read is
exactly the unattended failure that design exists to prevent, and "someone will see it on
Monday" is not a property of a system, it is a hope about a person.

**What this file decides, and what it cannot.** It decides that `slate.yml` names, before its
commit step, exactly the contract modules that read the committed `site/data/` -- the list in
the workflow against the test tree, derived rather than remembered, because a hand-kept list of
test paths inside a YAML file is the same defect as `NOT_ROW_SHAPED` one layer out. It cannot
decide that GitHub runs that step, that the runner has a venv, or that the step passes on a
real slate: a workflow is not executable here, and this file must not be read as saying it is.
The only claim is about what the workflow says.
"""
from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = Path(__file__).resolve().parent
SLATE = ROOT / ".github" / "workflows" / "slate.yml"

# The published tree the slate writes. `site/index.html` is deliberately not this: the slate
# does not write the page, only the artifacts under it.
PUBLISHED = ("site", "data")

# The step that must run them, and the step it must precede.
CHECKS = "Check the artifacts against the contracts that read them"
COMMITS = "Commit what was published"

# Naming either of these would be running the whole suite, which is what the ticket asks this
# not to be: `ci.yml`'s job is `tests/unit tests/contracts` under coverage, minutes rather
# than the sub-second the two published-artifact modules take.
WHOLE_SUITE = ("tests/unit", "tests ", "tests\n", "tests/contracts ", "tests/contracts\n")


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope -- `ROOT`, `SITE`. Not `tmp_path`, which is a fixture."""
    names = set()
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _reads_the_published_tree(source: str) -> bool:
    """Does this module bind the *committed* `site/data/`, rather than one under `tmp_path`?

    `x / "site" / "data"` where `x` is a module-level constant. The distinction is the whole
    of the rule: `tests/contracts/test_live_loop.py` writes `tmp_path / "site" / "data"` and
    is a harness over a fixture, so a slate run has nothing for it to read.
    """
    tree = ast.parse(source)
    module_level = _module_level_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        parts: list[str] = []
        cur: ast.expr = node
        while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
            if not (isinstance(cur.right, ast.Constant) and isinstance(cur.right.value, str)):
                break
            parts.insert(0, cur.right.value)
            cur = cur.left
        if tuple(parts) == PUBLISHED and isinstance(cur, ast.Name) and cur.id in module_level:
            return True
    return False


def _modules_reading_the_published_tree() -> set[str]:
    return {f"tests/contracts/{p.name}" for p in sorted(CONTRACTS.glob("test_*.py"))
            if _reads_the_published_tree(p.read_text(encoding="utf-8"))}


def _steps() -> list[dict]:
    return yaml.safe_load(SLATE.read_text(encoding="utf-8"))["jobs"]["refresh"]["steps"]


def _named(step: str) -> dict:
    got = [s for s in _steps() if s.get("name") == step]
    assert len(got) == 1, f"{SLATE.name} has {len(got)} steps named {step!r}; expected one"
    return got[0]


def test_the_derivation_can_tell_the_published_tree_from_a_fixture_one() -> None:
    """Non-vacuity, and the one distinction the derivation rests on.

    A rule that answered "yes" to every module would make the comparison below demand the
    whole contract directory run on every slate; one that answered "no" to every module would
    let the workflow name nothing and pass. Both are checked here rather than inferred from
    the real tree, where a single unnoticed change of shape would move the answer.
    """
    real = 'from pathlib import Path\nROOT = Path(".")\nSITE = ROOT / "site" / "data"\n'
    fixture = 'def test_x(tmp_path):\n    site = tmp_path / "site" / "data"\n'
    page = 'from pathlib import Path\nROOT = Path(".")\nP = ROOT / "site" / "index.html"\n'
    assert _reads_the_published_tree(real), "a module-level `ROOT / site / data` is the tree"
    assert not _reads_the_published_tree(fixture), (
        "`tmp_path / site / data` is a harness over a fixture; a slate run writes nothing "
        "there and running it at publish time would prove nothing about what was published")
    assert not _reads_the_published_tree(page), (
        "the slate writes the artifacts, not `site/index.html`")


def test_some_contract_actually_reads_the_published_tree() -> None:
    """The premise for the comparison. If nothing read `site/data/`, the workflow would
    correctly name nothing and this file would be asserting about an empty set -- green, and
    saying nothing, which is what `test_guards_are_load_bearing.py` exists about."""
    found = _modules_reading_the_published_tree()
    assert "tests/contracts/test_published_envelopes.py" in found, (
        f"the envelope contract is the one this ticket was filed out of; found {found}")
    assert len(found) >= 2, f"only {found} reads the published tree"


def test_the_slate_runs_the_contracts_that_read_what_it_publishes() -> None:
    """The list in `slate.yml` against the test tree, so neither can drift from the other.

    A module that starts reading the published site and is not named in the workflow would be
    a contract that only ever runs on somebody else's push -- which is the defect #228 was
    filed about, reintroduced one layer up in a YAML file nothing reads.
    """
    run = _named(CHECKS).get("run", "")
    named = {tok for tok in run.split() if tok.startswith("tests/")}
    required = _modules_reading_the_published_tree()
    assert named == required, (
        f"{SLATE.name}'s {CHECKS!r} step runs {sorted(named)}; the contract modules that read "
        f"the committed site/data/ are {sorted(required)}. Publishing without running one of "
        f"them is how site/data/cfbd.json reached the tree unread on 2026-09-09.")


def test_the_check_runs_before_the_slate_commits() -> None:
    """Ordering is the whole of the fix. Run after the commit and the artifact is published
    and deployed before anything reads it, which is where this started."""
    order = [s.get("name") for s in _steps()]
    assert order.index(CHECKS) < order.index(COMMITS), (
        f"{CHECKS!r} runs after {COMMITS!r}; a malformed artifact would be committed, and "
        f"`pages.yml` deploys on the slate's completion")


def test_the_slate_runs_the_contract_subset_and_not_the_whole_suite() -> None:
    """The cost constraint the ticket states, held as a check rather than as a comment.

    The slate runs on a schedule and the whole suite is minutes; the modules that read
    published artifacts are sub-second. A future edit that widened this to `tests/` would be
    invisible until someone read the billing.
    """
    run = _named(CHECKS).get("run", "")
    widened = [t for t in WHOLE_SUITE if t in run]
    assert not widened, (
        f"the slate's contract step names {widened}, which is the whole suite rather than the "
        f"modules that read what it published. `ci.yml` is where the whole suite belongs.")
    assert "pytest" in run, f"{CHECKS!r} runs no tests at all: {run!r}"
