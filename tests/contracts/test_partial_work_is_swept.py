"""`Advances #n` has to be swept, because no trailer rule can reach the commit that matters.

`Closes #n` on the default branch is acted on by GitHub. `Advances #n` is inert on purpose:
it is the honest trailer for a coherent half. What it cannot do is close the ticket later,
and the reason is not laziness -- **the commit that finishes the residue is usually about
something else and does not know it inherited an obligation.**

That is not a hypothesis. Measured 2026-09-05: four tickets had ever carried `Advances`, none
ever received a `Closes`, and two of them (#34, #53) were finished and still open. #34's last
criterion landed under a commit message calling it "the catch on the way through"; #53's
landed inside two commits about review findings. Asking those commits to have known better is
asking for the run of luck `test_guards_are_load_bearing.py` was written to replace. #34 was
the head of the dependency graph, blocking 23 tickets while being complete.

So the mechanism is a queue, and this file guards the two halves that can rot:

* **The sweep's extraction**, against a synthetic history -- a trailer is found, a resolved
  one leaves, and a mid-sentence mention is not a trailer. No network.
* **The documentation**, on the same seam argument as
  `test_close_reason_is_documented.py`: the agents that write these trailers learn them from
  `docs/agents/`, and a rule that stops being written there stops being followed.

The sweep is deliberately not a refusal. Everything it prints is either still partial, which
is fine, or finished and open, which needs a person's judgement about acceptance criteria --
and a check that went red on honest partial work is one that would be switched off in a week.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "scripts" / "partial_work.sh"
AGENT_DOCS = ROOT / "docs" / "agents"


def _repo(tmp_path: Path, *messages: str) -> Path:
    """A throwaway history whose commit messages are exactly `messages`."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    for i, msg in enumerate(messages):
        (tmp_path / f"f{i}").write_text(str(i))
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.email=t@example.com",
             "-c", "user.name=Test", "commit", "-q", "-m", msg], check=True)
    return tmp_path


def _numbers(repo: Path) -> list[str]:
    """The offline half of the sweep: candidate issue numbers, no `gh` call."""
    out = subprocess.run([str(SWEEP), "--numbers"], cwd=repo, check=True,
                         capture_output=True, text=True).stdout
    return [ln for ln in out.split() if ln]


def test_the_harness_can_see_the_sweep_it_guards():
    """A moved or renamed script would make every behavioural test below vacuous."""
    assert SWEEP.exists(), f"{SWEEP} is missing; this file guards nothing"
    assert SWEEP.stat().st_mode & 0o111, f"{SWEEP} is not executable, so nobody can run it"


def test_an_advanced_ticket_is_on_the_queue(tmp_path):
    assert _numbers(_repo(tmp_path, "Did half of it\n\nAdvances #7")) == ["7"]


def test_a_ticket_a_later_commit_closed_has_left_the_queue(tmp_path):
    """The queue stays short by construction, which is the only reason it gets read."""
    repo = _repo(tmp_path, "Did half of it\n\nAdvances #7", "Did the rest\n\nCloses #7")
    assert _numbers(repo) == []


@pytest.mark.parametrize("verb", ["Closes", "Fixes", "Resolves"])
def test_every_keyword_github_acts_on_takes_a_ticket_off_the_queue(tmp_path, verb):
    """GitHub closes on all three. A sweep that knew only `Closes` would report a ticket
    that had already closed itself, which is how a queue earns being ignored."""
    repo = _repo(tmp_path, "Half\n\nAdvances #7", f"Rest\n\n{verb} #7")
    assert _numbers(repo) == []


def test_a_mention_inside_a_sentence_is_not_a_trailer(tmp_path):
    """`Advances` is a trailer, and a trailer starts its line. Prose mentioning the ticket is
    a sentence about the work, not a claim about its state.

    The fixture is capitalised on purpose. Written lower-case it passed with the anchor
    removed -- the case-sensitive match was carrying the assertion and the anchor was
    untested, which the excision found and the green suite did not."""
    repo = _repo(tmp_path, "Something else\n\nNothing here Advances #7, despite appearances.")
    assert _numbers(repo) == []


def test_a_history_with_no_partial_work_reports_nothing(tmp_path):
    assert _numbers(_repo(tmp_path, "All of it\n\nCloses #7")) == []


def test_the_agent_docs_still_carry_the_rule():
    """The seam `test_close_reason_is_documented.py` names: agents learn the convention from
    `docs/agents/`, so a rule that stops being written there stops being followed. Both
    trailers and the sweep have to be findable, or the next agent invents `Advances` again
    and nothing sweeps it."""
    docs = sorted(AGENT_DOCS.glob("*.md"))
    assert docs, f"no agent docs under {AGENT_DOCS}; this assertion guards nothing"
    text = "\n".join(d.read_text() for d in docs)
    for needed, why in [
        ("Advances #", "the trailer for partial work is not documented"),
        ("Closes #", "the trailer GitHub acts on is not documented"),
        ("scripts/partial_work.sh", "nothing points at the sweep, so nobody runs it"),
    ]:
        assert needed in text, (
            f"{why}: {needed!r} appears nowhere in docs/agents/. "
            "An agent reading these docs would not learn it.")
