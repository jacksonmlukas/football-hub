"""The documented way to close a ticket must always name a reason.

This repo's done-state is GitHub's native close reason, not a label -- see
`docs/agents/triage-labels.md` for the argument. The mechanism cannot drift, because the
reason is written by the same call that closes the ticket. But the *documentation* of it
can, and there is one specific way it drifts back to nothing:

    gh issue close <n> --comment "..."

That is not an error. It silently records `completed`. So a doc example that drops
`--reason` does not fail loudly for the agent that copies it -- it quietly re-creates the
exact defect the convention exists to fix, marking an abandoned or `wontfix` ticket as
finished work. On 2026-09-05 all thirty closed issues in this repo read `COMPLETED`, every
one of them from that default, which is why the value had never once discriminated.

The agents that close tickets learn how to do it from `docs/agents/`. That is the seam, so
that is what this guards.

**What counts as an invocation.** A bare `gh issue close` naming the subcommand in prose
("`gh issue close` does not expose `duplicate`") is not an invocation and is not checked. A
`gh issue close` followed by an issue argument -- `<n>`, `<number>`, or a literal number --
is a thing an agent will copy, and must carry `--reason`.

**Non-vacuity.** A scan that matched nothing would pass every assertion below while proving
nothing, which is the failure mode `test_guards_are_load_bearing.py` was written about. So
the scan is required to find invocations at all, and to find *both* reasons: a convention
that only ever documents `completed` is the default wearing a costume.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT_DOCS = ROOT / "docs" / "agents"

# `gh issue close` plus an issue argument: `<n>`, `<number>`, `<issue>`, or a literal number.
INVOCATION = re.compile(r"gh issue close\s+(?:<[a-z-]+>|\d+|https?://\S+)")

# `--reason completed` / `-r "not planned"`, quoted or not.
REASON = re.compile(r"(?:--reason|-r)[=\s]+(\"[^\"]*\"|'[^']*'|\S+)")

# What `gh issue close --reason` itself accepts. `duplicate` is a real close reason but the
# CLI does not expose it -- it needs the REST API -- so a doc example passing it to
# `gh issue close` would be a copy-pasteable error.
CLI_REASONS = {"completed", "not planned"}

FENCE = re.compile(r"^\s*(?:```|~~~)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")


def _commands(text: str) -> list[str]:
    """Every shell command in a markdown file, as a flat list of strings.

    Fenced-block lines are commands on their own; elsewhere only inline code spans count,
    so that prose following a closing backtick can never satisfy an assertion about the
    command that preceded it.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(line)
        else:
            out.extend(INLINE_CODE.findall(line))
    return out


def _docs() -> list[Path]:
    return sorted(AGENT_DOCS.glob("*.md"))


def test_the_harness_can_see_the_docs_it_guards():
    """A moved or renamed docs/agents/ would make every test below vacuously green."""
    docs = _docs()
    assert docs, f"no agent docs found under {AGENT_DOCS}; this harness guards nothing"


def test_the_scan_finds_close_invocations_to_check():
    """The regex matching nothing looks identical to a clean sweep."""
    found = [
        (doc.name, cmd)
        for doc in _docs()
        for cmd in _commands(doc.read_text())
        if INVOCATION.search(cmd)
    ]
    assert len(found) >= 3, (
        "found fewer than three `gh issue close` invocations in docs/agents/; either the "
        f"docs stopped documenting how to close a ticket, or the scan broke. Found: {found}"
    )


@pytest.mark.parametrize("doc", _docs(), ids=lambda p: p.name)
def test_every_documented_close_names_a_reason(doc: Path):
    """A reason-less close is not an error -- it silently records `completed`."""
    offenders = [
        cmd.strip()
        for cmd in _commands(doc.read_text())
        if INVOCATION.search(cmd) and not REASON.search(cmd)
    ]
    assert not offenders, (
        f"{doc.name} documents a `gh issue close` with no --reason. A bare close silently "
        "records `completed`, so this example marks abandoned and wontfix tickets as "
        f"finished work. Add --reason completed or --reason \"not planned\":\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("doc", _docs(), ids=lambda p: p.name)
def test_every_documented_reason_is_one_the_cli_accepts(doc: Path):
    """`gh issue close --reason duplicate` is rejected; duplicate needs the REST API."""
    bad = []
    for cmd in _commands(doc.read_text()):
        if not INVOCATION.search(cmd):
            continue
        m = REASON.search(cmd)
        if m and m.group(1).strip("\"'") not in CLI_REASONS:
            bad.append(cmd.strip())
    assert not bad, (
        f"{doc.name} passes a reason `gh issue close` does not accept "
        f"(it takes {sorted(CLI_REASONS)}; `duplicate` needs `gh api`):\n  " + "\n  ".join(bad)
    )


def test_both_outcomes_are_documented():
    """`completed` only carries information once `not planned` is in use.

    Every closed issue in this repo read COMPLETED because that is the default, not because
    anyone asserted it. Documenting only the `completed` half would leave the state exactly
    as uninformative as it was before.
    """
    reasons = set()
    for doc in _docs():
        for cmd in _commands(doc.read_text()):
            if not INVOCATION.search(cmd):
                continue
            m = REASON.search(cmd)
            if m:
                reasons.add(m.group(1).strip("\"'"))
    assert reasons == CLI_REASONS, (
        "docs/agents/ must show both close reasons -- a vocabulary that only ever says "
        f"`completed` is the default wearing a costume. Documented: {sorted(reasons)}"
    )
