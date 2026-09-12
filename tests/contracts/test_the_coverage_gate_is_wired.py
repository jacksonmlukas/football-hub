"""The coverage harness is a gate because the slate runs it (#273).

`hub.models.coverage --gate` exits non-zero against a pre-registered band, and until this
was written nothing invoked it: the publisher read a file that was gitignored and so never
reached the runner, and the survivor verdict was computed and read by nothing. This holds
`slate.yml` to running the measurement, committing what it wrote, and then letting the gate
fail the run -- in that order, so a red gate is a red run and not an unpublished slate.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLATE = ROOT / ".github" / "workflows" / "slate.yml"


def test_the_slate_measures_writes_commits_then_gates():
    text = SLATE.read_text()
    measure = text.index("hub.models.coverage --measure --survivor --write")
    commit = text.index("interval_coverage.json", measure)
    gate = text.index("hub.models.coverage --gate", commit)
    assert measure < commit < gate, "measure, commit the artifact, then gate -- in that order"


def test_the_gate_step_is_not_soft_failed():
    text = SLATE.read_text()
    i = text.index("hub.models.coverage --gate")
    step = text[text.rfind("- name:", 0, i): i]
    assert "continue-on-error" not in step and "|| true" not in text[i: i + 200]
