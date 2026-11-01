"""The model-change issue form carries the deviations-log fields, and the weekly-slate skill checks pins (#454)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "model-change.yml"
SKILL = ROOT / "plugins" / "football-hub" / "skills" / "weekly-slate" / "SKILL.md"

FIELDS = {
    "contestant",
    "old_model_id",
    "new_model_id",
    "date_noticed",
    "reason",
    "leaves_holm_family",
    "resets_contestant",
}


def _fields(text: str) -> dict[str, dict]:
    return {b["id"]: b for b in yaml.safe_load(text)["body"] if "id" in b}


def test_template_exists_with_the_deviations_log_fields():
    assert TEMPLATE.exists()
    assert FIELDS <= set(_fields(TEMPLATE.read_text()))


def test_contestant_and_reason_options_cover_the_study():
    body = _fields(TEMPLATE.read_text())
    assert {"llm1", "llm2", "llm3", "llm4", "retrieval arm"} <= set(body["contestant"]["attributes"]["options"])
    assert {"provider deprecation", "alias moved", "outage"} <= set(body["reason"]["attributes"]["options"])


def test_weekly_slate_names_the_pinned_model_check():
    text = SKILL.read_text()
    assert "Pinned-model check" in text
    assert "model-change.yml" in text
    assert "forecast-dry-run.yml" in text


def test_a_removed_field_is_detected():
    broken = TEMPLATE.read_text().replace("id: reason", "id: why")
    assert not FIELDS <= set(_fields(broken))
