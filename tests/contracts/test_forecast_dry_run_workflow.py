"""`.github/workflows/forecast-dry-run.yml` cannot push, and is not the study's switch (#534).

The dry run spends real money on real keys, so what stops it from writing anywhere is held here
rather than trusted: read-only token permissions, a checkout that keeps no credentials, no git
write step, manual dispatch only, no dependence on `LLM_FORECASTS_ENABLED`, every secret the
live run maps, and the output saved as an artifact.
"""
from __future__ import annotations

import re
from pathlib import Path

from hub.forecasters import providers

ROOT = Path(__file__).resolve().parents[2]
WF = (ROOT / ".github" / "workflows" / "forecast-dry-run.yml").read_text()
LIVE = (ROOT / ".github" / "workflows" / "forecast.yml").read_text()


def _code(text: str) -> str:
    """The workflow without its comment lines, so prose cannot satisfy or trip an assertion."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


CODE = _code(WF)


def test_the_token_is_read_only_at_every_level():
    assert re.search(r"^permissions:\n\s+contents: read\n", CODE, re.M)
    perms = re.findall(r"^\s*([a-z-]+):\s*(read|write|none)\s*$", CODE, re.M)
    assert ("contents", "read") in perms
    assert not [p for p in perms if p[1] == "write"], perms
    assert CODE.count("permissions:") == 1                     # no wider job-level block


def test_the_checkout_keeps_no_credentials_and_nothing_runs_git_or_pushes():
    assert re.search(r"actions/checkout@v4\n\s+with:\n\s+persist-credentials: false", CODE)
    # No git command in any spelling (`git -C dir push` included), no `gh`, no push anywhere.
    assert not re.search(r"\bgit\b|\bgh\b|\bpush\b", CODE.replace("actions/", ""))
    for forbidden in ("forecasts-branch", "GITHUB_TOKEN", "secrets.GITHUB"):
        assert forbidden not in CODE, forbidden
    assert CODE.count("uses: actions/checkout") == 1


def test_manual_dispatch_only_and_not_gated_by_the_study_switch():
    on = CODE[CODE.index("on:"):CODE.index("permissions:")]
    assert "workflow_dispatch:" in on
    for trigger in ("schedule:", "push:", "pull_request", "workflow_run", "cron"):
        assert trigger not in on, trigger
    assert "LLM_FORECASTS_ENABLED" not in CODE and "\n    if:" not in CODE
    assert re.search(r"^\s+day:\n(?:.*\n)*?\s+required: true", on, re.M)
    assert "ceiling_usd:" in on


def test_it_maps_exactly_the_secrets_the_live_run_does_and_takes_inputs_through_env():
    mapped = set(re.findall(r"secrets\.([A-Z_]+)", CODE))
    assert mapped == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                      "TAVILY_API_KEY", "ODDS_API_KEY"}
    assert mapped <= set(re.findall(r"secrets\.([A-Z_]+)", LIVE))
    assert set(providers.KEY_ENV.values()) | {providers.TAVILY_KEY_ENV} <= mapped
    # Inputs reach the shell as environment variables, never spliced into the command line.
    found = re.search(r"run: (uv run [^\n]+)", CODE)
    assert found is not None
    run = found.group(1)
    assert "${{" not in run and '"$DAY"' in run and '"$CEILING_USD"' in run


def test_it_runs_the_dry_run_command_into_one_directory_and_uploads_that_directory():
    assert "hub.forecasters.cli dry-run --out dry-run-out" in CODE
    assert "hub.forecasters.cli run" not in CODE and "mark-done" not in CODE
    up = CODE[CODE.index("actions/upload-artifact"):]
    assert "path: dry-run-out/" in up and "if: always()" in CODE
