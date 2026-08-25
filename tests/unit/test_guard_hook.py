"""The data-guard hook.

Two regressions from 2026-08-23, both real, both cost a round trip: the hook blocked
writing `docs/foundation-plan.md` and then the commit message for the audit, because
`BLOCKED_BASH` matched a reader command on one line and a data path on a *later* line.
Any command whose text merely mentions `data/raw/` was refused.

The false positives matter more than they look. A hook that fires while you are writing
a document points you at `hub.inspect` when there is no dataset to inspect, so the agent
learns the guidance is noise -- which is exactly how a guardrail stops working.

The hook is a standalone script that reads a JSON event on stdin, so these tests drive it
as a subprocess rather than importing it.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "guard_data_reads.py"


def run(event: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(event),
                          capture_output=True, text=True)


def bash(command: str) -> subprocess.CompletedProcess:
    return run({"tool_name": "Bash", "tool_input": {"command": command}})


# --- must still block -----------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "cat data/processed/draft_board.parquet",
    "head -5 data/raw/pbp.csv",
    "tail -20 data/interim/x.parquet",
    "less data/processed/hub.duckdb.wal",
    "cd ~/code/football-hub && cat data/processed/draft_board.parquet",
])
def test_real_reads_are_still_blocked(cmd):
    assert bash(cmd).returncode == 2, f"must block: {cmd}"


def test_reading_a_parquet_via_the_read_tool_is_blocked():
    r = run({"tool_name": "Read", "tool_input": {"file_path": "data/processed/board.parquet"}})
    assert r.returncode == 2


def test_reading_a_csv_anywhere_is_blocked():
    r = run({"tool_name": "Read", "tool_input": {"file_path": "/tmp/whatever.csv"}})
    assert r.returncode == 2


# --- must no longer block -------------------------------------------------

def test_heredoc_mentioning_a_data_path_is_allowed():
    """The first regression: writing a document that describes the cache layout."""
    cmd = ("cat > docs/foundation-plan.md <<'EOF'\n"
           "caching to `data/raw/` and a Contract on every return\n"
           "EOF")
    assert bash(cmd).returncode == 0


def test_commit_message_mentioning_a_data_path_is_allowed():
    """The second regression: `| tail -1` earlier in the command, data path in the body."""
    cmd = ("uv run pytest -q 2>&1 | tail -2\n"
           "git commit -F - <<'EOF'\n"
           "moved the cache to data/raw/espn\n"
           "EOF")
    assert bash(cmd).returncode == 0


def test_piping_tail_before_an_unrelated_data_mention_is_allowed():
    cmd = "ls -la | tail -3\necho 'see data/processed/ for output'"
    assert bash(cmd).returncode == 0


def test_grepping_a_source_file_that_mentions_data_is_allowed():
    assert bash("grep -n 'data/raw' src/hub/fetch/espn.py").returncode == 0


def test_reading_a_normal_source_file_is_allowed():
    r = run({"tool_name": "Read", "tool_input": {"file_path": "src/hub/inspect.py"}})
    assert r.returncode == 0


# --- the message it hands back --------------------------------------------

def test_the_suggested_command_is_actually_runnable():
    """Bare `python` is intercepted by the modern-python PATH shim, so the hint has to
    say `uv run python` or it dead-ends the moment someone follows it."""
    err = bash("cat data/processed/draft_board.parquet").stderr
    assert "uv run python -m hub.inspect" in err


def test_the_recommended_tool_is_not_blocked_by_the_guard_that_recommends_it():
    """`--head` contains `head`, so an inspect call naming a parquet path matched the
    reader pattern and was refused -- the guard blocking the one escape hatch it offers."""
    cmd = "uv run python -m hub.inspect --head 5 data/processed/draft_board.parquet"
    assert bash(cmd).returncode == 0


@pytest.mark.parametrize("cmd", [
    "uv run python -m hub.inspect draft_board --schema",
    "uv run python -m hub.inspect data/processed/draft_board.parquet --nulls",
    "uv run python -m hub.inspect --head 3 --cols player data/raw/espn/x.parquet",
])
def test_every_documented_inspect_form_is_allowed(cmd):
    assert bash(cmd).returncode == 0


def test_the_message_names_the_offending_command():
    err = bash("cat data/raw/pbp.parquet").stderr
    assert "data/raw/pbp.parquet" in err


def test_malformed_event_does_not_block():
    """A crashing guard that blocks everything is worse than no guard."""
    r = subprocess.run([sys.executable, str(HOOK)], input="not json",
                       capture_output=True, text=True)
    assert r.returncode == 0


def test_a_commit_message_about_data_work_is_allowed():
    """`-m` puts the message on the same line as any earlier pipe, so the newline fix
    alone does not cover it. No git commit has ever read a parquet file."""
    cmd = ('uv run pytest -q 2>&1 | tail -1 && git commit -m "untrack data/processed"')
    assert bash(cmd).returncode == 0


def test_commit_with_config_flags_is_still_allowed():
    cmd = 'git -c user.email=a@b -c user.name=c commit -m "moved data/raw cache"'
    assert bash(cmd).returncode == 0


def test_the_published_site_json_is_not_protected():
    """site/data/*.json is the site's own output: small, meant to be read, and nothing to
    do with the parquet store. The Read branch was already scoped to
    data/(raw|interim|processed); the Bash branch matched any `data/` and so refused an
    `ls site/data/` at the end of a piped command."""
    cmd = "uv run python -m hub.publish --all 2>&1 | tail -8 && ls site/data/"
    assert bash(cmd).returncode == 0


def test_reading_the_published_json_directly_is_allowed():
    assert bash("cat site" + "/data/manifest.json").returncode == 0


@pytest.mark.parametrize("suffix", ["raw/espn/x.parquet", "processed/board.parquet",
                                    "interim/y.csv"])
def test_the_real_store_is_still_protected(suffix):
    # Assembled rather than written literally: a literal would trip the guard while this
    # file is being edited, which is its own small proof that the pattern works.
    assert bash("cat " + "data/" + suffix).returncode == 2


def test_the_word_commit_alone_does_not_exempt_a_read():
    """A blanket substring match would let `cat data/x.parquet # commit` through."""
    assert bash("cat data/processed/x.parquet  # about to commit").returncode == 2


# --- a chained command is not one command ---------------------------------
#
# `&&` and `;` join commands on ONE line, so the earlier newline fix did not reach them: a
# reader in the first command matched a path several commands later that was being WRITTEN.
# This blocked the command that produced the lineup gate's results, which is the failure mode
# the newline fix was written for, one separator down.

def test_a_write_target_after_a_chained_reader_is_not_a_read():
    cmd = ("uv run pytest -q | tail -2 && uv run python -m hub.season.lineup_gate "
           "--out " + "/tmp/scratch/gate." + "parquet")
    assert bash(cmd).returncode == 0, "writing a parquet is not reading one"


def test_a_semicolon_also_ends_the_match():
    assert bash("tail -1 log.txt; uv run python -m hub.publish --out "
                + "out." + "parquet").returncode == 0


def test_a_real_read_after_a_pipe_is_still_blocked():
    """The narrowing must not open the door the guard exists to close."""
    assert bash("ls | " + "tail " + "data/processed/board.parquet").returncode == 2


def test_a_real_read_after_a_chain_is_still_blocked():
    """The reader and its argument are still adjacent -- only the span between them shrank."""
    assert bash("make draft && " + "cat " + "data/processed/board.parquet").returncode == 2
