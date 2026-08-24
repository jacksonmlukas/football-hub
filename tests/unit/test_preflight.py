"""The pre-public gate.

On 2026-08-23 this script reported PASS while 45 parquet files -- 3.5MB of nflverse
play-by-play -- sat in history one commit back. The index check saw a clean tree and the
credential scan does not look at data, so nothing caught it. Flipping public exposes every
commit, not the tip, and a file removed in commit N is still reachable in commit N-1.

The case that matters is therefore the one that slipped: committed, then untracked, and
the working tree clean. Each test builds a throwaway repo rather than asserting against
this one, so a green run here means the gate works rather than that today happens to be
fine.
"""
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "preflight_public.sh"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-b", "main")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    (r / "README.md").write_text("hello\n")
    git(r, "add", "-A")
    git(r, "commit", "-m", "init")
    return r


def preflight(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT)], cwd=repo, capture_output=True, text=True)


def test_a_clean_repo_passes(repo):
    r = preflight(repo)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_a_parquet_still_tracked_fails(repo):
    d = repo / "data" / "processed"
    d.mkdir(parents=True)
    (d / "board.parquet").write_bytes(b"PAR1")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "add data")
    assert preflight(repo).returncode == 1


def test_a_parquet_committed_then_untracked_still_fails(repo):
    """The exact miss. Working tree clean, index clean, file alive in history."""
    d = repo / "data" / "processed"
    d.mkdir(parents=True)
    (d / "board.parquet").write_bytes(b"PAR1")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "add data")
    git(repo, "rm", "-r", "--cached", "data")
    (repo / ".gitignore").write_text("data/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "untrack data")

    r = preflight(repo)
    assert r.returncode == 1, "a clean tree is not a clean history"
    assert "history" in (r.stdout + r.stderr).lower()


def test_the_failure_says_untracking_is_not_enough(repo):
    d = repo / "data" / "raw"
    d.mkdir(parents=True)
    (d / "pbp.parquet").write_bytes(b"PAR1")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "add raw")
    err = preflight(repo).stderr
    assert "filter-repo" in err, "name the remedy, not just the problem"


def test_a_json_under_site_data_is_not_flagged(repo):
    """site/data/*.json is the published artifact. Flagging it would train you to ignore
    this gate, which is how the 2026-08-23 miss became possible."""
    d = repo / "site" / "data"
    d.mkdir(parents=True)
    (d / "draft_board.json").write_text("[]")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "publish")
    assert preflight(repo).returncode == 0


def test_history_rewritten_clean_passes_again(repo):
    """After a genuine rewrite the gate must go green, or it is not actionable."""
    d = repo / "data" / "processed"
    d.mkdir(parents=True)
    (d / "board.parquet").write_bytes(b"PAR1")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "add data")
    assert preflight(repo).returncode == 1
    # amend it away: one commit, so this is equivalent to a filter-repo run
    git(repo, "rm", "-r", "--cached", "data")
    (repo / ".gitignore").write_text("data/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "--amend", "-m", "init")
    assert preflight(repo).returncode == 0
