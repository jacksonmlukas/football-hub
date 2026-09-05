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


# --- the credential scan, proven to fail (issue #28) ------------------------
#
# Every test above plants a *data file*. Nothing planted a credential, so the only thing
# asserted about the credential scan was that it exits 0 on a clean tree -- which a scan
# matching nothing at all would also satisfy. It was matching nothing at all.
#
# `PATTERNS` is written with BRE interval syntax (`.\{40,\}`) and passed to `grep -E`, where
# `\{` is a literal brace. So every pattern in the gate was inert: measured 2026-09-04 by
# planting three synthetic credentials in a throwaway repo, against which the script printed
# "ok: no credential patterns in any commit" and exited PASS.
#
# Each value below is assembled from parts rather than written out, so this file does not
# itself become a hit when the gate scans this repo's history. None is a real credential.

SWID_BODY = "1A2B3C4D-5E6F-7081-9203-A4B5C6D7E8F9"
S2_BODY = "AEB" + "a1B2c3D4e5" * 6
KEY_BODY = "0f1e2d3c4b5a" * 3


def _plant(repo: Path, name: str, body: str) -> None:
    (repo / "leak.txt").write_text(f"{name}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", f"planted {body}")


@pytest.mark.parametrize("shape,line", [
    pytest.param("espn cookie", "ESPN_S2=" + S2_BODY, id="espn-cookie"),
    pytest.param("espn cookie, quoted", 'espn_s2: "' + S2_BODY + '"', id="espn-quoted"),
    pytest.param("braced swid", "SWID={" + SWID_BODY + "}", id="swid-braced"),
    pytest.param("brace-less swid", "SWID=" + SWID_BODY, id="swid-bare"),
    pytest.param("cfbd key", "CFBD_API_KEY=" + KEY_BODY, id="cfbd-key"),
    pytest.param("odds key", "ODDS_API_KEY=" + KEY_BODY, id="odds-key"),
])
def test_a_planted_credential_blocks_the_flip(repo, shape, line):
    """The assertion that was missing. Exit non-zero, or the gate is decoration."""
    _plant(repo, line, shape)
    got = preflight(repo)
    assert got.returncode == 1, f"{shape} passed the gate: {got.stdout}"
    assert "credential" in (got.stdout + got.stderr).lower()


def test_the_brace_less_swid_is_caught_too(repo):
    """ESPN's cookie carries braces and the pattern required them, so the form actually
    pasted into a secret -- which is how it was got wrong on 2026-09-04 -- was invisible."""
    _plant(repo, "ESPN_SWID=" + SWID_BODY, "brace-less swid")
    assert preflight(repo).returncode == 1


def test_a_credential_committed_and_then_removed_still_blocks(repo):
    """Flipping public exposes every commit. The same lesson the parquet tests carry, for
    the thing the scan is actually named after."""
    _plant(repo, "ESPN_S2=" + S2_BODY, "espn cookie")
    (repo / "leak.txt").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "remove it")
    got = preflight(repo)
    assert got.returncode == 1, "a clean tree is not a clean history"


def test_the_scan_proves_it_can_match_before_reporting_clean(repo):
    """The gate's own premise, checked by the gate rather than only here. A scan that
    reports 'no credential patterns in any commit' is making two claims -- that it looked,
    and that it found nothing -- and until now only the second was ever true."""
    got = preflight(repo)
    assert got.returncode == 0
    out = got.stdout.lower()
    assert "patterns match" in out or "self-check" in out, (
        f"the scan does not demonstrate that it can match anything:\n{got.stdout}")


def test_ordinary_prose_is_not_flagged(repo):
    """A gate that fires on the word ESPN in a docstring is a gate you learn to ignore --
    which is how the 2026-08-23 miss became possible."""
    (repo / "docs.md").write_text(
        "Set ESPN_S2 and SWID in .env. The CFBD_API_KEY is optional.\n"
        "See ODDS_API_KEY= in SETUP.md for where to get one.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "docs")
    assert preflight(repo).returncode == 0


# --- the schedules, read rather than described ------------------------------
#
# c984746 uncommented both `schedule:` blocks and edited neither the SCHEDULE DISABLED
# comment heading each one nor the two steps of this script telling you to go and uncomment
# them. The script reads the blocks now. These pin the states it has to tell apart --
# including a repo with no `.github/` at all, which is what every test above builds.

LIVE = 'name: w\non:\n  schedule:\n    - cron: "*/10 17-23 * * 0"\n  push:\n'
DEAD = 'name: w\non:\n  # schedule:\n  #   - cron: "*/10 17-23 * * 0"\n  push:\n'

# Dormant, but containing the string `schedule:` and the string `- cron:` -- which is all a
# pair of independent greps can ask, and the first version of this check asked exactly that.
# Neither occurrence is a trigger: `schedule:` is a `with:` input three levels down, and the
# cron line is heredoc payload. The only real schedule block is commented out.
HOSTILE = """name: watchdog
on:
  # schedule:
  #   - cron: "*/10 17-23 * * 0"
  workflow_dispatch:
jobs:
  heartbeat:
    steps:
      - run: |
          cat >> notes <<EOT
          - cron: "*/10 17-23 * * 0"
          EOT
        with:
          schedule: nightly
"""


def _workflow(repo: Path, name: str, body: str) -> None:
    d = repo / ".github" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body)
    git(repo, "add", "-A")
    git(repo, "commit", "-m", f"add {name}")


def _schedule_lines(out: str) -> list[str]:
    """Only the lines the schedule check emits. The author-address check warns on these
    throwaway repos too, so `WARNING appears somewhere` proves nothing."""
    return [ln for ln in out.splitlines() if "workflows/" in ln]


def test_a_live_schedule_is_reported_live(repo):
    _workflow(repo, "watchdog.yml", LIVE)
    _workflow(repo, "ci.yml", LIVE)
    got = preflight(repo)
    assert got.returncode == 0, got.stdout + got.stderr
    lines = _schedule_lines(got.stdout)
    assert any("watchdog.yml" in ln and "ok" in ln for ln in lines), got.stdout
    assert any("ci.yml" in ln and "ok" in ln for ln in lines), got.stdout


def test_a_commented_out_schedule_is_reported(repo):
    _workflow(repo, "watchdog.yml", DEAD)
    got = preflight(repo)
    assert any("watchdog.yml" in ln and "WARNING" in ln
               for ln in _schedule_lines(got.stdout)), got.stdout


def test_a_cron_outside_the_schedule_block_is_not_a_live_schedule(repo):
    """The check's own premise. `schedule:` and `- cron:` both occur in this file and it has
    no schedule trigger at all -- so a check that greps for the two independently reports a
    dormant workflow as live, which is the failure the check exists to prevent."""
    _workflow(repo, "watchdog.yml", HOSTILE)
    got = preflight(repo)
    assert any("watchdog.yml" in ln and "WARNING" in ln
               for ln in _schedule_lines(got.stdout)), (
        "a `with:` input and a heredoc line were accepted as a schedule trigger:\n"
        + got.stdout)


def test_a_dormant_cron_does_not_block_the_gate(repo):
    """A warning, for the same reason the author-address check is one: a cron that is off is
    something to know, not a credential in the history. Grading it FAIL would put a workflow
    someone switched off on purpose behind the same red line as a leaked cookie."""
    _workflow(repo, "watchdog.yml", DEAD)
    got = preflight(repo)
    assert got.returncode == 0, got.stdout + got.stderr
    assert "PASS" in got.stdout


def test_a_repo_with_no_workflows_is_not_a_finding(repo):
    """Every other test in this file builds a repo with no `.github/` at all. A check that
    read absence as a defect would fail all of them."""
    got = preflight(repo)
    assert got.returncode == 0, got.stdout + got.stderr
    assert not [ln for ln in _schedule_lines(got.stdout) if "WARNING" in ln], got.stdout


def test_the_gate_does_not_ask_for_work_that_is_already_done(repo):
    """The defect this check was added to remove, which the checklist then reproduced twice
    over. `Uncomment the schedule: block in ci.yml` outlived the commit that uncommented it;
    `Flip the repo public` and `Enable Pages` outlived the flip and the Pages build."""
    out = preflight(repo).stdout.lower()
    for done in ("uncomment", "is stale", "flip the repo public", "enable pages"):
        assert done not in out, f"{done!r} is an instruction to redo finished work:\n{out}"
