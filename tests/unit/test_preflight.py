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
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The excision harness at the bottom of this file re-runs this module against a *mutated copy*
# of the gate, so the script under test has to be an input rather than a constant. Reading it
# from the environment is the whole reason the mutant runs are trustworthy: the Python harness
# in tests/contracts/test_guards_are_load_bearing.py first passed its mutant tree through
# PYTHONPATH, which `pythonpath = ["src"]` in pyproject silently shadowed, so every mutant ran
# against the real source and all four guards reported green. `test_the_child_run_uses_the_
# script_it_is_handed` is the equivalent proof for this file.
SCRIPT = Path(os.environ.get("PREFLIGHT_SCRIPT") or ROOT / "scripts" / "preflight_public.sh")

# True inside a child run started by the excision harness. The excision tests skip themselves
# there: they would otherwise spawn their own children, forever.
MUTANT_RUN = bool(os.environ.get("PREFLIGHT_SCRIPT"))


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


# --- every alternation the pattern set declares, not one of five (issue #54) ---
#
# The canary added on 2026-09-04 planted a single ESPN cookie and, when it failed, printed
# "the patterns match nothing, so the scan below cannot fail" -- a claim about all five
# alternations from evidence about the first. The SWID branch, the quoted lowercase branch
# and the two key branches were as unproven inside the script as they had been while every
# pattern was inert BRE. These tests kill each pattern in turn and require the run to block
# and to name the one that died.
#
# The pattern declarations are discovered rather than listed, so a sixth shape added to the
# script is exercised here without anyone remembering to add it. `test_the_script_declares
# _the_patterns_this_file_expects` is the premise: an empty discovery would parametrize this
# to nothing and pass, which is the exact vacuum this whole issue is about.

PATTERN_DECL = re.compile(r"^(?P<var>P_[A-Z0-9_]+)=(?P<q>['\"])(?P<body>.*)(?P=q)$", re.M)


def _pattern_vars() -> list[str]:
    return [m.group("var") for m in PATTERN_DECL.finditer(SCRIPT.read_text())]


def test_the_script_declares_the_patterns_this_file_expects():
    """The premise. If the declarations stop being discoverable the parametrize below
    silently becomes empty, and an empty parametrize is a green suite proving nothing."""
    found = _pattern_vars()
    assert len(found) >= 5, f"only found {found}; the gate declares five credential shapes"


@pytest.mark.parametrize("var", _pattern_vars(), ids=lambda v: v.lower())
def test_a_dead_pattern_is_named_and_blocks_the_flip(repo, var, tmp_path):
    """Kill one pattern; the self-check must fail the run and say which one.

    Nothing is planted in the repo -- a clean history. The point is that the gate refuses to
    report a clean history through a scanner that has gone partly blind."""
    dead = PATTERN_DECL.sub(
        lambda m: f"{m['var']}={m['q']}zzz-this-matches-nothing-zzz{m['q']}"
        if m["var"] == var else m.group(0),
        SCRIPT.read_text())
    broken = tmp_path / "preflight_public.sh"
    broken.write_text(dead)
    got = subprocess.run(["bash", str(broken)], cwd=repo, capture_output=True, text=True)
    assert got.returncode == 1, (
        f"{var} matches nothing and the gate still reported a clean history:\n{got.stdout}")
    both = got.stdout + got.stderr
    assert var in both, (
        f"the failure does not say which pattern went dead:\n{both}")
    for other in _pattern_vars():
        if other != var:
            assert f"FAIL: {other} " not in both, (
                f"killing {var} was reported as {other} dying too; that is the "
                f"'the patterns match nothing' claim again, one pattern wide")


def test_a_pattern_with_no_synthetic_sample_is_a_failure(repo, tmp_path):
    """An alternation added to PATTERNS without a sample is an unproven alternation, which
    is the whole defect. The gate counts what it joined against what it exercised."""
    text = SCRIPT.read_text()
    widened = text.replace(
        'PATTERNS="$P_ESPN_S2|', 'PATTERNS="NEVER_MATCHES_ANYTHING_AT_ALL_XYZ=[0-9]{9,}|$P_ESPN_S2|')
    assert widened != text, "PATTERNS is no longer assembled the way this test widens it"
    wide = tmp_path / "preflight_public.sh"
    wide.write_text(widened)
    got = subprocess.run(["bash", str(wide)], cwd=repo, capture_output=True, text=True)
    assert got.returncode == 1, (
        "a sixth alternation went into PATTERNS with nothing proving it can match, and the "
        f"gate still called the history clean:\n{got.stdout}")


def test_the_gate_and_its_own_tests_are_not_credential_hits(repo):
    """Every synthetic value in the script and in this file is assembled from parts so that
    neither is a hit when the scan reads its own history. That property is easy to break by
    writing one plausible-looking literal, and breaking it makes the gate fail forever on
    itself -- so it is checked by committing both files and running the real scan over them
    rather than by remembering."""
    for src, rel in ((SCRIPT, "scripts/preflight_public.sh"),
                     (Path(__file__).resolve(), "tests/unit/test_preflight.py")):
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "vendor the gate and its tests")
    got = preflight(repo)
    assert got.returncode == 0, (
        "the gate or its tests contain a credential-shaped literal:\n" + got.stdout + got.stderr)


# --- the other refusals, given planted inputs of their own -------------------
#
# The excision harness below needs, for each refusal, an input that *only* that refusal
# catches. Two of the six had none: a committed `.env` and a data file staged but not yet
# committed. Without them, deleting the refusal leaves another one covering the same input
# and the excision reports a load-bearing guard that is not.


def test_a_committed_env_file_blocks_the_flip(repo):
    """`.env` is where every credential in this project lives. Its contents here are inert
    on purpose: this must fail because the file is named .env, not because of what is in it,
    which is what makes it the input only the .env refusal catches."""
    (repo / ".env").write_text("PLACEHOLDER=1\n")
    git(repo, "add", "-f", ".env")
    git(repo, "commit", "-m", "oops")
    got = preflight(repo)
    assert got.returncode == 1, f".env passed the gate:\n{got.stdout}"
    assert ".env" in got.stderr


def test_a_staged_but_uncommitted_data_file_blocks_the_flip(repo):
    """Staged and not yet committed: the index check sees it, the history check cannot, and
    it is the moment before the 2026-08-23 miss rather than the moment after."""
    d = repo / "data" / "processed"
    d.mkdir(parents=True)
    (d / "board.parquet").write_bytes(b"PAR1")
    git(repo, "add", "-f", "data/processed/board.parquet")
    got = preflight(repo)
    assert got.returncode == 1, f"a staged parquet passed the gate:\n{got.stdout}"
    assert "tracked" in got.stderr.lower()


# --- the excision property (issue #54) --------------------------------------
#
# Everything above plants an input and requires a non-zero exit. That is the right shape and
# it is not the property that matters: a refusal that has been quietly weakened -- a pattern
# that stops matching, a check whose two halves are never tied together -- still passes it,
# because some *other* refusal catches the planted input, or because the input was never
# reaching the refusal at all. This file has failed exactly that way twice.
#
# The stronger property, the one tests/contracts/test_guards_are_load_bearing.py already
# holds the Python source to: **delete the refusal and the planted input must stop being
# caught.** Same marker syntax, in shell comments:
#
#     # GUARD name [pytest -k expression]: one line on what it refuses.
#     ...
#     # /GUARD
#
# The `-k` expression is written into the marker rather than derived, because there is no
# filename convention to derive it from in shell -- one script, one test module, six
# refusals. It is a second thing to keep in step, so the control run below fails loudly when
# it selects nothing.
#
# Two traps, both already paid for on the Python side:
#
#   1. A mutant that no longer parses is a HARNESS error, never a pass. Excising a block can
#      leave a dangling `fi`, and `bash` then fails on every input -- indistinguishable from
#      a load-bearing refusal.
#   2. The harness must actually be running against the mutant. The Python version's first
#      cut passed its mutant tree through PYTHONPATH, which pyproject's `pythonpath = ["src"]`
#      shadows, so every mutant ran against the real source and all four guards reported
#      green. Here the equivalent proof is `test_the_child_run_uses_the_script_it_is_handed`.

SHELL_GUARD = re.compile(
    r"^[ \t]*# GUARD (?P<name>[a-z0-9][a-z0-9-]*)"
    r" \[(?P<k>[^\]]+)\]: (?P<why>[^\n]+)\n"
    r"(?P<body>.*?)"
    r"^[ \t]*# /GUARD[^\n]*\n",
    re.S | re.M,
)


@dataclass(frozen=True)
class Refusal:
    """One declared refusal in the gate: what it refuses, and what must notice its absence."""

    name: str
    k: str
    why: str
    span: tuple[int, int]

    def __repr__(self) -> str:                          # the parametrize id
        return self.name


def _refusals() -> list[Refusal]:
    text = SCRIPT.read_text()
    return [Refusal(m.group("name"), m.group("k"), m.group("why"), m.span())
            for m in SHELL_GUARD.finditer(text)]


def _without(refusal: Refusal) -> str:
    text = SCRIPT.read_text()
    lo, hi = refusal.span
    return text[:lo] + text[hi:]


def _child(script: Path, k: str) -> subprocess.CompletedProcess:
    """Run this module's own tests against the script at `script`."""
    env = os.environ.copy()
    env["PREFLIGHT_SCRIPT"] = str(script)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()), "-k", k,
         "-q", "-x", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)


@pytest.mark.skipif(MUTANT_RUN, reason="a child run of the excision harness")
def test_the_gate_declares_the_refusals_it_makes():
    """The premise, and the reason a new refusal cannot land unproven: every `fail=1` in the
    script must sit inside a declared guard, or this harness quietly covers five of six."""
    text = SCRIPT.read_text()
    found = _refusals()
    assert len(found) >= 5, f"only found {[r.name for r in found]}"
    covered = "".join(text[lo:hi] for lo, hi in (r.span for r in found))
    outside = text.count("fail=1") - covered.count("fail=1")
    # `fail=0` initialises and the gitleaks line is `|| fail=1` on a tool that is skipped
    # when absent, so it is not excised here; count it and no more.
    assert outside <= 1, (
        f"{outside} `fail=1` refusals sit outside any # GUARD block. An unmarked refusal is "
        f"an unproven one -- that is what this file is for.")


@pytest.mark.skipif(MUTANT_RUN, reason="a child run of the excision harness")
def test_the_child_run_uses_the_script_it_is_handed(tmp_path):
    """Trap 2. Hand the child a stub that passes everything; the planting tests must go red.

    If PREFLIGHT_SCRIPT were ignored -- the shape of failure that made the first Python
    harness report four green mutants against untouched source -- the child would run the
    real gate, catch the planted credential, and come back green."""
    stub = tmp_path / "preflight_public.sh"
    stub.write_text("#!/usr/bin/env bash\necho PASS\nexit 0\n")
    got = _child(stub, "test_a_planted_credential_blocks_the_flip")
    assert got.returncode != 0, (
        "a stub that refuses nothing passed the planting tests, so the child run is not "
        "using the script it was handed and every excision below proves nothing:\n"
        + got.stdout[-2000:])


@pytest.mark.skipif(MUTANT_RUN, reason="a child run of the excision harness")
@pytest.mark.parametrize("refusal", _refusals(), ids=repr)
def test_deleting_the_refusal_stops_the_planted_input_being_caught(refusal, tmp_path):
    """Delete one refusal, run the tests that must notice, require red.

    Three outcomes have to be told apart. A red child run means the refusal is load-bearing.
    A green one means the planted input is caught by something else, or by nothing -- either
    way nothing proves this refusal fires. A mutant that will not parse, or a control run
    that is not already green, is a harness error and is reported as one."""
    mutant_text = _without(refusal)
    mutant = tmp_path / "preflight_public.sh"
    mutant.write_text(mutant_text)
    parses = subprocess.run(["bash", "-n", str(mutant)], capture_output=True, text=True)
    if parses.returncode != 0:                          # trap 1
        pytest.fail(
            f"removing refusal {refusal.name!r} leaves a script bash will not parse "
            f"({parses.stderr.strip()}). Widen the markers so the block stands alone -- as "
            f"written, a red run below would prove nothing.")

    control = _child(SCRIPT, refusal.k)
    assert control.returncode == 0, (
        f"the control run for {refusal.name!r} is not green before anything is removed, so "
        f"a red mutant run would prove nothing. `-k {refusal.k}` selected "
        f"{'no tests' if control.returncode == 5 else 'failing tests'}:\n"
        + control.stdout[-2000:])

    got = _child(mutant, refusal.k)
    assert got.returncode != 0, (
        f"{refusal.name!r} was deleted and `-k {refusal.k}` still passes, so nothing proves "
        f"it fires. It refuses: {refusal.why}\n"
        f"Either something else catches the same planted input, or the refusal is dead.")
    # Trap 1 again, one level out: `bash -n` clears a mutant that parses, but a child run can
    # still go non-zero by erroring during collection rather than by failing an assertion --
    # and a red run for that reason proves nothing either. Require an actual test failure.
    assert "failed" in got.stdout, (
        f"the mutant run for {refusal.name!r} is red without a test failing, which is a "
        f"harness error and not evidence:\n" + got.stdout[-2000:])
