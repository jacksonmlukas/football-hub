"""The worktree-base hook.

On the night of 2026-09-06/07 three agents each reported an isolated worktree created from a
stale base -- 13 commits behind in one case, 17 in two others. Every one of them noticed and
fast-forwarded before starting, and two said the ticket was *unimplementable* at that base
because the change it followed on from had landed on `main` in the meantime.

They caught it. Nothing caught it for them, and that is the defect: an agent that did not
notice would have written against code that no longer exists, run its targeted tests green
against that stale tree, reported clean gates honestly, and produced a merge that silently
reverts whatever landed in between. The tests, the type gate, the mutation checks and the
review all run *inside* the worktree and would all agree with it.

So the check has to compare against something the worktree cannot see from inside, and it has
to run before the first edit rather than after -- an agent told at merge time has already
spent its context.

**Why the hook blocks rather than warns.** A warning printed into a transcript is a check whose
firing depends on the reader, which is the same shape as the defect. Blocking the first `Edit`
is refusable-by-fixing: the moment the worktree fast-forwards, `behind` is zero and the hook
stops firing. It clears itself.

Driven as a subprocess against throwaway repositories, because the thing under test is a
standalone script reading a JSON event on stdin, and because the states it distinguishes are
states of a repository rather than of a string.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "worktree_base.sh"

EDIT = {"tool_name": "Edit", "tool_input": {"file_path": "src/hub/thing.py"}}


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout.strip()


def fire(cwd: Path, event: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HOOK)], cwd=cwd, input=json.dumps(event or EDIT),
                          capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with a `main` carrying one commit, and nothing else."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "T")
    (root / "a.txt").write_text("one\n")
    git(root, "add", "a.txt")
    git(root, "commit", "-qm", "one")
    return root


def worktree_at(repo: Path, name: str, base: str) -> Path:
    """An agent worktree, in the place and shape the harness makes them."""
    path = repo / ".claude" / "worktrees" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "-q", "-b", f"worktree-{name}", str(path), base)
    return path


def advance(repo: Path, text: str) -> None:
    (repo / "a.txt").write_text(text)
    git(repo, "commit", "-qam", text)


# --- the case the ticket was filed about ----------------------------------

def test_a_worktree_behind_main_is_refused_before_the_first_edit(repo):
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    advance(repo, "three\n")
    wt = worktree_at(repo, "agent-1", stale)

    done = fire(wt)
    assert done.returncode == 2, (
        f"a worktree two commits behind main was allowed to edit: {done.stdout}{done.stderr}")
    said = done.stdout + done.stderr
    assert "2 commit" in said, (
        f"the refusal does not say how far behind: {said}")
    assert "main" in said, f"the refusal does not name the branch to catch up to: {said}"


def test_the_refusal_names_a_command_that_fixes_it(repo):
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-2", stale)

    done = fire(wt)
    said = done.stdout + done.stderr
    runnable = [ln.strip() for ln in said.splitlines()
                if re.match(r"^\s*git (merge|rebase)\b", ln) and "main" in ln]
    assert runnable, (
        f"an agent is told it is stale and not how to stop being stale: {said}")
    assert all("$" not in ln and "<" not in ln for ln in runnable), (
        f"the command is a template rather than something to run: {runnable}")


def test_it_clears_itself_once_the_worktree_catches_up(repo):
    """The property that makes blocking acceptable rather than obstructive."""
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-3", stale)
    assert fire(wt).returncode == 2, "premise: it must refuse before the merge"

    git(wt, "merge", "-q", "--no-edit", "main")
    done = fire(wt)
    assert done.returncode == 0, (
        f"still refusing a worktree that has caught up: {done.stdout}{done.stderr}")


# --- the states it must not fire in ---------------------------------------

def test_a_fresh_worktree_at_the_tip_is_silent(repo):
    wt = worktree_at(repo, "agent-4", "main")
    done = fire(wt)
    assert done.returncode == 0, f"refused a worktree created at the tip: {done.stdout}"
    assert done.stdout.strip() == "", f"a clean worktree should say nothing: {done.stdout!r}"


def test_the_primary_checkout_is_not_a_worktree_and_is_left_alone(repo):
    """The hook is wired into the same settings the primary checkout uses. Firing there would
    block edits in the checkout this session does its own work in.

    The checkout has to be genuinely *behind* `main` for this to test anything. An earlier
    version advanced `main` while HEAD was on it, so `behind` was zero and the test passed
    without the worktree check existing at all -- mutation caught it. A feature branch left
    behind `main` is both the realistic state and the one that separates "not a worktree" from
    "not stale".
    """
    git(repo, "checkout", "-qb", "feature")
    git(repo, "checkout", "-q", "main")
    advance(repo, "two\n")
    git(repo, "checkout", "-q", "feature")
    assert git(repo, "rev-list", "--count", "HEAD..main") == "1", "premise: behind by one"
    assert fire(repo).returncode == 0, "the hook fired in the primary checkout"


def test_a_worktree_ahead_of_main_is_not_stale(repo):
    """Ahead is the normal state of a worktree that has done its work. Only *behind* matters,
    so the count has to be one-sided rather than a symmetric divergence."""
    wt = worktree_at(repo, "agent-5", "main")
    (wt / "b.txt").write_text("mine\n")
    git(wt, "add", "b.txt")
    git(wt, "commit", "-qm", "mine")
    assert fire(wt).returncode == 0, "a worktree ahead of main was called stale"


def test_a_worktree_both_ahead_and_behind_is_still_refused(repo):
    """The realistic mid-work case: the agent has committed, and `main` moved underneath."""
    wt = worktree_at(repo, "agent-6", "main")
    (wt / "b.txt").write_text("mine\n")
    git(wt, "add", "b.txt")
    git(wt, "commit", "-qm", "mine")
    advance(repo, "two\n")
    assert fire(wt).returncode == 2, "divergence with commits on both sides was not refused"


# --- it must not become a check that cannot fail ---------------------------

def test_a_repository_with_no_main_does_not_silently_pass(repo, tmp_path):
    """`git rev-list` against a ref that does not exist fails, and a hook that reads that
    failure as "zero commits behind" is a guard that passes hardest exactly when it is least
    able to answer. This repo has that defect on record three times over.
    """
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-7", stale)
    git(repo, "branch", "-m", "main", "trunk")

    done = fire(wt)
    said = done.stdout + done.stderr
    assert done.returncode == 2, (
        f"no `main` to compare against, and the hook allowed the edit anyway: {said}")
    assert "main" in said.lower()
    # The exit code alone is not enough. Deleting the branch that handles this left the exit
    # code at 2 anyway -- bash errored on comparing an empty string with `-eq`, fell through,
    # and printed the *stale* message, which names a commit count it never obtained. Mutation
    # caught that. So this asserts on which of the two refusals was given.
    assert "there is no" in said, (
        f"refused, but with the message for a different problem: {said}")
    assert "behind" not in said.split("Establish")[0].lower(), (
        f"reported a staleness count it could not have measured: {said}")


@pytest.mark.parametrize("event", [
    {"tool_name": "Bash", "tool_input": {"command": "cat > src/hub/x.py <<EOF\nx = 1\nEOF"}},
    {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/a/b/' src/hub/x.py"}},
    {"tool_name": "Read", "tool_input": {"file_path": "src/hub/x.py"}},
    {"tool_name": "Write", "tool_input": {"file_path": "src/hub/x.py"}},
], ids=["heredoc", "sed-i", "read", "write"])
def test_every_wired_tool_is_refused_not_just_the_edit_tools(repo, event):
    """The gap the first draft had, and the reason it is not a nitpick.

    `tdd_gate.sh`'s header records that agents in this repo "had begun making their edits
    through Bash heredocs, which this hook never sees, so the gate was already not covering
    the people who had worked out how to avoid it." A staleness check wired to `Edit|Write`
    alone inherits that hole exactly, and the agents most likely to slip through it are the
    ones moving fastest.

    `Read` is in here for a different reason: refusing at the first edit still lets an agent
    read a dozen stale files and build a plan on them first, and the ticket asks for the
    divergence to reach the agent *before it starts work*.
    """
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-8", stale)

    assert fire(wt, event).returncode == 2, (
        f"a stale worktree accepted {event['tool_name']}; the edit could go through it")


def test_an_unreadable_event_fails_closed(repo):
    """`tdd_gate.sh` fails closed when its input surprises it, for the reason its header gives.
    The first draft of this hook fell through an unmatched `case` to exit 0 -- so a changed
    event shape would have silently disarmed it, which is the failure mode the whole file
    exists to prevent, one level up.
    """
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-9", stale)

    blank = subprocess.run(["bash", str(HOOK)], cwd=wt, input="",
                           capture_output=True, text=True)
    assert blank.returncode == 2, "an empty event disarmed the check"

    junk = subprocess.run(["bash", str(HOOK)], cwd=wt, input="not json at all",
                          capture_output=True, text=True)
    assert junk.returncode == 2, "an unparseable event disarmed the check"


def test_a_directory_that_is_not_a_repository_is_left_alone(tmp_path):
    """Distinct from the cases above: not-a-repository is genuinely none of this hook's
    business, where cannot-answer refuses. Conflating the two is what made the first draft
    exit 0 on a missing git."""
    plain = tmp_path / "elsewhere"
    plain.mkdir()
    assert fire(plain).returncode == 0, "refused an edit outside any repository"


def test_no_git_on_path_refuses_rather_than_assuming_the_best(repo):
    """The one refusal the repository's own state cannot reach.

    Every other path here is exercised by arranging the repo; this one needs the tool itself
    to be missing, so it is arranged by emptying `PATH`. Without a test it is a branch that
    mutation cannot kill -- and it is the branch guarding the case where the hook has the least
    information, which is exactly where this repo's guards have historically reported success.
    """
    bash = shutil.which("bash")
    assert bash, "premise: a bash to run the hook with"
    empty = repo / "no-tools"
    empty.mkdir()
    # `bash` is launched by absolute path, because an empty PATH would leave subprocess unable
    # to find the launcher itself and the test would fail before reaching the hook.
    done = subprocess.run([bash, str(HOOK)], cwd=repo, input=json.dumps(EDIT),
                          capture_output=True, text=True, env={"PATH": str(empty)})
    assert done.returncode == 2, (
        f"no git, and the hook waved the edit through: {done.stdout}{done.stderr}")
    assert "git" in (done.stdout + done.stderr).lower()


# --- the guard must never block its own remedy -----------------------------

@pytest.mark.parametrize("command", [
    "git merge --no-edit main",
    "git merge main",
    "git rebase main",
    "git -C /some/worktree merge --no-edit main",
    "cd /some/worktree && git merge --no-edit main",
    "git fetch origin && git merge --no-edit main",
    "git status --short",
    "git log --oneline -5",
], ids=["plain", "bare", "rebase", "dash-C", "cd-then", "fetch-then", "status", "log"])
def test_the_prescribed_remedy_is_never_refused(repo, command):
    """The deadlock, and it was not hypothetical for four minutes.

    The first version of this hook refused every tool call with no exception, so it also
    refused `git merge --no-edit main` -- the command its own refusal message instructs you to
    run. On 2026-09-09 three agents hit that within minutes of each other, each tried the
    prescribed merge four separate times, and each abandoned its ticket with work uncommitted.

    A guard that blocks its own fix does not fail safe. It converts a one-command correction
    into a lost session, and it teaches the next agent that the guard is an obstacle to route
    around rather than a fact to act on -- which is how `tdd_gate.sh` lost its coverage.
    """
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-remedy", stale)

    done = fire(wt, {"tool_name": "Bash", "tool_input": {"command": command}})
    assert done.returncode == 0, (
        f"refused {command!r}, which an agent needs to stop being stale: {done.stderr}")


@pytest.mark.parametrize("command", [
    "sed -i 's/a/b/' src/hub/x.py",
    "cat > src/hub/x.py <<EOF\nx = 1\nEOF",
    "python -c \"open('src/hub/x.py','w').write('x')\"",
    "echo 'digitally' > src/hub/x.py",
], ids=["sed-i", "heredoc", "python-write", "redirect"])
def test_the_escape_hatch_does_not_reopen_the_bash_hole(repo, command):
    """The exemption is for `git`, and `git` cannot write source the way these can.

    Widening this hook to Bash existed to close the evasion `tdd_gate.sh` documents -- agents
    editing through heredocs that an `Edit|Write` hook never sees. An exemption broad enough to
    let those back through would undo the reason the hook covers Bash at all.
    """
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-hole", stale)

    done = fire(wt, {"tool_name": "Bash", "tool_input": {"command": command}})
    assert done.returncode == 2, (
        f"a stale worktree accepted {command!r}; the edit could go through it")


def test_a_word_that_merely_contains_git_is_not_an_exemption(repo):
    """`digital`, `legitimate`, a path named `gitlab-ci` -- none of them run git."""
    stale = git(repo, "rev-parse", "HEAD")
    advance(repo, "two\n")
    wt = worktree_at(repo, "agent-word", stale)

    for command in ["echo legitimate > src/hub/x.py",
                    "sed -i 's/x/y/' .gitlab/config.yml",
                    "cat digital-merge-notes.txt > src/hub/x.py"]:
        done = fire(wt, {"tool_name": "Bash", "tool_input": {"command": command}})
        assert done.returncode == 2, f"{command!r} was exempted as if it ran git"
