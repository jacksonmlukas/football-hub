"""The unattended refresher, run rather than reasoned about.

The window used to be a `*/10` cron on the deploy workflow. On 2026-09-05 those runs were
measured landing 97-126 minutes apart, median 106, each executing in under a minute -- so the
published overlay was stale by the watchdog's own 1500s definition for most of every game
window, an incident was filed, the next deploy closed it, and it read as noise. Issue #91.

**Why this file exists in this shape.** The refresher's failure mode is silence: a loop that
stops, or that publishes a fresh timestamp over frozen scores, looks exactly like a working
one from the outside -- and the second is worse than the first, because the heartbeat the
watchdog reads is the very field it overwrites. A test that asserted the *arrangement* (a
cron exists, a flag is passed) would pass in both cases. So each of the three fetch outcomes
is made to happen, through the real script and the real CLI, and what came out is read back:

  * ESPN answers with games      -> the overlay is written and the deploy command runs.
  * ESPN answers with no games   -> the same. ADR-0018: an empty board is ESPN saying there
                                    are no games, and relaying it is the relay working.
  * ESPN cannot be reached       -> **no deploy, and `generated_at` does not move**, even
                                    though the fetch layer's cache could have answered.

Only two things are doubles: the socket, and the deploy. The socket, because the outcomes are
defined by what ESPN does and a test that waited for ESPN to fail would never run; the deploy,
for the reason `heartbeat.sh` takes a URL. Everything between them -- `live-loop.sh`,
`hub.publish --live`, `hub.publish.live`, `hub.fetch.espn._get` and its cache -- is the code
that will run on a Sunday.
"""
import json
import re
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "live-loop.sh"

# One in-progress game, in ESPN's own shape. Enough for `_overlay_row` to resolve a row, which
# is what makes "games" different from "no games" all the way through.
ONE_GAME = {"events": [{"id": "401", "competitions": [{
    "status": {"type": {"state": "in", "shortDetail": "Q3 4:12"}},
    "competitors": [{"homeAway": "home", "team": {"abbreviation": "PHI"}, "score": "21"},
                    {"homeAway": "away", "team": {"abbreviation": "DAL"}, "score": "17"}]}]}]}
NO_GAMES: dict = {"events": []}

# Imported at interpreter start-up by every `uv run python` the loop spawns, which is the only
# way into a subprocess the script starts itself. It redirects the fetch layer's cache and
# decides what the network does; where the overlay is written is the CLI's own `--out`.
#
# `hub.publish` is deliberately *not* patched here: `python -m hub.publish` runs the module as
# `__main__`, a second module object, so patching the imported one would silently miss.
SHIM = '''
import json
import os
import pathlib

import requests

import hub.fetch.espn

hub.fetch.espn.CACHE = pathlib.Path(os.environ["FAKE_CACHE"])
hub.fetch.espn.CACHE.mkdir(parents=True, exist_ok=True)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _get(*args, **kwargs):
    board = pathlib.Path(os.environ["FAKE_ESPN"]).read_text()
    if board == "down":
        raise OSError("espn unreachable")
    return _Resp(json.loads(board))


requests.get = _get
'''


class Run:
    """One run of the loop, and everything it left behind."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.site = tmp_path / "site" / "data"
        self.cache = tmp_path / "cache"
        self.board = tmp_path / "board.json"
        self.log = tmp_path / "deploys.log"
        shim = tmp_path / "shim"
        shim.mkdir()
        (shim / "sitecustomize.py").write_text(SHIM)
        self.shim = shim
        # The deploy double. It records the moment it was asked and the stamp of the overlay
        # it was asked to publish, which is what makes "deployed cached scores" visible rather
        # than merely absent.
        self.deploy = tmp_path / "deploy.sh"
        self.deploy.write_text(
            "#!/usr/bin/env bash\n"
            # Read with sed rather than another interpreter: this runs once per cycle and the
            # cadence assertion below is about the loop's pacing, not the double's start-up.
            f"stamp=$(sed -n 's/.*\"generated_at\": \"\\([^\"]*\\)\".*/\\1/p' "
            f'"{self.site}/live.json" | head -1)\n'
            f'echo "$(date +%s) $stamp" >> "{self.log}"\n'
            f'exit ${{DEPLOY_EXIT:-0}}\n')
        self.deploy.chmod(0o755)

    def espn(self, payload) -> None:
        """What ESPN does from now on: a payload, or `down` for unreachable."""
        self.board.write_text("down" if payload == "down" else json.dumps(payload))

    def go(self, seconds: int, interval: int, league: str = "nfl",
           deploy_exit: int = 0) -> subprocess.CompletedProcess:
        got = subprocess.run(
            [str(SCRIPT), str(seconds), str(interval), league, str(self.site),
             str(self.deploy)],
            capture_output=True, text=True, timeout=300,
            env={**_env(), "PYTHONPATH": str(self.shim), "FAKE_ESPN": str(self.board),
                 "FAKE_CACHE": str(self.cache), "DEPLOY_EXIT": str(deploy_exit)})
        assert got.returncode == 0, got.stdout + got.stderr
        return got

    @property
    def deploys(self) -> list[tuple[int, str]]:
        """(when, the `generated_at` that was published) for every deploy asked for."""
        if not self.log.exists():
            return []
        return [(int(line.split()[0]), line.split()[1])
                for line in self.log.read_text().splitlines() if line.strip()]

    @property
    def overlay(self) -> dict:
        return json.loads((self.site / "live.json").read_text())


def _env() -> dict:
    import os
    return {k: v for k, v in os.environ.items() if k not in ("FAKE_ESPN", "FAKE_CACHE")}


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def test_the_script_branches_on_the_exit_code_python_actually_returns():
    """The one fact the script and the CLI both hold, so the one that can drift.

    It fails safe if it does -- an unrecognised code does not deploy either -- but the loop
    would report every outage as the program being broken, which sends the reader looking in
    the wrong place during the one hour anybody is watching.
    """
    from hub import publish
    text = SCRIPT.read_text()
    assert f"\n    {publish.NOTHING_FRESH})\n" in text, (
        f"the loop has no branch for exit {publish.NOTHING_FRESH}, which is what "
        f"`hub.publish --live` returns when it could not reach ESPN")
    assert "NOTHING_FRESH" in text, "the script does not say whose constant that number is"


# --- one loop, and a deploy that is never cut in half -------------------------

def test_one_loop_at_a_time_and_no_deploy_is_ever_cancelled():
    """Two settings that look contradictory and are both right.

    A second loop would double the deploy rate and interleave two answers to one question, and
    cancelling one costs nothing -- the next cycle asks ESPN again. A cancelled *deploy* is a
    half-applied site, which is worse than a redundant one. So the loop cancels its
    predecessor and the deploy queues behind its own.

    Read out of the files because it is a two-line arrangement nothing else would notice: a
    `cancel-in-progress: true` copied onto the deploy group would show up as an occasional
    broken page, months later, on a Sunday.
    """
    import yaml
    live = yaml.safe_load((ROOT / ".github" / "workflows" / "live.yml").read_text())
    pages = yaml.safe_load((ROOT / ".github" / "workflows" / "pages.yml").read_text())

    assert live["concurrency"]["cancel-in-progress"] is True, (
        "a newer window start must replace the running loop, not run beside it")
    assert live["concurrency"]["group"] != pages["concurrency"]["group"], (
        "the loop shares the deploy's group, so starting a loop would cancel or queue behind "
        "a deployment")
    assert pages["concurrency"]["cancel-in-progress"] is False, (
        "a deployment in flight can now be cancelled: a half-applied site is worse than a "
        "redundant deploy")


# --- the premise: the doubles have to be able to record something ------------

def test_the_recorder_records_and_the_socket_can_be_cut(run):
    """Every assertion below is about what did or did not reach the deploy log, so a
    recorder that never records would make the interesting half of this file vacuous --
    which is the shape the repo keeps finding in its own checks."""
    run.espn(ONE_GAME)
    run.go(seconds=1, interval=2)
    assert len(run.deploys) == 1, "the deploy double did not record its one call"
    assert run.overlay["rows"][0]["home"] == "PHI", "the fake ESPN did not reach the overlay"


# --- outcome 1: ESPN answered, with games ------------------------------------

def test_games_are_written_and_deployed(run):
    run.espn(ONE_GAME)
    run.go(seconds=1, interval=2)
    assert run.overlay["n"] == 1
    assert run.overlay["rows"][0]["state"] == "in"
    assert len(run.deploys) == 1
    assert run.deploys[0][1] == run.overlay["generated_at"], (
        "the deploy published a different stamp than the one on disk")


# --- outcome 2: ESPN answered, with nothing ----------------------------------

def test_an_empty_board_is_still_deployed(run):
    """ADR-0018. A live score is someone else's fact, so an empty scoreboard is ESPN saying
    there are no games -- a Tuesday, or a slate that has finished -- and relaying that is the
    relay working, not an empty answer replacing a full one. The refusal below must be about
    reaching ESPN and nothing else, or it becomes the last-good guard ADR-0018 refuses."""
    run.espn(ONE_GAME)
    run.go(seconds=1, interval=2)
    had = run.overlay

    time.sleep(1.1)          # `generated_at` has second resolution; this is what "advanced" means
    run.espn(NO_GAMES)
    run.go(seconds=1, interval=2)
    assert run.overlay["n"] == 0 and run.overlay["rows"] == []
    assert run.overlay["generated_at"] > had["generated_at"], "an answer advances the stamp"
    assert len(run.deploys) == 2, "the empty board was not deployed"


# --- outcome 3: ESPN was not reached -----------------------------------------

def test_an_unreachable_espn_holds_the_stamp_and_does_not_deploy(run):
    """The heart of issue #91, and the one that erases its own evidence if it is wrong.

    `hub.fetch.espn._get` falls back to a last-good cache, deliberately, so the dashboard
    degrades instead of erroring. Here that cache is *warm* -- the first cycle filled it --
    and a refresher that took it would write a payload of frozen scores under a brand new
    `generated_at`. The page would sit still while the heartbeat said it was fresh, and the
    watchdog, which reads exactly that field, could never fire again.

    So: no write, no deploy, and the previously published overlay untouched, byte for byte.
    """
    run.espn(ONE_GAME)
    run.go(seconds=1, interval=2)
    published = (run.site / "live.json").read_text()
    assert (run.cache / "sb_nfl_now.json").exists(), (
        "the cache is cold, so this test would pass without the refusal it is about")

    run.espn("down")
    got = run.go(seconds=5, interval=2)

    assert (run.site / "live.json").read_text() == published, (
        "generated_at moved over scores nobody refreshed")
    assert len(run.deploys) == 1, f"a cache-served cycle deployed: {run.deploys}"
    assert "ESPN was not reached" in got.stdout
    assert "held" in got.stdout.splitlines()[-1]


def test_the_loop_keeps_running_after_a_failure_and_recovers(run):
    """A refresher that stops on the first outage is the defect this replaces wearing a
    stricter face. The window keeps going, so the loop does too."""
    run.espn("down")
    got = run.go(seconds=5, interval=2)
    # Asserted on how the loop *ended*, not on how many cycles fitted. The comment here used
    # to say that a cycle count is a fact about the machine and then count cycles anyway, and
    # under load one cycle can outlast the whole window (issue #115). The summary line is
    # printed after the `while` breaks on its deadline; a loop that died at the refusal --
    # the defect this is about -- never reaches it.
    assert _ran_to_its_deadline(got), "the loop stopped at the first refusal"
    assert "ESPN was not reached" in got.stdout, "it did not see the refusal at all"
    assert run.deploys == []

    run.espn(ONE_GAME)
    run.go(seconds=1, interval=2)
    assert len(run.deploys) == 1, "the loop did not resume publishing when ESPN came back"


def test_a_failed_deploy_does_not_end_the_window(run):
    """The dispatch can fail on its own -- a rate limit, a token, a bad day at GitHub. The
    overlay is written and honest by then; only its publication failed, and the next cycle
    publishes the same thing or something newer."""
    run.espn(ONE_GAME)
    got = run.go(seconds=5, interval=2, deploy_exit=1)
    assert _ran_to_its_deadline(got), "a failed dispatch ended the loop"
    assert "the deploy command failed" in got.stdout


def _ran_to_its_deadline(got) -> bool:
    """Whether the loop exited through its own deadline check rather than dying mid-window.

    The summary line is the last thing the script echoes, after the `while` breaks. A loop
    that exited on a refusing fetch or a failed dispatch -- the two defects these tests exist
    to catch -- never gets there. This is what "carried on" means without counting cycles
    against a wall clock that belongs to the machine, not to the loop (issue #115).
    """
    tail = got.stdout.rstrip().splitlines()
    return bool(tail) and tail[-1].startswith("live-loop: ") and "cycles over" in tail[-1]


# --- the cadence, measured -------------------------------------------------

def test_the_cycles_land_at_the_interval_asked_for(run):
    """The number the design rests on: the loop paces from the *start* of each cycle, so the
    cadence is the interval rather than the interval plus however long ESPN and the dispatch
    took. Five minutes in production, two seconds here.

    **Asserted on the schedule the loop computes, not on how many cycles finished.** The
    previous form counted deploys inside a wall-clock window -- "at least four in nine
    seconds" -- which passes alone and fails under load, and failed repeatedly with several
    suites running at once. That is not a weaker version of this property, it is a different
    one: when the machine is busy the work outgrows the interval, the nap goes to zero, and
    the count falls while the pacing stays exactly right. Three agents diagnosed it
    independently in one session (issue #115).

    What separates the two arrangements is arithmetic the loop already does and now prints:
    pacing from the cycle start makes `nap` shrink as the work grows, and the two always sum
    to the interval. Sleeping the interval *after* the work makes `nap` the interval every
    time, whatever the work cost. Load moves both numbers and cannot break the identity.
    """
    run.espn(ONE_GAME)
    got = run.go(seconds=9, interval=2)

    paced = [tuple(int(n) for n in m)
             for m in re.findall(r"paced: work (\d+)s, nap (\d+)s, interval (\d+)s",
                                 got.stdout)]
    assert paced, f"the loop printed no schedule at all\n{got.stdout}"
    for work, nap, interval in paced:
        assert nap == max(interval - work, 0), (
            f"a cycle costing {work}s napped {nap}s at a {interval}s interval. The loop is "
            f"sleeping the interval on top of the work rather than including it."
            f"\n{got.stdout}")
    # The identity above holds trivially if the work is always zero, so at least one cycle
    # has to have cost something -- otherwise this passes on a loop that does no work.
    assert any(work > 0 for work, _nap, _interval in paced) or len(paced) > 1, (
        f"every cycle was instantaneous, so nothing distinguished the two arrangements"
        f"\n{got.stdout}")


def test_the_loop_still_completes_cycles_at_a_short_interval(run):
    """The companion to the one above, and deliberately loose.

    The identity is about intent; this is the weakest statement that the loop *acts* on it,
    and it is weak on purpose -- how many cycles fit in nine seconds is a fact about the
    machine. One completed cycle is what separates a paced loop from one that never runs, and
    that much is true on any machine.

    Remaining dependence on real time, written where a reader meets it: `run.go` blocks for
    the duration it is given, so this test costs nine seconds of wall clock whatever the load.
    """
    run.espn(ONE_GAME)
    started = time.time()
    got = run.go(seconds=9, interval=2)
    elapsed = time.time() - started
    assert run.deploys, f"the loop completed no cycle at all\n{got.stdout}"
    assert elapsed < 60, "the cap did not stop the loop anywhere near when it should have"


def test_the_loop_stops_when_its_time_is_up(run):
    """The cap is what keeps a loop started at a window's close from running for hours past
    it, and what lets the next scheduled start replace this one cleanly."""
    run.espn(ONE_GAME)
    started = time.time()
    run.go(seconds=4, interval=2)
    assert time.time() - started < 20, "the loop ran well past its cap"
    assert len(run.deploys) <= 3, "more cycles than the cap allows"
