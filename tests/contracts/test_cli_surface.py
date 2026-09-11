"""Every CLI must answer with a sentence, never a traceback.

Three bugs of exactly this shape were found on 2026-08-25, all in code at or above 80%
coverage:

  * `hub.draft.board` raised ConnectionError with no network, instead of serving the board
    already on disk.
  * `hub.models.conformal --recalibrate` died on `BinderException: "margin_actual" not
    found`. That column has never existed; every unit test handed the function its own frame.
  * `hub.season.lineup` raised a bare FileNotFoundError for a roster file nothing writes.

**Coverage did not catch any of them, because coverage measures lines executed, not whether
the seam between a module and the real world exists.** All three lived at that seam, and all
three surface only when a CLI meets absent input -- which is the state of a fresh clone, and
the state most of this repo is in right now.

These tests are deliberately shallow and wide: they assert the *shape* of the failure, not
what any command computes. Nothing here touches the network.

The store-backed CLIs were initially excluded, because `main()` read the repo's own `ROOT`
with no override -- and that is exactly where the conformal bug lived, so the file did not
close the class it was written for. Threading `--store` through them fixed that, and doing so
immediately found a further bug of the same family: against a genuinely empty store, both
`conformal --recalibrate` and `eval --compare` raised

    _duckdb.CatalogException: Table with name preds does not exist!

An empty store does not have an empty `preds` table, it has no `preds` view at all, because
`store.connect` builds a view per directory that exists. **That is the state of a fresh
clone** -- which is what the public gets when this repo flips on 2026-09-04.

**What #111 fixed is the other half of that same idea.** `CLI_MODULES` was held against the
source tree by a scan; the list of modules actually *driven* against absent input was held
against nothing, and it had four of the twenty-nine on it -- chosen by hand, and neither of
the two worth the most among them. One property now runs over every entry point in the tree:
non-zero exit, the missing thing named, no traceback. Eighteen of them answered a failed
fetch with a traceback until it did, and answer with `hub.cli.unavailable` now.
"""
import ast
import importlib
import inspect
import socket

import dotenv
import pytest
import requests.adapters
from nflreadpy import config as nflconfig

from hub import store
from hub.draft import board, state
from hub.fetch import nflverse

# Every module with a CLI. A new one missing from this list is caught by
# `test_every_cli_module_is_covered_here` below rather than by nobody.
CLI_MODULES = (
    "hub.draft.adherence", "hub.draft.backtest", "hub.draft.board", "hub.draft.calibrate", "hub.draft.evaluate",
    "hub.draft.fit_corrections",
    "hub.draft.leverage", "hub.draft.live", "hub.draft.tune", "hub.fetch.bigten",
    "hub.fetch.cfbd", "hub.fetch.nflverse", "hub.fetch.odds", "hub.inspect", "hub.models.conformal",
    "hub.models.coverage",
    "hub.models.correlate", "hub.models.eval", "hub.models.injury", "hub.models.margin",
    "hub.models.ratings", "hub.models.spread",
    "hub.models.component_error", "hub.models.weekly", "hub.models.weekly_screen", "hub.publish", "hub.season.lineup",
    "hub.season.lineup_gate", "hub.season.roster", "hub.season.weekly_gate",
    "hub.season.survivor", "hub.season.pool", "hub.season.journal", "hub.store",
)


@pytest.mark.parametrize("name", CLI_MODULES)
def test_every_cli_exposes_main_taking_argv(name):
    """`board.main()` was once the only entry point in the repo that did not take argv, and
    was therefore the least reachable code in the module with the most churn."""
    mod = importlib.import_module(name)
    assert hasattr(mod, "main"), f"{name} has no main()"
    import inspect
    params = inspect.signature(mod.main).parameters
    assert params, f"{name}.main() takes no argv, so it cannot be driven from a test"


@pytest.mark.parametrize("name", CLI_MODULES)
def test_help_needs_no_network_and_no_data(name, capsys):
    """`--help` must never reach a fetch. It is also the weakest check in this file: it is
    exactly what passed on all three of the bugs above."""
    mod = importlib.import_module(name)
    with pytest.raises(SystemExit) as e:
        mod.main(["--help"])
    assert e.value.code == 0
    assert "usage:" in capsys.readouterr().out


# --- the flag stage 2 of the power measurement is run with --------------------
#
# `docs/gate-power.md` stage 2 compares each gate's MDE against **its own** foresight ceiling
# and closes the tickets under a gate that cannot clear it. Getting those three numbers means
# running three commands, so the flag that produces one is part of this repo's CLI surface in
# the way `--help` is: recorded here, not left to each gate.
#
# It was not, and that is what #134 was. #43 gave the weekly gate and the lineup gate a
# ceiling arm each, both reachable only as a keyword argument to a `compare` no operator
# calls -- so the measurement stage 2 exists to take could be taken for one of the three.

GATES_WITH_A_CEILING = (
    "hub.draft.backtest", "hub.season.lineup_gate", "hub.season.weekly_gate",
)


@pytest.mark.parametrize("name", GATES_WITH_A_CEILING)
def test_every_gate_offers_its_ceiling_under_the_one_flag_name(name, capsys):
    """One name across all three, because a measurement taken three ways is three
    measurements. What each gate's ceiling *means* is its own -- a foresight drafter, a
    perfect weekly projection, a perfect spread -- and each says so on the line it prints;
    what an operator types to ask for one must not also be per-gate."""
    mod = importlib.import_module(name)
    with pytest.raises(SystemExit):
        mod.main(["--help"])
    assert "--ceiling" in capsys.readouterr().out, (
        f"{name} builds a ceiling column that no command line can reach, so "
        f"docs/gate-power.md stage 2 cannot be run for it")


@pytest.mark.parametrize("name", GATES_WITH_A_CEILING)
def test_the_ceiling_flag_reaches_the_comparison_it_asks_for(name):
    """Accepting the flag is not answering it.

    The test above proves an operator can *type* `--ceiling`; a `main` that parsed it and
    dropped it would pass that and print the same summary it always did, which is the shape
    of the defect #134 was filed for -- a ceiling built and unreachable -- moved one step
    later. Read off the source because these `main`s are `# pragma: no cover - network`:
    what can be checked without a season of data is that the parsed flag is handed to the
    call that computes the arm.
    """
    mod = importlib.import_module(name)
    src = inspect.getsource(mod)
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    # The parsed flag is read somewhere in `main`. Deliberately not "forwarded as
    # `ceiling=`": the draft gate branches on it (`if a.ceiling:`) and the two season gates
    # pass it down, and pinning either spelling would make this a test of house style. What
    # it refuses is the one thing that cannot be right -- a flag declared, parsed, and never
    # looked at again.
    read = [n for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and n.attr == "ceiling"
            and isinstance(n.ctx, ast.Load)]
    assert read, (
        f"{name}.main declares --ceiling and never reads it back. The flag is accepted and "
        f"the ceiling is still unreachable, which is issue #134 one step further in: "
        f"docs/gate-power.md stage 2 would run, print, and measure nothing.")


# --- absent input, for every entry point ------------------------------------
#
# A fresh clone is not only a directory with no parquet in it. It has no key in the
# environment, no `.env` to supply one, nothing cached from upstream, and -- on the runner
# that will publish this repo, and on a laptop on a plane -- no network either. Those
# arrive together, and any one of them alone is a state no operator is ever in.
#
# So the world is built once, here, and every module below is driven through the same one.
# The alternative is a fixture that knows which constant each CLI reads, which is the
# per-CLI knowledge this file exists to avoid: the roots below are the whole repo's, not any
# module's. `hub.fetch.nflverse.RAW` is where every cached upstream release lands and
# `hub.store.DATA` is where every partition does, so redirecting those two empties the store
# and the cache for all of the store-backed commands at once.
#
# It also stops this file reading its answer off the machine it runs on, in both directions.
# Every one of `board`, `ratings`, `roster` and `survivor` exits 0 on a developer's clone and
# refuses on a fresh one, because each degrades to local state that a developer has and a
# fresh clone does not -- a green tick meaning "this laptop has a `data/` directory". And in
# the other direction, nine of these commands passed alone and exited 0 under the full
# suite until the last two lines of the fixture below existed, because something earlier in
# the run had already fetched what they wanted.


@pytest.fixture
def a_fresh_clone(monkeypatch, tmp_path):
    """Nothing on the network, no credential, and nothing cached anywhere."""
    def refuse(*_a, **_k):
        raise OSError("the network is absent (no test here reaches one)")

    # Two levers on the network, because the socket alone is not one. Blocking it stops a
    # connection being *made*, and `requests.Session` -- which `nflreadpy`, `espn_api` and
    # this repo's own fetch layer all hold on to -- pools connections that are already open.
    # A test that fetched for real earlier in the session leaves one there.
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)
    # Patched rather than merely unset: `load_dotenv` is called *inside* the functions that
    # read a key, so a developer's own `.env` would otherwise decide whether a CLI refuses.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for key in ("CFBD_API_KEY", "ODDS_API_KEY", "ESPN_S2", "ESPN_SWID", "ESPN_LEAGUE_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(nflverse, "RAW", tmp_path / "raw")
    monkeypatch.setattr(store, "DATA", tmp_path / "processed")
    # The board and the draft state, patched on the modules that *bind* them rather than on
    # `hub.paths`. `hub.draft.board` does `from hub.paths import BOARD_PARQUET` at import, so
    # rebinding the source module never reaches it; `hub.draft.state` resolves `STATE` at call
    # time on purpose, and says so where it does it.
    #
    # Without these two, the poller read the developer's own board and draft state. Its
    # absent-input case passed by taking a pre-existing branch and never reaching the handler
    # it was added for -- and it *inverted* once the draft state held picks, because replay
    # then succeeds and exits zero. Green on a clone that had not drafted, red on one that
    # had, for a reason with nothing to do with absent input.
    monkeypatch.setattr(board, "BOARD_PARQUET", tmp_path / "processed" / "draft_board.parquet")
    monkeypatch.setattr(board, "BOARD_JSON", tmp_path / "site" / "draft_board.json")
    monkeypatch.setattr(state, "STATE", tmp_path / "processed" / "draft_state.json")
    # The third cache, and the one that is not this repo's: `nflreadpy` keeps every release
    # it has downloaded in this *process*, so an earlier test's fetch answers a later
    # command with no network needed at all. Turned off through its own configuration rather
    # than by emptying it, so a developer's warm cache survives this file having run.
    nflconfig.update_config(cache_mode="off")
    yield
    nflconfig.reset_config()


# Every CLI, each with the argv that puts it in front of the input it needs. The argv
# differs because the commands differ; what is asserted about the result does not, and that
# is the point -- see `test_absent_input_is_reported_not_raised` below. A command that names
# a file takes an absent one, a command that names a store takes an empty one, and the rest
# are driven on the path they actually run, with the world above around them.
#
# `--status-path`, `--quota-path`, `--state-path` and `--out` are here for a second reason:
# they keep a refusal from writing its record over the real one. A contract test that
# overwrites `site/data/cfbd.json` to prove a point has broken the thing it was checking.
ABSENT_INPUT = [
    ("hub.draft.adherence", ["--board", "{tmp}/nope.parquet"]),
    ("hub.draft.backtest", []),
    ("hub.draft.calibrate", []),
    ("hub.draft.evaluate", ["--sweep"]),
    ("hub.draft.fit_corrections", ["--fit", "--seasons", "2024,2025"]),
    ("hub.draft.live", ["--replay", "2024"]),
    ("hub.draft.tune", ["--sweep"]),
    ("hub.fetch.bigten", ["--capture", "--status-path", "{tmp}/bigten.json",
                          "--index", "{tmp}/captures.json", "--archive", "{tmp}/archive",
                          "--lines-dir", "{tmp}/lines", "--quota-path", "{tmp}/quota.json"]),
    ("hub.fetch.cfbd", ["--week", "1", "--status-path", "{tmp}/cfbd.json",
                        "--quota-path", "{tmp}/quota.json"]),
    ("hub.fetch.nflverse", ["--refresh"]),
    ("hub.fetch.odds", ["--snapshot", "--state-path", "{tmp}/odds.json"]),
    ("hub.inspect", ["{tmp}/nope"]),
    ("hub.models.component_error", ["--run"]),
    ("hub.models.conformal", ["--recalibrate", "--store", "{tmp}"]),
    ("hub.models.coverage", ["--measure", "--cache", "{tmp}"]),
    ("hub.models.correlate", []),
    ("hub.models.eval", ["--compare", "a,b", "--store", "{tmp}"]),
    ("hub.models.injury", ["--fit"]),
    ("hub.models.margin", ["--fit"]),
    ("hub.models.ratings", ["--fit"]),
    ("hub.models.spread", ["--fit"]),
    ("hub.models.weekly", ["--fit"]),
    ("hub.models.weekly_screen", ["--run"]),
    ("hub.publish", ["--live", "--out", "{tmp}"]),
    ("hub.season.lineup", ["--opp-mu", "110", "--roster", "{tmp}/nope.parquet"]),
    ("hub.season.lineup_gate", []),
    ("hub.season.roster", ["--out", "{tmp}/roster.parquet"]),
    ("hub.season.survivor", []),
    # The money layer and its journal (#163). `--store` keeps the pool's last-good read --
    # the journal -- and the journal's own read off the developer's store, so a fresh clone
    # is what both meet: no schedule, and no decision recorded to serve in its place.
    ("hub.season.pool", ["--week", "1", "--store", "{tmp}"]),
    ("hub.season.journal", ["--store", "{tmp}"]),
    ("hub.season.weekly_gate", ["--run"]),
    ("hub.store", ["--verify"]),
]


@pytest.mark.parametrize("name,argv", ABSENT_INPUT)
def test_absent_input_is_reported_not_raised(name, argv, tmp_path, capsys, a_fresh_clone):
    """One property, asserted identically of every entry point: it exits non-zero, it says
    what it could not read, and it does not hand back a traceback.

    Deliberately nothing stronger. Which sentence a command prints, which exit code it picks
    out of the non-zero ones, and which of its inputs it happens to miss first are all
    per-CLI facts, and a test that pinned them here would be twenty-seven things to maintain
    that no operator is helped by. The three assertions below are the whole of what a person
    at a terminal needs: that the command failed, that they can tell what to go and get, and
    that they are reading a sentence rather than a stack.
    """
    mod = importlib.import_module(name)
    filled = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    try:
        code = mod.main(filled)
    except SystemExit as e:                      # argparse's own exit is a sentence too
        code = e.code
    except Exception as e:
        pytest.fail(f"{name} raised {type(e).__name__}: {e}\n"
                    f"Absent input is what a fresh clone hands every command here, and the "
                    f"answer to it is a sentence and a non-zero exit, not a traceback.")
    assert isinstance(code, int) and code != 0, f"{name} should fail, and say so"
    out = capsys.readouterr()
    assert "Traceback" not in (out.out + out.err)
    assert (out.out + out.err).strip(), f"{name} failed silently, which is worse"


def test_every_cli_is_driven_against_absent_input():
    """The list above, held against the repo -- which is what nothing did until issue #111.

    `CLI_MODULES` has been kept complete by a scan of the source since this file was
    written. What was driven against absent input was four of those twenty-nine, chosen by
    hand, and the gap was invisible: every test in here passed, `--help` passed for all
    twenty-nine, and the two entry points worth the most -- the board's, whose docstring
    records that all three bugs found in the draft-night rehearsal lived below its `main`,
    and the publisher's -- were not among the four. A list nothing holds against the repo is
    the one part of a contract file that decays, which is the failure the rest of it exists
    to prevent.

    Two modules are not driven, and the two reasons are different in kind:

      * `hub.draft.leverage` has no input for absence to arrive at. It reads no file, opens
        no socket and takes no credential: it simulates this league's bracket from the
        constants in `hub.draft.season`, and `--sims` is the only thing it is given. Driving
        it would mean inventing a refusal -- a floor on `--sims` written for this test and
        for no caller -- and a green tick over a guard that guards nothing is the exact
        shape this repo spent the week of 2026-09-04 removing.
      * `hub.draft.board` is a missing source change rather than a missing test, and it is
        the one entry point here that most needs it. Offline with no board on disk, `main`
        reaches `last_good`, which raises `FileNotFoundError` -- a sentence, but raised, so
        a fresh clone gets the traceback in place of it. It also has no flag naming the
        board it reads, so nothing can point it at an absent one the way `adherence
        --board` can. Both are changes to `src/hub/draft/board.py`, which was being edited
        elsewhere when #111 landed; the entry belongs in `ABSENT_INPUT` the day they land,
        and this subtraction goes with it.
    """
    driven = {name for name, _ in ABSENT_INPUT}
    assert driven == set(CLI_MODULES) - {"hub.draft.leverage", "hub.draft.board"}, (
        f"not driven against absent input: "
        f"{sorted(set(CLI_MODULES) - driven - {'hub.draft.leverage', 'hub.draft.board'})}; "
        f"driven but not a CLI: {sorted(driven - set(CLI_MODULES))}")


def test_every_cli_module_is_covered_here():
    """The list above must not drift from the repo. A new CLI that nobody thought to add is
    precisely the one that will ship a traceback."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    found = set()
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if "\ndef main(" in path.read_text():
            found.add("hub." + str(path.relative_to(root).with_suffix("")).replace("/", "."))
    assert found == set(CLI_MODULES), (
        f"CLI_MODULES is stale. missing={sorted(found - set(CLI_MODULES))} "
        f"stale={sorted(set(CLI_MODULES) - found)}")


# --- module paths must be resolvable at call time ---------------------------
#
# `def f(path: Path = STATE)` binds the module constant when the function is DEFINED, so
# monkeypatching the module attribute never reaches it. A test that believes it redirected
# its output writes to the real file instead -- which happened on 2026-08-27, overwriting the
# live draft board with a 320-row fixture and the draft state with a test pick.

def test_no_module_level_path_is_bound_as_a_default_argument():
    """The whole class, not the two instances that were found."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    # Module-level names that are filesystem paths. Binding one as a default makes the
    # function write somewhere a caller cannot redirect.
    pathish = {"STATE", "BOARD_PARQUET", "OUT", "ROOT", "DATA", "CATALOG", "ARCHIVE",
               "AS_DRAFTED", "STORE"}
    bad = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for default in (*args.defaults, *(d for d in args.kw_defaults if d)):
                if isinstance(default, ast.Name) and default.id in pathish:
                    bad.append(f"{path.name}:{node.name}() default={default.id}")
    assert not bad, (
        "module-level paths bound as default arguments -- monkeypatching the module "
        f"attribute will not reach these, so tests silently write to the real file: {bad}")


def test_no_private_name_crosses_a_module_line():
    """A private name imported elsewhere is a seam without an address.

    `hub.draft.state._norm` was imported by ten modules across three packages -- two of them
    inside a function body, which is what a caller does when an import feels wrong. It is now
    `hub.names.player_key`, with a term in CONTEXT.md. `market._norm_cdf` went public at the
    same time. Nothing should reintroduce the pattern.
    """
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    bad = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hub"):
                bad += [f"{path.name} imports {node.module}.{a.name}"
                        for a in node.names if a.name.startswith("_")]
    assert not bad, f"private names imported across modules: {bad}"


def test_models_does_not_reach_into_draft():
    """The tree's one consistent direction is `draft -> models`, in six places. A module
    under `models/` importing `draft/` inverts it, and both instances that existed needed a
    function-local import to get away with it."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "models"
    bad = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hub.draft"):
                bad.append(f"{path.name} -> {node.module}")
    assert not bad, f"models/ reaches into draft/: {bad}"


def test_the_league_module_is_a_leaf():
    """`hub.league` may import `hub.config` and nothing else from this repo.

    That is the whole point of it. `STARTERS` and `starting_lineup` used to live in
    `hub.draft.season`, so anything wanting the roster shape imported a draft simulator --
    which is why `models/` could not have them at all (the test above) and why four of the six
    `hub.season` modules imported `draft/`. A leaf that grows a dependency stops being usable
    from the places that needed it.
    """
    import ast
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "league.py"
    bad = [n.module for n in ast.walk(ast.parse(path.read_text()))
           if isinstance(n, ast.ImportFrom) and n.module
           and n.module.startswith("hub.") and n.module != "hub.config"]
    assert not bad, f"hub.league must import only hub.config; it imports {bad}"


def test_season_does_not_reach_into_draft_for_league_rules():
    """`hub.season` sets lineups on real Sundays. It legitimately imports `draft/` to *simulate*
    drafts -- the weekly gate builds the rosters it scores that way -- but it must not import
    one to learn how many receivers it starts.

    `season -> draft` was twelve imports across four modules, five of them for `STARTERS`,
    `FLEX_*`, `starting_lineup` and `REG_SEASON_WEEKS`. Those are `hub.league` now. The seven
    that remain are `board`, `backtest`, `optimize` and `state`, which are genuinely the draft.
    """
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "season"
    bad = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.ImportFrom) and node.module == "hub.draft.season"):
                bad.append(f"{path.name} -> {node.module}")
    assert not bad, (
        f"season/ imports the draft's season simulator for league rules: {bad}. "
        f"The roster shape and `starting_lineup` are in `hub.league`.")


def test_nothing_reads_the_preds_view_around_the_one_reader():
    """`preds` holds several fitted versions of the same game on purpose, and every consumer
    that read it directly treated them as separate games.

    A raw `SELECT ... FROM preds` looks exactly like the correct code, which is why the rule
    is enforced here rather than offered as a flag on `store.predictions`. `hub.store` owns
    the query; anything genuinely wanting every version writes the SQL and is visible in
    review doing it.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    bad = []
    for path in root.rglob("*.py"):
        if path.name == "store.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bFROM\s+preds\b", line, re.IGNORECASE):
                bad.append(f"{path.relative_to(root)}:{i}")
    assert not bad, (
        f"these query `preds` directly instead of `store.predictions()`: {bad}. "
        f"That returns one row per fitted version, not one per game.")


# --- the general form of the defect above -------------------------------------
#
# Redirecting the two roots the fixture started with was not enough, and the way that failed
# is the point: the poller read a path nobody had thought about, its absent-input case passed
# by reaching a different branch, and the failure only became visible when the draft state
# changed underneath it. Any path constant added later can do the same thing silently.
#
# So every path under the repo root is accounted for here -- redirected by the fixture, or
# named with the reason it does not need to be. A new one fails this test until someone
# decides which it is.

REAL_ROOTS_NOT_REDIRECTED: dict[str, str] = {
    "ROOT": "the repo itself. Reading it is reading the source, which is the same on any "
            "clone and is what the CLIs are made of.",
    "SITE": "committed and in git, so it is identical on every clone -- reading it cannot "
            "make a verdict depend on the machine. `BOARD_JSON` under it *is* redirected, "
            "because an absent board is exactly what these cases are about.",
    "STATE_DIR": "committed, and read only by the odds fetcher's credit floor, which never "
                 "reaches it here -- the key is deleted, so it refuses upstream.",
    "DATA": "the parent of PROCESSED. Nothing binds it directly; the CLIs bind the leaves.",
    "ROSTER_PARQUET": "bound at import by `hub.season.roster` and `hub.publish`, and NOT "
                      "redirected. Its case is safe for a different reason: the CLI is driven "
                      "with `--out` at a tmp path, and `--out` moves the last-good *read* as "
                      "well as the write, so it reads an absent file. That is a real "
                      "dependency -- if `--out` ever stops moving the read, this case starts "
                      "reading the developer's own roster and this note is wrong.",
}


def test_every_path_under_the_repo_root_is_redirected_or_explained(a_fresh_clone, tmp_path):
    """A CLI must not be able to read the machine it runs on without someone deciding so.

    The fixture cannot redirect what it does not know about, and the modules that matter bind
    their paths at import -- so patching `hub.paths` reaches nothing. This holds the list
    itself: a constant added to `hub.paths` is either redirected on the module that binds it,
    or listed above with why it is safe to read for real.
    """
    from pathlib import Path

    from hub import paths as paths_mod
    from hub import store as store_mod
    from hub.draft import board as board_mod
    from hub.draft import state as state_mod

    redirected = {
        "BOARD_PARQUET": board_mod.BOARD_PARQUET,
        "BOARD_JSON": board_mod.BOARD_JSON,
        "PROCESSED": store_mod.DATA,
    }
    for name, value in redirected.items():
        assert tmp_path in Path(value).parents or Path(value) == tmp_path or \
               str(tmp_path) in str(value), (
            f"{name} still points at {value}, which is on the machine running the test. A CLI "
            f"reading it decides its own verdict from ambient state.")
    assert str(tmp_path) in str(state_mod.STATE), (
        "the draft state is not redirected, so a clone that has drafted and one that has not "
        "give different verdicts -- which is exactly how the poller's case inverted")

    declared = set(redirected) | set(REAL_ROOTS_NOT_REDIRECTED)
    actual = {n for n, v in vars(paths_mod).items() if n.isupper() and isinstance(v, Path)}
    assert actual <= declared, (
        f"new path constants in `hub.paths`: {sorted(actual - declared)}. Redirect each on the "
        f"module that binds it, or add it to REAL_ROOTS_NOT_REDIRECTED with why a CLI may read "
        f"the real one. The fixture cannot protect a path nobody listed.")
