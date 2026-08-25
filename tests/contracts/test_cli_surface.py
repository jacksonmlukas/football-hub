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
"""
import importlib

import pytest

# Every module with a CLI. A new one missing from this list is caught by
# `test_every_cli_module_is_covered_here` below rather than by nobody.
CLI_MODULES = (
    "hub.draft.backtest", "hub.draft.board", "hub.draft.calibrate", "hub.draft.evaluate",
    "hub.draft.leverage", "hub.draft.live", "hub.draft.tune", "hub.fetch.cfbd",
    "hub.fetch.nflverse", "hub.fetch.odds", "hub.inspect", "hub.models.conformal",
    "hub.models.correlate", "hub.models.eval", "hub.models.injury", "hub.models.margin",
    "hub.models.ratings", "hub.models.spread", "hub.publish", "hub.season.lineup",
    "hub.season.lineup_gate", "hub.season.survivor", "hub.store",
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


# The paths that actually touch local state. Each is run with its input absent, which is the
# case that produced every bug this file exists for.
ABSENT_INPUT = [
    ("hub.season.lineup", ["--opp-mu", "110", "--roster", "{tmp}/nope.parquet"]),
    ("hub.inspect", ["{tmp}/nope"]),
    ("hub.models.conformal", ["--recalibrate", "--store", "{tmp}"]),
    ("hub.models.eval", ["--compare", "a,b", "--store", "{tmp}"]),
]


@pytest.mark.parametrize("name,argv", ABSENT_INPUT)
def test_absent_input_is_reported_not_raised(name, argv, tmp_path, capsys):
    mod = importlib.import_module(name)
    filled = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    try:
        code = mod.main(filled)
    except SystemExit as e:                      # argparse's own exit is a sentence too
        code = e.code
    assert isinstance(code, int) and code != 0, f"{name} should fail, and say so"
    out = capsys.readouterr()
    assert "Traceback" not in (out.out + out.err)
    assert (out.out + out.err).strip(), f"{name} failed silently, which is worse"


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
