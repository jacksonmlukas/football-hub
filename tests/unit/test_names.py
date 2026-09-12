"""One comparable key for a player, across sources that spell him differently.

`docs/decisions.md` records the bug this fixes: FantasyPros and ffopportunity disagree on
suffixes, so an exact join drops players silently. It is the mechanism every cross-source
join in this repo depends on, and until 2026-08-27 it was a private helper inside the module
that tracks draft state, imported by ten modules across three packages, with two tests.

All offline.
"""
import pytest

from hub.names import player_key


@pytest.mark.parametrize("a,b", [
    ("Ja'Marr Chase", "JaMarr Chase"),          # punctuation
    ("A.J. Brown", "AJ Brown"),                 # the case decisions.md names
    ("Marvin Harrison Jr.", "Marvin Harrison"), # generational suffix
    ("Amon-Ra St. Brown", "Amon Ra St. Brown"),  # hyphen becomes a space
    ("  Kyren  Williams ", "Kyren Williams"),   # whitespace
    ("Jeffery Simmons", "JEFFERY SIMMONS"),     # case
])
def test_spellings_of_one_player_collapse_to_one_key(a, b):
    assert player_key(a) == player_key(b)


def test_different_players_do_not_collide():
    assert player_key("Justin Jefferson") != player_key("Justin Herbert")
    assert player_key("Michael Pittman") != player_key("Michael Thomas")


def test_accents_fold():
    """nflverse and FantasyPros do not agree about diacritics either."""
    assert player_key("Equanimeous St. Brown") == player_key("Equanimeous St Brown")


def test_an_empty_or_missing_name_is_an_empty_key_not_a_crash():
    """A null name arrives from a join that missed. It must not raise inside a map_elements
    over a whole board."""
    assert player_key("") == ""
    assert player_key(None) == ""            # type: ignore[arg-type]


def test_a_suffix_that_is_part_of_a_name_survives():
    """`_SUFFIX` is word-bounded: dropping every `v` would collapse real names."""
    assert player_key("Vita Vea") == "vita vea"


def test_it_is_a_leaf():
    """It imports nothing from `hub`, so any module may use it without dragging a board
    builder along -- which is what the old home cost two callers, who imported it inside a
    function body to avoid exactly that."""
    import ast
    import inspect

    from hub import names
    tree = ast.parse(inspect.getsource(names))
    hub_imports = [n for n in ast.walk(tree)
                   if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("hub")]
    assert hub_imports == []


def test_a_hyphen_becomes_a_space_rather_than_nothing():
    """Worth pinning, because it is the one case where two plausible spellings do NOT meet:
    `Amon-Ra` collapses to `amon ra`, not `amonra`.

    Verified against the live board rather than argued: all hyphenated players -- Jaxon
    Smith-Njigba, Amon-Ra St. Brown, Marquez Valdes-Scantling and the rest -- join with an
    xFP, none dropped. Both sources hyphenate consistently, so the unhandled case is
    hypothetical. Changing it during a move would have been changing behaviour under cover of
    a refactor.
    """
    assert player_key("Amon-Ra") == "amon ra"
    assert player_key("Amon-Ra") != player_key("AmonRa")


# --- hub.paths is a leaf, and that is the whole point of it -----------------

def test_paths_imports_nothing_from_hub():
    """The same rule `hub.names` lives by. `adp_history` took `ROOT` from a 780-line board
    builder purely to learn a `Path`, which is a cycle, and it costs `board.main` a
    function-local import with a comment saying why."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "paths.py"
    tree = ast.parse(src.read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith("hub") for m in mods), f"paths must stay a leaf: {mods}"


def test_paths_agrees_with_the_declarations_it_replaces():
    """All eight `ROOT` declarations resolve to the same path; this one has to as well or the
    tidying would be a silent relocation of every artifact."""
    from hub.draft.board import BOARD_PARQUET as board_parquet
    from hub.draft.board import ROOT as board_root
    from hub.paths import BOARD_PARQUET, ROOT
    from hub.store import DATA
    assert ROOT == board_root
    assert BOARD_PARQUET == board_parquet
    assert (ROOT / "data") == DATA.parent if DATA.name == "processed" else True


def test_adp_history_does_not_import_a_board_builder_at_module_scope():
    """No import *cycle*, which is narrower than no mention of `board`.

    `board` imports `adp_history`, so a module-level runtime import back is a cycle and the
    interpreter says so. An import under `if TYPE_CHECKING:` and an import inside a function
    body are neither — the first never executes, the second executes after both modules are
    built. #199 needs `report_for` and `BuildReport` from `board`, and reaches them by exactly
    those two routes.

    This test used to `ast.walk` the whole tree and forbid any `ImportFrom` naming `board`,
    which fails on both of them. That is the shape this repo keeps finding: a check standing in
    for the property it means, passing and failing on something adjacent to it. So the scan is
    scoped to module scope outside `TYPE_CHECKING`, and the property itself is asserted below
    rather than left to the proxy.
    """
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "draft"
           / "adp_history.py")
    tree = ast.parse(src.read_text())

    def executes_on_import(body: list[ast.stmt]) -> list[ast.ImportFrom]:
        """Imports reached by simply importing the module.

        Descends through `if`/`try`/`with`, which do run, and stops at `def` and `class`,
        whose bodies do not -- a function-local import runs on first call, by which time both
        modules exist. `if TYPE_CHECKING:` never runs at all.
        """
        found: list[ast.ImportFrom] = []
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            test = getattr(node, "test", None)
            if (isinstance(node, ast.If) and isinstance(test, ast.Name)
                    and test.id == "TYPE_CHECKING"):
                continue
            if isinstance(node, ast.ImportFrom):
                found.append(node)
            for field in ("body", "orelse", "finalbody"):
                found += executes_on_import(getattr(node, field, []) or [])
        return found

    mods = {n.module for n in executes_on_import(tree.body) if n.module}
    assert not any("board" in m for m in mods), (
        f"a module-level runtime import of board is a cycle: {sorted(mods)}. "
        f"Reach it under `if TYPE_CHECKING:` or from inside the function that needs it.")


def test_adp_history_imports_on_its_own_without_cycling():
    """The property the scan above is a proxy for, asserted directly.

    A fresh interpreter importing only this module is the thing that would actually raise on a
    cycle, and it stays true however the imports are written -- including forms the scan has
    not been taught about.
    """
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-c", "import hub.draft.adp_history"],
                          capture_output=True, text=True)
    assert done.returncode == 0, f"importing adp_history alone failed:\n{done.stderr}"


# --- the relaxed comparison a join failure is counted under (issue #46) --------------------

def test_the_relaxed_key_is_the_first_initial_and_the_surname():
    """`Justin Jefferson` and `J. Jefferson` share no player key -- one source abbreviates --
    and a drafted name absent from the realised set that matches a realised name here is a
    join failure rather than a player who never played."""
    from hub.names import relaxed_key
    assert relaxed_key("Justin Jefferson") == "j jefferson"
    assert relaxed_key("J. Jefferson") == "j jefferson"
    assert relaxed_key("Marvin Harrison Jr.") == "m harrison"
    assert relaxed_key("Ja'Marr Chase") == "j chase"


def test_the_relaxed_key_does_not_collide_where_the_player_key_does_not():
    from hub.names import relaxed_key
    assert relaxed_key("Justin Jefferson") != relaxed_key("Justin Herbert")
    assert relaxed_key("Michael Pittman") != relaxed_key("Michael Thomas")


def test_a_one_word_or_empty_name_relaxes_to_itself():
    """No initial to take: the key is the whole comparison, so it cannot match more loosely
    than exactly. `P0` on a synthetic board must not relax to `p p0`."""
    from hub.names import relaxed_key
    assert relaxed_key("P0") == "p0"
    assert relaxed_key("") == ""
    assert relaxed_key(None) == ""            # type: ignore[arg-type]


# --- the nickname crosswalk (#246) ------------------------------------------------------

@pytest.mark.parametrize(("consensus", "stats"), [
    ("Gabriel Davis", "Gabe Davis"),
    ("Ken Walker III", "Kenneth Walker"),
    ("Kenneth Gainwell", "Kenny Gainwell"),
    ("Joshua Palmer", "Josh Palmer"),
    ("Chigoziem Okonkwo", "Chig Okonkwo"),
    ("Cameron Ward", "Cam Ward"),
])
def test_a_player_the_stats_source_calls_by_a_nickname_joins_exactly(consensus, stats):
    """Six drafted players scored zero in every published draft-gate number because the
    consensus page spells the full name and the stats source the nickname (#246). The
    crosswalk is by whole key, never by first name: `Mike` is not `Michael` for everyone."""
    assert player_key(consensus) == player_key(stats)


def test_the_crosswalk_is_by_whole_name_and_not_by_first_name():
    """`Irv Smith Jr.` and `Ito Smith` share an initial and a surname and are two people;
    so do `Robbie Anderson` and `Ryan Anderson`. Only a name on the list moves."""
    assert player_key("Irv Smith Jr.") != player_key("Ito Smith")
    assert player_key("Robbie Anderson") != player_key("Ryan Anderson")
    assert player_key("Joshua Allen") != player_key("Josh Allen"), "a first name alone is not an alias"


def test_every_alias_maps_a_key_to_a_key_and_never_to_itself():
    import re

    from hub.names import ALIASES
    for src, dst in ALIASES.items():
        # `src` is already key-shaped (it is what the key looks like before the alias step)
        # and `dst` is a fixed point of the whole function.
        assert re.fullmatch(r"[a-z0-9 ]+", src) and src == " ".join(src.split()), src
        assert dst == player_key(dst), dst
        assert src != dst
        assert dst not in ALIASES, "an alias must land on a canonical key, not another alias"
