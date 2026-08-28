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
