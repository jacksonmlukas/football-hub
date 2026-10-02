"""Every nflverse read in `src/hub` goes through the loader, and nothing else imports `nflreadpy`.

A published gate number that rests on an unpinned fetch cannot say what data produced it, and
#71 made that concrete: every gate now stamps a `data_digest` folded from what the run actually
loaded. A call site that goes straight to `nflreadpy` contributes nothing to that digest, so the
stamp would quietly describe less than the run read -- provenance that looks present and is
partial, which is worse than none.

**The whole tree, not a list.** This used to name the modules that had been routed -- "a module
joins it when it is routed and never leaves" -- because thirteen still imported `nflreadpy`
directly and a list claiming the tree would have had to be a list of exceptions. #398 routed the
last of them, research and exhibit modules included, so the rule is the plain one: the loader
(`hub/fetch/nflverse.py`, whose `Network` adapter is the first of two at the seam) is the only
file that may import it. There is no exception list to grow, and the premise test below fails if
the loader itself ever stops being the one place the import lives.

The scan is `direct_importers`, in `tests/nflverse_routing.py` for the reason the other shared
scans are top-level modules: pyrefly cannot resolve an import between two test modules here.
"""
from __future__ import annotations

from pathlib import Path

from nflverse_routing import LOADER, direct_importers

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"


def test_nothing_under_src_imports_nflreadpy_except_the_loader():
    """The AST, not a text search: this file names the import it forbids, and a substring check
    would match its own prose the way `test_every_optional_stage_goes_through_the_helper`
    found `report.adp = True` inside a docstring explaining why it no longer exists."""
    found = {p: lines for p, lines in direct_importers(SRC).items() if p != LOADER}
    assert not found, (
        "these modules import nflreadpy directly, which contributes nothing to the run's data "
        "digest, so the published stamp describes less data than the run actually read: "
        + "; ".join(f"src/hub/{p} line(s) {lines}" for p, lines in sorted(found.items()))
        + ". Read it through `hub.fetch.nflverse.load` -- declare a `Source` if it is not one.")


def test_the_scan_sees_the_one_import_that_is_allowed():
    """The premise. A scan that matched nothing would make the test above vacuously true --
    the failure mode of every guard of this kind -- so the loader's own import is asserted
    found: the day it stops being (a rename of the file, a rewrite of the adapter) this says so
    rather than passing over an empty set."""
    assert LOADER in direct_importers(SRC)


def test_a_planted_direct_import_is_found(tmp_path):
    """Rule 18: the check is trusted only once it has been shown to fail. Every shape an import
    of the library takes, planted in a tree that looks like the real one, and each is found."""
    for name, body in {
        "models/plain.py": "import nflreadpy\n",
        "models/aliased.py": "def f():\n    import nflreadpy as nfl\n    return nfl\n",
        "models/from_form.py": "from nflreadpy import load_pbp\n",
        "models/dynamic.py": "import importlib\nx = importlib.import_module('nflreadpy')\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (tmp_path / "models" / "clean.py").write_text("import polars as pl\n")
    found = direct_importers(tmp_path)
    assert sorted(found) == ["models/aliased.py", "models/dynamic.py",
                             "models/from_form.py", "models/plain.py"]
