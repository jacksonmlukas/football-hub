"""Where the dashboard's script lives, for the tests that read it as text.

Since #476 the page's script is native ES modules under `site/js/`, not one inline block, so a
test that scans the script for a property (every feed value escaped, one `panel("p-slate"`
exit, the keys the page reads off each artifact) reads all of them. Tests that *run* the code
import the modules in node instead; see `tests/contracts/test_dashboard_escapes.py`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def js_files(site: Path = SITE) -> list[Path]:
    """Every module the page loads, in a stable order."""
    return sorted((site / "js").rglob("*.js"))


def script_text(site: Path = SITE) -> str:
    """All the page's script as one string, for a scan that does not care which file a line is in."""
    return "\n".join(f.read_text() for f in js_files(site))
