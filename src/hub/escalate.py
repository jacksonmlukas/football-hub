"""Surface every published artifact that says `escalate: true` somewhere a person will look (#498).

**The finding.** `hub.fetch.cfbd` has set `escalate` since #424, and the slate's last step turns
it into a red run. That worked exactly once: the Wednesday 2026-10-14 run went red at "Escalate
a CFBD source that has stayed unfetched". The red run was the whole surface. The next scheduled
runs were the odds-only polls (a different job, which never reads `escalate`), the run list
showed green on top of the red one, and nothing was filed, so `escalate: true` sat in the
committed artifact for ten days with a single failed run as its only trace.

**The fix** is the pattern `watchdog-lookback` already uses for a state that should not be
missed: an open, labelled issue. This module only *decides* (which artifacts escalate, and what
the incident says); `slate.yml` files, updates, or closes the issue with `gh`. The incident
closes itself when no artifact escalates, because the source recovering is the only thing that
clears `escalate`.

Reads every `*.json` directly under the artifact directory and counts a top-level
`escalate: true` -- not the string "true", not a nested key. An unreadable file is skipped, not
fatal: this runs `always()` after a publish and must not be the thing that breaks a run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hub import atomic

MARKER = "<!-- artifact-escalate -->"


def escalated(site_data: Path) -> list[dict]:
    """The artifacts in `site_data` whose top-level `escalate` is the boolean true."""
    out = []
    for path in sorted(site_data.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("escalate") is True:
            out.append({"artifact": path.name, "since": doc.get("unfetched_since"),
                        "generated_at": doc.get("generated_at")})
    return out


def render(items: Sequence[dict]) -> str:
    lines = [MARKER, "An artifact on the page has escalated: its source has stayed unfetched "
             "past the warning window. Opened by `hub.escalate` from the slate run; it closes "
             "itself when no artifact escalates. The cause is in the artifact's `reason` and "
             "the run's log, not here: the artifact is committed and its reason may quote "
             "payload values.", ""]
    for it in items:
        lines.append(f"- `{it['artifact']}`: unfetched since {it['since'] or 'unknown'}, "
                     f"record written {it['generated_at'] or 'unknown'}.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.escalate",
        description="Print `escalated=N` for $GITHUB_OUTPUT; write the incident text to "
                    "--report when N > 0.")
    ap.add_argument("--site-data", type=Path, default=Path("site/data"))
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.site_data.is_dir():
        # Not "nothing escalated": a missing directory has established nothing, and the
        # workflow must not read silence here as a reason to close an open incident.
        print(f"hub.escalate: {args.site_data} is not a directory; nothing could be checked",
              file=sys.stderr)
        return 2
    items = escalated(args.site_data)
    print(f"escalated={len(items)}")
    if items and args.report:
        atomic.write_text(args.report, render(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
