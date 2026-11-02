"""`site/data/leaderboard.json` -- the interim LLM-forecaster leaderboard (#543).

`forecast-score.yml` (#453) writes `leaderboard/interim.json` to the **forecasts branch** and
commits nothing to main. The Study view fetches `data/leaderboard.json` and has nowhere else
to look, so until something copied the one into the other it showed its empty state forever.

**Where the copy happens: here, in the slate's publish.** The slate already checks out full
history (`fetch-depth: 0`, so `origin/forecasts` is present when the branch exists), commits
`site/data` to main, and `pages.yml` deploys that commit via `workflow_run`. So the
leaderboard rides the one deploy source that every other artifact uses. It needs no new
workflow permission and no write to another branch: this module only *reads* a ref.
`forecast-score.yml` stays as it is, writing to `forecasts` alone, and neither `pages.yml` nor
`live.yml` gains a trigger to race the other.

**Graceful degradation.** A scoring run that failed, or never ran, leaves the forecasts
branch's copy as it was; a branch that does not exist, a file that is missing, one that is not
JSON, or one with no contestants all yield `Kept`, and whatever was last published stands. The
copy is a straight relay of the scorer's document -- the contestants, `interim`, `label`,
`through_week` and the rest are the scorer's own fields, wrapped in the `summary` envelope. In
particular `min_games` is published only if the scorer wrote one: the prereg
(`docs/prereg/llm-forecasters.md`) defines no minimum game count for ranking, so nothing here
invents one.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hub import jsonio

ROOT = Path(__file__).resolve().parents[2]

NAME = "leaderboard"
REF = "origin/forecasts"
PATH = "leaderboard/interim.json"
SOURCE = f"forecasts:{PATH}"
COUNT_KEY = "n_contestants"


def interim_from_git(ref: str = REF, path: str = PATH, root: Path | None = None) -> Any:
    """The interim leaderboard as committed on `ref`, parsed; None if it cannot be read.

    Reads the git object and does not check anything out, so the working tree is untouched.
    Every failure -- no git, no such ref, no such file, not JSON -- is None, because the
    caller's answer to all of them is the same: keep what is published.
    """
    try:
        done = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=root or ROOT,
                              capture_output=True,
                              text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except ValueError:
        return None


def _usable(doc: Any) -> bool:
    # GUARD an-empty-or-malformed-source-is-not-published [unit/test_publish_leaderboard.py]:
    # deleting it lets an empty or malformed forecasts-branch file replace the published copy.
    return (isinstance(doc, dict) and isinstance(doc.get("contestants"), list)
            and bool(doc["contestants"]))
    # /GUARD


def leaderboard(out: Path | None = None,
                read: Callable[[], Any] | None = None) -> Any:
    """Write `leaderboard.json`, or keep last-good. Never raises over a bad source document.

    `read` is the seam for tests; unset it is `interim_from_git`, looked up at call time.
    """
    from hub import publish  # lazy: publish wires this module in, so the import is circular

    out = out or publish.SITE
    doc = (read or interim_from_git)()
    if not _usable(doc):
        return publish._keeping(
            out, NAME, "no usable interim leaderboard on the forecasts branch", COUNT_KEY)
    try:
        fields: dict[str, Any] = {COUNT_KEY: len(doc["contestants"]), **doc}
        payload = jsonio.summary(NAME, SOURCE, **fields)
    except (ValueError, TypeError) as exc:
        # A scorer document that restates an envelope key; refuse it rather than publish it.
        return publish._keeping(out, NAME, f"the interim leaderboard was refused ({exc})",
                                COUNT_KEY)
    return publish._publish(out, NAME, payload, count_key=COUNT_KEY)
