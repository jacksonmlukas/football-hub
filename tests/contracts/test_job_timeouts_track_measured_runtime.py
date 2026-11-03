"""A CI job's `timeout-minutes` is at most twice its measured median wall time (#514 item 4).

A timeout is a backstop for a hang, and a backstop set at 60 when the job takes 12 reports a
deadlock an hour late. #514's td_dispersion deadlock held every CI run at 98% until a long limit
cancelled it; the limit was generous enough that the hang looked like slowness. A job with no
`timeout-minutes` runs to GitHub's six-hour default.

The medians cannot come from the network at test time, so they are a declared table below, each
row naming the run IDs it was measured on (GitHub run metadata via `gh run view --json jobs`,
2026-11-03; wall time is the job's `startedAt` to `completedAt`). Re-measure and edit the row
when a job's work changes; the contract then moves the limit with it.

The limit is `floor(2 x median)` whole minutes, never below 1 -- GitHub's smallest timeout.
Sub-minute jobs therefore sit at 1 or 2 minutes, a thin margin over setup and checkout; that is
what the rule yields, and `MEASURED` is where to argue a different median.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

_HB = (41456396719, 41394901187, 41378546050, 41361501210, 41344315616, 41297953335, 41268502261,
       41261353332, 41249857640, 41235152020)

# (workflow file, job id) -> (median minutes, run IDs the median is over)
MEASURED: dict[tuple[str, str], tuple[float, tuple[int, ...]]] = {
    ("bigten.yml", "capture"): (0.6, (41286291080, 41254547807, 41232414084, 41179734500, 41020617401,
                                      40835433196, 40649065115, 40169312667, 40140398622, 40117804131)),
    # 11:04 and 13:37 on the runner, after the td_dispersion spawn fix; earlier green runs predate it.
    ("ci.yml", "test"): (12.3, (41571184996, 41572986000)),
    ("ci.yml", "golden"): (0.8, (38748031945,)),  # scheduled, so few runs exist
    ("consensus.yml", "capture"): (0.6, (41209338422, 41188232904, 41149674767, 41106558228, 40093918432,
                                         40075976244, 40041366074, 39997009964, 39705561405, 39670061859)),
    ("forecast-dry-run.yml", "dry-run"): (2.9, (41377255960, 41282863357)),
    ("live.yml", "refresh"): (110.6, (41520258028, 41437249634, 41382883510, 41361108727, 41294812198,
                                      41281208715, 41094769483, 41044254374, 41002046921, 40920337221)),
    ("odds-close.yml", "capture"): (0.6, (41370591781, 40247829681, 40229619925, 40228037089, 39179226102,
                                          39162158052, 39162111273, 39160606999, 39146554920)),
    ("pages.yml", "deploy"): (0.6, (41576137422, 41575633153, 41575226007, 41538271879, 41536900077,
                                    41536167512, 41535424652, 41534690498, 41533936881, 41533193947)),
    ("screenshots.yml", "screenshots"): (1.1, (41554023291, 41539933677, 41538280405, 41378483424, 41338622870,
                                               41337696478, 41331587278, 41330621005, 41317158235, 41316222596)),
    ("slate.yml", "refresh"): (1.2, (41229277416, 40731107930, 40115540589)),
    ("slate.yml", "poll_odds"): (0.6, (41520104328, 41335155348, 41094693191, 40920217231, 40364800646,
                                       40213797264, 39980463640)),
    ("watchdog-lookback.yml", "gaps"): (1.2, (41510543032, 41324339275, 41214453924, 41085439726, 40908779045,
                                              40722554809, 40198942151, 40097777265, 39968115236, 39796467423)),
    ("watchdog.yml", "heartbeat"): (0.1, _HB),
    ("watchdog.yml", "bigten"): (0.2, _HB),
}

# Jobs with a timeout and no honest median yet. Each names why; the contract still demands the
# timeout exist, and fails when the job also appears in MEASURED.
UNMEASURED: dict[tuple[str, str], str] = {
    ("forecast.yml", "forecast"): "gated on LLM_FORECASTS_ENABLED; the only green runs (0.2 min) forecast nothing",
    ("forecast-close.yml", "capture"): "gated on LLM_FORECASTS_ENABLED; the only green runs (0.2 min) captured nothing",
    ("forecast-score.yml", "score"): "gated on LLM_FORECASTS_ENABLED; no green run exists",
}

# Jobs allowed to have no timeout. Empty on purpose: a job with none runs for six hours.
NO_TIMEOUT_OK: dict[tuple[str, str], str] = {}


def limit_minutes(median: float) -> int:
    return max(1, math.floor(2 * median))


def problems(workflows: Path, measured=MEASURED, unmeasured=UNMEASURED, no_timeout=NO_TIMEOUT_OK) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(workflows.glob("*.yml")):
        for job, spec in (yaml.safe_load(path.read_text()).get("jobs") or {}).items():
            key = (path.name, job)
            seen.add(key)
            timeout = spec.get("timeout-minutes")
            if timeout is None:
                if key not in no_timeout:
                    out.append(f"{key}: no timeout-minutes (GitHub default is 360)")
            elif key in measured:
                median = measured[key][0]
                if timeout > limit_minutes(median):
                    out.append(f"{key}: timeout-minutes {timeout} > {limit_minutes(median)} (2 x median {median})")
            elif key not in unmeasured:
                out.append(f"{key}: timeout-minutes {timeout} but no row in MEASURED or UNMEASURED")
    for key in (*measured, *unmeasured, *no_timeout):
        if key not in seen:
            out.append(f"{key}: declared here but no such job")
    out += [f"{key}: in both MEASURED and UNMEASURED" for key in unmeasured if key in measured]
    return out


def test_every_job_timeout_is_within_twice_its_measured_median():
    assert problems(WORKFLOWS) == []


@pytest.mark.parametrize("key", sorted(MEASURED))
def test_each_median_names_its_runs(key):
    median, runs = MEASURED[key]
    assert median > 0 and runs and len(set(runs)) == len(runs)


def _wf(tmp_path, timeout_line: str):
    (tmp_path / "w.yml").write_text(f"jobs:\n  j:\n    runs-on: x\n{timeout_line}    steps: []\n")
    return tmp_path


ROW = {("w.yml", "j"): (12.0, (1,))}


def test_planted_over_limit_is_red(tmp_path):
    assert problems(_wf(tmp_path, "    timeout-minutes: 25\n"), ROW, {}, {}) != []
    assert problems(_wf(tmp_path, "    timeout-minutes: 24\n"), ROW, {}, {}) == []


def test_planted_missing_timeout_is_red(tmp_path):
    assert problems(_wf(tmp_path, ""), {}, {}, {}) != []
    assert problems(_wf(tmp_path, ""), {}, {}, {("w.yml", "j"): "argued"}) == []


def test_planted_unregistered_job_is_red(tmp_path):
    assert problems(_wf(tmp_path, "    timeout-minutes: 5\n"), {}, {}, {}) != []
