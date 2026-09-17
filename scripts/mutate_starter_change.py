"""Mutation testing for `hub.models.starter_change`, committed so the count can be re-taken.

`docs/audits/2026-09-16-audit-iv.json`'s `mutation_testing.starter_change` was audit IV's own
run -- fourteen mutants, seven survived, named in `.survivors` -- but it was ad hoc: "run by
the reviewer in a /tmp copy of the tree" (`.mutation_testing.note`), never committed, so #304's
second acceptance criterion is this script.

**Twelve mutants, not fourteen.** The audit's JSON counts `"mutants": 14, "survived": 7` but
its `.killed` list names only five of the seven it counted as killed, and no other record of
the run exists -- `docs/method.md`'s rule-15 note (2026-09-16) repeats the two counts and one
survivor's shape but adds no further names. The two missing killed mutants have no text
anywhere in the repo to reconstruct from, and inventing a plausible-looking pair would print a
count this script cannot stand behind (the same discipline `not_an_input` enforces on the
module's own constants). So this runs the twelve mutants the record actually names -- the
seven survivors, condition for condition, plus five of the seven killed -- and says so in its
own total rather than silently padding to fourteen.

**No mutation framework.** `pyproject.toml`'s `[dependency-groups]` carries none, and #304
asks not to add one. Each mutant here is one exact source-text substitution, applied straight
to `src/hub/models/starter_change.py`, `tests/unit/test_starter_change.py` run against the
mutated file with a plain subprocess, and the original text restored -- in a `finally`, so a
crash mid-run leaves the module exactly as it found it -- before the next mutant is tried. The
substitution's old text is required to appear exactly once in the unmutated file; a drifted
line or a rewritten function fails loudly instead of mutating the wrong spot or silently
mutating nothing.

    uv run python scripts/mutate_starter_change.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "hub" / "models" / "starter_change.py"
TEST = "tests/unit/test_starter_change.py"

# Shared between the postseason mutant's old and new text (unchanged either way) -- split
# across two literals only to stay under ruff's line-length bar, not because the source is.
_WEEK_GUARD_RAISE = (
    '        raise ValueError("the rows carry no \'week\' column, and the nflverse game id '
    'the "\n'
    '                         "archive is keyed by cannot be rebuilt without it")\n'
)

# (label, audit's own disposition ["survived" | "killed, named"], old text, new text). `old`
# is matched verbatim once against the real file. Ordered to match
# `docs/audits/2026-09-16-audit-iv.json`'s own `survivors` then `killed` lists.
MUTANTS: list[tuple[str, str, str, str]] = [
    # --- the seven named survivors -----------------------------------------------------
    (
        "_poll_day: the Eastern conversion deleted",
        "survived",
        '    return (pl.col(col).dt.replace_time_zone("UTC")\n'
        '              .dt.convert_time_zone(nfeloqb.GAME_DAY_ZONE).dt.date().cast(pl.Utf8))\n',
        '    return (pl.col(col).dt.replace_time_zone("UTC")\n'
        '              .dt.date().cast(pl.Utf8))\n',
    ),
    (
        "event_games: frozen_before .min() -> .max()",
        "survived",
        '                  pl.col("prev_date").min().alias("frozen_before"))\n',
        '                  pl.col("prev_date").max().alias("frozen_before"))\n',
    ),
    (
        "team_games: the postseason filter deleted",
        "survived",
        '    if "week" not in rows.columns:\n'
        + _WEEK_GUARD_RAISE +
        '    if "game_type" in rows.columns:\n'
        '        rows = rows.filter(pl.col("game_type") == REGULAR_SEASON)\n'
        '    week = pl.col("week").cast(pl.Utf8).cast(pl.Float64).cast(pl.Int64)\n'
        '    home = pl.col("team1").replace(nfeloqb.ABBREVIATIONS)\n',
        '    if "week" not in rows.columns:\n'
        + _WEEK_GUARD_RAISE +
        '    week = pl.col("week").cast(pl.Utf8).cast(pl.Float64).cast(pl.Int64)\n'
        '    home = pl.col("team1").replace(nfeloqb.ABBREVIATIONS)\n',
    ),
    (
        "pilot: abs() deleted",
        "survived",
        '"target": abs(statistics.fmean(means)) if means else float("nan"),',
        '"target": statistics.fmean(means) if means else float("nan"),',
    ),
    (
        "gate_rows: the not-null filter deleted",
        "survived",
        '    have = _adjusted(have, rows, tg).filter(pl.col("adjusted_by").is_not_null()\n'
        '                                            & pl.col("oracle_by").is_not_null())\n',
        '    have = _adjusted(have, rows, tg)\n',
    ),
    (
        "run: the NOT-RUNNABLE guard deleted",
        "survived",
        '    k = paired["season"].n_unique() if paired.height else 0\n'
        '    if k < EVENT_SEASONS_MINIMUM:\n'
        '        need = (f"the pilot says {needed} event-seasons are needed at 80% power"\n'
        '                if needed is not None else "the pilot cannot say how many '
        'event-seasons are "\n'
        '                                           "needed (no spread across seasons yet)")\n'
        '        verdict = (\n'
        '            "NOT-RUNNABLE",\n'
        '            f"NOT RUNNABLE: {k} event-season(s) in the archive against a '
        'pre-registered "\n'
        '            f"minimum of {EVENT_SEASONS_MINIMUM} (ADR-0019: no gate runs at fewer '
        'than three "\n'
        '            f"seasons); {need}. No verdict is read. Since #300 this '
        'diagnostic\'s NOT-RUNNABLE "\n'
        '            f"is an exemption from firing below the minimum, not a bar to the '
        'module\'s ADOPT "\n'
        '            f"condition, which is #221\'s line-move coefficient '
        '(docs/qb-adjustment.md); this "\n'
        '            f"is not-runnable, not a null.")\n'
        '        return experiment.GateRun(summary={}, seasons=pl.DataFrame(), verdict=verdict,\n'
        '                                  lines=[], stamped=paired)\n'
        '    arm = (experiment.Ceiling(CEILING_ARM, paired["ceiling"].to_numpy())\n',
        '    arm = (experiment.Ceiling(CEILING_ARM, paired["ceiling"].to_numpy())\n',
    ),
    (
        "events: the .sort() deleted",
        "survived",
        '    out = (starters.sort("team", "season", "week")\n'
        '                   .with_columns(shifted)\n',
        '    out = (starters\n'
        '                   .with_columns(shifted)\n',
    ),
    # --- five of the seven the audit counted as killed (see the module docstring above) -
    (
        "_adjusted: the shipped and oracle arms swapped",
        "killed",
        '        out.append(_moved(_moved(part, shipped, "adjusted"), known, "oracle"))\n',
        '        out.append(_moved(_moved(part, known, "adjusted"), shipped, "oracle"))\n',
    ),
    (
        "_last_before: < changed to <=",
        "killed",
        '                   .filter(pl.col("poll_day") < pl.col(day))\n',
        '                   .filter(pl.col("poll_day") <= pl.col(day))\n',
    ),
    (
        "event_games: net_gap sign flipped",
        "killed",
        '              .with_columns((pl.col("home_gap").fill_null(0.0)\n'
        '                             - pl.col("away_gap").fill_null(0.0)).alias("net_gap"))\n',
        '              .with_columns((pl.col("away_gap").fill_null(0.0)\n'
        '                             - pl.col("home_gap").fill_null(0.0)).alias("net_gap"))\n',
    ),
    (
        "study_fit: the residual's SE used instead of the floor's",
        "killed",
        '    se = floor_window / (sd_gap * math.sqrt(n - 1)) if sd_gap > 0 else nan\n',
        '    _intercept = float(np.polyfit(x, y, 1)[1]) if sd_gap > 0 else nan\n'
        '    _resid = y - beta * x - _intercept\n'
        '    se = (float(np.std(_resid, ddof=2)) / (sd_gap * math.sqrt(n - 1))\n'
        '          if sd_gap > 0 and n > 2 else nan)\n',
    ),
    (
        "_change_points: days measured from the frozen poll, not frozen_before",
        "killed",
        '                seen = float((dt.date.fromisoformat(p["poll_day"])\n'
        '                              - dt.date.fromisoformat(r["frozen_before"])).days)\n',
        '                seen = float((dt.date.fromisoformat(p["poll_day"])\n'
        '                              - dt.date.fromisoformat(r["frozen_day"])).days)\n',
    ),
]


def _apply(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected the mutant's old text exactly once in "
                         f"{MODULE.name}, found {count} -- the source has drifted from what "
                         f"this script was written against")
    return text.replace(old, new, 1)


def _run_tests() -> bool:
    """True if the unit suite passes against whatever is currently on disk at MODULE."""
    got = subprocess.run(["uv", "run", "pytest", TEST, "-q"], cwd=ROOT,
                         capture_output=True, text=True)
    return got.returncode == 0


def main() -> int:
    original = MODULE.read_text()
    if not _run_tests():
        print(f"  {TEST} does not pass against the unmutated module; refusing to mutate a "
              f"tree that is not green", file=sys.stderr)
        return 2

    results: list[tuple[str, str, str]] = []       # (label, audit disposition, outcome)
    for label, disposition, old, new in MUTANTS:
        try:
            MODULE.write_text(_apply(original, old, new, label))
            passed = _run_tests()
        finally:
            MODULE.write_text(original)
        outcome = "KILLED" if not passed else "SURVIVED"
        results.append((label, disposition, outcome))
        print(f"  {outcome:8} ({disposition:8}) {label}")

    survivors = [r for r in results if r[2] == "SURVIVED"]
    print(f"\n  {len(results) - len(survivors)}/{len(results)} killed, {len(survivors)} "
          f"survived, of the twelve mutants the audit's record names by text (it counted "
          f"fourteen total and seven survived; the other two killed mutants have no recorded "
          f"text to reproduce -- see this script's module docstring)")
    if survivors:
        print("\n  survivors:")
        for label, disposition, _outcome in survivors:
            print(f"    - {label} (audit: {disposition})")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
