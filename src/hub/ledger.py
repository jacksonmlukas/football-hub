"""The Ledger: one seam between a gate's run and the interval-width history it compares
against, so a caller reads `record(entry) -> Comparison` rather than a file's shape (#385).

`experiment.review_width` (86 lines) read `state/gate-width.json`, picked the comparable
entry, built the narrowing sentence, built the entry and wrote the file back -- one function
whose interface was the file: its callers had to know the path, the write, the JSON's shape
and the "comparable at an identical digest" rule, and nine tests read the file back with
`json.loads` to assert on it. Two defects of 2026-09-21 were exactly that shape -- the
unreadable-file overwrite and the cross-digest comparison (`docs/method.md`'s rule 18 names
both as this module's required controls, in `tests/unit/test_ledger.py`).

**The key is one value.** `WidthEntry.key` is `(name, recipe, config_digest, data_digest)`,
and two entries compare iff their keys are equal *and* both recipes are known --
`WidthEntry.comparable` is the one line that answers both the digest condition (2026-09-21)
and #384 (a recipe carries the run's arm, so two recipes at one digest pair do not compare).
`recipe` defaults to `None`, which is a *known* recipe -- "no arm declared" -- and compares
equal to another `None`; that is what keeps every caller that has not adopted #384 yet (all
of them, today) reading exactly the comparison `review_width` gave them. `known=False` is
reserved for an entry read off disk with no `"recipe"` key in it at all -- every entry ever
written before this module existed -- and a `known=False` entry never compares, not even
against another one: `state/README.md` says why.

**What the Ledger owns.** The equality on `key` that decides comparability; the append; the
atomic write (`hub.atomic`, so a killed process leaves the previous file exactly as it was);
the refusal to write over a file that is present but does not parse (CLAUDE.md's degradation
rule: a gate that cannot read its own history still has a verdict, and the history is not
destroyed on the way to reporting that); and the pre-#362 dict shape on read, so a file this
module has not yet rewritten does not read as empty.

A leaf with respect to `hub.models.experiment`: it takes `hub.atomic`, `hub.jsonio` and
`hub.paths` and nothing from `hub.models`, so `experiment.run_gate` can hold a `Ledger`
without a cycle. `narrowing`'s wording (the sentence, not the decision) is still
`experiment`'s own -- read back in here through a function-local import inside `record`,
the same pattern `experiment._stats` and `experiment.stamped_for_publication` already use
to keep a leaf a leaf while still sharing one sentence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

from hub import atomic
from hub.jsonio import stamp as _now
from hub.paths import STATE_DIR

# Same path `experiment.WIDTH_STATE` names -- under `state/` with the odds poller and the
# CFBD quota, not `data/processed/`, which is measured output rather than a small fact one
# run leaves the next.
WIDTH_STATE = STATE_DIR / "gate-width.json"


@dataclass(frozen=True)
class WidthEntry:
    """One gate run's interval width, and the key that decides what it may be compared
    against.

    `name`, `recipe`, `config_digest` and `data_digest` are the key (`.key`); `width`,
    `clusters`, `lo`, `hi`, `verdict` and the optional `seasons` (#382's per-season records,
    additive since it predates this module) are what the run measured. `timestamp` is filled
    in by `Ledger.record` when left `None`, so a caller never has to know the clock's own
    format to construct one.
    """

    name: str
    config_digest: str
    data_digest: str
    width: float
    clusters: float
    lo: float
    hi: float
    verdict: str
    recipe: str | None = None
    seasons: list[dict] | None = None
    timestamp: str | None = None
    # False only for an entry read off disk with no `"recipe"` key at all -- every entry
    # written before this module existed. A fresh `WidthEntry` a caller constructs is always
    # `known=True`, even when `recipe` is `None`: "no arm declared" is a known fact about the
    # run, not a hole in what was read back.
    known: bool = True

    @property
    def key(self) -> tuple[str, str | None, str, str]:
        """`(name, recipe, config_digest, data_digest)` -- #385's key, #384's `recipe` in it."""
        return (self.name, self.recipe, self.config_digest, self.data_digest)

    def comparable(self, other: WidthEntry) -> bool:
        """Two entries compare iff their keys are equal and both recipes are known.

        The digest condition (2026-09-21) and #384 (a recipe carries the run's arm) were two
        checks on two different histories; here they are the one line above, because both are
        the same fact -- a width measured under a different key is a width measured on a
        different question, and a width whose key nobody recorded is a width nobody can say
        that about either way.
        """
        return self.known and other.known and self.key == other.key

    def _as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "gate": self.name, "recipe": self.recipe,
            "config_digest": self.config_digest, "data_digest": self.data_digest,
            "timestamp": self.timestamp, "width": self.width, "clusters": self.clusters,
            "lo": self.lo, "hi": self.hi, "verdict": self.verdict,
        }
        if self.seasons is not None:
            out["seasons"] = self.seasons
        return out

    @staticmethod
    def _from_dict(d: dict) -> WidthEntry:
        """The read side of `_as_dict`, plus the two backward-compatible shapes.

        A dict with no `"recipe"` key at all -- every entry `review_width` ever wrote, and the
        pre-#362 one-record-per-gate shape below it -- reads `known=False`: of an unknown
        recipe, named in the ledger and never compared, per `state/README.md`. A dict that
        does carry the key, even as `null`, reads it as a known `None` -- "no arm declared" --
        which is what lets a run that has not adopted #384 continue comparing the way
        `review_width` always did.
        """
        known = "recipe" in d
        width = d.get("width")
        return WidthEntry(
            name=str(d.get("gate", "")),
            config_digest=str(d.get("config_digest", "")),
            data_digest=str(d.get("data_digest", "")),
            width=float(width) if isinstance(width, int | float) else float("nan"),
            clusters=float(d.get("clusters", 0) or 0),
            lo=float(d.get("lo", float("nan"))),
            hi=float(d.get("hi", float("nan"))),
            verdict=str(d.get("verdict", "")),
            recipe=d.get("recipe"),
            seasons=d.get("seasons"),
            timestamp=d.get("timestamp"),
            known=known,
        )


class Comparison(NamedTuple):
    """What `Ledger.record` hands back: the entry as recorded, what it compared against (if
    anything), and the lines a reader gets. `run_gate` prints `.lines`; nothing else reads
    `state/gate-width.json` to learn what a run decided -- this is the interface.
    """

    entry: WidthEntry              # the entry as written (or as it would have been, unwritten)
    previous: WidthEntry | None    # the most recent comparable entry, or none
    elsewhere: int                 # earlier entries of this gate that did not compare, and why
    ratio: float                   # width / previous.width; NaN with no comparison
    requires_review: bool
    lines: list[str]


class Ledger:
    """Where a gate's interval-width history is kept, read back, and compared against.

    `path=None` is the in-memory adapter: entries live in `self._entries` for the life of the
    object and nothing touches disk, which is what a test injects in place of a tmp file --
    `tests/unit/test_gate_run.py` shares one `Ledger()` across two `run_gate` calls the same
    way two runs of a real gate share one `state/gate-width.json`. `write=False` keeps the
    file-backed reads and comparisons but skips the append -- `hub.draft.backtest`'s noise
    sweep reads this way, deliberately: a sensitivity sweep over `opp_noise` is not a run of
    the gate whose history it would otherwise crowd.
    """

    def __init__(self, path: Path | None = WIDTH_STATE, *, write: bool = True,
                places: int = 2):
        self.path = path
        self.write = write
        self.places = places
        self._entries: list[WidthEntry] = []

    def _read(self) -> list[WidthEntry] | None:
        """Every entry, oldest first, or `None` for present-and-unreadable.

        Never raises: CLAUDE.md's degradation rule, restated for this module -- a gate that
        cannot read its own history still has a verdict to report, and what it loses is one
        comparison line. Reads the pre-#362 dict shape (`{gate: {width, ...}}`, one record per
        gate) as entries with no `"recipe"` key, which is `WidthEntry._from_dict`'s `known`
        rule doing the "old shape is of unknown recipe" work on its own.
        """
        if self.path is None:
            return list(self._entries)
        if not self.path.exists():
            return []
        try:
            got = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        if isinstance(got, dict) and isinstance(got.get("entries"), list):
            return [WidthEntry._from_dict(e) for e in got["entries"] if isinstance(e, dict)]
        if isinstance(got, dict):
            return [WidthEntry._from_dict({"gate": gate, **rec})
                    for gate, rec in got.items() if isinstance(rec, dict)]
        return []

    def record(self, entry: WidthEntry) -> Comparison:
        """Compare `entry` against this gate's most recent comparable one, and append it.

        The comparison is against the *most recent* previous entry with an equal, known key,
        read back to front so three gates' histories interleaved in one file (or one
        in-memory list) do not confuse each other, and so a gate run a third time compares
        against its second run, not its first. Earlier entries that share the name but not
        the full key are counted as `elsewhere` and named in a line rather than silently
        dropped -- a width against a different config, data, or recipe is not a narrowing.

        An unreadable ledger costs exactly the comparison line: `entry` comes back stamped as
        it would have been written, `previous` is `None`, and nothing is appended -- the file
        on disk is untouched, per the positive control in `tests/unit/test_ledger.py`.
        """
        from hub.models.experiment import narrowing  # local: experiment imports this module

        stamped = replace(entry, timestamp=entry.timestamp or _now())
        entries = self._read()
        if entries is None:
            name = self.path.name if self.path is not None else "the ledger"
            line = (f"  interval width {stamped.width:.{self.places}f}; {name} is on disk "
                    f"and does not parse, so this run is not recorded and nothing is "
                    f"compared -- the ledger is left as it is for a reader to recover, not "
                    f"replaced")
            return Comparison(stamped, None, 0, float("nan"), False, [line])

        previous: WidthEntry | None = None
        elsewhere = 0
        for e in reversed(entries):
            if e.name != stamped.name:
                continue
            if e.comparable(stamped):
                previous = e
                break
            elsewhere += 1

        said = narrowing(stamped.width, previous.width if previous is not None else None,
                         places=self.places)
        lines = list(said.lines)
        if previous is None and elsewhere:
            lines.append(f"  interval width {stamped.width:.{self.places}f}; {elsewhere} "
                         f"earlier run(s) of this gate at another config or data digest, "
                         f"not compared -- two runs are comparable only at an identical "
                         f"digest (docs/gate-power.md)")

        if self.write:
            entries.append(stamped)
            if self.path is None:
                self._entries = entries
            else:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    atomic.write_text(self.path, json.dumps(
                        {"entries": [e._as_dict() for e in entries]},
                        indent=2, sort_keys=True) + "\n")
                except OSError:
                    pass

        return Comparison(stamped, previous, elsewhere, said.ratio, said.requires_review, lines)
