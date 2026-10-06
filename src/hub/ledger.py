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
from hub.declare import decision
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
    # The review flag a row written before #385 carried (`review_width` stored it; this module
    # derives it per run and returns it in `Comparison`). Kept as a pass-through on an entry
    # read off disk so a rewrite of the file does not erase it (#427); `None` on every entry
    # this module writes, which then carries no such key.
    requires_review: bool | None = None
    # #381: how many of the run's seasons resolved (a win or a loss) and how many abstained
    # (CONTEXT.md, **Abstention**) -- the two numbers the verdict is read over, recorded beside
    # it so a verdict on fewer seasons than were run says so in the record and not only in the
    # sentence. `None` on an entry written before #381, or by a caller with no seasons frame to
    # count; then neither key is written, the same additive shape as `seasons`.
    resolved: int | None = None
    abstained: int | None = None

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
            "gate": self.name,
            "config_digest": self.config_digest, "data_digest": self.data_digest,
            "timestamp": self.timestamp, "width": self.width, "clusters": self.clusters,
            "lo": self.lo, "hi": self.hi, "verdict": self.verdict,
        }
        # **Unknown stays unknown (#427).** A `known=False` row had no `"recipe"` key on disk,
        # and `_from_dict` reads the key's *presence* as "an arm was declared, or declared
        # none". Writing `"recipe": null` for it turned "unknown" into "no arm declared" on
        # the first rewrite of the file -- a silent change to what the record says about its
        # own history. So the key is written only for a known entry.
        if self.known:
            out["recipe"] = self.recipe
        if self.seasons is not None:
            out["seasons"] = self.seasons
        if self.requires_review is not None:
            out["requires_review"] = self.requires_review
        if self.resolved is not None:
            out["resolved"] = self.resolved
        if self.abstained is not None:
            out["abstained"] = self.abstained
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
        return WidthEntry(
            name=str(d.get("gate", "")),
            config_digest=str(d.get("config_digest", "")),
            data_digest=str(d.get("data_digest", "")),
            width=_num(d.get("width")),
            clusters=_num(d.get("clusters"), 0.0),
            lo=_num(d.get("lo")),
            hi=_num(d.get("hi")),
            verdict=str(d.get("verdict", "")),
            recipe=d.get("recipe"),
            seasons=d.get("seasons"),
            timestamp=d.get("timestamp"),
            known=known,
            requires_review=(d["requires_review"]
                             if isinstance(d.get("requires_review"), bool) else None),
            resolved=_count(d.get("resolved")),
            abstained=_count(d.get("abstained")),
        )


def _spell(v: object) -> str:
    """One value in a recipe: a bool as `yes`/`no`, `None` as `none`, a float without
    trailing zeros, a set sorted, a list or tuple in the order the caller gave it."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, set | frozenset):
        return "+".join(sorted(_spell(x) for x in v))
    if isinstance(v, list | tuple):
        return "+".join(_spell(x) for x in v)
    return str(v)


def recipe(**arm: object) -> str:
    """The run's arm as the ledger key's `recipe` (#384): `k=v` pairs, sorted by `k`, joined
    by commas -- the one place any gate's recipe string is spelled.

    **The rule each gate applies.** Name every flag of the run that reaches the paired frame,
    the draw behind it, or the verdict, and that the two digests do not already cover:
    `config_digest` is the resolved conf and fitted constants and `data_digest` is the pins,
    so neither sees `--churn`, `--drafts`, `--seed`, `--ceiling-arm` or the like. A flag that
    only changes where output goes (`--out`, `--progress`, `--workers`) is not part of the arm.
    Sorted so two spellings of one arm are one string; a gate with no such flag passes no
    recipe at all, which is `None` -- "no arm declared" -- rather than `""`.
    """
    return ",".join(f"{k}={_spell(v)}" for k, v in sorted(arm.items()))


def _num(v: object, missing: float = float("nan")) -> float:
    """A number read off disk, or `missing` -- never a raise: `null`, a string or an absent
    key in one row must not take a gate run down (`Ledger.record` never raises)."""
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else missing


def _count(v: object) -> int | None:
    """A season count read off disk, or `None` -- never a raise, for the reason `_num` gives."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None


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

    @decision
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
        unknown = 0
        rerecipe = 0
        for e in reversed(entries):
            if e.name != stamped.name:
                continue
            if e.comparable(stamped):
                previous = e
                break
            elsewhere += 1
            # Skipped only for want of a recipe: same digests, no "recipe" key on disk. A row
            # whose digests also differ keeps the digest line, which is the true reason for it.
            same_digests = ((e.config_digest, e.data_digest)
                            == (stamped.config_digest, stamped.data_digest))
            unknown += not e.known and same_digests
            # Known, same digests, another arm: #384's case, named as such rather than as a
            # digest difference, which it is not.
            rerecipe += e.known and same_digests and e.recipe != stamped.recipe

        said = narrowing(stamped.width, previous.width if previous is not None else None,
                         places=self.places)
        lines = list(said.lines)
        if previous is None and elsewhere - unknown - rerecipe:
            lines.append(f"  interval width {stamped.width:.{self.places}f}; "
                         f"{elsewhere - unknown - rerecipe} "
                         f"earlier run(s) of this gate at another config or data digest, "
                         f"not compared -- two runs are comparable only at an identical "
                         f"digest (docs/gate-power.md)")
        if previous is None and unknown:
            # Named apart from the digest line: these rows are skipped because nobody recorded
            # their recipe (written before #385), not because a digest differs.
            lines.append(f"  interval width {stamped.width:.{self.places}f}; {unknown} "
                         f"earlier run(s) of this gate of unknown recipe (written before the "
                         f"ledger key carried one), not compared -- state/README.md")

        if previous is None and rerecipe:
            lines.append(f"  interval width {stamped.width:.{self.places}f}; {rerecipe} "
                         f"earlier run(s) of this gate at another recipe (a different arm "
                         f"of the same gate), not compared -- state/README.md")

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
