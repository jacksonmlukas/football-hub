"""The committed odds archive is readable from a clean checkout, with no `data/` (#389, #428).

This is #389's persistence criterion held as a standing check rather than a one-off: clone the
repository's committed `HEAD` into a scratch directory, read `state/odds/` with *that clone's*
`hub.store`, and require every poll #428 back-filled to be there. A snapshot that exists only in
the working tree of the machine that wrote it -- the fault #383 ended -- is absent from the
clone, so deleting one from the index, or forgetting to `git add` the next, turns this red.

The plant (one capture removed, one left untracked) is `tests/unit/test_instrument_check.py`;
this file is the same check pointed at the real tree.
"""
from hub import instrument_check as ic
from hub.paths import ROOT


def test_the_backfilled_polls_are_readable_from_a_clean_checkout():
    got = ic.backfill_readable(ROOT, 2026)
    assert got.status == ic.PASS, got.detail
    assert int(str(got.facts["rows"])) >= 1834,"the back-fill held 1,834 rows when #428 landed"


def test_the_clean_checkout_had_no_data_directory():
    """If the scratch clone carried a `data/`, what it read would prove nothing about a fresh
    one; `persistence` refuses in that case, and this holds that it did not need to."""
    got = ic.persistence(ROOT, 2026, 1, ic.PERSISTENCE_BEGINS, ic.PERSISTENCE_BEGINS,
                         now=ic.PERSISTENCE_BEGINS)
    assert got.facts["fresh_checkout_had_a_data_dir"] is False
