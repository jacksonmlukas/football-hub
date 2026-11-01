"""Timestamp receipts for the forecast files (#449).

The study's claim is that each forecast was fixed *before* the game. A file on a branch the
maintainer controls proves nothing of the kind, so each day's forecast file is submitted to
the public OpenTimestamps calendars and the calendar's answer is kept beside it as
`<file>.ots`, on the `forecasts` branch (never main).

**Why the protocol, not the `opentimestamps` client.** The calendar protocol is one HTTP POST:
the SHA-256 of the file, 32 raw bytes, to `<calendar>/digest`; the body of the reply is the
serialised timestamp (the commitment path up to a *pending* attestation naming the calendar).
The client library adds `python-bitcoinlib` and a full `ots` CLI around that, is unmaintained,
and its Bitcoin verification needs a node we do not run. The repo already speaks HTTP through
`requests`; a receipt written here is the standard `.ots` container (magic, version, SHA-256
op, digest, then the calendar's bytes), so `ots upgrade` / `ots verify` from the stock client
read it unchanged. Upgrading a pending receipt to a Bitcoin attestation is left to that
client; nothing in the study's gates waits on it.

**What `verify` proves.** That the receipt is a well-formed `.ots` for *exactly these bytes*
(the digest in its header equals the file's SHA-256) and carries a calendar commitment. A
modified file fails. It does not contact a calendar or Bitcoin.

**Graceful degradation.** A calendar outage never blocks the forecast: `stamp_pending` logs
and moves on, the file stays unreceipted, and the next run stamps it (a file without a
receipt *is* the pending state). Receipts land atomically via `hub.atomic`.

CLI: `python -m hub.forecasters.receipts stamp|verify FILE` and `stamp-pending DIR`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import requests

from hub import atomic

CALENDARS = (
    "https://alice.btc.calendar.opentimestamps.org",
    "https://bob.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
)
HEADER_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
VERSION = b"\x01"
OP_SHA256 = b"\x08"
PENDING_TAG = bytes.fromhex("83dfe30d2ef90c8e")     # the pending-attestation marker
PREFIX = HEADER_MAGIC + VERSION + OP_SHA256          # then the 32-byte digest
TIMEOUT_S = 20
SUFFIX = ".ots"
USER_AGENT = "football-hub/0.1 (+https://github.com/jacksonmlukas/football-hub)"

Post = Callable[[str, bytes], bytes]


class CalendarUnavailable(Exception):
    """No calendar answered with a usable timestamp."""


def receipt_path(path: Path) -> Path:
    return Path(str(path) + SUFFIX)


def digest(path: Path) -> bytes:
    return hashlib.sha256(Path(path).read_bytes()).digest()


def _post(url: str, body: bytes) -> bytes:
    r = requests.post(url, data=body, timeout=TIMEOUT_S,
                      headers={"Accept": "application/vnd.opentimestamps.v1",
                               "User-Agent": USER_AGENT,
                               "Content-Type": "application/x-www-form-urlencoded"})
    r.raise_for_status()
    return r.content


def stamp(path: Path, *, post: Post = _post, calendars: Sequence[str] = CALENDARS) -> Path:
    """Submit `path`'s digest to the first calendar that answers; write `<path>.ots`.

    Raises `CalendarUnavailable` (naming each failure) if none does; no receipt is written.
    """
    d = digest(path)
    failures: list[str] = []
    for cal in calendars:
        try:
            reply = post(cal.rstrip("/") + "/digest", d)
        except Exception as e:                       # any transport failure is an outage
            failures.append(f"{cal}: {type(e).__name__}: {e}")
            continue
        if not reply or PENDING_TAG not in reply:
            failures.append(f"{cal}: reply is not a pending timestamp")
            continue
        return atomic.write_bytes(receipt_path(path), PREFIX + d + reply)
    raise CalendarUnavailable("; ".join(failures) or "no calendars configured")


def verify(path: Path, receipt: Path) -> bool:
    """True iff `receipt` is a well-formed `.ots` for exactly the bytes of `path`."""
    try:
        blob = Path(receipt).read_bytes()
        d = digest(path)
    except OSError:
        return False
    if not blob.startswith(PREFIX) or len(blob) <= len(PREFIX) + 32:
        return False
    return blob[len(PREFIX):len(PREFIX) + 32] == d and PENDING_TAG in blob[len(PREFIX) + 32:]


def forecast_files(root: Path) -> list[Path]:
    """The per-day forecast logs, `<root>/<YYYY>/<YYYY-MM-DD>.jsonl`."""
    return sorted(Path(root).glob("[0-9][0-9][0-9][0-9]/*.jsonl"))


def stamp_pending(root: Path, *, post: Post = _post,
                  calendars: Sequence[str] = CALENDARS) -> tuple[list[Path], list[Path]]:
    """Stamp every forecast file with no receipt. Returns (stamped, still_pending).

    Never raises on an outage: the forecast is already written, and the next run retries.
    """
    stamped: list[Path] = []
    pending: list[Path] = []
    for f in forecast_files(root):
        if receipt_path(f).exists():
            continue
        try:
            stamped.append(stamp(f, post=post, calendars=calendars))
        except CalendarUnavailable as e:
            print(f"receipts: {f} stays pending ({e})", file=sys.stderr)
            pending.append(f)
    return stamped, pending


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.receipts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stamp", "verify"):
        sub.add_parser(name).add_argument("file", type=Path)
    sub.add_parser("stamp-pending").add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "stamp":
        try:
            print(stamp(a.file))
        except CalendarUnavailable as e:
            print(f"receipts: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "verify":
        ok = verify(a.file, receipt_path(a.file))
        print("ok" if ok else "FAILED: no receipt, or it does not match the file")
        return 0 if ok else 1
    stamped, pending = stamp_pending(a.dir)
    print(f"stamped {len(stamped)}, pending {len(pending)}")
    return 0                                          # an outage never fails the run


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
