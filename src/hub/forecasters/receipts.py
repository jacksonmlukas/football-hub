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
client, or by `upgrade` below (#535); week 10's first scoring waits on it.

**What `verify` proves.** That the receipt is a well-formed `.ots` for *exactly these bytes*
(the digest in its header equals the file's SHA-256) and carries a calendar commitment. A
modified file fails. It does not contact a calendar or Bitcoin.

**Graceful degradation.** A calendar outage never blocks the forecast: `stamp_pending` logs
and moves on, the file stays unreceipted, and the next run stamps it (a file without a
receipt *is* the pending state). Receipts land atomically via `hub.atomic`.

**Upgrading (#535).** A pending receipt is a timestamp tree whose leaf is a *pending
attestation* naming a calendar; the calendar later holds the completed path to a Bitcoin block
header. `upgrade` parses the receipt, computes the commitment (the message at each pending
leaf, by replaying the tree's ops from the file digest), asks the calendar named by the leaf
-- and only if it is in our calendar list -- for `GET <calendar>/timestamp/<hex commitment>`
(the documented calendar endpoint), and splices the reply (itself a serialised timestamp for
that commitment) in place of the leaf, which is what stock `ots upgrade` does. A 404 (or any
reply without a Bitcoin attestation) means "not yet confirmed": the receipt is left byte for
byte as it was, with no error. An outage changes nothing. The wire format follows
python-opentimestamps' `Timestamp.serialize` and `DetachedTimestampFile` (LEB128 varuints,
`0xff` before all but the last item of a node, `0x00` + 8-byte tag + varbytes for an
attestation, op tags 0x08 sha256 / 0xf0 append / 0xf1 prepend).

CLI: `python -m hub.forecasters.receipts stamp|verify FILE`, `stamp-pending DIR` and
`upgrade-pending DIR`.
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
BITCOIN_TAG = bytes.fromhex("0588960d73d71901")     # the Bitcoin block-header attestation
PREFIX = HEADER_MAGIC + VERSION + OP_SHA256          # then the 32-byte digest
TIMEOUT_S = 20
SUFFIX = ".ots"
USER_AGENT = "football-hub/0.1 (+https://github.com/jacksonmlukas/football-hub)"

Post = Callable[[str, bytes], bytes]
# GET a calendar URL: the body on 200, None on 404 (not yet confirmed); raises on an outage.
Get = Callable[[str], bytes | None]


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
    tail = blob[len(PREFIX) + 32:]
    return blob[len(PREFIX):len(PREFIX) + 32] == d and (PENDING_TAG in tail or BITCOIN_TAG in tail)


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


# --- the .ots timestamp tree (just enough of the format to upgrade a receipt) -----------------
_APPEND, _PREPEND, _SHA256, _REVERSE, _HEXLIFY = 0xF0, 0xF1, 0x08, 0xF2, 0xF3
_MORE = 0xFF                      # "another item follows" marker before all but the last item


class _Malformed(ValueError):
    """Not a timestamp tree we can read (or replay)."""


class _Reader:
    """Reads python-opentimestamps' serialisation: LEB128 varuints, varbytes, ops, attestations."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise _Malformed("truncated")
        self.pos += 1
        return self.data[self.pos - 1]

    def raw(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise _Malformed("truncated")
        self.pos += n
        return self.data[self.pos - n:self.pos]

    def varuint(self) -> int:
        n = shift = 0
        while True:
            b = self.byte()
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return n
            if shift > 63:
                raise _Malformed("varuint too long")

    def varbytes(self) -> bytes:
        return self.raw(self.varuint())


class _Node:
    """One message in the tree: attestations that hold at it, and ops leading to child nodes."""

    def __init__(self) -> None:
        self.attestations: list[tuple[bytes, bytes]] = []        # (8-byte tag, payload)
        self.edges: list[tuple[bytes, _Node]] = []              # (serialised op, child)


def _varuint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _parse_node(r: _Reader, depth: int = 0) -> _Node:
    if depth > 400:
        raise _Malformed("tree too deep")
    node = _Node()
    while True:
        tag = r.byte()
        more = tag == _MORE
        if more:
            tag = r.byte()
        if tag == 0x00:
            node.attestations.append((r.raw(8), r.varbytes()))
        elif tag in (_APPEND, _PREPEND):
            arg = r.varbytes()
            node.edges.append((bytes([tag]) + _varuint(len(arg)) + arg, _parse_node(r, depth + 1)))
        elif tag in (_SHA256, _REVERSE, _HEXLIFY):
            node.edges.append((bytes([tag]), _parse_node(r, depth + 1)))
        else:
            raise _Malformed(f"unsupported op 0x{tag:02x}")
        if not more:
            return node


def _serialize_node(node: _Node) -> bytes:
    items = [b"\x00" + tag + _varuint(len(p)) + p for tag, p in sorted(node.attestations)]
    items += [op + _serialize_node(child) for op, child in sorted(node.edges, key=lambda e: e[0])]
    if not items:
        raise _Malformed("empty node")
    return b"".join(bytes([_MORE]) + it for it in items[:-1]) + items[-1]


def _apply(op: bytes, msg: bytes) -> bytes:
    kind = op[0]
    if kind == _SHA256:
        return hashlib.sha256(msg).digest()
    if kind == _REVERSE:
        return msg[::-1]
    if kind == _HEXLIFY:
        return msg.hex().encode()
    arg = _Reader(op[1:]).varbytes()
    return msg + arg if kind == _APPEND else arg + msg


def _parse_whole(data: bytes, start: int = 0) -> _Node:
    r = _Reader(data)
    r.pos = start
    node = _parse_node(r)
    if r.pos != len(data):
        raise _Malformed("trailing bytes")
    return node


def _parse_receipt(blob: bytes) -> tuple[bytes, _Node]:
    """(file digest, tree) of a SHA-256 detached `.ots`; raises `_Malformed` otherwise."""
    if not blob.startswith(PREFIX) or len(blob) <= len(PREFIX) + 32:
        raise _Malformed("not a SHA-256 .ots")
    return blob[len(PREFIX):len(PREFIX) + 32], _parse_whole(blob, len(PREFIX) + 32)


def _has(node: _Node, tag: bytes) -> bool:
    return any(t == tag for t, _ in node.attestations) or any(_has(c, tag) for _, c in node.edges)


def _splice(node: _Node, msg: bytes, get: Get, allowed: set[str], notes: list[str]) -> bool:
    """Replace each confirmable pending leaf under `node` by its calendar's completed path."""
    changed = False
    for op, child in node.edges:
        changed |= _splice(child, _apply(op, msg), get, allowed, notes)
    for att in list(node.attestations):
        tag, payload = att
        if tag != PENDING_TAG:
            continue
        uri = _Reader(payload).varbytes().decode("utf-8").rstrip("/")
        if uri not in allowed:                    # never fetch a URL the file merely names
            notes.append(f"{uri}: not a configured calendar")
            continue
        try:
            reply = get(f"{uri}/timestamp/{msg.hex()}")
        except Exception as e:                    # an outage changes nothing
            notes.append(f"{uri}: {type(e).__name__}: {e}")
            continue
        if not reply:
            continue                              # not yet confirmed
        try:
            sub = _parse_whole(reply)
        except _Malformed as e:
            notes.append(f"{uri}: unusable reply ({e})")
            continue
        if not _has(sub, BITCOIN_TAG):
            continue                              # still a calendar-internal hop, not confirmed
        node.attestations.remove(att)
        node.attestations.extend(sub.attestations)
        node.edges.extend(sub.edges)
        changed = True
    return changed


def _get(url: str) -> bytes | None:
    r = requests.get(url, timeout=TIMEOUT_S,
                     headers={"Accept": "application/vnd.opentimestamps.v1",
                              "User-Agent": USER_AGENT})
    if r.status_code == 404:                      # "pending confirmation in the Bitcoin blockchain"
        return None
    r.raise_for_status()
    return r.content


def upgrade(receipt_path: Path, *, get: Get = _get, calendars: Sequence[str] = CALENDARS) -> bool:
    """Upgrade pending leaves of `receipt_path` whose calendar now has a Bitcoin attestation.

    True iff the file was rewritten (atomically). Not-yet-confirmed leaves, an outage, an
    unreadable receipt: the file is untouched and the answer False -- never an exception.
    """
    receipt_path = Path(receipt_path)
    notes: list[str] = []
    out = b""
    try:
        d, tree = _parse_receipt(receipt_path.read_bytes())
        if _splice(tree, d, get, {c.rstrip("/") for c in calendars}, notes):
            out = PREFIX + d + _serialize_node(tree)
    except (OSError, _Malformed, UnicodeDecodeError) as e:
        notes.append(f"{type(e).__name__}: {e}")
    for n in notes:
        print(f"receipts: {receipt_path} not upgraded ({n})", file=sys.stderr)
    if not out:
        return False
    atomic.write_bytes(receipt_path, out)
    return True


def is_pending(receipt_path: Path) -> bool:
    """True iff the receipt still has a pending leaf (an unreadable receipt is not 'pending')."""
    try:
        return _has(_parse_receipt(Path(receipt_path).read_bytes())[1], PENDING_TAG)
    except (OSError, _Malformed):
        return False


def upgrade_pending(root: Path, *, get: Get = _get,
                    calendars: Sequence[str] = CALENDARS) -> tuple[list[Path], list[Path]]:
    """Try to upgrade every pending receipt. Returns (upgraded, still_pending). Never raises."""
    upgraded: list[Path] = []
    pending: list[Path] = []
    for f in forecast_files(root):
        r = receipt_path(f)
        if not r.exists() or not is_pending(r):
            continue
        if upgrade(r, get=get, calendars=calendars):
            upgraded.append(r)
        if is_pending(r):
            pending.append(r)
    return upgraded, pending


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.receipts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stamp", "verify"):
        sub.add_parser(name).add_argument("file", type=Path)
    sub.add_parser("stamp-pending").add_argument("dir", type=Path)
    sub.add_parser("upgrade-pending").add_argument("dir", type=Path)
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
    if a.cmd == "upgrade-pending":
        upgraded, still = upgrade_pending(a.dir)
        print(f"upgraded {len(upgraded)}, pending {len(still)}")
        return 0                                      # not-yet-confirmed and outages never fail
    stamped, pending = stamp_pending(a.dir)
    print(f"stamped {len(stamped)}, pending {len(pending)}")
    return 0                                          # an outage never fails the run


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
