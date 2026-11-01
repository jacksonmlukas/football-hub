"""Timestamp receipts (#449): the calendar servers are stubbed; nothing here touches a network."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hub.forecasters import receipts

PENDING = b"\xf0\x04abcd\x08\x00" + receipts.PENDING_TAG + b"\x16\x14https://cal.example"


def _calendar(log: list[str]):
    def post(url: str, body: bytes) -> bytes:
        log.append(url)
        assert len(body) == 32
        return PENDING
    return post


def _down(url: str, body: bytes) -> bytes:
    raise ConnectionError("calendar down")


def _day(tmp_path: Path, name: str = "2026-10-28") -> Path:
    f = tmp_path / "2026" / f"{name}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"game_id": "g1", "p": 0.6}\n')
    return f


def test_a_stamped_file_round_trips(tmp_path):
    f = _day(tmp_path)
    seen: list[str] = []
    r = receipts.stamp(f, post=_calendar(seen), calendars=["https://cal.example/"])
    assert r == tmp_path / "2026" / "2026-10-28.jsonl.ots"
    assert seen == ["https://cal.example/digest"]
    assert r.read_bytes().startswith(receipts.HEADER_MAGIC)
    assert receipts.verify(f, r) is True


def test_a_modified_file_fails_verification(tmp_path):
    f = _day(tmp_path)
    r = receipts.stamp(f, post=_calendar([]), calendars=["https://cal.example"])
    f.write_text('{"game_id": "g1", "p": 0.9}\n')
    assert receipts.verify(f, r) is False


def test_a_missing_or_malformed_receipt_fails(tmp_path):
    f = _day(tmp_path)
    assert receipts.verify(f, tmp_path / "nope.ots") is False
    bad = tmp_path / "bad.ots"
    bad.write_bytes(b"not a receipt")
    assert receipts.verify(f, bad) is False
    bare = tmp_path / "bare.ots"                      # right digest, no calendar commitment
    bare.write_bytes(receipts.PREFIX + receipts.digest(f))
    assert receipts.verify(f, bare) is False


def test_a_calendar_outage_leaves_the_forecast_written_and_the_stamp_pending(tmp_path, capsys):
    f = _day(tmp_path)
    before = f.read_bytes()
    with pytest.raises(receipts.CalendarUnavailable):
        receipts.stamp(f, post=_down, calendars=["https://a", "https://b"])
    stamped, pending = receipts.stamp_pending(tmp_path, post=_down, calendars=["https://a"])
    assert (stamped, pending) == ([], [f])
    assert f.read_bytes() == before and not receipts.receipt_path(f).exists()
    assert "stays pending" in capsys.readouterr().err
    # the next run, calendar back, fills it in
    stamped, pending = receipts.stamp_pending(tmp_path, post=_calendar([]), calendars=["https://a"])
    assert stamped == [receipts.receipt_path(f)] and pending == []


def test_falls_through_to_the_next_calendar_and_rejects_junk_replies(tmp_path):
    f = _day(tmp_path)
    replies = {"https://a/digest": b"<html>", "https://b/digest": PENDING}
    def post(url: str, body: bytes) -> bytes:
        if url == "https://c/digest":
            raise AssertionError("stopped at the first usable calendar")
        return replies[url]
    r = receipts.stamp(f, post=post, calendars=["https://a", "https://b", "https://c"])
    assert receipts.verify(f, r)


def test_already_receipted_files_are_not_restamped(tmp_path):
    f = _day(tmp_path)
    receipts.stamp(f, post=_calendar([]), calendars=["https://a"])
    seen: list[str] = []
    stamped, pending = receipts.stamp_pending(tmp_path, post=_calendar(seen), calendars=["https://a"])
    assert (stamped, pending, seen) == ([], [], [])


def test_only_the_day_logs_are_stamped(tmp_path):
    _day(tmp_path)
    (tmp_path / "exclusions.jsonl").write_text("{}\n")
    assert [p.name for p in receipts.forecast_files(tmp_path)] == ["2026-10-28.jsonl"]


def test_the_cli_stamps_verifies_and_never_fails_a_run_on_an_outage(tmp_path, monkeypatch, capsys):
    f = _day(tmp_path)
    monkeypatch.setattr(receipts, "_post", _down)
    monkeypatch.setattr(receipts.stamp, "__kwdefaults__", {"post": _down, "calendars": receipts.CALENDARS})
    monkeypatch.setattr(receipts.stamp_pending, "__kwdefaults__",
                        {"post": _down, "calendars": receipts.CALENDARS})
    assert receipts.main(["stamp", str(f)]) == 1
    assert receipts.main(["stamp-pending", str(tmp_path)]) == 0
    assert "pending 1" in capsys.readouterr().out
    assert receipts.main(["verify", str(f)]) == 1
    good = {"post": _calendar([]), "calendars": ["https://a"]}
    monkeypatch.setattr(receipts.stamp, "__kwdefaults__", good)
    assert receipts.main(["stamp", str(f)]) == 0
    assert receipts.main(["verify", str(f)]) == 0


def test_the_default_transport_posts_the_raw_digest_with_a_timeout(monkeypatch):
    seen: dict = {}
    class R:
        content = PENDING
        def raise_for_status(self): pass
    def fake(url, **kw):
        seen.update(url=url, **kw)
        return R()
    monkeypatch.setattr(receipts.requests, "post", fake)
    assert receipts._post("https://a/digest", b"x" * 32) == PENDING
    assert seen["data"] == b"x" * 32 and seen["timeout"] == receipts.TIMEOUT_S


# --- upgrading pending receipts (#535) ---------------------------------------------------------
# Fixtures are written byte by byte, NOT through the module's own serialiser, from the format
# python-opentimestamps (the stock `ots` client) reads and writes -- Timestamp.serialize and
# DetachedTimestampFile: op tag 0xf0 append / 0xf1 prepend + varbytes, 0x08 sha256, `0x00` +
# 8-byte tag + varbytes(payload) for an attestation, `0xff` before every item but the last.
# A pending attestation's payload is varbytes(calendar uri); a Bitcoin one's is varuint(height).
# The calendar's `GET /timestamp/<hex commitment>` body is a bare serialised timestamp for that
# commitment (no header), which `ots upgrade` merges at the pending leaf.
CAL = "https://cal.example"
OTHER = "https://other.example"


def _pending_leaf(uri: str) -> bytes:
    u = uri.encode()
    return b"\x00" + receipts.PENDING_TAG + bytes([len(u) + 1, len(u)]) + u


BITCOIN_LEAF = b"\x00" + receipts.BITCOIN_TAG + b"\x01\x64"       # block height 100
X = b"\xaa" * 8
COMPLETED = b"\xf1\x08" + X + b"\x08" + BITCOIN_LEAF              # prepend X, sha256, bitcoin


def _pending_receipt(f: Path, uri: str = CAL, nonce: bytes = b"\x01\x02\x03\x04"):
    """A receipt for `f`: digest -> append nonce -> sha256 -> pending(uri). Returns (path, commitment)."""
    d = hashlib.sha256(f.read_bytes()).digest()
    r = Path(str(f) + ".ots")
    r.write_bytes(receipts.PREFIX + d + b"\xf0\x04" + nonce + b"\x08" + _pending_leaf(uri))
    return r, hashlib.sha256(d + nonce).digest()


def _server(responses: dict, log: list[str] | None = None):
    def get(url: str):
        if log is not None:
            log.append(url)
        r = responses[url]
        if isinstance(r, Exception):
            raise r
        return r
    return get


def test_a_confirmed_receipt_is_upgraded_in_place_to_a_standard_bitcoin_timestamp(tmp_path):
    f = _day(tmp_path)
    r, commitment = _pending_receipt(f)
    d = hashlib.sha256(f.read_bytes()).digest()
    seen: list[str] = []
    get = _server({f"{CAL}/timestamp/{commitment.hex()}": COMPLETED}, seen)
    assert receipts.upgrade(r, get=get, calendars=[CAL + "/"]) is True
    assert seen == [f"{CAL}/timestamp/{commitment.hex()}"]
    # exactly the tree the stock client would hold: the pending leaf replaced by the calendar's path
    assert r.read_bytes() == receipts.PREFIX + d + b"\xf0\x04\x01\x02\x03\x04\x08" + COMPLETED
    assert receipts.BITCOIN_TAG in r.read_bytes() and receipts.PENDING_TAG not in r.read_bytes()
    assert receipts.verify(f, r) is True and receipts.is_pending(r) is False
    assert sorted(p.name for p in r.parent.iterdir()) == sorted([f.name, r.name])   # no temp left


def test_the_upgrade_is_written_through_the_atomic_writer(tmp_path, monkeypatch):
    f = _day(tmp_path)
    r, c = _pending_receipt(f)
    written: list[Path] = []
    real = receipts.atomic.write_bytes

    def spy(p, b):
        written.append(p)
        return real(p, b)
    monkeypatch.setattr(receipts.atomic, "write_bytes", spy)
    receipts.upgrade(r, get=_server({f"{CAL}/timestamp/{c.hex()}": COMPLETED}), calendars=[CAL])
    assert written == [r]


def test_an_unconfirmed_receipt_stays_pending_without_an_error(tmp_path, capsys):
    f = _day(tmp_path)
    r, c = _pending_receipt(f)
    before = r.read_bytes()
    # 404 (the calendar's "pending confirmation"), or a hop that is still not a Bitcoin attestation
    for answer in (None, b"", b"\xf1\x08" + X + b"\x08" + _pending_leaf(CAL)):
        assert receipts.upgrade(r, get=_server({f"{CAL}/timestamp/{c.hex()}": answer}),
                                calendars=[CAL]) is False
        assert r.read_bytes() == before
    assert receipts.is_pending(r) is True and capsys.readouterr().err == ""


def test_a_calendar_outage_or_a_junk_reply_changes_nothing(tmp_path, capsys):
    f = _day(tmp_path)
    r, c = _pending_receipt(f)
    before = r.read_bytes()
    url = f"{CAL}/timestamp/{c.hex()}"
    for answer in (ConnectionError("down"), b"<html>502</html>", COMPLETED[:-3], COMPLETED + b"\x00"):
        assert receipts.upgrade(r, get=_server({url: answer}), calendars=[CAL]) is False
        assert r.read_bytes() == before
    assert "not upgraded" in capsys.readouterr().err


def test_a_calendar_the_file_merely_names_is_never_fetched(tmp_path):
    f = _day(tmp_path)
    r, _ = _pending_receipt(f, uri=OTHER)
    seen: list[str] = []
    assert receipts.upgrade(r, get=_server({}, seen), calendars=[CAL]) is False
    assert seen == []


def test_an_unreadable_receipt_is_left_alone(tmp_path):
    f = _day(tmp_path)
    bad = tmp_path / "bad.ots"
    bad.write_bytes(b"not a receipt")
    assert receipts.upgrade(bad, get=_server({}), calendars=[CAL]) is False
    assert receipts.upgrade(tmp_path / "missing.ots", get=_server({}), calendars=[CAL]) is False
    r, _ = _pending_receipt(f)
    r.write_bytes(r.read_bytes()[:-5])                                # truncated mid-attestation
    assert receipts.is_pending(r) is False
    assert receipts.upgrade(r, get=_server({}), calendars=[CAL]) is False
    r.write_bytes(receipts.PREFIX + receipts.digest(f) + b"\x67\x00")  # an op we do not replay
    assert receipts.upgrade(r, get=_server({}), calendars=[CAL]) is False


def test_only_the_confirmed_leaf_of_a_two_calendar_receipt_is_replaced(tmp_path):
    f = _day(tmp_path)
    d = hashlib.sha256(f.read_bytes()).digest()
    n1, n2 = b"\x01\x01\x01\x01", b"\x02\x02\x02\x02"
    tree = (b"\xff\xf0\x04" + n1 + b"\x08" + _pending_leaf(CAL)
            + b"\xf0\x04" + n2 + b"\x08" + _pending_leaf(OTHER))
    r = Path(str(f) + ".ots")
    r.write_bytes(receipts.PREFIX + d + tree)
    c1 = hashlib.sha256(d + n1).digest()
    c2 = hashlib.sha256(d + n2).digest()
    get = _server({f"{CAL}/timestamp/{c1.hex()}": COMPLETED, f"{OTHER}/timestamp/{c2.hex()}": None})
    assert receipts.upgrade(r, get=get, calendars=[CAL, OTHER]) is True
    blob = r.read_bytes()
    assert receipts.BITCOIN_TAG in blob and _pending_leaf(OTHER) in blob
    assert _pending_leaf(CAL) not in blob and receipts.is_pending(r) is True


def test_upgrade_pending_walks_the_receipts_and_skips_the_finished(tmp_path, capsys):
    a, b = _day(tmp_path, "2026-10-28"), _day(tmp_path, "2026-10-29")
    ra, ca = _pending_receipt(a)
    rb, _ = _pending_receipt(b, nonce=b"\x09\x09\x09\x09")
    _day(tmp_path, "2026-10-30")                                      # unreceipted: not our business
    seen: list[str] = []
    ua = f"{CAL}/timestamp/{ca.hex()}"

    def get(url):
        seen.append(url)
        return COMPLETED if url == ua else None
    assert receipts.upgrade_pending(tmp_path, get=get, calendars=[CAL]) == ([ra], [rb])
    assert len(seen) == 2
    seen.clear()
    assert receipts.upgrade_pending(tmp_path, get=get, calendars=[CAL]) == ([], [rb])
    assert len(seen) == 1                                             # the finished one is not asked again
    assert capsys.readouterr().err == ""


def test_the_cli_upgrades_and_never_fails_a_run(tmp_path, monkeypatch, capsys):
    f = _day(tmp_path)
    cal = receipts.CALENDARS[0]
    r, c = _pending_receipt(f, uri=cal)
    url = f"{cal}/timestamp/{c.hex()}"
    kw = dict(receipts.upgrade_pending.__kwdefaults__ or {})
    monkeypatch.setattr(receipts.upgrade_pending, "__kwdefaults__",
                        {**kw, "get": _server({url: ConnectionError("down")})})
    assert receipts.main(["upgrade-pending", str(tmp_path)]) == 0
    assert "upgraded 0, pending 1" in capsys.readouterr().out
    monkeypatch.setattr(receipts.upgrade_pending, "__kwdefaults__",
                        {**kw, "get": _server({url: COMPLETED})})
    assert receipts.main(["upgrade-pending", str(tmp_path)]) == 0
    assert "upgraded 1, pending 0" in capsys.readouterr().out
    assert receipts.verify(f, r)


def test_the_default_get_maps_404_to_not_yet_and_raises_on_other_failures(monkeypatch):
    class R:
        def __init__(self, code, content=b""):
            self.status_code, self.content = code, content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise ConnectionError(self.status_code)
    answers = iter([R(404), R(200, COMPLETED), R(503)])
    seen: dict = {}

    def fake(url, **kw):
        seen.update(url=url, **kw)
        return next(answers)
    monkeypatch.setattr(receipts.requests, "get", fake)
    assert receipts._get("https://a/timestamp/ab") is None
    assert receipts._get("https://a/timestamp/ab") == COMPLETED
    assert seen["timeout"] == receipts.TIMEOUT_S
    with pytest.raises(ConnectionError):
        receipts._get("https://a/timestamp/ab")


def test_one_calendar_down_does_not_stop_the_other_leaf_upgrading(tmp_path):
    f = _day(tmp_path)
    d = hashlib.sha256(f.read_bytes()).digest()
    n1, n2 = b"\x01\x01\x01\x01", b"\x02\x02\x02\x02"
    tree = (b"\xff\xf0\x04" + n1 + b"\x08" + _pending_leaf(OTHER)
            + b"\xf0\x04" + n2 + b"\x08" + _pending_leaf(CAL))
    r = Path(str(f) + ".ots")
    r.write_bytes(receipts.PREFIX + d + tree)
    get = _server({f"{OTHER}/timestamp/{hashlib.sha256(d + n1).digest().hex()}": OSError("down"),
                   f"{CAL}/timestamp/{hashlib.sha256(d + n2).digest().hex()}": COMPLETED})
    assert receipts.upgrade(r, get=get, calendars=[CAL, OTHER]) is True
    assert _pending_leaf(OTHER) in r.read_bytes() and receipts.BITCOIN_TAG in r.read_bytes()


def test_malformed_trees_are_refused_not_crashed_on():
    with pytest.raises(receipts._Malformed):
        receipts._Reader(b"").byte()
    with pytest.raises(receipts._Malformed):
        receipts._Reader(b"\x05ab").varbytes()                 # length runs past the end
    with pytest.raises(receipts._Malformed):
        receipts._Reader(b"\xff" * 12).varuint()                 # endless continuation bits
    with pytest.raises(receipts._Malformed):
        receipts._parse_whole(b"\x08" * 1000 + BITCOIN_LEAF)     # deeper than any real path
    with pytest.raises(receipts._Malformed):
        receipts._serialize_node(receipts._Node())


def test_reverse_and_hexlify_ops_are_replayed_into_the_commitment(tmp_path):
    f = _day(tmp_path)
    d = hashlib.sha256(f.read_bytes()).digest()
    r = Path(str(f) + ".ots")
    r.write_bytes(receipts.PREFIX + d + b"\xf2\xf3" + _pending_leaf(CAL))
    commitment = d[::-1].hex().encode()
    seen: list[str] = []
    get = _server({f"{CAL}/timestamp/{commitment.hex()}": None}, seen)
    assert receipts.upgrade(r, get=get, calendars=[CAL]) is False
    assert seen == [f"{CAL}/timestamp/{commitment.hex()}"]


def test_a_leaf_that_is_not_pending_is_left_as_it_is(tmp_path):
    f = _day(tmp_path)
    d = hashlib.sha256(f.read_bytes()).digest()
    r = Path(str(f) + ".ots")
    r.write_bytes(receipts.PREFIX + d + b"\xff" + BITCOIN_LEAF + _pending_leaf(CAL))
    assert receipts.upgrade(r, get=_server({f"{CAL}/timestamp/{d.hex()}": None}), calendars=[CAL]) is False
    assert receipts.BITCOIN_TAG in r.read_bytes()
