"""Timestamp receipts (#449): the calendar servers are stubbed; nothing here touches a network."""
from __future__ import annotations

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
