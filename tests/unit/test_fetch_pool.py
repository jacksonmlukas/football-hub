"""Pool state ingest (#85): the field, the pot and every entry's Ledger, read off the host.

Each acceptance criterion in the ticket is one test here, named for it. The host's payload
shape is documented nowhere in this repo -- the money-layer plan records only that it is JSON
behind a session cookie and answers 401 unauthenticated -- so every test drives the parser
through `tests/golden/fixtures/pool_payload.synthetic.json`, a hand-built payload of the
shape `hub.fetch.pool` assumes. What the first live run has to confirm is listed in that
module's docstring; nothing here can stand in for it.

Two words carry their `CONTEXT.md` meanings throughout. A **Ledger** is the set of teams an
entry has already spent, one per entry; the **Decision journal** is a different record and
is not touched by this module.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hub.contracts import ContractViolation
from hub.fetch import pool

FIXTURES = Path(__file__).resolve().parents[1] / "golden" / "fixtures"

# Assembled from parts so the file itself is never a hit for the secret scan in
# `scripts/preflight_public.sh`, which reads every commit this file is in.
COOKIE = "s%3A" + "0f1e2d3c4b5a" * 3 + "." + "9a8b7c6d5e4f" * 3
# Our own entry on the host: Member A in the fixture, whose id sorts first there, which is
# why `test_our_entry_is_index_0_however_its_id_sorts` moves it.
OURS = "ent-8841"


def payload() -> dict:
    return json.loads((FIXTURES / "pool_payload.synthetic.json").read_text())


def parse(p: dict, **kw) -> pool.PoolState:
    """The state alone; the index map `parse_payload` returns beside it is tested by name."""
    state, _ = pool.parse_payload(p, ours=kw.pop("ours", OURS), **kw)
    return state


@pytest.fixture
def store(tmp_path):
    return tmp_path / "processed"


@pytest.fixture
def session(monkeypatch):
    """A cookie and a host in the environment, and `.env` kept out of it."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv(pool.SESSION_ENV, COOKIE)
    monkeypatch.setenv(pool.URL_ENV, "https://pool.example.test/api/pools/1")
    monkeypatch.setenv(pool.ENTRY_ENV, OURS)


@pytest.fixture
def transport(monkeypatch):
    """The host, answering with a status and a body; or raising, when told to."""
    def _install(body=None, status=200, raises=None):
        calls = []

        def _fake(url, cookie):
            calls.append((url, cookie))
            if raises is not None:
                raise raises
            return status, (payload() if body is None else body)
        monkeypatch.setattr(pool, "_http_get", _fake)
        return calls
    return _install


def _no_secret_in(text: str, where: str) -> None:
    assert COOKIE not in text, f"the session cookie reached {where}"
    assert COOKIE[:24] not in text, f"a prefix of the session cookie reached {where}"


def _stored_text(store: Path) -> list[tuple[Path, str]]:
    """Every file under the store as text a scan can read: a parquet partition through
    polars, since a byte scan of a columnar file is not a check; anything else as written.
    The archive (#280) is what made the store hold parquet at all."""
    import polars as pl
    out = []
    for p in sorted(q for q in store.rglob("*") if q.is_file()):
        if p.suffix == ".parquet":
            out.append((p, pl.read_parquet(p).write_json()))
        elif p.suffix != ".duckdb":
            out.append((p, p.read_text()))
    return out


# --- 1. a payload missing a required field is refused, not partially served ---------------

def test_a_payload_missing_a_required_field_is_refused_and_nothing_partial_is_written(
        store, session, transport, capsys):
    """The pot is gone from the payload. The parser refuses with a contract violation that
    names the field, and the store is exactly as it was: no file appears with the field
    size and the entries in it and a pot of nothing."""
    broken = payload()
    del broken["pool"]["pot"]
    with pytest.raises(ContractViolation, match="pot"):
        parse(broken)

    transport(body=broken)
    assert pool.main(["--refresh", "--store", str(store)]) != 0
    assert not list(store.glob("**/*")), "a refused payload left a partial state behind"
    err = capsys.readouterr().err
    assert "pot" in err and "Traceback" not in err


def test_an_entry_missing_its_picks_is_refused_rather_than_read_as_an_empty_ledger():
    """The other shape of the same failure. An entry with no `picks` key would otherwise
    parse as an entry that has spent nothing, which is a Ledger the host never sent."""
    broken = payload()
    del broken["entries"][1]["picks"]
    with pytest.raises(ContractViolation, match="picks"):
        parse(broken)


def test_a_field_size_that_disagrees_with_the_entries_listed_is_refused():
    """A field of five with four entries in the body is a truncated or paginated payload,
    and reading the four as the field is the partial state the criterion forbids."""
    broken = payload()
    broken["pool"]["field_size"] = 6
    with pytest.raises(ContractViolation, match="field_size"):
        parse(broken)


def test_a_pick_in_a_week_the_payload_does_not_describe_is_refused():
    """A pick can only be counted or not counted by the status of its week, so a week with
    no status is a pick that cannot be placed either side of the line."""
    broken = payload()
    broken["entries"][0]["picks"].append({"week": 4, "team": "MIA", "result": None})
    with pytest.raises(ContractViolation, match="week 4"):
        parse(broken)


# --- 2. field size, pot and per-entry used teams --------------------------------------------

def test_field_size_pot_and_each_entrys_used_teams_are_parsed_from_the_payload():
    state = parse(payload())
    assert state.season == 2026
    assert state.week == 3
    assert state.field_size == 5
    assert state.pot == 100.0
    assert state.alive == 4
    # Indexed 0..4 in a stable order that is not the host's, so the Ledgers below are
    # addressed by index and nothing else. Member A is the first id in sort order.
    assert [e.index for e in state.entries] == [0, 1, 2, 3, 4]
    assert [e.used for e in state.entries] == [
        ("DAL", "KC"), ("BUF", "PHI"), ("NYG",), ("DAL", "KC"), ()]
    assert [e.alive for e in state.entries] == [True, True, False, True, True]


def test_a_team_is_upper_cased_and_an_empty_one_is_refused():
    p = payload()
    p["entries"][1]["picks"][0]["team"] = "phi"
    assert parse(p).entries[1].used == ("BUF", "PHI")
    p["entries"][1]["picks"][0]["team"] = " "
    with pytest.raises(ContractViolation, match="team"):
        parse(p)


# --- 3. an unauthenticated response is an auth failure, distinct from an empty pool --------

def test_an_unauthenticated_response_is_an_auth_failure_and_not_an_empty_pool(
        store, session, transport, capsys):
    """Two answers the host can give that both carry no entries, and they must not be read
    as one: a 401 is the cookie being refused, and a pool with nobody in it is a fact about
    the pool. Different exceptions, and the entry point says which."""
    transport(body={"detail": "not authenticated"}, status=401)
    with pytest.raises(pool.AuthFailure, match="401"):
        pool.refresh(store=store)
    assert pool.main(["--refresh", "--store", str(store)]) != 0
    said = capsys.readouterr().err
    assert pool.SESSION_ENV in said and "no entries" not in said

    empty = payload()
    empty["entries"], empty["pool"]["field_size"] = [], 0
    transport(body=empty)
    with pytest.raises(pool.EmptyPool):
        pool.refresh(store=store)
    assert not issubclass(pool.EmptyPool, pool.AuthFailure)
    assert pool.main(["--refresh", "--store", str(store)]) != 0
    said = capsys.readouterr().err
    assert "no entries" in said and "401" not in said


# --- 4. a network failure serves the last-known state ---------------------------------------

def test_a_network_failure_serves_the_last_known_state_rather_than_failing_the_refresh(
        store, session, transport, capsys):
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    capsys.readouterr()

    transport(raises=OSError("connection refused"))
    assert pool.main(["--refresh", "--store", str(store)]) == 0, (
        "the weekly refresh failed on a host that was merely unreachable")
    out = capsys.readouterr()
    assert "last-known" in out.err and "connection refused" in out.err
    assert "5 entries" in out.out and "pot $100.00" in out.out

    # The same failure on a fresh clone is the one case with nothing to serve, and it is
    # reported the way every entry point reports absent input.
    fresh = store.parent / "fresh"
    assert pool.main(["--refresh", "--store", str(fresh)]) == 1
    assert "unavailable" in capsys.readouterr().err


def test_a_missing_cookie_serves_the_last_known_state_and_names_the_variable(
        store, session, transport, monkeypatch, capsys):
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    capsys.readouterr()
    monkeypatch.delenv(pool.SESSION_ENV)
    calls = transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    assert calls == [], "no cookie, and the host was asked anyway"
    err = capsys.readouterr().err
    assert pool.SESSION_ENV in err and "last-known" in err


# --- 5. revealed picks from a completed week populate rival Ledgers -------------------------

def test_revealed_picks_from_a_final_week_populate_rival_ledgers_and_an_open_week_adds_none():
    """Member A's week-3 pick is in the payload -- our own pick is visible to us before the
    deadline even under Hidden Picks -- and week 3 is open, so SF is not spent. Weeks 1 and
    2 are final and every pick in them is."""
    state = parse(payload())
    ledgers = pool.ledgers(state)
    assert ledgers[0] == {"DAL", "KC"}, "an open week's pick reached the Ledger"
    assert ledgers[3] == {"DAL", "KC"}
    assert ledgers[2] == {"NYG"}
    assert len(ledgers) == state.field_size

    # And the line moves with the status, not with the week number: the same payload with
    # week 3 final counts SF.
    p = payload()
    p["weeks"][2]["status"] = "final"
    assert pool.ledgers(parse(p))[0] == {"DAL", "KC", "SF"}


def test_ledgers_are_in_entry_index_order_for_the_simulator():
    """`hub.season.pool.simulate` takes `ledgers` in entry order and checks the count
    against `entries`, so the list has to be one per entry, index for index."""
    state = parse(payload())
    got = pool.ledgers(state)
    assert len(got) == len(state.entries)
    assert all(got[e.index] == set(e.used) for e in state.entries)


# --- 6. the cookie is read from the environment and reaches nothing ------------------------

def test_the_cookie_comes_from_the_environment_and_reaches_no_artifact_or_message(
        store, session, transport, capsys):
    """The cookie is sent to the host and nowhere else. Every file the refresh writes and
    every sentence it prints -- the success path, a transport error that quotes its own
    request, and a refused cookie -- is searched for the value."""
    calls = transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    assert calls and calls[0][1] == COOKIE, "the cookie in the environment was not the one sent"

    # The error path's message. A transport that echoes its request -- which `requests`
    # exceptions do, header and all -- must be scrubbed before it is printed.
    transport(raises=OSError(f"GET failed with Cookie: session={COOKIE}"))
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    with pytest.raises(OSError) as e:
        pool.refresh(store=store)
    _no_secret_in(str(e.value), "the exception text")

    transport(body={"detail": f"bad session {COOKIE}"}, status=401)
    with pytest.raises(pool.AuthFailure) as auth:
        pool.refresh(store=store)
    _no_secret_in(str(auth.value), "the auth failure's text")
    pool.main(["--refresh", "--store", str(store)])
    pool.main(["--status", "--store", str(store)])

    out = capsys.readouterr()
    _no_secret_in(out.out + out.err, "stdout or stderr")
    files = _stored_text(store)
    assert files, "nothing was written, so nothing was searched"
    for f, text in files:
        _no_secret_in(text, str(f))


def test_the_cookie_is_read_through_dotenv_like_every_other_credential(monkeypatch):
    seen = []
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: seen.append(True))
    monkeypatch.setenv(pool.SESSION_ENV, COOKIE)
    assert pool._cookie() == COOKIE
    assert seen, "the cookie was read without loading .env, so a laptop's .env is ignored"


# --- 7. entry identifiers are replaced with an internal index before caching --------------

def test_entry_identifiers_are_replaced_with_an_internal_index_before_caching(
        store, session, transport):
    """Other members are real people. Their ids and names go no further than the parser:
    nothing under the store carries either, and the Ledger is addressed by index."""
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    files = _stored_text(store)
    assert files and any(p.suffix == ".parquet" for p, _ in files), "the archive is scanned too"
    text = "\n".join(t for _, t in files)
    for member in ("ent-884", "Member A", "Member B", "Member C", "Member D", "Member E",
                   "Survivor 2026", "pool-0001"):
        assert member not in text, f"{member!r} reached the store"
    state = pool.read_state(store)
    assert state is not None
    assert [e.index for e in state.entries] == [0, 1, 2, 3, 4]
    assert not any(hasattr(e, "name") or hasattr(e, "id") for e in state.entries)


def test_the_index_is_stable_under_the_hosts_ordering_but_carries_no_identity():
    """Sorted by the host's id so two fetches of the same field agree on who is who, and
    then the id is dropped: the index is the only name an entry has from here on."""
    p = payload()
    shuffled = copy.deepcopy(p)
    shuffled["entries"] = list(reversed(shuffled["entries"]))
    a, b = parse(p), parse(shuffled)
    assert [e.used for e in a.entries] == [e.used for e in b.entries]


# --- our own entry is index 0, whatever its id --------------------------------------------

def test_our_entry_is_index_0_however_its_id_sorts():
    """`hub.season.pool.simulate` takes our Ledger at index 0. Member E's id sorts last in
    the fixture; named as ours, it is index 0 and Member A moves off it."""
    state = parse(payload(), ours="ent-8845")
    assert state.entries[0].index == 0 and state.entries[0].used == ()
    assert [e.used for e in state.entries][1:] == [("DAL", "KC"), ("BUF", "PHI"), ("NYG",),
                                                    ("DAL", "KC")]
    assert pool.ledgers(state)[0] == set()


def test_our_id_absent_from_the_entries_is_refused_before_anything_is_written(
        store, session, transport, monkeypatch, capsys):
    monkeypatch.setenv(pool.ENTRY_ENV, "ent-0000")
    with pytest.raises(ContractViolation, match=pool.ENTRY_ENV) as e:
        parse(payload(), ours="ent-0000")
    _no_secret_in(str(e.value), "the refusal")
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) != 0
    assert not list(store.glob("**/*")), "a refused payload wrote something"
    err = capsys.readouterr().err
    assert pool.ENTRY_ENV in err
    _no_secret_in(err, "stderr")


def test_an_unset_entry_id_serves_the_last_known_state_and_names_the_variable(
        store, session, transport, monkeypatch, capsys):
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    capsys.readouterr()
    monkeypatch.delenv(pool.ENTRY_ENV)
    calls = transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    assert calls == [], "no entry id, and the host was asked anyway"
    err = capsys.readouterr().err
    assert pool.ENTRY_ENV in err and "last-known" in err


# --- the index survives a membership change ------------------------------------------------

def test_a_surviving_entry_keeps_its_index_when_another_is_dropped(
        store, session, transport, capsys):
    """Refresh, then refresh again with Member C gone from the payload. Every entry still
    there keeps the index it had, C's index is never handed to anybody, and a newcomer
    takes the next free one."""
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    first = pool.read_state(store)
    assert first is not None
    before = {e.index: e.used for e in first.entries}
    assert sorted(before) == [0, 1, 2, 3, 4]

    later = payload()
    later["entries"] = [e for e in later["entries"] if e["id"] != "ent-8843"]
    later["entries"].append({"id": "ent-9999", "name": "Member F", "alive": True,
                             "picks": [{"week": 1, "team": "MIA", "result": "win"}]})
    transport(body=later)
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    second = pool.read_state(store)
    assert second is not None
    after = {e.index: e.used for e in second.entries}
    assert 2 not in after, "the dropped entry's index was reused"
    assert after[5] == ("MIA",), "the newcomer did not take the next free index"
    for i in (0, 1, 3, 4):
        assert after[i] == before[i], f"entry {i} moved"
    assert "5 entries" in capsys.readouterr().out


def test_the_index_map_holds_hashes_and_never_a_raw_id_or_a_name(store, session, transport):
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    path = pool.index_map_path(store)
    assert path.exists()
    text = path.read_text()
    for member in ("ent-884", "Member", OURS):
        assert member not in text, f"{member!r} reached the index map"
    doc = json.loads(text)
    assert all(len(k) == 64 and int(k, 16) >= 0 for k in doc), "keys are not sha256 hex"
    assert sorted(doc.values()) == [0, 1, 2, 3, 4]
    assert doc[pool.entry_key(OURS)] == 0


def test_the_index_map_is_append_only(store):
    """An index once assigned is never changed and never dropped, whoever is in the payload
    this week: a rewrite that lost a row would hand the next newcomer a rival's Ledger."""
    _, index_map = pool.parse_payload(payload(), ours=OURS)
    pool.write_index_map(index_map, store)
    smaller = dict(list(index_map.items())[:2])
    pool.write_index_map(smaller, store)
    assert pool.read_index_map(store) == index_map
    moved = dict(index_map)
    moved[pool.entry_key("ent-8842")] = 9
    with pytest.raises(ContractViolation, match="already"):
        pool.write_index_map(moved, store)


def test_a_map_built_for_another_entry_id_is_refused_rather_than_moved():
    """Index 0 is ours by construction. A map that has somebody else there, or has our id
    somewhere else, was built under a different `POOL_ENTRY_ID` and is refused rather than
    quietly renumbered around."""
    _, index_map = pool.parse_payload(payload(), ours=OURS)
    with pytest.raises(ContractViolation, match="index 0"):
        pool.parse_payload(payload(), ours="ent-8842", index_map=index_map)


# --- the shape stored, and the entry point around it ---------------------------------------

def test_the_stored_state_reads_back_as_it_was_written(store):
    state = parse(payload())
    pool.write_state(state, store)
    assert pool.read_state(store) == state


def test_the_state_file_is_written_byte_for_byte_as_it_was_before_the_shared_fetcher(store):
    """#255 moved the stamp write into `hub.fetch.cached`. The document is what the site,
    the journal and `captured_at` read, so its keys, their order and the capture time's
    spelling -- aware, with its offset, to the second -- are pinned here."""
    import datetime as dt
    state = parse(payload())
    when = dt.datetime(2026, 9, 12, 12, 0, 7, 500, tzinfo=dt.UTC)
    doc = json.loads(pool.write_state(state, store, when=when).read_text())
    assert list(doc) == ["captured_at", "season", "week", "field_size", "pot", "entries"]
    assert doc["captured_at"] == "2026-09-12T12:00:07+00:00"
    assert pool.captured_at(store) == "2026-09-12T12:00:07+00:00"
    assert doc["entries"][0] == {"entry": 0, "alive": True, "used": ["DAL", "KC"]}
    assert pool.state_path(store).read_text().startswith('{\n  "captured_at": ')


def _later(p: dict, *, week: int, pot: float, alive: int) -> dict:
    """The fixture a week on: the host's week advanced, the pot grown, `alive` entries left.
    Members from the end of the list are the ones marked out."""
    q = copy.deepcopy(p)
    q["pool"]["current_week"], q["pool"]["pot"] = week, pot
    q["weeks"] = [{"week": w, "status": "final"} for w in range(1, week)] + [
        {"week": week, "status": "open"}]
    for e in q["entries"]:
        e["picks"] = [k for k in e["picks"] if k["week"] < week]
    for e in q["entries"][alive:]:
        e["alive"] = False
    return q


# --- the archive (#280): one partition per read, append-only, hashed -----------------------

def test_each_read_is_archived_under_its_week_with_its_capture_time_and_none_is_rewritten(
        store):
    """The state file is overwritten every refresh -- one capture time, no history -- so the
    field a week-3 decision was priced against was gone the moment week 4 was fetched. Every
    write now also lands a dated partition under the week, in the layout the lines archive
    uses, and a second write of the same moment with different contents is refused rather
    than rewritten."""
    import datetime as dt
    first = parse(payload())
    t1 = dt.datetime(2026, 9, 17, 9, 0, tzinfo=dt.UTC)
    pool.write_state(first, store, when=t1)
    later = parse(_later(payload(), week=4, pot=140.0, alive=3))
    t2 = dt.datetime(2026, 9, 24, 9, 0, tzinfo=dt.UTC)
    pool.write_state(later, store, when=t2)

    parts = sorted(p.relative_to(store).as_posix() for p in store.rglob("*.parquet"))
    assert parts == [
        "pool_state/league=nfl/season=2026/week=03/snap-20260917T090000000000.parquet",
        "pool_state/league=nfl/season=2026/week=04/snap-20260924T090000000000.parquet"]
    assert pool.read_state(store) == later, "the state file still serves the latest read"
    got = pool.archived(2026, base=store)
    assert [(c, s) for c, s in got] == [(t1.replace(tzinfo=None), first),
                                        (t2.replace(tzinfo=None), later)]
    with pytest.raises(FileExistsError):
        pool.write_state(later._replace(pot=141.0), store, when=t2)
    assert pool.read_state(store) == later, "a refused archive write left the state file alone"
    assert pool.archived(2026, base=store, week=4) == [(t2.replace(tzinfo=None), later)]
    assert pool.archived(2025, base=store) == []


def test_a_state_digest_names_one_field_and_finds_it_in_the_archive(store):
    """The hash a journal row carries: eight hex characters over what the state says --
    week, field size, pot, every entry's index, liveness and Ledger -- the same across a
    round trip through the archive, and different for a field that differs by a dollar.
    `archived_state` is how a row's digest is resolved back to the field it named."""
    import datetime as dt
    first = parse(payload())
    d = pool.state_digest(first)
    assert len(d) == 8 and int(d, 16) >= 0
    assert pool.state_digest(first._replace(pot=first.pot + 1.0)) != d
    assert pool.archived_state(d, season=2026, base=store) is None, "nothing archived yet"
    pool.write_state(first, store, when=dt.datetime(2026, 9, 17, 9, 0, tzinfo=dt.UTC))
    later = parse(_later(payload(), week=4, pot=140.0, alive=3))
    pool.write_state(later, store, when=dt.datetime(2026, 9, 24, 9, 0, tzinfo=dt.UTC))
    back = pool.archived_state(d, season=2026, base=store)
    assert back is not None and back == first and pool.state_digest(back) == d
    assert pool.archived_state(pool.state_digest(later), season=2026, base=store) == later
    assert pool.read_state(store) == later != first


def test_the_archive_carries_an_index_and_never_a_name(store, session, transport):
    """The privacy design, kept: a partition holds exactly what the contract validated plus
    the capture time, so nothing a member could be recognised by is on disk in the archive
    either. Read as parquet, since a byte scan of a columnar file is not a check."""
    import polars as pl
    transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 0
    files = list(store.rglob("*.parquet"))
    assert files
    for p in files:
        df = pl.read_parquet(p)
        assert set(df.columns) == {"entry", "alive", "used", "field_size", "pot", "captured_at"}
        text = df.write_json()
        for member in ("ent-884", "Member", "Survivor 2026", "pool-0001"):
            assert member not in text, f"{member!r} reached the archive"


def test_a_stored_state_is_validated_by_the_contract_on_the_way_out(store):
    """`read_state` refuses a file that has drifted from the declared shape rather than
    serving it as last-known: a partial cache is the one thing worse than none."""
    pool.write_state(parse(payload()), store)
    path = next(store.rglob("*.json"))
    doc = json.loads(path.read_text())
    del doc["entries"][0]["used"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ContractViolation):
        pool.read_state(store)


def test_a_field_of_the_wrong_kind_is_refused_by_name():
    """`alive` as the string "yes" and `pot` as `true` are both fields that are present and
    still not the fact the state needs; the refusal names the field and the kind wanted."""
    p = payload()
    p["entries"][0]["alive"] = "yes"
    with pytest.raises(ContractViolation, match=r"entries\[0\].alive is str, expected bool"):
        parse(p)
    p = payload()
    p["pool"]["pot"] = True
    with pytest.raises(ContractViolation, match=r"pool\.pot is bool"):
        parse(p)


def test_an_entry_id_listed_twice_is_refused():
    """Two entries under one id could not both be numbered, and the second would silently
    take the first's index."""
    p = payload()
    p["entries"][1]["id"] = p["entries"][2]["id"]
    with pytest.raises(ContractViolation, match="listed twice"):
        parse(p)


def test_a_week_described_twice_is_refused():
    p = payload()
    p["weeks"].append({"week": 2, "status": "open"})
    with pytest.raises(ContractViolation, match="week 2 is described twice"):
        parse(p)


def test_the_cookie_name_comes_from_the_environment_with_a_stated_default(monkeypatch):
    monkeypatch.delenv(pool.COOKIE_NAME_ENV, raising=False)
    assert pool._cookie_name() == pool.DEFAULT_COOKIE_NAME
    monkeypatch.setenv(pool.COOKIE_NAME_ENV, "sid")
    assert pool._cookie_name() == "sid"


def test_a_drifted_cache_is_not_served_over_a_failed_refresh(store, session, transport, capsys):
    """The last-known state is read through the contract on the way out, so a cache that
    no longer parses is reported as unavailable beside the refresh's own failure."""
    transport()
    pool.main(["--refresh", "--store", str(store)])
    path = pool.state_path(store)
    doc = json.loads(path.read_text())
    doc["entries"][0]["entry"] = doc["entries"][1]["entry"]
    path.write_text(json.dumps(doc))
    capsys.readouterr()
    transport(raises=OSError("down"))
    assert pool.main(["--refresh", "--store", str(store)]) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "not unique" in err and "down" in err


def test_a_saved_payload_that_does_not_parse_is_a_sentence_not_a_traceback(store, tmp_path, capsys):
    f = tmp_path / "payload.json"
    f.write_text("{not json")
    assert pool.main(["--payload", str(f), "--store", str(store)]) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "Traceback" not in err


def test_a_season_other_than_the_one_asked_for_is_refused():
    """A stale URL pointing at last year's pool would otherwise write last year's field as
    this year's state."""
    with pytest.raises(ContractViolation, match="2025"):
        parse(payload(), season=2025)


def test_a_saved_payload_file_is_ingested_offline(store, tmp_path, session, monkeypatch, capsys):
    """The first live run's escape hatch: a payload saved from the browser is read from
    disk, spends nothing, needs no cookie, and lands in the store as a refresh would."""
    monkeypatch.delenv(pool.SESSION_ENV)
    f = tmp_path / "payload.json"
    f.write_text(json.dumps(payload()))
    assert pool.main(["--payload", str(f), "--store", str(store)]) == 0
    assert pool.read_state(store) is not None
    assert "5 entries" in capsys.readouterr().out


def test_status_reports_the_last_known_state_and_the_arguments_the_money_layer_takes(
        store, session, transport, capsys):
    transport()
    pool.main(["--refresh", "--store", str(store)])
    capsys.readouterr()
    assert pool.main(["--status", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "--entries 4" in out and "--pot 100.00" in out
    assert "entry 0" in out and "DAL, KC" in out


def test_status_on_a_fresh_clone_is_a_sentence_and_a_non_zero_exit(store, capsys):
    assert pool.main(["--status", "--store", str(store)]) == 1
    err = capsys.readouterr().err
    assert "unavailable" in err and "Traceback" not in err


def test_an_unset_host_is_reported_before_any_cookie_is_read(store, session, transport,
                                                             monkeypatch, capsys):
    monkeypatch.delenv(pool.URL_ENV)
    calls = transport()
    assert pool.main(["--refresh", "--store", str(store)]) == 1
    assert calls == []
    assert pool.URL_ENV in capsys.readouterr().err


def test_the_transport_refuses_a_live_call_from_the_default_suite(monkeypatch):
    """`hub.fetch.cfbd`'s guard, at the one function here that touches the network."""
    import requests

    def _never(*a, **k):                                     # pragma: no cover - must not run
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(requests, "get", _never)
    with pytest.raises(pool.LiveCallRefused) as e:
        pool._http_get("https://pool.example.test/api/pools/1", COOKIE)
    assert "test_the_transport_refuses_a_live_call" in str(e.value)
    _no_secret_in(str(e.value), "the refusal")
