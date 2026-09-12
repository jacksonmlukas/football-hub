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


def payload() -> dict:
    return json.loads((FIXTURES / "pool_payload.synthetic.json").read_text())


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


# --- 1. a payload missing a required field is refused, not partially served ---------------

def test_a_payload_missing_a_required_field_is_refused_and_nothing_partial_is_written(
        store, session, transport, capsys):
    """The pot is gone from the payload. The parser refuses with a contract violation that
    names the field, and the store is exactly as it was: no file appears with the field
    size and the entries in it and a pot of nothing."""
    broken = payload()
    del broken["pool"]["pot"]
    with pytest.raises(ContractViolation, match="pot"):
        pool.parse_payload(broken)

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
        pool.parse_payload(broken)


def test_a_field_size_that_disagrees_with_the_entries_listed_is_refused():
    """A field of five with four entries in the body is a truncated or paginated payload,
    and reading the four as the field is the partial state the criterion forbids."""
    broken = payload()
    broken["pool"]["field_size"] = 6
    with pytest.raises(ContractViolation, match="field_size"):
        pool.parse_payload(broken)


def test_a_pick_in_a_week_the_payload_does_not_describe_is_refused():
    """A pick can only be counted or not counted by the status of its week, so a week with
    no status is a pick that cannot be placed either side of the line."""
    broken = payload()
    broken["entries"][0]["picks"].append({"week": 4, "team": "MIA", "result": None})
    with pytest.raises(ContractViolation, match="week 4"):
        pool.parse_payload(broken)


# --- 2. field size, pot and per-entry used teams --------------------------------------------

def test_field_size_pot_and_each_entrys_used_teams_are_parsed_from_the_payload():
    state = pool.parse_payload(payload())
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
    assert pool.parse_payload(p).entries[1].used == ("BUF", "PHI")
    p["entries"][1]["picks"][0]["team"] = " "
    with pytest.raises(ContractViolation, match="team"):
        pool.parse_payload(p)


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
    state = pool.parse_payload(payload())
    ledgers = pool.ledgers(state)
    assert ledgers[0] == {"DAL", "KC"}, "an open week's pick reached the Ledger"
    assert ledgers[3] == {"DAL", "KC"}
    assert ledgers[2] == {"NYG"}
    assert len(ledgers) == state.field_size

    # And the line moves with the status, not with the week number: the same payload with
    # week 3 final counts SF.
    p = payload()
    p["weeks"][2]["status"] = "final"
    assert pool.ledgers(pool.parse_payload(p))[0] == {"DAL", "KC", "SF"}


def test_ledgers_are_in_entry_index_order_for_the_simulator():
    """`hub.season.pool.simulate` takes `ledgers` in entry order and checks the count
    against `entries`, so the list has to be one per entry, index for index."""
    state = pool.parse_payload(payload())
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
    files = [p for p in store.rglob("*") if p.is_file()]
    assert files, "nothing was written, so nothing was searched"
    for f in files:
        _no_secret_in(f.read_text(), str(f))


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
    files = [p for p in store.rglob("*") if p.is_file()]
    assert files
    text = "\n".join(p.read_text() for p in files)
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
    a, b = pool.parse_payload(p), pool.parse_payload(shuffled)
    assert [e.used for e in a.entries] == [e.used for e in b.entries]


# --- the shape stored, and the entry point around it ---------------------------------------

def test_the_stored_state_reads_back_as_it_was_written(store):
    state = pool.parse_payload(payload())
    pool.write_state(state, store)
    assert pool.read_state(store) == state


def test_a_stored_state_is_validated_by_the_contract_on_the_way_out(store):
    """`read_state` refuses a file that has drifted from the declared shape rather than
    serving it as last-known: a partial cache is the one thing worse than none."""
    pool.write_state(pool.parse_payload(payload()), store)
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
        pool.parse_payload(p)
    p = payload()
    p["pool"]["pot"] = True
    with pytest.raises(ContractViolation, match=r"pool\.pot is bool"):
        pool.parse_payload(p)


def test_a_week_described_twice_is_refused():
    p = payload()
    p["weeks"].append({"week": 2, "status": "open"})
    with pytest.raises(ContractViolation, match="week 2 is described twice"):
        pool.parse_payload(p)


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
        pool.parse_payload(payload(), season=2025)


def test_a_saved_payload_file_is_ingested_offline(store, tmp_path, capsys):
    """The first live run's escape hatch: a payload saved from the browser is read from
    disk, spends nothing, needs no cookie, and lands in the store as a refresh would."""
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
