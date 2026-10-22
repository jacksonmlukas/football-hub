"""Every espn_api request is time-bounded, and a stall is an ordinary failure (#519)."""
import http.server
import threading

import polars as pl
import pytest
import requests

import hub.fetch.espn as E
from hub.fetch import free_agents as FA

DEADLINE = 8.0   # the hang tests must finish well inside this; a regression fails, never hangs


class _Stalled:
    """A local server that accepts the request and never answers until released."""

    def __enter__(self):
        self.release = threading.Event()
        release = self.release

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                release.wait(30)

            def log_message(self, format, *args):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.srv.shutdown()
        self.srv.server_close()


def _bounded(fn):
    """Run `fn` on a thread with a deadline: (returned, raised); fails if it outlives it."""
    box: dict = {}

    def run():
        try:
            box["ok"] = fn()
        except BaseException as e:           # the test inspects it
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(DEADLINE)
    assert not t.is_alive(), "the ESPN read hung past the deadline: no timeout is set"
    return box.get("ok"), box.get("err")


@pytest.fixture
def fast_timeout(monkeypatch):
    monkeypatch.setattr(E, "ESPN_API_TIMEOUT", 0.3)


def _fake_league(url):
    """Stands in for espn_api's League: builds its request object and does its first read."""
    from espn_api.requests.espn_requests import EspnFantasyRequests

    class FakeLeague:
        def __init__(self, **kw):
            r = EspnFantasyRequests("nfl", kw["year"], kw["league_id"], cookies={})
            r.LEAGUE_ENDPOINT = url
            r.league_get()
    return FakeLeague


def test_a_stalled_library_request_raises_promptly(fast_timeout):
    from espn_api.requests.espn_requests import EspnFantasyRequests
    E.bound_espn_api_requests()
    with _Stalled() as srv:
        r = EspnFantasyRequests("nfl", 2026, 1, cookies={})
        r.LEAGUE_ENDPOINT = srv.url
        _, err = _bounded(r.league_get)
    assert isinstance(err, requests.exceptions.Timeout)


def test_the_bound_is_scoped_to_espn_api_and_a_callers_timeout_wins(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "get", lambda *a, **k: seen.update(k) or "resp")
    E.bound_espn_api_requests()
    E.bound_espn_api_requests()                      # idempotent
    from espn_api.requests import espn_requests as m
    assert m.requests.get("u") == "resp" and seen["timeout"] == E.ESPN_API_TIMEOUT
    m.requests.get("u", timeout=3)
    assert seen["timeout"] == 3
    assert not isinstance(requests, E._BoundedRequests)   # the global module is untouched


def test_a_stalled_free_agent_read_serves_last_good_with_the_reason(
        fast_timeout, monkeypatch, tmp_path):
    parquet = tmp_path / "free_agents.parquet"
    pl.DataFrame({"pos": ["RB"], "player": ["Kept"]}).write_parquet(parquet)
    monkeypatch.setattr(E, "resolve_league_id", lambda override=None: 1)
    # Start unbound, so only `league_settings` itself can install the bound (order-independent).
    from espn_api.requests import espn_requests
    monkeypatch.setattr(espn_requests, "requests", requests)
    with _Stalled() as srv:
        monkeypatch.setattr("espn_api.football.League", _fake_league(srv.url))
        got, err = _bounded(lambda: FA.refresh(E.fetch_free_agents, parquet=parquet))
    assert err is None and got is not None
    line, df = got
    assert df is not None
    assert "serving last-good" in line and "Timeout" in line
    assert df["player"].to_list() == ["Kept"]
    assert "Timeout" in FA.read_meta(parquet)["last_error"]


def test_everything_but_get_passes_through_to_requests():
    assert E._BoundedRequests().exceptions is requests.exceptions
