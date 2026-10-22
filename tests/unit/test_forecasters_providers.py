"""`hub.forecasters.providers`: one thin adapter per provider behind `runner.Client`.

Each adapter reads exactly one environment variable and refuses with one sentence when it is
absent; each request is built from the pinned settings (temperature only where the config
gives one, the provider's own spelling of reasoning effort); each reply becomes a
`Completion` with the tokens the cost log prices. No adapter reaches the network here: every
call goes through a stubbed transport that records what it was handed, and the default
transport is refused under pytest before anything leaves the process.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from hub.fetch import cached
from hub.forecasters import config as study_config
from hub.forecasters import providers, runner
from hub.forecasters.config import LlmSpec, StudyConfig

KEY = "sk-test-0123456789abcdef"
ENV = {"ANTHROPIC_API_KEY": KEY, "OPENAI_API_KEY": KEY, "GEMINI_API_KEY": KEY}
ANSWER = '{"p_home_win": 0.61, "rationale": "home field"}'


def _spec(provider: str, **kw) -> LlmSpec:
    base = LlmSpec(name=f"{provider}-m", provider=provider, model_version=f"{provider}-model-x",
                   training_cutoff="2025-06", temperature=None, reasoning_effort=None,
                   max_output_tokens=2000, usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)
    return replace(base, **kw)


REPLIES = {
    "anthropic": {"content": [{"type": "thinking", "thinking": ""},
                              {"type": "text", "text": ANSWER}],
                  "usage": {"input_tokens": 900, "output_tokens": 120}},
    "openai": {"output": [{"type": "reasoning", "summary": []},
                          {"type": "message", "content": [{"type": "output_text",
                                                           "text": ANSWER}]}],
               "usage": {"input_tokens": 900, "output_tokens": 120}},
    "google": {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                     {"text": ANSWER}]}}],
               "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 80,
                                 "thoughtsTokenCount": 40}},
}


class Stub:
    """A transport that never leaves the process: records each call, answers from `reply`."""

    def __init__(self, reply: Mapping[str, Any] | None = None,
                 raises: Exception | None = None):
        self.reply, self.raises = reply, raises
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, headers: Mapping[str, str],
                 body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((url, dict(headers), dict(body)))
        if self.raises is not None:
            raise self.raises
        return self.reply or {}


def _client(provider: str, stub: Stub, env=ENV):
    return providers.ADAPTERS[provider](env=env, transport=stub)


# --- keys ---------------------------------------------------------------------------------

def test_the_key_names_are_exactly_the_workflow_secrets():
    assert providers.KEY_ENV == {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
                                 "google": "GEMINI_API_KEY"}
    assert providers.TAVILY_KEY_ENV == "TAVILY_API_KEY"


@pytest.mark.parametrize("provider", sorted(providers.ADAPTERS))
@pytest.mark.parametrize("value", [None, "", "   "])
def test_each_adapter_refuses_in_one_sentence_without_its_key(monkeypatch, provider, value):
    name = providers.KEY_ENV[provider]
    for k in providers.KEY_ENV.values():
        monkeypatch.setenv(k, KEY)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    stub = Stub(REPLIES[provider])
    with pytest.raises(providers.MissingKey) as exc:
        providers.ADAPTERS[provider](transport=stub)          # reads os.environ
    msg = str(exc.value)
    assert msg.startswith(name) and msg.count(". ") == 0 and msg.endswith(".")
    assert "\n" not in msg and KEY not in msg
    assert exc.value.__cause__ is None and stub.calls == []


def test_another_providers_key_does_not_stand_in(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    with pytest.raises(providers.MissingKey, match="GEMINI_API_KEY"):
        providers.Google(transport=Stub())


def test_the_router_refuses_before_any_call_when_one_provider_has_no_key():
    cfg = StudyConfig(llms=(_spec("anthropic"), _spec("google")), season_cap_usd=150,
                      home_rate=0.547, home_rate_seasons="2016-2025", est_input_tokens=2000)
    stub = Stub()
    with pytest.raises(providers.MissingKey, match="GEMINI_API_KEY"):
        providers.Router(cfg, env={"ANTHROPIC_API_KEY": KEY}, transport=stub)
    assert stub.calls == []


def test_the_router_refuses_an_unknown_provider():
    cfg = StudyConfig(llms=(_spec("xai"),), season_cap_usd=150, home_rate=0.547,
                      home_rate_seasons="2016-2025", est_input_tokens=2000)
    with pytest.raises(ValueError, match="xai"):
        providers.Router(cfg, env=ENV, transport=Stub())


# --- requests and replies -----------------------------------------------------------------

@pytest.mark.parametrize("provider", sorted(providers.ADAPTERS))
def test_each_adapter_reads_its_reply_into_a_completion(provider):
    stub = Stub(REPLIES[provider])
    c = _client(provider, stub).complete(_spec(provider), "SYSTEM", "USER")
    assert c == runner.Completion(ANSWER, 900, 120)
    assert runner.parse_sample(c.text) is not None and len(stub.calls) == 1


@pytest.mark.parametrize("provider", sorted(providers.ADAPTERS))
def test_the_key_goes_in_a_header_and_never_in_the_url_or_body(provider):
    stub = Stub(REPLIES[provider])
    _client(provider, stub).complete(_spec(provider), "SYSTEM", "USER")
    url, headers, body = stub.calls[0]
    assert KEY not in url and KEY not in repr(body)
    assert any(KEY in v for v in headers.values())
    assert url.startswith("https://")


def test_anthropic_request_shape():
    stub = Stub(REPLIES["anthropic"])
    _client("anthropic", stub).complete(_spec("anthropic", reasoning_effort="high"), "S", "U")
    url, headers, body = stub.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body == {"model": "anthropic-model-x", "max_tokens": 2000, "system": "S",
                    "messages": [{"role": "user", "content": "U"}],
                    "output_config": {"effort": "high"}}


def test_openai_request_shape():
    stub = Stub(REPLIES["openai"])
    _client("openai", stub).complete(_spec("openai", reasoning_effort="medium"), "S", "U")
    url, _, body = stub.calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert body == {"model": "openai-model-x", "instructions": "S", "input": "U",
                    "max_output_tokens": 2000, "store": False,
                    "reasoning": {"effort": "medium"}}


def test_google_request_shape_and_thoughts_are_billed_but_not_answered():
    stub = Stub(REPLIES["google"])
    c = _client("google", stub).complete(_spec("google", reasoning_effort="low"), "S", "U")
    url, _, body = stub.calls[0]
    assert url.endswith("/models/google-model-x:generateContent")
    assert body["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert body["generationConfig"] == {"maxOutputTokens": 2000,
                                        "thinkingConfig": {"thinkingLevel": "low"}}
    assert "thinking..." not in c.text and c.output_tokens == 120


@pytest.mark.parametrize("provider", sorted(providers.ADAPTERS))
def test_temperature_is_sent_only_where_the_config_gives_one(provider):
    stub = Stub(REPLIES[provider])
    a = _client(provider, stub)
    a.complete(_spec(provider), "S", "U")
    a.complete(_spec(provider, temperature=0.7), "S", "U")
    without, with_ = (repr(call[2]) for call in stub.calls)
    assert "temperature" not in without and "'temperature': 0.7" in with_


# --- failures -----------------------------------------------------------------------------

@pytest.mark.parametrize("provider", sorted(providers.ADAPTERS))
def test_a_transport_failure_never_carries_the_request_or_the_key(provider):
    leaky = ConnectionError(f"POST https://x/?key={KEY} headers={{'x-api-key': '{KEY}'}}")
    with pytest.raises(providers.ProviderError) as exc:
        _client(provider, Stub(raises=leaky)).complete(_spec(provider), "S", "U")
    assert KEY not in str(exc.value) and "x-api-key" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.parametrize("provider", sorted(providers.ADAPTERS))
def test_an_unreadable_reply_is_a_provider_error(provider):
    with pytest.raises(providers.ProviderError, match="unreadable"):
        _client(provider, Stub({"weird": 1})).complete(_spec(provider), "S", "U")


def test_the_router_turns_a_failed_call_into_a_lost_sample_not_a_crash():
    cfg = StudyConfig(llms=(_spec("openai"),), season_cap_usd=150, home_rate=0.547,
                      home_rate_seasons="2016-2025", est_input_tokens=2000)
    stub = Stub(raises=TimeoutError(KEY))
    c = providers.Router(cfg, env=ENV, transport=stub).complete(cfg.llms[0], "S", "U")
    assert runner.parse_sample(c.text) is None
    assert (c.input_tokens, c.output_tokens) == (0, 0) and KEY not in c.text


def test_the_router_dispatches_by_provider():
    cfg = StudyConfig(llms=(_spec("anthropic"), _spec("openai"), _spec("google")),
                      season_cap_usd=150, home_rate=0.547, home_rate_seasons="2016-2025",
                      est_input_tokens=2000)
    stub = Stub()
    router = providers.Router(cfg, env=ENV, transport=stub)
    for spec in cfg.llms:
        stub.reply = REPLIES[spec.provider]
        assert router.complete(spec, "S", "U").text == ANSWER
    hosts = [url.split("/")[2] for url, _, _ in stub.calls]
    assert hosts == ["api.anthropic.com", "api.openai.com", "generativelanguage.googleapis.com"]


def test_the_default_transport_is_refused_under_pytest_before_anything_leaves():
    a = providers.OpenAI(env=ENV)                              # no transport: the real one
    with pytest.raises(cached.LiveCallRefused):
        providers._post("https://api.openai.com/v1/responses", {}, {})
    with pytest.raises(providers.ProviderError):               # and complete() wraps it
        a.complete(_spec("openai"), "S", "U")


# --- the committed config ------------------------------------------------------------------

def test_the_committed_model_fields_are_still_tbd_and_refused():
    cfg = study_config.load()
    assert all(m.model_version == study_config.UNSET for m in cfg.llms)
    with pytest.raises(runner.NotPinned) as exc:
        runner.require_pinned(cfg)
    for m in cfg.llms:
        assert f"{m.name}.model_version" in str(exc.value)
        assert f"{m.name}.provider" in str(exc.value)
