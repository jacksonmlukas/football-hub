"""The provider clients: one thin adapter per model provider behind `runner.Client`.

`docs/prereg/llm-forecasters.md`, *Contestants*: four models from at least three providers --
the newest flagship from Anthropic, OpenAI and Google plus one cheaper model (maintainer,
2026-10-21, #448). Each adapter turns `(spec, system, user)` into one request, and the reply
into a `runner.Completion`: the text, and the input and output tokens the cost log multiplies
by the pinned prices. No tools and no web search on the model's side: the one retrieval
contestant (#511) searches *before* the call, through `hub.fetch.tavily`, and hands the model
its results in the prompt (`hub.forecasters.retrieval`); `TAVILY_KEY_ENV` is that adapter's
key name, restated here beside the others.

**Plain HTTPS through `requests`, not the three official SDKs.** `requests` is already a
dependency (`hub.fetch.odds`, `hub.fetch.espn`); `anthropic`, `openai` and `google-genai`
would add three clients, an HTTP stack each (httpx2/httpx, pydantic, anyio, google-auth) and
their own retry loops, for one non-streaming POST per sample. Their retries would also be a
second, invisible retry policy on top of the prereg's (a sample that fails is retried up to
twice, `runner.forecast_llm`), spending money the cost log never sees. One transport seam,
`Transport`, is also what makes "no network in tests" one assertion rather than three
monkeypatches of three SDKs' internals.

**Keys.** Each adapter reads exactly one environment variable -- `KEY_ENV` -- and refuses with
one sentence (`MissingKey`, no traceback, no key material) when it is absent or empty. The
key travels in a header, never in a URL (Google's API takes `x-goog-api-key` for that reason),
and no header is ever logged: a failed call becomes a `ProviderError` whose text this module
writes itself -- the exception's type name and the HTTP status -- never the transport's own
message, which can quote the request.

**A failed call is a lost sample, not a crashed run.** `Router` turns a `ProviderError` into an
empty `Completion` with zero tokens; `runner.parse_sample` rejects it like any unparseable
answer, the runner retries it within the prereg's limit, and with fewer than three valid
samples that contestant has no forecast for the game -- the prereg's "API failure" row. The
failure's one-line reason lands in the record's `parse_failures`.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from hub.fetch import cached, tavily
from hub.forecasters.config import LlmSpec, StudyConfig
from hub.forecasters.runner import Completion, call_settings

ANTHROPIC, OPENAI, GOOGLE = "anthropic", "openai", "google"

# The secret names, exactly as `.github/workflows/forecast.yml` maps them into the run step.
KEY_ENV: dict[str, str] = {ANTHROPIC: "ANTHROPIC_API_KEY", OPENAI: "OPENAI_API_KEY",
                           GOOGLE: "GEMINI_API_KEY"}
TAVILY_KEY_ENV = tavily.KEY_ENV            # the retrieval arm's search key (#511)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/responses"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Seconds one request may take. A run must finish inside 30 minutes (the prereg's arithmetic);
# a hung connection is the one thing that could spend that silently.
REQUEST_TIMEOUT_S = 120

# One POST: (url, headers, json body) -> the decoded JSON reply. Raises on any failure.
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]]


class MissingKey(RuntimeError):
    """The provider's key is not in the environment. Nothing was sent."""


class ProviderError(RuntimeError):
    """One call failed. The text is this module's own -- never the transport's, which can
    quote the request it failed on."""


def _post(url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> Mapping[str, Any]:
    cached.refuse_live_call(f"would call a model provider at {url.split('/')[2]} (a paid call)",
                            patch="transport=", tests="tests/unit/test_forecasters_providers.py")
    return _send(url, headers, body)


def _send(url: str, headers: Mapping[str, str],
          body: Mapping[str, Any]) -> Mapping[str, Any]:           # pragma: no cover - network
    import requests
    r = requests.post(url, headers=dict(headers), json=dict(body), timeout=REQUEST_TIMEOUT_S)
    if r.status_code >= 400:
        raise ProviderError(f"HTTP {r.status_code}")
    return r.json()


def _key(provider: str, env: Mapping[str, str] | None) -> str:
    name = KEY_ENV[provider]
    value = (os.environ if env is None else env).get(name, "")
    if not value.strip():
        raise MissingKey(f"{name} is not set, so the {provider} client cannot call a model; "
                         f"add it as a repository secret (never to .env).")
    return value


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


class _Adapter:
    provider = ""

    def __init__(self, *, env: Mapping[str, str] | None = None,
                 transport: Transport | None = None) -> None:
        self._key = _key(self.provider, env)
        self._transport = transport or _post

    def request(self, spec: LlmSpec, system: str, user: str) -> tuple[str, dict[str, str],
                                                                       dict[str, Any]]:
        raise NotImplementedError

    def reply(self, data: Mapping[str, Any]) -> Completion:
        raise NotImplementedError

    def complete(self, spec: LlmSpec, system: str, user: str) -> Completion:
        url, headers, body = self.request(spec, system, user)
        try:
            data = self._transport(url, headers, body)
        except ProviderError:
            raise
        except Exception as exc:
            # The transport's message is dropped on purpose: it can quote the request.
            raise ProviderError(f"{self.provider}: {type(exc).__name__}") from None
        try:
            return self.reply(data)
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.provider}: unreadable reply "
                                f"({type(exc).__name__})") from None


class Anthropic(_Adapter):
    """The Messages API. `reasoning_effort` is `output_config.effort`; temperature only where
    the model accepts it (the newest reject it, so their config says "none")."""
    provider = ANTHROPIC

    def request(self, spec, system, user):
        s = call_settings(spec)
        body: dict[str, Any] = {"model": spec.model_version, "max_tokens": s["max_output_tokens"],
                                "system": system, "messages": [{"role": "user", "content": user}]}
        if "temperature" in s:
            body["temperature"] = s["temperature"]
        if "reasoning_effort" in s:
            body["output_config"] = {"effort": s["reasoning_effort"]}
        headers = {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        return ANTHROPIC_URL, headers, body

    def reply(self, data):
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        u = data.get("usage") or {}
        return Completion(text, _int(u.get("input_tokens")), _int(u.get("output_tokens")))


class OpenAI(_Adapter):
    """The Responses API: the system text as `instructions`, the packet as `input`, and
    `reasoning.effort` for a reasoning model. Output tokens include reasoning tokens."""
    provider = OPENAI

    def request(self, spec, system, user):
        s = call_settings(spec)
        body: dict[str, Any] = {"model": spec.model_version, "instructions": system,
                                "input": user, "max_output_tokens": s["max_output_tokens"],
                                "store": False}
        if "temperature" in s:
            body["temperature"] = s["temperature"]
        if "reasoning_effort" in s:
            body["reasoning"] = {"effort": s["reasoning_effort"]}
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        return OPENAI_URL, headers, body

    def reply(self, data):
        text = "".join(c.get("text", "") for item in data["output"]
                       if item.get("type") == "message"
                       for c in item.get("content") or [] if c.get("type") == "output_text")
        u = data.get("usage") or {}
        return Completion(text, _int(u.get("input_tokens")), _int(u.get("output_tokens")))


class Google(_Adapter):
    """`generateContent`, keyed by header. Thinking tokens are billed as output, so they are
    added to the candidates' count; a thought part's text is never part of the answer."""
    provider = GOOGLE

    def request(self, spec, system, user):
        s = call_settings(spec)
        gen: dict[str, Any] = {"maxOutputTokens": s["max_output_tokens"]}
        if "temperature" in s:
            gen["temperature"] = s["temperature"]
        if "reasoning_effort" in s:
            gen["thinkingConfig"] = {"thinkingLevel": s["reasoning_effort"]}
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen}
        headers = {"x-goog-api-key": self._key, "content-type": "application/json"}
        return GOOGLE_URL.format(model=spec.model_version), headers, body

    def reply(self, data):
        # A reply with no candidate list is unreadable; a candidate with no content (a blocked
        # answer) is an empty one, which the runner rejects like any unparseable sample.
        parts = ((data["candidates"] or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = data.get("usageMetadata") or {}
        out = _int(u.get("candidatesTokenCount")) + _int(u.get("thoughtsTokenCount"))
        return Completion(text, _int(u.get("promptTokenCount")), out)


ADAPTERS: dict[str, type[_Adapter]] = {ANTHROPIC: Anthropic, OPENAI: OpenAI, GOOGLE: Google}


class Router:
    """`runner.Client` over the adapters the config names, built (and so keyed) up front:
    a missing key refuses before the betting-market snapshot spends a credit or any model is
    called."""

    def __init__(self, config: StudyConfig, *, env: Mapping[str, str] | None = None,
                 transport: Transport | None = None) -> None:
        unknown = sorted({m.provider for m in config.llms} - ADAPTERS.keys())
        if unknown:
            raise ValueError(f"no client for provider(s) {unknown}; known: {sorted(ADAPTERS)}")
        self._by = {p: ADAPTERS[p](env=env, transport=transport)
                    for p in sorted({m.provider for m in config.llms})}

    def complete(self, spec: LlmSpec, system: str, user: str) -> Completion:
        try:
            return self._by[spec.provider].complete(spec, system, user)
        except ProviderError as exc:
            return Completion(f"[provider error] {exc}", 0, 0)
