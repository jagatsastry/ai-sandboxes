"""LLM client interface + a deterministic offline mock + a real HTTP client.

The whole sandbox depends ONLY on the tiny `LLMClient` protocol -- one
method, `.complete(prompt, system, temperature) -> str`. Want a real model?
Use `OpenAIChatLLM` (works with any OpenAI-Chat-compatible endpoint:
OpenAI, vLLM, Ollama, LM Studio, ...).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol


class LLMClient(Protocol):
    name: str
    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = 0.0) -> str: ...


class EchoLLM:
    """A fake 'LLM' that returns canned responses based on prompt keywords.

    Not smart -- deterministic and fast, so the sandbox runs offline. Each
    instance carries a 'persona' so council members produce predictable
    disagreement.
    """
    def __init__(self, name: str = "echo", persona: str = "default",
                 latency_ms: int = 5):
        self.name = name
        self.persona = persona
        self.latency_ms = latency_ms

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = 0.0) -> str:
        time.sleep(self.latency_ms / 1000.0)
        return f"[{self.name}:{self.persona}] ok"


# Retry-worthy HTTP status codes (transient). Everything else (4xx that isn't
# 429, anything we don't recognize) raises immediately.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAIChatLLM:
    """OpenAI Chat Completions client (zero external deps -- urllib only).

    Works with any provider that speaks the OpenAI Chat schema:

        OpenAI:     base_url='https://api.openai.com/v1', model='gpt-4o-mini'
        Ollama:     base_url='http://localhost:11434/v1', model='qwen2.5:0.5b'
        vLLM:       base_url='http://localhost:8000/v1', model=<served model>
        LM Studio:  base_url='http://localhost:1234/v1'

    Auth: pass api_key=<str> explicitly, or set OPENAI_API_KEY. Local servers
    that don't check the key still want SOMETHING, so we send a placeholder.

    Resilience:
        - retries 5xx and 429 only (4xx is permanent -> raise fast)
        - honours Retry-After header on 429 when present
        - bounded exponential backoff with jitter
        - raises RuntimeError on hard failure; callers (e.g. LLMCoder)
          catch this and fall back

    Thread-safety: stateless per call -> safe to share across threads.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        name: str | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        # `None` -> read env; explicit '' -> respect the user's choice (no key).
        if api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.name = name or f"openai:{model}"

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        last_err: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read()
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, e)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"OpenAIChatLLM HTTP {e.code} from {url}: {e.reason}"
                ) from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise RuntimeError(
                    f"OpenAIChatLLM network failure to {url}: {e!r}"
                ) from e

            # Parse outside the urlopen context so we don't hold the conn.
            try:
                text = body.decode("utf-8")
                data = json.loads(text)
                choices = data["choices"]
                if not choices:
                    raise ValueError("response had empty 'choices'")
                content = choices[0]["message"]["content"]
                if content is None:
                    raise ValueError("response content was null")
                return content
            except (ValueError, KeyError, IndexError,
                    UnicodeDecodeError) as e:
                # Bad payloads are not transient; do not retry.
                raise RuntimeError(
                    f"OpenAIChatLLM bad response from {url}: {e!r}"
                ) from e

        # unreachable in practice
        raise RuntimeError(f"OpenAIChatLLM unknown failure: {last_err!r}")

    @staticmethod
    def _retry_delay(attempt: int,
                     http_err: urllib.error.HTTPError | None = None) -> float:
        # Respect Retry-After on 429 if present.
        if http_err is not None:
            ra = http_err.headers.get("Retry-After") if http_err.headers else None
            if ra:
                try:
                    return float(ra)
                except (TypeError, ValueError):
                    pass
        return 0.5 * (2 ** attempt)
