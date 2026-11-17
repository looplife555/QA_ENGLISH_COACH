"""
LLM provider abstraction for SpeakUp - QA English Coach.

Supports:
- Local Ollama (existing behaviour, no API key)
- OpenAI (ChatGPT) and any OpenAI-compatible API (OpenRouter, Groq, Together, local LLM studio, etc.)
- Anthropic (Claude)

The frontend chooses a provider and sends an API key. Keys are stored only in the
browser's localStorage and are forwarded to the provider you picked - never to
any other service. Set COACH_PROVIDER / COACH_API_KEY / COACH_BASE_URL /
COACH_MODEL as environment variables to hard-code defaults (useful for deployments).
"""
from __future__ import annotations

import json
import os
import re
import requests

# Default model per provider (fallbacks, overridable via env or the UI).
PROVIDERS = {
    "ollama": {
        "label": "Local Ollama",
        "defaultModel": "qwen3:4b",
        "defaultBase": "http://127.0.0.1:11434",
        "keyRequired": False,
        "note": "Runs on your computer. No tokens, no cloud.",
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "defaultModel": "gpt-4o-mini",
        "defaultBase": "https://api.openai.com/v1",
        "keyRequired": True,
        "note": "Fast, capable models great for interview role-play and coaching.",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "defaultModel": "claude-3-haiku-20240307",
        "defaultBase": "https://api.anthropic.com/v1",
        "keyRequired": True,
        "note": "Strong, natural conversationalist - good for mock interviews.",
    },
    "openai-compatible": {
        "label": "OpenAI-compatible API (OpenRouter, Groq, local servers, …)",
        "defaultModel": "",
        "defaultBase": "",
        "keyRequired": True,
        "note": "Any service with an OpenAI-style /v1/chat/completions endpoint.",
    },
}

# Recommended models shown in the UI per provider.
RECOMMENDED_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
    "openai-compatible": [],
    "ollama": [],
}

ENV_PROVIDER = "COACH_PROVIDER"
ENV_API_KEY = "COACH_API_KEY"
ENV_BASE_URL = "COACH_BASE_URL"
ENV_MODEL = "COACH_MODEL"


def resolve_settings(req=None):
    """(provider, api_key, base_url, model) from request overrides then env defaults."""
    provider = "ollama"
    api_key = ""
    base_url = ""
    model = ""

    if req is not None:
        provider = (getattr(req, "provider", None) or "").strip() or provider
        api_key = (getattr(req, "apiKey", None) or "").strip()
        base_url = (getattr(req, "baseUrl", None) or "").strip()
        model = (getattr(req, "model", None) or "").strip()

    provider = provider or os.environ.get(ENV_PROVIDER, "ollama")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        provider = "ollama"

    prov = PROVIDERS[provider]
    api_key = api_key or os.environ.get(ENV_API_KEY, "")
    base_url = base_url or os.environ.get(ENV_BASE_URL, "") or prov["defaultBase"]
    model = model or os.environ.get(ENV_MODEL, "") or prov["defaultModel"]

    return provider, api_key, base_url.rstrip("/"), model


def provider_info(provider=None, api_key=None, base_url=None, model=None):
    """Small status dict for /api/status."""
    p, key, base, mdl = resolve_settings(
        type("R", (), {"provider": provider, "apiKey": api_key, "baseUrl": base_url, "model": model})()
        if provider is not None else None
    )
    prov = PROVIDERS[p]
    return {
        "provider": p,
        "label": prov["label"],
        "keyRequired": prov["keyRequired"],
        "keyConfigured": bool(key),
        "defaultModel": mdl,
        "baseUrl": base,
        "models": [{"id": m, "name": m} for m in RECOMMENDED_MODELS.get(p, [])],
        "note": prov["note"],
    }


class LLMError(Exception):
    """Raised when the LLM provider returns an error."""


class LLMClient:
    def __init__(self, provider, api_key, base_url, model):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    # ------------------------------------------------------------------ chat
    def chat(self, system, messages, temperature=0.6, max_tokens=260, think=False):
        """Non-streaming chat. Returns the assistant text."""
        if self.provider == "ollama":
            return self._ollama_chat(system, messages, temperature, max_tokens, think)
        if self.provider in ("openai", "openai-compatible"):
            return self._openai_chat(system, messages, temperature, max_tokens, stream=False)
        if self.provider == "anthropic":
            return self._anthropic_chat(system, messages, temperature, max_tokens, stream=False)
        raise LLMError(f"Unknown provider: {self.provider}")

    # ---------------------------------------------------------------- stream
    def stream_chat(self, system, messages, temperature=0.4, max_tokens=1200):
        """Yield assistant text chunks (used for long reports/scorecards)."""
        if self.provider == "ollama":
            yield from self._ollama_stream(system, messages, temperature, max_tokens)
        elif self.provider in ("openai", "openai-compatible"):
            yield from self._openai_stream(system, messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            yield from self._anthropic_stream(system, messages, temperature, max_tokens)
        else:
            raise LLMError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------ Ollama
    def _ollama_chat(self, system, messages, temperature, max_tokens, think):
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}] + messages,
        }
        if self._supports_think():
            body["think"] = False
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=180)
        except requests.exceptions.ConnectionError:
            raise LLMError("Can't reach Ollama. Start Ollama and try again.")
        except Exception as e:
            raise LLMError(f"Model error: {e}")
        if r.status_code != 200:
            detail = ""
            try:
                detail = r.json().get("error", "")
            except Exception:
                pass
            missing = r.status_code == 404 or "not found" in detail.lower()
            if missing:
                raise LLMError(f"Model '{self.model}' is not installed. Download it in the Setup tab.", code="MODEL_MISSING")
            raise LLMError(detail or "Ollama error", code="OLLAMA_ERROR")
        data = r.json()
        raw = (data.get("message") or {}).get("content") or ""
        return re.sub(r"\u2329.*?\u232a", "", raw, flags=re.S).strip()  # strip any think tags

    def _ollama_stream(self, system, messages, temperature, max_tokens):
        body = {
            "model": self.model,
            "stream": True,
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}] + messages,
        }
        if self._supports_think():
            body["think"] = False
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=body, stream=True, timeout=None)
        except requests.exceptions.ConnectionError:
            raise LLMError("Can't reach Ollama. Start Ollama and try again.")
        except Exception as e:
            raise LLMError(f"Report failed: {e}")
        if r.status_code != 200:
            try:
                detail = r.json().get("error", "Ollama error")
            except Exception:
                detail = "Ollama error"
            raise LLMError(detail)
        for line in r.iter_lines():
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("error"):
                raise LLMError(j["error"])
            content = (j.get("message") or {}).get("content") or ""
            if content:
                yield content
            if j.get("done"):
                break

    def _supports_think(self):
        m = self.model.lower()
        return m.startswith("qwen3") or m.startswith("deepseek-r1")

    # ------------------------------------------------------------ OpenAI
    def _openai_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _openai_chat(self, system, messages, temperature, max_tokens, stream):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._openai_headers(),
                json=body,
                timeout=180 if not stream else None,
            )
        except requests.exceptions.ConnectionError:
            raise LLMError("Can't reach the API provider. Check your base URL and internet connection.")
        except Exception as e:
            raise LLMError(f"API error: {e}")
        if r.status_code == 401:
            raise LLMError("Invalid API key. Check your provider API key in Settings.")
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found by the provider.")
        if r.status_code != 200:
            detail = ""
            try:
                detail = (r.json().get("error") or {}).get("message", "") or r.text[:200]
            except Exception:
                detail = r.text[:200]
            raise LLMError(detail or "API provider error")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("Unexpected response from the API provider.")

    def _openai_stream(self, system, messages, temperature, max_tokens):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._openai_headers(),
                json=body,
                stream=True,
                timeout=None,
            )
        except requests.exceptions.ConnectionError:
            raise LLMError("Can't reach the API provider. Check your base URL and internet connection.")
        except Exception as e:
            raise LLMError(f"API error: {e}")
        if r.status_code == 401:
            raise LLMError("Invalid API key. Check your provider API key in Settings.")
        if r.status_code != 200:
            detail = ""
            try:
                detail = (r.json().get("error") or {}).get("message", "") or r.text[:200]
            except Exception:
                detail = r.text[:200]
            raise LLMError(detail or "API provider error")
        for line in r.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            if not text.startswith("data:"):
                continue
            payload = text[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            try:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or ""
            except (KeyError, IndexError, TypeError):
                content = ""
            if content:
                yield content

    # ----------------------------------------------------------- Anthropic
    def _anthropic_headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _anthropic_chat(self, system, messages, temperature, max_tokens, stream):
        body = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        try:
            r = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self._anthropic_headers(),
                json=body,
                timeout=180 if not stream else None,
            )
        except requests.exceptions.ConnectionError:
            raise LLMError("Can't reach Anthropic. Check your internet connection.")
        except Exception as e:
            raise LLMError(f"API error: {e}")
        if r.status_code == 401:
            raise LLMError("Invalid API key. Check your provider API key in Settings.")
        if r.status_code != 200:
            detail = ""
            try:
                detail = r.json().get("error", {}).get("message", "") or r.text[:200]
            except Exception:
                detail = r.text[:200]
            raise LLMError(detail or "Anthropic error")
        data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _anthropic_stream(self, system, messages, temperature, max_tokens):
        body = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        try:
            r = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self._anthropic_headers(),
                json=body,
                stream=True,
                timeout=None,
            )
        except requests.exceptions.ConnectionError:
            raise LLMError("Can't reach Anthropic. Check your internet connection.")
        except Exception as e:
            raise LLMError(f"API error: {e}")
        if r.status_code == 401:
            raise LLMError("Invalid API key. Check your provider API key in Settings.")
        if r.status_code != 200:
            detail = ""
            try:
                detail = r.json().get("error", {}).get("message", "") or r.text[:200]
            except Exception:
                detail = r.text[:200]
            raise LLMError(detail or "Anthropic error")
        event = None
        for line in r.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            if text.startswith("event:"):
                event = text[6:].strip()
            elif text.startswith("data:"):
                payload = text[5:].strip()
                if event == "content_block_delta" and payload:
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_delta = delta.get("text", "")
                            if text_delta:
                                yield text_delta
                    except Exception:
                        continue
