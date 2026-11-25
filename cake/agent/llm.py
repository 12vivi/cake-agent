"""LLM backends for the kernel-evolution agent.

* ``AnthropicLLM`` -- Claude through the official ``anthropic`` SDK (default model
  ``claude-opus-5-5``, adaptive thinking, ``output_config.effort``; the paper
  ran its agents at reasoning effort xhigh).  Credentials resolve the usual way
  (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` / ``ant auth login``).
* ``OpenAICompatLLM`` -- any OpenAI-compatible Chat Completions endpoint (the
  paper's GPT models, or DeepSeek / Qwen / vLLM via ``base_url``).
* ``ScriptedLLM`` -- an offline, deterministic kernel-engineer policy (see
  :mod:`cake.agent.scripted`) used for tests, CI and token-free demos.

Every backend returns the response text plus provider token usage, which the
evolution loop accounts against its token budget (paper: "we report
provider-token consumption").
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""
    stop_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


class LLMError(RuntimeError):
    pass


class LLM:
    name = "llm"

    def generate(self, system: str, prompt: str, context: Any = None) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class AnthropicLLM(LLM):
    """Claude via the Anthropic SDK (streaming, adaptive thinking, effort control)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "xhigh", max_tokens: int = 64000,
                 fallbacks: bool = True):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.name = f"anthropic:{model}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - optional dependency
                raise LLMError("the `anthropic` package is not installed: pip install 'cake-agent[anthropic]'") from e
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, system: str, prompt: str, context: Any = None) -> LLMResponse:
        import anthropic

        client = self._get_client()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # the long, stable system prompt (language reference + contract) is cached across rounds
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
        )
        try:
            if self.fallbacks:
                # server-side refusal fallback (routes a declined request to a fallback model)
                with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                 **kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                with client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise LLMError(f"the model declined the request (category: {cat})")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        return LLMResponse(text=text, input_tokens=getattr(u, "input_tokens", 0) or 0,
                           output_tokens=getattr(u, "output_tokens", 0) or 0,
                           cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                           cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                           model=getattr(msg, "model", self.model), stop_reason=str(msg.stop_reason))


class OpenAICompatLLM(LLM):
    """OpenAI-compatible Chat Completions endpoint (OpenAI, DeepSeek, Qwen, vLLM, ...)."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key_env = api_key_env
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.name = f"openai-compat:{model}"
        self._client = None

    def generate(self, system: str, prompt: str, context: Any = None) -> LLMResponse:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - optional dependency
                raise LLMError("the `openai` package is not installed: pip install 'cake-agent[openai]'") from e
            key = os.environ.get(self.api_key_env)
            if not key:
                raise LLMError(f"set {self.api_key_env} for the OpenAI-compatible backend")
            self._client = OpenAI(base_url=self.base_url, api_key=key)
        kwargs: dict = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                        {"role": "user", "content": prompt}])
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens:
            kwargs["max_completion_tokens"] = self.max_tokens
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # pragma: no cover - network
            raise LLMError(f"{type(e).__name__}: {e}") from e
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return LLMResponse(text=text, input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                           output_tokens=getattr(u, "completion_tokens", 0) or 0, model=self.model,
                           stop_reason=str(resp.choices[0].finish_reason))


def make_llm(spec: str, **kw) -> LLM:
    """``scripted`` | ``anthropic[:model]`` | ``openai:<model>`` (base URL from OPENAI_BASE_URL)."""
    kind, _, model = spec.partition(":")
    if kind == "scripted":
        from .scripted import ScriptedLLM
        return ScriptedLLM(seed=kw.get("seed", 0))
    if kind in ("anthropic", "claude"):
        return AnthropicLLM(model=model or "claude-opus-5-5", effort=kw.get("effort") or "xhigh")
    if kind in ("openai", "openai-compat"):
        if not model:
            raise LLMError("openai backend needs a model: openai:<model>")
        return OpenAICompatLLM(model=model, reasoning_effort=kw.get("effort"))
    raise LLMError(f"unknown LLM backend `{spec}`")
