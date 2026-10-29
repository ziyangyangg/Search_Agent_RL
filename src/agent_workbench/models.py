"""OpenAI-compatible chat transport and a clearly labelled offline demo backend.

Compatibility is endpoint/model/template specific. No model name is hardcoded.
API keys are read from the environment and never included in traces.
"""

import json
import math
import os
from collections import deque
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .types import Completion, ToolCall


class ModelError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ModelError("model endpoint redirects are disabled; use the final trusted URL")


def _open(request, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def parse_completion(payload: dict) -> Completion:
    """Validate transport data before constructing executable tool proposals."""
    try:
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
            raise ValueError("choices must be a nonempty array")
        choice = payload["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("choice and message must be objects")
        message = choice["message"]
        content = message.get("content")
        content = "" if content is None else content
        if not isinstance(content, str):
            raise ValueError("only text message content is supported")
        calls = []
        ids = set()
        raw_calls = message.get("tool_calls")
        raw_calls = [] if raw_calls is None else raw_calls
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be an array")
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
                raise ValueError("tool calls and functions must be objects")
            fn = raw["function"]
            call_id, name = raw["id"], fn["name"]
            if not isinstance(call_id, str) or not call_id or call_id in ids:
                raise ValueError("tool call IDs must be nonempty and unique")
            if not isinstance(name, str) or not name:
                raise ValueError("tool name must be nonempty")
            ids.add(call_id)
            if not isinstance(fn["arguments"], str):
                raise ValueError("tool arguments must be JSON-encoded text")
            arguments = json.loads(fn["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            # The stdlib decoder accepts NaN/Infinity, which are not JSON values.
            json.dumps(arguments, allow_nan=False)
            calls.append(ToolCall(call_id, name, arguments))
        reason = choice.get("finish_reason", "stop")
        if reason not in ("stop", "tool_calls", "length", "content_filter"):
            raise ValueError("unsupported finish_reason")
        if reason == "tool_calls" and not calls:
            raise ValueError("tool_calls finish reason without calls")
        usage = payload.get("usage")
        usage = {} if usage is None else usage
        if not isinstance(usage, dict):
            raise ValueError("usage must be an object")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage and (isinstance(usage[key], bool) or
                                 not isinstance(usage[key], int) or usage[key] < 0):
                raise ValueError("token usage must contain nonnegative integers")
        usage = {k: v for k, v in usage.items()
                 if k in ("prompt_tokens", "completion_tokens", "total_tokens")
                 and isinstance(v, int) and not isinstance(v, bool) and v >= 0}
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("reasoning_content must be text when present")
        return Completion(content, tuple(calls), reason, usage, reasoning)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelError(f"invalid chat completion: {exc}") from exc


class ChatCompletionsBackend:
    """Bounded synchronous client; caller chooses the model and endpoint.

HTTPS is required except for loopback hosts. No automatic POST retries: an
ambiguous transport failure must not silently duplicate paid requests.
"""

    def __init__(self, base_url: str, model: str, *, api_key_env: str = "AGENT_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 1024,
                 timeout: float = 90.0, opener=_open,
                 enable_thinking: bool | None = None, reasoning_effort: str | None = None):
        parsed = urlparse(base_url)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base URL cannot contain credentials, query or fragment")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("use HTTPS, or HTTP on loopback")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model ID is required")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or not 0 <= temperature <= 2 or
                isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1 or
                isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("invalid generation configuration")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("API key environment variable name is required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.opener = opener
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean or None")
        if reasoning_effort not in (None, "low", "medium", "xhigh"):
            raise ValueError("unsupported reasoning_effort")
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with self.opener(request, timeout=self.timeout) as response:
                body = response.read(8_000_001)
            if len(body) > 8_000_000:
                raise ModelError("model response exceeds 8 MB")
            return parse_completion(json.loads(body))
        except HTTPError as exc:
            # Response bodies can contain echoed prompts, keys, or private docs.
            raise ModelError(f"model endpoint returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelError(f"model transport failed ({type(exc).__name__})") from exc


class ScriptedBackend:
    """Deterministic fixtures for tests/demo; never evidence of model capability."""

    def __init__(self, completions: list[Completion]):
        self.completions = deque(completions)

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        if not self.completions:
            raise ModelError("scripted fixture exhausted")
        return self.completions.popleft()
