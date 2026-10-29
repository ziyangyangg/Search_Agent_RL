"""TypeSafe Jev decisions, separate from text-generating model backends.

Protocol: https://docs.typesafe.ai/api/system-one
Choice: https://docs.typesafe.ai/primitives/choice
Confidence is a concentration statistic, not a correctness guarantee.
"""

from dataclasses import dataclass
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request

from .models import ModelError, _open


@dataclass(frozen=True)
class ChoiceDecision:
    option: str
    probabilities: dict[str, float]
    confidence: float | None
    usage: dict[str, int]
    model: str


def _probability(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"Jev {field} must be a probability")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ModelError(f"Jev {field} must be finite and between 0 and 1")
    return value


def _choice(answer: dict, candidates: dict) -> None:
    if (answer.get("type") != "choice" or not isinstance(answer.get("choice"), str)
            or answer["choice"] not in candidates):
        raise ModelError("Jev returned an invalid choice")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(candidates):
        raise ModelError("Jev probabilities must cover exactly the requested options")
    values = {key: _probability(value, "probabilities") for key, value in probabilities.items()}
    if not math.isclose(sum(values.values()), 1.0, abs_tol=0.002, rel_tol=0):
        raise ModelError("Jev probabilities do not sum to one")
    if values[answer["choice"]] + 0.002 < max(values.values()):
        raise ModelError("Jev choice does not maximize its probability")
    if answer.get("confidence") is not None:
        _probability(answer["confidence"], "confidence")


class JevDecisionClient:
    """Bounded synchronous client for Choice and Noul questions.

    Explicit response validation fails closed. The caller decides whether errors
    should escalate to its reasoning model. There is no automatic paid retry.
    Experimental image/Box and Score primitives are not exposed in this client.
    """

    def __init__(self, model: str, base_url: str = "https://api.typesafe.ai/v1",
                 api_key_env: str = "JEV_API_KEY", timeout: float = 20.0,
                 opener=_open):
        parsed = urlparse(base_url)
        self.local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Jev base URL cannot contain credentials, query or fragment")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and self.local):
            raise ValueError("use HTTPS, or HTTP on loopback")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Jev model ID is required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("API key environment variable name is required")
        self.url = base_url.rstrip("/") + "/systemone"
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.opener = timeout, opener

    def decide(self, state: dict, questions: dict) -> dict:
        if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
            raise ValueError("state and nonempty questions must be objects")
        for name, question in questions.items():
            if not isinstance(name, str) or not name or not isinstance(question, dict):
                raise ValueError("questions require nonempty names and object definitions")
            if question.get("type") not in ("choice", "noul"):
                raise ValueError("this client supports Choice and Noul questions only")
            if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
                raise ValueError("question instructions must be nonempty text")
            if question["type"] == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                    raise ValueError("Choice requires 1 to 255 named options")
                if any(not isinstance(k, str) or not k for k in criteria):
                    raise ValueError("Choice option IDs must be nonempty strings")
                if any(not isinstance(v, str) or not v.strip() for v in criteria.values()):
                    raise ValueError("Choice option descriptions must be nonempty strings")
        try:
            encoded = json.dumps({"model": self.model, "state": state, "questions": questions},
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Jev request must contain finite JSON values") from exc
        if len(encoded) > 4_000_000:
            raise ValueError("Jev request exceeds the client byte limit")
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        elif not self.local:
            raise ModelError(f"missing Jev API key in {self.api_key_env}")
        request = Request(self.url, data=encoded, headers=headers, method="POST")
        try:
            with self.opener(request, timeout=self.timeout) as response:
                body = response.read(4_000_001)
            if len(body) > 4_000_000:
                raise ModelError("Jev response exceeds the client byte limit")
            result = json.loads(body)
        except HTTPError as exc:
            raise ModelError(f"Jev endpoint returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelError(f"Jev transport failed ({type(exc).__name__})") from exc
        if not isinstance(result, dict) or not isinstance(result.get("model"), str) or not result["model"]:
            raise ModelError("Jev response requires an actual model ID")
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ModelError("Jev answer IDs must match the requested question IDs")
        for name, question in questions.items():
            answer = answers[name]
            if not isinstance(answer, dict):
                raise ModelError("Jev answers must be objects")
            if question["type"] == "choice":
                _choice(answer, question["criteria"])
            else:
                if answer.get("type") != "noul":
                    raise ModelError("Jev answer type does not match its question")
                _probability(answer.get("noul"), "noul")
        usage = result.get("usage", {})
        if not isinstance(usage, dict) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for k, v in usage.items() if k in ("input_tokens", "output_tokens")):
            raise ModelError("Jev usage must contain nonnegative integer token counts")
        return result

    def choose(self, state: dict, candidates: dict[str, str]) -> ChoiceDecision:
        if not isinstance(candidates, dict) or any(not isinstance(value, str) or not value.strip()
                                                   for value in candidates.values()):
            raise ValueError("candidate descriptions must be nonempty strings")
        result = self.decide(state, {"action": {
            "type": "choice",
            "instructions": "Which of the described candidate actions best fits the supplied state? Select one supplied option.",
            "criteria": candidates,
        }})
        answer = result["answers"]["action"]
        return ChoiceDecision(answer["choice"], dict(answer["probabilities"]),
                              answer.get("confidence"), dict(result.get("usage", {})), result["model"])
