"""Selective execution: Jev chooses prepared read actions; Qwen handles the rest.

This is an explicit candidate-restricted prototype, not an autonomous planner.
An external planner/recipe can supply fully bound actions. Jev never fabricates
free-text arguments or grants permissions. The normal registry remains executor.
"""

import copy
import json
import math
import re
import time
from dataclasses import dataclass

from .jev import _choice
from .tools import validate_arguments
from .types import Completion, ModelBackend, ToolCall

READ_TOOLS = frozenset({"workspace_list_files", "workspace_read_text", "corpus_search"})


@dataclass(frozen=True)
class ActionCandidate:
    id: str
    description: str
    tool: str
    arguments: dict

    def __post_init__(self):
        if (not isinstance(self.id, str) or
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", self.id) or self.id == "reason"):
            raise ValueError("candidate ID must be simple and cannot be 'reason'")
        if (not isinstance(self.description, str) or not self.description.strip()
                or not isinstance(self.arguments, dict)):
            raise ValueError("candidate needs a description and complete argument object")
        if not isinstance(self.tool, str) or self.tool not in READ_TOOLS:
            raise ValueError("fast-path candidates are limited to the built-in read tools")
        try:
            json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate arguments must contain finite JSON values") from exc
        # Freeze the prepared action against mutation by its caller.
        object.__setattr__(self, "arguments", copy.deepcopy(self.arguments))


def _fingerprint(name: str, arguments: dict) -> str:
    return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False, allow_nan=False)


class CooperativeBackend:
    """Drop-in ModelBackend; default slow backend can be a thinking Qwen model.

    min_probability is an experimental threshold, not a correctness guarantee.
    Missing candidates, uncertainty, transport errors, or context limits escalate.
    Route logs contain probabilities/usage, not the private state sent to Jev.
    """

    def __init__(self, reasoner: ModelBackend, decision_client,
                 candidates: list[ActionCandidate], *, min_probability: float = 0.9,
                 max_state_chars: int = 24000):
        if (isinstance(min_probability, bool) or not isinstance(min_probability, (int, float))
                or not math.isfinite(min_probability) or not 0.5 < min_probability <= 1
                or isinstance(max_state_chars, bool) or not isinstance(max_state_chars, int)
                or max_state_chars < 128):
            raise ValueError("invalid routing configuration")
        if not isinstance(candidates, list) or any(not isinstance(c, ActionCandidate) for c in candidates):
            raise ValueError("candidates must be a list of ActionCandidate objects")
        if len({c.id for c in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be unique")
        self.reasoner = reasoner
        self.decisions = decision_client
        self.candidates = copy.deepcopy(candidates)
        self.min_probability = min_probability
        self.max_state_chars = max_state_chars
        self.routing_events: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        specs = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        seen = set()
        for message in messages:
            for call in message.get("tool_calls") or []:
                try:
                    seen.add(_fingerprint(call["function"]["name"], json.loads(call["function"]["arguments"])))
                except (ValueError, KeyError, TypeError):
                    continue
        eligible = []
        for candidate in self.candidates:
            if candidate.tool not in specs or _fingerprint(candidate.tool, candidate.arguments) in seen:
                continue
            # A misconfigured recipe must not trigger an API call or tool execution.
            validate_arguments(candidate.arguments, specs[candidate.tool])
            eligible.append(candidate)
        if not eligible:
            self.routing_events.append({"route": "qwen", "reason": "no_unused_candidate"})
            return self.reasoner.complete(messages, tools)
        state = {"conversation": messages, "prepared_actions": [
            {"id": c.id, "tool": c.tool, "arguments": c.arguments} for c in eligible]}
        if len(json.dumps(state, ensure_ascii=False)) > self.max_state_chars:
            self.routing_events.append({"route": "qwen", "reason": "decision_state_budget"})
            return self.reasoner.complete(messages, tools)
        options = {c.id: c.description for c in eligible}
        options["reason"] = ("None of these prepared actions is clearly the best next step, "
                             "or planning, new arguments, answering, or uncertainty requires the reasoning model.")
        started = time.monotonic()
        try:
            decision = self.decisions.choose(state, options)
            # Custom selectors must satisfy the same contract as the HTTP client.
            # Keep this validation inside the escalation boundary.
            _choice({"type": "choice", "choice": decision.option,
                     "probabilities": decision.probabilities,
                     "confidence": decision.confidence}, options)
            if not isinstance(decision.model, str) or not decision.model.strip():
                raise ValueError("decision model must be nonempty text")
            if not isinstance(decision.usage, dict) or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 0
                    for k, v in decision.usage.items() if k in ("input_tokens", "output_tokens")):
                raise ValueError("decision usage must contain nonnegative integer token counts")
        except Exception as exc:
            self.routing_events.append({"route": "qwen", "reason": "decision_error",
                                        "error_type": type(exc).__name__,
                                        "latency_seconds": time.monotonic() - started})
            return self.reasoner.complete(messages, tools)
        probability = decision.probabilities[decision.option]
        fast = decision.option != "reason" and probability >= self.min_probability
        self.routing_events.append({"route": "jev" if fast else "qwen",
                                    "reason": "accepted_candidate" if fast else "escalate",
                                    "option": decision.option, "selected_probability": probability,
                                    "probabilities": decision.probabilities, "confidence": decision.confidence,
                                    "jev_model": decision.model, "jev_usage": decision.usage,
                                    "latency_seconds": time.monotonic() - started})
        if not fast:
            return self.reasoner.complete(messages, tools)
        candidate = next(c for c in eligible if c.id == decision.option)
        return Completion(tool_calls=(ToolCall(f"jev-{len(self.routing_events)}-{candidate.id}",
                                               candidate.tool, candidate.arguments),), finish_reason="tool_calls")
