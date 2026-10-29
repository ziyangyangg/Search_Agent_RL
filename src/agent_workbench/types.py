"""Shared contracts. Training frameworks are deliberately optional consumers."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def wire(self) -> dict:
        import json
        return {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)}}


@dataclass(frozen=True)
class Completion:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None


class ModelBackend(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        """Produce one assistant turn; never execute a tool."""
        ...


@dataclass
class Trajectory:
    task: str
    messages: list[dict]
    events: list[dict] = field(default_factory=list)
    status: str = "running"
    final_answer: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1"

    def to_dict(self) -> dict:
        return asdict(self)
