import unittest
from types import SimpleNamespace

from agent_workbench.cooperation import ActionCandidate, CooperativeBackend
from agent_workbench.models import ScriptedBackend
from agent_workbench.runtime import Agent
from agent_workbench.tools import ToolRegistry, ToolSpec
from agent_workbench.types import Completion


class FakeDecision:
    def __init__(self, option="read", probability=0.98, fail=False):
        self.option, self.probability, self.fail, self.calls = option, probability, fail, 0

    def choose(self, state, candidates):
        self.calls += 1
        if self.fail:
            raise TimeoutError()
        return SimpleNamespace(option=self.option,
                               probabilities={"read": self.probability, "reason": 1-self.probability},
                               confidence=None, usage={"input_tokens": 10}, model="jev-test")


class CooperationTests(unittest.TestCase):
    def tools(self):
        registry = ToolRegistry()
        registry.register(ToolSpec("workspace_read_text", "read", {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
            "additionalProperties": False}), lambda args: "meeting notes")
        return registry

    def candidate(self):
        return ActionCandidate("read", "Read the requested meeting notes.", "workspace_read_text", {"path": "notes.md"})

    def test_fast_action_skips_a_reasoner_call_and_is_not_repeated(self):
        # Only one scripted model response is available; two-turn execution must
        # succeed because the first turn uses the actual fast selector contract.
        decisions = FakeDecision()
        backend = CooperativeBackend(ScriptedBackend([Completion("summary")]), decisions, [self.candidate()])
        trace = Agent(backend, self.tools()).run("Summarize notes.md")
        self.assertEqual(trace.status, "completed")
        self.assertEqual(trace.final_answer, "summary")
        self.assertEqual(decisions.calls, 1)
        self.assertEqual([e["route"] for e in backend.routing_events], ["jev", "qwen"])

    def test_low_probability_escalates(self):
        backend = CooperativeBackend(ScriptedBackend([Completion("slow")]), FakeDecision(probability=0.6), [self.candidate()])
        trace = Agent(backend, self.tools()).run("task")
        self.assertEqual(trace.final_answer, "slow")
        self.assertFalse(any(e["kind"] == "tool" for e in trace.events))

    def test_decision_failure_escalates_without_executing(self):
        backend = CooperativeBackend(ScriptedBackend([Completion("slow")]), FakeDecision(fail=True), [self.candidate()])
        trace = Agent(backend, self.tools()).run("task")
        self.assertEqual(trace.status, "completed")
        self.assertEqual(backend.routing_events[0]["reason"], "decision_error")

    def test_fast_path_cannot_authorize_writes(self):
        with self.assertRaises(ValueError):
            ActionCandidate("write", "write", "workspace_write_text", {"path": "file.md", "content": "x"})

    def test_reasoning_content_survives_tool_followup(self):
        from agent_workbench.types import ToolCall
        class Capture:
            def __init__(self):
                self.calls = 0
            def complete(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return Completion(tool_calls=(ToolCall("one", "workspace_read_text", {"path": "notes.md"}),),
                                      finish_reason="tool_calls", reasoning_content="provider test field")
                assert messages[-2]["reasoning_content"] == "provider test field"
                return Completion("summary")
        self.assertEqual(Agent(Capture(), self.tools()).run("task").status, "completed")
