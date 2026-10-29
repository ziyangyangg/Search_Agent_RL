"""Oversized tasks and growing observations stop before another model call."""
import unittest
from agent_workbench.models import ScriptedBackend
from agent_workbench.runtime import Agent, RunConfig
from agent_workbench.tools import ToolRegistry, ToolSpec
from agent_workbench.types import Completion, ToolCall


class RuntimeLimitTests(unittest.TestCase):
    def test_oversized_task_never_contacts_backend(self):
        class NeverCalled:
            def complete(self, messages, tools):
                raise AssertionError("must reject before backend use")
        trace = Agent(NeverCalled(), ToolRegistry(), RunConfig(max_context_chars=1000)).run("x" * 2000)
        self.assertEqual(trace.status, "context_budget")
        self.assertEqual(trace.events[0]["kind"], "context_budget")

    def test_tool_output_growth_stops_before_next_generation(self):
        tools = ToolRegistry()
        tools.register(ToolSpec("read", "Read a fixture", {"type": "object"}), lambda args: "x" * 2000)
        backend = ScriptedBackend([Completion(tool_calls=(ToolCall("r1", "read", {}),), finish_reason="tool_calls")])
        trace = Agent(backend, tools, RunConfig(max_context_chars=1500)).run("Read.")
        self.assertEqual(trace.status, "context_budget")
        self.assertEqual(sum(e["kind"] == "model" for e in trace.events), 1)
        self.assertEqual(sum(e["kind"] == "tool" for e in trace.events), 1)

    def test_within_budget_completes(self):
        trace = Agent(ScriptedBackend([Completion("done")]), ToolRegistry()).run("hello")
        self.assertEqual((trace.status, trace.final_answer), ("completed", "done"))

    def test_invalid_budget_types_fail_before_run(self):
        for key in ("max_steps", "max_tool_calls", "max_tool_output_chars", "max_context_chars"):
            for value in (True, 1.5, float("inf"), 0):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RunConfig(**{key: value})
        with self.assertRaises(ValueError):
            RunConfig(allow_workspace_writes="yes")
