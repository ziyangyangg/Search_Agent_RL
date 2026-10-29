"""Malformed transports and selectors cannot bypass execution boundaries."""

import copy
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

from agent_workbench.cooperation import ActionCandidate, CooperativeBackend
from agent_workbench.jev import JevDecisionClient
from agent_workbench.models import ChatCompletionsBackend, ModelError, ScriptedBackend, parse_completion
from agent_workbench.runtime import Agent
from agent_workbench.tools import ToolRegistry, ToolSpec
from agent_workbench.types import Completion


class AdapterValidationTests(unittest.TestCase):
    def test_invalid_generation_limits_rejected_before_transport(self):
        for field, values in {
            "timeout": (True, float("nan"), float("inf"), 0, "90"),
            "temperature": (True, float("nan"), float("inf"), -0.1, "0"),
            "max_tokens": (True, 1.5, float("inf"), 0, "100"),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ChatCompletionsBackend("http://127.0.0.1:8000/v1", "fixture", **{field: value})

    def test_host_and_model_required(self):
        for url in ("https:///v1", "https://", "http:///v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ChatCompletionsBackend(url, "fixture")
        for model in (None, 3, " "):
            with self.subTest(model=model), self.assertRaises(ValueError):
                ChatCompletionsBackend("http://127.0.0.1:8000/v1", model)

    def test_malformed_completion_structures_become_model_errors(self):
        payloads = [None, [], {"choices": {}}, {"choices": []}, {"choices": [None]},
                    {"choices": [{"message": []}]},
                    {"choices": [{"message": {"content": False}}]},
                    {"choices": [{"message": {"tool_calls": {"id": "x"}}}]},
                    {"choices": [{"message": {"tool_calls": [None]}}]}]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ModelError):
                parse_completion(payload)

    def test_invalid_usage_is_not_silently_reported_as_zero(self):
        for usage in ([], "7", {"total_tokens": True}, {"total_tokens": -1}, {"total_tokens": 1.5}):
            with self.subTest(usage=usage), self.assertRaises(ModelError):
                parse_completion({"choices": [{"message": {"content": "answer"}}], "usage": usage})

    def test_nonfinite_tool_arguments_are_rejected(self):
        for arguments in ('{"topk": NaN}', '{"topk": Infinity}', {"topk": 3}):
            payload = {"choices": [{"message": {"tool_calls": [
                {"id": "x", "function": {"name": "search", "arguments": arguments}}]},
                "finish_reason": "tool_calls"}]}
            with self.subTest(arguments=arguments), self.assertRaises(ModelError):
                parse_completion(payload)

    def test_credentialed_requests_never_follow_redirects(self):
        # Loopback-only HTTP proves that the default opener never forwards a key
        # to a redirect target. The fixture key is deliberately non-sensitive.
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, self.headers.get("Authorization")))
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(307)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/target")
                self.end_headers()

            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from unittest.mock import patch
            with patch.dict("os.environ", {"AGENT_API_KEY": "fixture-only-key"}):
                client = ChatCompletionsBackend(f"http://127.0.0.1:{server.server_port}/v1", "fixture")
                with self.assertRaisesRegex(ModelError, "redirects are disabled"):
                    client.complete([{"role": "user", "content": "fixture"}], [])
            self.assertEqual(seen, [("/v1/chat/completions", "Bearer fixture-only-key")])
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_invalid_jev_question_text_never_sends_request(self):
        def never_called(request, timeout):
            raise AssertionError("invalid question must not be transmitted")

        client = JevDecisionClient("fixture", "http://127.0.0.1:8001/v1", opener=never_called)
        for question in ({"type": "noul", "instructions": {}},
                         {"type": "noul", "instructions": " "},
                         {"type": "choice", "instructions": "choose", "criteria": {"read": {}}},
                         {"type": "choice", "instructions": "choose", "criteria": {"read": " "}}):
            with self.subTest(question=question), self.assertRaises(ValueError):
                client.decide({}, {"action": question})

    def test_unhashable_jev_choice_becomes_model_error(self):
        for choice in ([], {}, None, False):
            response = {"model": "fixture", "answers": {"action": {
                "type": "choice", "choice": choice, "probabilities": {"read": 1}}}}
            client = JevDecisionClient("fixture", "http://127.0.0.1:8001/v1",
                                       opener=lambda request, timeout: io.BytesIO(json.dumps(response).encode()))
            with self.subTest(choice=choice), self.assertRaises(ModelError):
                client.choose({}, {"read": "Read evidence"})

    def test_malformed_selector_response_escalates_without_tools(self):
        valid = SimpleNamespace(option="read", probabilities={"read": 0.99, "reason": 0.01},
                                confidence=None, usage={"input_tokens": 1}, model="fixture")
        decisions = [None]
        for field, value in (("option", "unknown"), ("option", []),
                             ("probabilities", {"read": 0.99}),
                             ("probabilities", {"read": float("nan"), "reason": 0.01}),
                             ("model", []), ("usage", {"input_tokens": True})):
            decision = copy.deepcopy(valid)
            setattr(decision, field, value)
            decisions.append(decision)
        for decision in decisions:
            with self.subTest(decision=decision):
                selector = SimpleNamespace(choose=lambda state, options: decision)
                registry = ToolRegistry()
                registry.register(ToolSpec("workspace_read_text", "read", {"type": "object"}),
                                  lambda args: self.fail("invalid selector must not execute a tool"))
                backend = CooperativeBackend(ScriptedBackend([Completion("reasoner answer")]), selector,
                    [ActionCandidate("read", "Read evidence", "workspace_read_text", {})])
                trace = Agent(backend, registry).run("Read evidence")
                self.assertEqual(trace.final_answer, "reasoner answer")
                self.assertEqual(backend.routing_events[0]["reason"], "decision_error")

    def test_candidate_fields_have_readable_validation_errors(self):
        args = {"id": "read", "description": "Read evidence", "tool": "workspace_read_text", "arguments": {}}
        for field, value in (("id", []), ("description", None), ("tool", []),
                             ("arguments", {"x": float("nan")})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ActionCandidate(**{**args, field: value})

    def test_caller_mutation_cannot_change_prepared_actions(self):
        arguments = {"path": "approved.md"}
        candidate = ActionCandidate("read", "Read approved evidence", "workspace_read_text", arguments)
        arguments["path"] = "different.md"
        self.assertEqual(candidate.arguments, {"path": "approved.md"})
        backend = CooperativeBackend(ScriptedBackend([]), None, [candidate])
        candidate.arguments["path"] = "different.md"
        self.assertEqual(backend.candidates[0].arguments, {"path": "approved.md"})

    def test_routing_budget_configuration_requires_finite_values(self):
        for kwargs in ({"min_probability": True}, {"min_probability": float("nan")},
                       {"max_state_chars": 128.5}, {"max_state_chars": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CooperativeBackend(ScriptedBackend([]), None, [], **kwargs)


if __name__ == "__main__":
    unittest.main()
