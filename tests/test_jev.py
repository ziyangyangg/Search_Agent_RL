import copy
import io
import json
import os
import unittest
from unittest.mock import patch

from agent_workbench.jev import JevDecisionClient
from agent_workbench.models import ModelError


CANDIDATES = {"search": "Search the corpus for evidence", "_reason": "Escalate to the reasoning model"}
RESPONSE = {"model": "jev-1.13.0", "answers": {"action": {
    "type": "choice", "choice": "search", "probabilities": {"search": 0.9, "_reason": 0.1},
    "confidence": 0.7}}, "usage": {"input_tokens": 100, "output_tokens": 20}}


class JevTests(unittest.TestCase):
    def client(self, response):
        return JevDecisionClient("jev-1.13.0", "http://127.0.0.1:8001/v1",
                                 opener=lambda request, timeout: io.BytesIO(json.dumps(response).encode()))

    def test_official_payload_and_validated_choice(self):
        seen = {}
        def opener(request, timeout):
            seen.update(json.loads(request.data))
            self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
            self.assertEqual(timeout, 20)
            return io.BytesIO(json.dumps(RESPONSE).encode())
        with patch.dict(os.environ, {"JEV_API_KEY": "test-secret"}):
            client = JevDecisionClient("jev-1.13.0", opener=opener)
            decision = client.choose({"task": "Find evidence"}, CANDIDATES)
        self.assertEqual(seen["questions"]["action"]["type"], "choice")
        self.assertEqual(seen["questions"]["action"]["criteria"], CANDIDATES)
        self.assertNotIn("messages", seen)
        self.assertEqual(decision.option, "search")
        self.assertEqual(decision.model, "jev-1.13.0")
        self.assertEqual(decision.confidence, 0.7)

    def test_remote_request_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelError):
                JevDecisionClient("jev-1.13.0").choose({}, CANDIDATES)

    def test_probability_distribution_validation(self):
        for probabilities in ({"search": 0.8, "_reason": 0.8},
                              {"search": 0.9},
                              {"search": float("nan"), "_reason": 0.1},
                              {"search": True, "_reason": 0},
                              {"search": -0.2, "_reason": 1.2}):
            with self.subTest(probabilities=probabilities):
                response = copy.deepcopy(RESPONSE)
                response["answers"]["action"]["probabilities"] = probabilities
                with self.assertRaises(ModelError):
                    self.client(response).choose({}, CANDIDATES)

    def test_answer_id_choice_and_type_validation(self):
        mutations = [lambda r: r["answers"].update(other=r["answers"]["action"]),
                     lambda r: r["answers"]["action"].update(choice="invented"),
                     lambda r: r["answers"]["action"].update(choice="_reason"),
                     lambda r: r["answers"]["action"].update(type="score"),
                     lambda r: r["answers"]["action"].update(confidence=1.5)]
        for mutate in mutations:
            response = copy.deepcopy(RESPONSE)
            mutate(response)
            with self.assertRaises(ModelError):
                self.client(response).choose({}, CANDIDATES)

    def test_confidence_optional_not_fabricated(self):
        response = copy.deepcopy(RESPONSE)
        response["answers"]["action"].pop("confidence")
        self.assertIsNone(self.client(response).choose({}, CANDIDATES).confidence)

    def test_noul_and_invalid_requests(self):
        client = self.client({"model": "jev-1.13.0", "answers": {"check": {"type": "noul", "noul": 0.8}}})
        self.assertEqual(client.decide({}, {"check": {"type": "noul", "instructions": "Is evidence sufficient?"}})["answers"]["check"]["noul"], 0.8)
        with self.assertRaises(ValueError):
            client.choose({}, {})
        with self.assertRaises(ValueError):
            client.decide({}, {"check": {"type": "box"}})

    def test_url_and_transport_errors(self):
        for url in ("http://example.com/v1", "https://key@example.com/v1", "https://example.com/v1?key=x"):
            with self.assertRaises(ValueError):
                JevDecisionClient("jev-1.13.0", base_url=url)
        def fail(request, timeout):
            raise TimeoutError("private state must not appear in error")
        client = JevDecisionClient("jev-1.13.0", "http://127.0.0.1:8001/v1", opener=fail)
        with self.assertRaisesRegex(ModelError, "TimeoutError") as caught:
            client.choose({}, CANDIDATES)
        self.assertNotIn("private state", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
