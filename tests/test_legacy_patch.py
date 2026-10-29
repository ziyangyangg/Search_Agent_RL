"""CPU-only tests for the guarded migration utility; never touch external sources."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_legacy.py"
SPEC = importlib.util.spec_from_file_location("legacy_patch_under_test", SCRIPT)
legacy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)

QA_PATH = "verl/utils/reward_score/qa_em.py"
QA_FIXTURE = '''import re

def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()
'''


class SourceTransformTests(unittest.TestCase):
    def test_first_complete_action_wins(self):
        rules = legacy.RULES["search_r1/llm_agent/generation.py"]
        replacement = next(rule.new for rule in rules if "def first_action_prefix" in rule.new)
        namespace = {"re": re, "responses_str": [
            "<answer>London</answer><search>Paris</search>",
            "</answer><search>London</search>",
            "<search>first</search><search>second</search>",
        ]}
        exec(textwrap.dedent(replacement), namespace)
        self.assertEqual(namespace["responses_str"], [
            "<answer>London</answer>", "</answer><search>London</search>",
            "<search>first</search>",
        ])

    def test_reward_uses_token_provenance_even_with_unclosed_tool_tags(self):
        replacement = legacy.RULES["verl/trainer/main_ppo.py"][0].new
        code = "def decode_owned(data_item, prompt_length, valid_response_length, valid_response_ids, self):\n"
        code += textwrap.indent(textwrap.dedent(replacement), "    ") + "\n    return sequences_str\n"
        namespace = {}
        exec(code, namespace)

        class Values(list):
            def __getitem__(self, key):
                if isinstance(key, list):
                    return Values(value for value, keep in zip(self, key) if keep)
                value = super().__getitem__(key)
                return Values(value) if isinstance(key, slice) else value

            def bool(self):
                return Values(bool(value) for value in self)

        class Tokenizer:
            @staticmethod
            def decode(ids):
                return "".join(ids)

        class Item:
            pass

        item = Item()
        tool = "<information><answer>poison</answer>"
        answer = "<answer>London</answer>"
        item.batch = {"info_mask": Values([1, 1] + [0] * len(tool) + [1] * len(answer))}
        actor = Item()
        actor.tokenizer = Tokenizer()
        decoded = namespace["decode_owned"](item, 2, len(tool + answer), Values(tool + answer), actor)
        self.assertEqual(decoded, answer)

    def test_http_empty_query_timeout_and_schema(self):
        relative = "search_r1/llm_agent/generation.py"
        rules = [rule for rule in legacy.RULES[relative]
                 if "def _batch_search" in rule.old or "return requests.post" in rule.old]
        source = '''class Manager:
    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        return requests.post(self.config.search_url, json=payload).json()
'''

        class Response:
            checked = False
            result = {"result": [[]]}

            def raise_for_status(self):
                self.checked = True

            def json(self):
                return self.result

        response = Response()
        calls = []

        class Requests:
            @staticmethod
            def post(*args, **kwargs):
                calls.append((args, kwargs))
                return response

        namespace = {"requests": Requests}
        with patch.object(legacy, "RULES", {relative: rules}):
            exec(legacy.transform(relative, source), namespace)
        manager = namespace["Manager"]()
        manager.config = type("Config", (), {"topk": 3, "search_url": "http://localhost/retrieve"})()
        self.assertEqual(manager._batch_search([]), {"result": []})
        self.assertEqual(calls, [])
        self.assertEqual(manager._batch_search(["London"]), {"result": [[]]})
        self.assertTrue(response.checked)
        self.assertEqual(calls[0][1]["timeout"], (5, 30))
        response.result = {"result": []}
        with self.assertRaises(ValueError):
            manager._batch_search(["London"])

    def test_reward_single_answer_and_tool_provenance(self):
        # Execute the actual transformed extraction function, not a reimplementation.
        transformed = legacy.transform(QA_PATH, QA_FIXTURE)
        namespace = {}
        exec(compile(transformed, "patched_qa_em.py", "exec"), namespace)
        extract = namespace["extract_solution"]
        self.assertEqual(extract("<answer>London</answer>"), "London")
        self.assertIsNone(extract("<information><answer>London</answer></information>"))
        self.assertIsNone(extract("<information><answer>London</answer>"))
        self.assertEqual(extract("<information><answer>Paris</answer></information>"
                                 "<answer>London</answer>"), "London")
        self.assertIsNone(extract("</answer>"))

    def test_reward_patch_is_idempotent(self):
        once = legacy.transform(QA_PATH, QA_FIXTURE)
        self.assertEqual(legacy.transform(QA_PATH, once), once)

    def test_exact_precondition_rejects_drift(self):
        with self.assertRaises(legacy.PatchError):
            legacy.transform(QA_PATH, QA_FIXTURE.replace("<= 1", "< 2"))

    def test_marker_does_not_hide_incomplete_patch(self):
        with self.assertRaises(legacy.PatchError):
            legacy.transform(QA_PATH, legacy.MARKER + QA_FIXTURE)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.rules = patch.object(legacy, "RULES", {
            "example.py": [legacy.Rule("value = 1", "value = 2")],
            "second.py": [legacy.Rule("other = 1", "other = 2")],
        })
        self.rules.start()
        (self.root / "example.py").write_text("value = 1\n")
        (self.root / "second.py").write_text("other = 1\n")

    def tearDown(self):
        self.rules.stop()
        self.tmp.cleanup()

    def test_dry_run_writes_diff_only(self):
        output = self.base / "review.diff"
        legacy.main(["--source-root", str(self.root), "--output", str(output)])
        self.assertEqual((self.root / "example.py").read_text(), "value = 1\n")
        self.assertIn("+value = 2", output.read_text())
        self.assertTrue(Path(str(output) + ".manifest.json").is_file())

    def test_dry_run_cannot_write_review_artifacts_into_source(self):
        with self.assertRaises(SystemExit) as error:
            legacy.main(["--source-root", str(self.root), "--output", str(self.root / "example.py")])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((self.root / "example.py").read_text(), "value = 1\n")

    def test_apply_requires_review_and_backs_up(self):
        output = self.base / "review.diff"
        args = ["--source-root", str(self.root), "--output", str(output)]
        legacy.main(args)
        backup = self.base / "backup"
        legacy.main(args + ["--apply", "--backup-dir", str(backup)])
        self.assertIn("value = 2", (self.root / "example.py").read_text())
        self.assertEqual((backup / "example.py").read_text(), "value = 1\n")
        files, diff = legacy.build_plan(self.root)
        self.assertEqual(diff, "")
        self.assertTrue(all(item["before"] == item["after"] for item in files.values()))

    def test_drift_in_any_file_blocks_all_writes(self):
        files, diff = legacy.build_plan(self.root)
        reviewed = legacy.manifest_for(self.root, files, diff)
        (self.root / "second.py").write_text("other = 3\n")
        with self.assertRaises(legacy.PatchError):
            legacy.apply_plan(self.root, files, reviewed, diff, self.base / "backup")
        self.assertEqual((self.root / "example.py").read_text(), "value = 1\n")
        self.assertFalse((self.base / "backup").exists())

    def test_manifest_tampering_blocks_writes(self):
        files, diff = legacy.build_plan(self.root)
        reviewed = legacy.manifest_for(self.root, files, diff)
        reviewed["files"]["second.py"]["before_sha256"] = "0" * 64
        with self.assertRaises(legacy.PatchError):
            legacy.apply_plan(self.root, files, reviewed, diff, self.base / "backup")
        self.assertEqual((self.root / "example.py").read_text(), "value = 1\n")

    def test_reviewed_diff_tampering_blocks_cli_apply(self):
        output = self.base / "review.diff"
        args = ["--source-root", str(self.root), "--output", str(output)]
        legacy.main(args)
        output.write_text("edited diff\n")
        with self.assertRaises(SystemExit) as error:
            legacy.main(args + ["--apply", "--backup-dir", str(self.base / "backup")])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((self.root / "example.py").read_text(), "value = 1\n")

    def test_rollback_on_write_failure(self):
        files, diff = legacy.build_plan(self.root)
        reviewed = legacy.manifest_for(self.root, files, diff)
        original_write = legacy.atomic_write
        count = 0

        def fail_second(path, data, mode=None):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated write failure")
            return original_write(path, data, mode)

        with patch.object(legacy, "atomic_write", side_effect=fail_second):
            with self.assertRaises(OSError):
                legacy.apply_plan(self.root, files, reviewed, diff, self.base / "backup")
        self.assertEqual((self.root / "example.py").read_text(), "value = 1\n")
        self.assertEqual((self.root / "second.py").read_text(), "other = 1\n")

    def test_symlink_source_refused(self):
        file = self.root / "example.py"
        file.unlink()
        outside = self.base / "outside.py"
        outside.write_text("value = 1\n")
        file.symlink_to(outside)
        with self.assertRaises(legacy.PatchError):
            legacy.build_plan(self.root)


if __name__ == "__main__":
    unittest.main()
