"""Behavioral checks for argument validation and filesystem boundaries."""

from pathlib import Path
import os
import tempfile
import unittest

from agent_workbench.tools import ToolRegistry, ToolSpec, ValidationError, WorkspaceTools


class ToolRegistryTests(unittest.TestCase):
    def test_schema_rejects_invalid_arguments_before_handler(self):
        calls = []
        registry = ToolRegistry()
        registry.register(ToolSpec("sample", "Example", {
            "type": "object", "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 4},
                "mode": {"type": "string", "enum": ["a", "b"]},
                "items": {"type": "array", "items": {"type": "string", "minLength": 2}, "uniqueItems": True},
            }, "required": ["count"], "additionalProperties": False,
        }), lambda args: calls.append(args) or args)
        for args in ({}, {"count": True}, {"count": 5}, {"count": 1, "extra": 2},
                     {"count": 1, "mode": "c"}, {"count": 1, "items": ["x"]},
                     {"count": 1, "items": ["xx", "xx"]}, {"count": float("nan")}):
            self.assertFalse(registry.execute("sample", args)["ok"], args)
        self.assertEqual(calls, [])
        self.assertTrue(registry.execute("sample", {"count": 2, "items": ["ok"]})["ok"])

    def test_bad_schema_duplicate_and_missing_tool(self):
        registry = ToolRegistry()
        with self.assertRaises(ValidationError):
            registry.register(ToolSpec("bad", "Bad", {"type": "object", "$ref": "external"}), lambda args: None)
        spec = ToolSpec("test", "Test", {"type": "object"})
        registry.register(spec, lambda args: args)
        with self.assertRaises(ValueError):
            registry.register(spec, lambda args: args)
        self.assertEqual(registry.execute("missing", {})["error"]["type"], "ToolNotFound")
        self.assertFalse(registry.execute("test", "{}")["ok"])

    def test_schemas_are_copied_and_outputs_bounded(self):
        schema = {"type": "object", "additionalProperties": False}
        registry = ToolRegistry(max_result_chars=20)
        registry.register(ToolSpec("test", "Test", schema), lambda args: "x" * 100)
        schema["additionalProperties"] = True
        exported = registry.specifications()
        exported[0]["function"]["parameters"]["additionalProperties"] = True
        self.assertEqual(registry.execute("test", {"x": 1})["error"]["type"], "ValidationError")
        self.assertEqual(registry.execute("test", {})["error"]["type"], "OutputTooLarge")


class WorkspaceToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.registry = ToolRegistry()
        WorkspaceTools(self.root).register(self.registry)

    def execute(self, name, args, **kwargs):
        return self.registry.execute("workspace_" + name, args, **kwargs)

    def test_read_list_and_truncation(self):
        (self.root / "notes.md").write_text("hello world", encoding="utf-8")
        (self.root / "image.png").write_bytes(b"not text")
        (self.root / ".env").write_text("secret", encoding="utf-8")
        listing = self.execute("list_files", {})
        self.assertTrue(listing["ok"])
        self.assertEqual([x["path"] for x in listing["value"]["entries"]], ["notes.md"])
        result = self.execute("read_text", {"path": "notes.md", "max_chars": 5})
        self.assertEqual(result["value"]["text"], "hello")
        self.assertTrue(result["value"]["truncated"])

    def test_traversal_absolute_credentials_and_symlink_escape(self):
        outside = Path(self.temp.name) / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        (self.root / "link.txt").symlink_to(outside)
        (self.root / "linked-dir").symlink_to(outside.parent, target_is_directory=True)
        (self.root / "credentials.json").write_text("secret", encoding="utf-8")
        for path in ("../secret.txt", str(outside), "link.txt", "linked-dir/secret.txt", ".env", ".ssh/key.txt", "credentials.json"):
            result = self.execute("read_text", {"path": path})
            self.assertEqual(result["error"]["type"], "PermissionDenied", path)
        self.assertEqual(self.execute("list_files", {"path": "../"})["error"]["type"], "PermissionDenied")

    def test_writes_require_authorization_and_never_overwrite(self):
        args = {"path": "report.md", "content": "new report"}
        self.assertEqual(self.execute("write_text", args)["error"]["type"], "approval_required")
        self.assertFalse((self.root / "report.md").exists())
        self.assertEqual(self.execute("write_text", {"path": "report.md"}, allow_writes=True)["error"]["type"], "ValidationError")
        self.assertFalse((self.root / "report.md").exists())
        self.assertTrue(self.execute("write_text", args, allow_writes=True)["ok"])
        self.assertEqual(self.execute("write_text", {**args, "content": "overwrite"}, allow_writes=True)["error"]["type"], "FileExistsError")
        self.assertEqual((self.root / "report.md").read_text(), "new report")
        self.assertFalse(self.execute("write_text", {"path": "../escape.md", "content": "x"}, allow_writes=True)["ok"])

    def test_arguments_and_file_types_fail_closed(self):
        for args in ({}, {"path": "x.md", "max_chars": True}, {"path": "x.md", "max_chars": 100001}, {"path": "x.md", "unused": 1}):
            self.assertEqual(self.execute("read_text", args)["error"]["type"], "ValidationError")
        self.assertEqual(self.execute("write_text", {"path": "program.py", "content": "x"}, allow_writes=True)["error"]["type"], "PermissionDenied")
        self.assertFalse(self.execute("write_text", {"path": "missing/out.md", "content": "x"}, allow_writes=True)["ok"])

    def test_root_must_already_exist(self):
        with self.assertRaises(FileNotFoundError):
            WorkspaceTools(self.root / "absent")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "named pipes are unavailable")
    def test_named_pipe_is_rejected_without_blocking(self):
        os.mkfifo(self.root / "stream.txt")
        self.assertEqual(self.execute("read_text", {"path": "stream.txt"})["error"]["type"], "PermissionDenied")


if __name__ == "__main__":
    unittest.main()
