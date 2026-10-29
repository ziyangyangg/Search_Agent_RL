"""Typed tools and a deliberately small, confined workspace toolset.

Path checks are best effort application guards, not an operating-system sandbox.
Run untrusted tools in an isolated process/container with separate credentials.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable


class ValidationError(ValueError):
    """Arguments or a schema do not satisfy the supported JSON Schema subset."""


_SCHEMA_KEYS = {
    "type", "description", "title", "default", "properties", "required",
    "additionalProperties", "items", "enum", "const", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
    "maxLength", "pattern", "minItems", "maxItems", "uniqueItems", "minProperties",
    "maxProperties", "anyOf", "oneOf", "allOf",
}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _is_type(value: Any, kind: str) -> bool:
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    return checks[kind]()


def _check_schema(schema: Any, path: str = "schema") -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValidationError(f"{path}: expected a schema object or boolean")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise ValidationError(f"{path}: unsupported schema keywords: {sorted(unknown)}")
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if not isinstance(kinds, list) or any(k not in _TYPES for k in kinds):
        raise ValidationError(f"{path}.type: invalid JSON type")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise ValidationError(f"{path}.properties: expected an object")
        for key, sub in schema["properties"].items():
            if not isinstance(key, str):
                raise ValidationError(f"{path}.properties: keys must be strings")
            _check_schema(sub, f"{path}.properties.{key}")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
            raise ValidationError(f"{path}.required: expected an array of strings")
    for key in ("items", "additionalProperties"):
        if key in schema:
            _check_schema(schema[key], f"{path}.{key}")
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise ValidationError(f"{path}.{key}: expected a nonempty schema array")
            for sub in schema[key]:
                _check_schema(sub, f"{path}.{key}")
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if key in schema and (not _is_type(schema[key], "integer") or schema[key] < 0):
            raise ValidationError(f"{path}.{key}: expected a nonnegative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in schema and (not _is_type(schema[key], "number") or not math.isfinite(schema[key])):
            raise ValidationError(f"{path}.{key}: expected a finite number")
    if schema.get("multipleOf", 1) <= 0:
        raise ValidationError(f"{path}.multipleOf: must be positive")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValidationError(f"{path}.enum: expected a nonempty array")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValidationError(f"{path}.uniqueItems: expected a boolean")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (TypeError, re.error) as exc:
            raise ValidationError(f"{path}.pattern: invalid regular expression") from exc


def _json_equal(a: Any, b: Any) -> bool:
    # Python considers True == 1; JSON Schema does not.
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def validate_arguments(value: Any, schema: dict | bool, path: str = "arguments") -> None:
    """Validate the documented subset; no coercion, defaults, or remote references."""
    if schema is True:
        return
    if schema is False:
        raise ValidationError(f"{path}: value is not allowed")
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if kinds and not any(_is_type(value, kind) for kind in kinds):
        raise ValidationError(f"{path}: expected {' or '.join(kinds)}")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise ValidationError(f"{path}: not one of the allowed values")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValidationError(f"{path}: does not match the required constant")
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            matches = 0
            for sub in schema[key]:
                try:
                    validate_arguments(value, sub, path)
                    matches += 1
                except ValidationError:
                    pass
            expected = len(schema[key]) if key == "allOf" else 1
            if (key == "anyOf" and matches == 0) or (key != "anyOf" and matches != expected):
                raise ValidationError(f"{path}: does not satisfy {key}")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValidationError(f"{path}: object keys must be strings")
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValidationError(f"{path}: missing required fields: {sorted(missing)}")
        props = schema.get("properties", {})
        for key, item in value.items():
            validate_arguments(item, props.get(key, schema.get("additionalProperties", True)), f"{path}.{key}")
        for key, comparison in (("minProperties", lambda n: len(value) >= n), ("maxProperties", lambda n: len(value) <= n)):
            if key in schema and not comparison(schema[key]):
                raise ValidationError(f"{path}: violates {key}")
    if isinstance(value, list):
        for i, item in enumerate(value):
            validate_arguments(item, schema.get("items", True), f"{path}[{i}]")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValidationError(f"{path}: too many items")
        if schema.get("uniqueItems") and any(_json_equal(v, old) for i, v in enumerate(value) for old in value[:i]):
            raise ValidationError(f"{path}: duplicate items")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: string too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(f"{path}: string too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValidationError(f"{path}: does not match pattern")
    if _is_type(value, "number"):
        if not math.isfinite(value):
            raise ValidationError(f"{path}: number must be finite")
        for key, bad in (("minimum", lambda x: value < x), ("maximum", lambda x: value > x),
                         ("exclusiveMinimum", lambda x: value <= x), ("exclusiveMaximum", lambda x: value >= x)):
            if key in schema and bad(schema[key]):
                raise ValidationError(f"{path}: violates {key}")
        if "multipleOf" in schema:
            quotient = value / schema["multipleOf"]
            if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
                raise ValidationError(f"{path}: violates multipleOf")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    effect: str = "read"


class ToolRegistry:
    """Explicit registration, schema checks, effects, and bounded JSON results."""

    def __init__(self, max_result_chars: int = 120_000):
        if max_result_chars < 1:
            raise ValueError("max_result_chars must be positive")
        self.max_result_chars = max_result_chars
        self._tools: dict[str, tuple[ToolSpec, Callable[[dict], Any]]] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict], Any]) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec.name):
            raise ValueError("Tool names must contain 1–64 letters, digits, underscores, or hyphens")
        if spec.effect not in {"read", "write"}:
            raise ValueError("Tool effect must be 'read' or 'write'")
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool: {spec.name}")
        if not callable(handler):
            raise TypeError("Tool handler must be callable")
        if not isinstance(spec.parameters, dict) or spec.parameters.get("type") != "object":
            raise ValidationError("Tool parameters must have type 'object'")
        _check_schema(spec.parameters)
        # Freeze caller-owned mutable schema data by copying it.
        self._tools[spec.name] = (ToolSpec(spec.name, spec.description, deepcopy(spec.parameters), spec.effect), handler)

    def specifications(self) -> list[dict]:
        return [{"type": "function", "function": {"name": s.name, "description": s.description,
                 "parameters": deepcopy(s.parameters)}} for s, _ in self._tools.values()]

    def execute(self, name: str, args: dict, *, allow_writes: bool = False) -> dict:
        def error(kind: str, message: str) -> dict:
            return {"ok": False, "error": {"type": kind, "message": message[:1000]}}

        if name not in self._tools:
            return error("ToolNotFound", f"Unknown tool: {name}")
        spec, handler = self._tools[name]
        if spec.effect == "write" and not allow_writes:
            return error("approval_required", "Write tools require explicit caller authorization")
        try:
            if not isinstance(args, dict):
                raise ValidationError("arguments: expected an object")
            # Reject non-JSON objects and nonfinite numbers before the handler runs.
            serialized = json.dumps(args, ensure_ascii=False, allow_nan=False)
            if len(serialized) > 150_000:
                raise ValidationError("arguments exceed the size limit")
            validate_arguments(args, spec.parameters)
            result = handler(deepcopy(args))
            result_text = json.dumps(result, ensure_ascii=False, allow_nan=False)
            if len(result_text) > self.max_result_chars:
                return error("OutputTooLarge", "Tool result exceeds the configured limit")
            return {"ok": True, "value": result}
        except ValidationError as exc:
            return error("ValidationError", str(exc))
        except PermissionError as exc:
            return error("PermissionDenied", str(exc))
        except FileExistsError:
            return error("FileExistsError", "Destination already exists; overwrite is disabled")
        except (TypeError, ValueError) as exc:
            return error("ValidationError", str(exc))
        except Exception as exc:
            return error(type(exc).__name__, str(exc))


class WorkspaceTools:
    """Read and create text artifacts inside one existing directory.

    All symlinks and hidden paths are denied. Concurrent hostile filesystem
    changes can defeat application-level checks: this is not a security sandbox.
    """

    EXTENSIONS = {".md", ".txt", ".csv", ".json"}
    PROTECTED_NAMES = {"credentials.json", "secrets.json", "token.json", "tokens.json",
                       "service-account.json", "id_rsa", "id_ed25519", "credentials"}

    def __init__(self, root: Path):
        root = Path(root).expanduser()
        if root.is_symlink():
            raise PermissionError("Workspace root must not be a symlink")
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError("Workspace root must be an existing directory")

    def _resolve(self, relative: str, *, file: bool = False) -> Path:
        if not isinstance(relative, str) or "\x00" in relative or "\\" in relative:
            raise PermissionError("Invalid workspace path")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PermissionError("Only paths inside the workspace are allowed")
        for part in candidate.parts:
            if part.startswith(".") or part.lower() in self.PROTECTED_NAMES:
                raise PermissionError("Hidden and credential paths are unavailable")
        target = self.root / candidate
        cursor = self.root
        for part in candidate.parts:
            cursor /= part
            if cursor.is_symlink():
                raise PermissionError("Symlink paths are unavailable")
        resolved = target.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise PermissionError("Path escapes the workspace")
        if file and resolved.suffix.lower() not in self.EXTENSIONS:
            raise PermissionError("Supported file extensions: .md, .txt, .csv, .json")
        return resolved

    def list_files(self, args: dict) -> dict:
        directory = self._resolve(args.get("path", ""))
        if not directory.is_dir():
            raise NotADirectoryError("Requested path is not a directory")
        entries = []
        truncated = False
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.name.startswith(".") or path.name.lower() in self.PROTECTED_NAMES or path.is_symlink():
                continue
            if not path.is_dir() and path.suffix.lower() not in self.EXTENSIONS:
                continue
            if len(entries) >= 500:
                truncated = True
                break
            entry = {"path": path.relative_to(self.root).as_posix(), "kind": "directory" if path.is_dir() else "file"}
            if path.is_file():
                entry["size_bytes"] = path.stat().st_size
            entries.append(entry)
        return {"path": directory.relative_to(self.root).as_posix(), "entries": entries, "truncated": truncated}

    def read_text(self, args: dict) -> dict:
        path = self._resolve(args["path"], file=True)
        # Opening with O_NOFOLLOW also protects the final component at open time.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise PermissionError("Only regular text files can be read")
        limit = args.get("max_chars", 20_000)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            text = stream.read(limit + 1)
        return {"path": path.relative_to(self.root).as_posix(), "text": text[:limit], "truncated": len(text) > limit}

    def write_text(self, args: dict) -> dict:
        path = self._resolve(args["path"], file=True)
        if not path.parent.is_dir():
            raise FileNotFoundError("Destination parent directory must already exist")
        # O_EXCL guarantees create-only behavior even if another writer races us.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(args["content"])
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"path": path.relative_to(self.root).as_posix(), "bytes_written": len(args["content"].encode("utf-8"))}

    def register(self, registry: ToolRegistry) -> None:
        def params(properties: dict, required: list[str] | None = None) -> dict:
            return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}

        path = {"type": "string", "maxLength": 2048}
        registry.register(ToolSpec("workspace_list_files", "List visible text files and directories within the workspace.",
                                  params({"path": {**path, "default": ""}})), self.list_files)
        registry.register(ToolSpec("workspace_read_text", "Read a UTF-8 .md, .txt, .csv, or .json file in the workspace.",
                                  params({"path": path, "max_chars": {"type": "integer", "minimum": 1, "maximum": 100_000, "default": 20_000}}, ["path"])), self.read_text)
        registry.register(ToolSpec("workspace_write_text", "Create a new UTF-8 text artifact. Existing files are never overwritten.",
                                  params({"path": path, "content": {"type": "string", "maxLength": 100_000}}, ["path", "content"]), "write"), self.write_text)
