#!/usr/bin/env python3
"""Generate a reviewed patch for the audited Search-R1 snapshot; never apply by default.

Dry run: python scripts/patch_legacy.py --source-root /path/Search-R1 --output audit.diff
Apply: repeat with --apply --backup-dir /path/to/new-backup-directory.
The apply command requires the manifest produced by the earlier dry run. Changes to
any source file or the reviewed diff invalidate approval. GPU execution is untested.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass

MARKER = "# search-agent-workbench legacy patch v1\n"


class PatchError(ValueError):
    """Source drift, incomplete application, or an invalid review manifest."""


@dataclass(frozen=True)
class Rule:
    old: str
    new: str
    count: int = 1


RULES = {
    "search_r1/llm_agent/generation.py": [
        Rule("import torch\n", "import copy\nimport torch\n"),
        Rule("padded_active_batch = DataProto.from_dict(padded_batch)",
             "padded_active_batch = DataProto.from_dict(\n            padded_batch, meta_info=copy.deepcopy(active_batch.meta_info)\n        )"),
        Rule("                k: v[active_mask] for k, v in rollings.batch.items()\n            })",
             "                k: v[active_mask] for k, v in rollings.batch.items()\n            }, meta_info=copy.deepcopy(rollings.meta_info))", 2),
        Rule("""        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]""",
             """        def first_action_prefix(response):
            # Stop at the earliest complete, well-formed action, regardless of type.
            match = re.search(r'<(search|answer)>.*?</\\1>', response, re.DOTALL)
            return response[:match.end()] if match else response

        responses_str = [first_action_prefix(resp) for resp in responses_str]"""),
        Rule("    def _batch_search(self, queries):\n        \n        payload = {",
             "    def _batch_search(self, queries):\n        if not queries:\n            return {\"result\": []}\n        payload = {"),
        Rule("        return requests.post(self.config.search_url, json=payload).json()",
             """        response = requests.post(self.config.search_url, json=payload, timeout=(5, 30))
        response.raise_for_status()
        result = response.json()
        if (not isinstance(result, dict) or not isinstance(result.get("result"), list)
                or len(result["result"]) != len(queries)):
            raise ValueError("Retriever response must contain one result list per query")
        if any(not isinstance(items, list) for items in result["result"]):
            raise ValueError("Each retriever result must be a list")
        return result"""),
    ],
    "verl/trainer/main_ppo.py": [
        Rule("            sequences = torch.cat((valid_prompt_ids, valid_response_ids))\n            sequences_str = self.tokenizer.decode(sequences)",
             """            # Score assistant-owned tokens only: neither prompt examples nor
            # tool text can supply an answer. Provenance remains valid even if
            # observation truncation removed a textual </information> delimiter.
            if 'info_mask' in data_item.batch:
                owned_mask = data_item.batch['info_mask'][
                    prompt_length:prompt_length + valid_response_length
                ].bool()
                valid_response_ids = valid_response_ids[owned_mask]
            sequences_str = self.tokenizer.decode(valid_response_ids)"""),
    ],
    "verl/utils/reward_score/qa_em.py": [
        Rule("    answer_pattern = r'<answer>(.*?)</answer>'\n    match = re.finditer(answer_pattern, solution_str, re.DOTALL)",
             """    # Tool observations cannot supply the model's scored answer. An unclosed
    # information block consumes the remainder, rather than exposing its contents.
    solution_str = re.sub(r'<information>.*?(?:</information>|$)', '',
                          solution_str, flags=re.DOTALL)
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)"""),
        Rule("    # If there are 0 or exactly 1 matches, return None\n    if len(matches) <= 1:",
             "    # A single complete assistant answer is sufficient.\n    if not matches:"),
        Rule("    # If there are 2 or more matches, return the last one",
             "    # Preserve the legacy last-answer policy within the assistant response."),
    ],
    "verl/trainer/ppo/ray_trainer.py": [
        Rule("                batch: DataProto = DataProto.from_single_dict(batch_dict)\n                batch = batch.repeat",
             """                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # Assign groups before repetition. Source/index alone can collide
                # across merged files; the batch occurrence makes every group unique.
                batch.non_tensor_batch['uid'] = np.array([
                    f'{source}:{index}:{position}'
                    for position, (source, index) in enumerate(zip(
                        batch.non_tensor_batch['data_source'], batch.non_tensor_batch['index']))
                ], dtype=object)
                batch = batch.repeat"""),
        Rule("                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],\n                                                                dtype=object)",
             "                        # Keep the pre-repeat group ID through rollout repetition."),
        Rule("                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()",
             "                        # Keep the pre-repeat group ID through batch balancing."),
    ],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(relative: str, text: str) -> str:
    """Exact preconditions fail closed; a fully patched source is an idempotent no-op."""
    rules = RULES[relative]
    if text.startswith(MARKER):
        for rule in rules:
            if text.count(rule.new) != rule.count:
                raise PatchError(f"Incomplete or altered patched file: {relative}")
        ast.parse(text, filename=relative)
        return text
    original = text
    for number, rule in enumerate(rules, 1):
        if text.count(rule.old) != rule.count:
            raise PatchError(f"Source drift in {relative}, rule {number}: expected {rule.count} occurrence(s)")
        text = text.replace(rule.old, rule.new)
    if text == original:
        raise PatchError(f"No transformation for {relative}")
    ast.parse(MARKER + text, filename=relative)
    return MARKER + text


def build_plan(root: Path) -> tuple[dict, str]:
    root = root.resolve(strict=True)
    files = {}
    diffs = []
    for relative in RULES:
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise PatchError(f"Expected a regular source file beneath source root: {relative}")
        before = path.read_bytes()
        after = transform(relative, before.decode("utf-8")).encode("utf-8")
        files[relative] = {"before": before, "after": after,
                           "before_sha256": digest(before), "after_sha256": digest(after)}
        diffs.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),
                                         after.decode().splitlines(keepends=True),
                                         fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    return files, "".join(diffs)


def manifest_for(root: Path, files: dict, patch: str) -> dict:
    return {"schema_version": 1, "patch_version": 1, "source_root": str(root.resolve()),
            "patch_sha256": digest(patch.encode("utf-8")),
            "files": {name: {key: data[key] for key in ("before_sha256", "after_sha256")}
                      for name, data in files.items()}}


def atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_plan(root: Path, files: dict, reviewed: dict, patch: str, backup_dir: Path) -> None:
    """Require a reviewed manifest, validate all hashes, back up all sources, then write."""
    expected = manifest_for(root, files, patch)
    if reviewed != expected:
        raise PatchError("Reviewed manifest does not match current sources and generated diff; run dry-run again")
    if not any(item["before"] != item["after"] for item in files.values()):
        return
    for relative, item in files.items():
        if digest((root / relative).read_bytes()) != item["before_sha256"]:
            raise PatchError(f"Source changed after planning: {relative}")
    backup_dir = backup_dir.resolve()
    if backup_dir.exists() or backup_dir.is_relative_to(root.resolve()):
        raise PatchError("Backup directory must be new and outside the source tree")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for relative in files:
        target = backup_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    (backup_dir / "manifest.json").write_text(json.dumps(reviewed, indent=2) + "\n", encoding="utf-8")
    changed = []
    try:
        for relative, item in files.items():
            path = root / relative
            if digest(path.read_bytes()) != item["before_sha256"]:
                raise PatchError(f"Source changed before write: {relative}")
            if item["before"] != item["after"]:
                atomic_write(path, item["after"], path.stat().st_mode & 0o777)
                changed.append(relative)
    except BaseException:
        for relative in reversed(changed):
            path = root / relative
            atomic_write(path, files[relative]["before"], (backup_dir / relative).stat().st_mode & 0o777)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Unified diff to review")
    parser.add_argument("--manifest", type=Path, help="Defaults to OUTPUT.manifest.json")
    parser.add_argument("--apply", action="store_true", help="Apply the previously generated and reviewed diff")
    parser.add_argument("--backup-dir", type=Path, help="New directory outside source root; required for --apply")
    args = parser.parse_args(argv)
    root = args.source_root.resolve()
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    try:
        if (args.output.resolve().is_relative_to(root)
                or manifest_path.resolve().is_relative_to(root)
                or args.output.resolve() == manifest_path.resolve()
                or args.output.is_symlink() or manifest_path.is_symlink()):
            raise PatchError("Diff and manifest must be distinct regular paths outside the source tree")
        files, patch = build_plan(root)
        current_manifest = manifest_for(root, files, patch)
        if args.apply:
            if args.backup_dir is None or not manifest_path.is_file() or not args.output.is_file():
                raise PatchError("Apply requires --backup-dir and an existing dry-run diff plus manifest")
            if digest(args.output.read_bytes()) != current_manifest["patch_sha256"]:
                raise PatchError("Reviewed diff differs from generated diff")
            apply_plan(root, files, json.loads(manifest_path.read_text()), patch, args.backup_dir)
            print("Applied reviewed patch; original files backed up. GPU behavior requires separate validation.")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(patch, encoding="utf-8")
            manifest_path.write_text(json.dumps(current_manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Dry run: {sum(v['before'] != v['after'] for v in files.values())} changed files; sources untouched.")
        return 0
    except (PatchError, OSError, UnicodeError, SyntaxError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Patch refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
