#!/usr/bin/env python3
"""Install only the new source tree beside the original project; assets stay put.

The source snapshot, weights and watermarked material are never copied or moved.
No network requests, dependency installation, or legacy patches are applied.
"""

import argparse
import json
import shutil
from pathlib import Path


# Copy a release, not arbitrary contents of a developed checkout. Local secrets,
# weights and traces must never become part of a side-by-side installation.
ROOT_FILES = {
    ".gitignore", "README.md", "LICENSE", "NOTICE", "pyproject.toml",
    "Dockerfile", "MANIFEST.json", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
}
SOURCE_DIRECTORIES = {"src", "scripts", "tests", "docs", "configs", "examples", "patches"}
EXCLUDED_NAMES = {".git", ".venv", "__pycache__", "build", "dist", "assets", "legacy", "runs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".safetensors", ".parquet", ".db", ".sqlite", ".sqlite3"}


def release_files(bundle: Path) -> list[Path]:
    """Preflight selected regular files before creating the destination.

    Selected symlinks are rejected instead of followed. This copy safeguard is
    not a sandbox against concurrent hostile filesystem changes.
    """
    selected = []

    def visit(path: Path) -> None:
        if (path.name in EXCLUDED_NAMES or path.name.startswith(".env")
                or path.name.endswith(".egg-info") or path.suffix.lower() in EXCLUDED_SUFFIXES):
            return
        if path.is_symlink():
            raise PermissionError(f"Symlinks are not allowed in release sources: {path.relative_to(bundle)}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child)
        elif path.is_file():
            selected.append(path.relative_to(bundle))
        else:
            raise PermissionError(f"Only regular release files are allowed: {path.relative_to(bundle)}")

    for path in sorted(bundle.iterdir()):
        if path.name in ROOT_FILES or path.name in SOURCE_DIRECTORIES:
            visit(path)
        elif path.name == ".github":
            if path.is_symlink():
                raise PermissionError("The .github directory must not be a symlink")
            workflows = path / "workflows"
            if workflows.exists() or workflows.is_symlink():
                visit(workflows)
    return selected


def install(bundle: Path, source: Path, destination: Path) -> None:
    if bundle.is_symlink():
        raise PermissionError("The release root must not be a symlink")
    bundle, source = bundle.resolve(strict=True), source.resolve(strict=True)
    destination = destination.absolute()
    if not (source / "Search-R1/search_r1/llm_agent/generation.py").is_file():
        raise ValueError("source must be the original project container, not the Search-R1 subdirectory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination must be a new directory")
    if destination.resolve().is_relative_to(bundle) or bundle.is_relative_to(destination.resolve()):
        raise ValueError("installation and bundle paths must not contain each other")
    if destination.resolve().is_relative_to(source / "Search-R1"):
        raise ValueError("install beside, not inside, the original Search-R1 snapshot")
    # No destination writes occur until every selected source passes preflight.
    files = release_files(bundle)
    references = {
        "schema_version": "1", "legacy_source": str(source / "Search-R1"),
        "checkpoint": str(source / "global_step_100"),
        "qa_dataset": str(source / "Search-R1/eval/data/eval/hotpotqa_dev.jsonl"),
        "small_corpus": str(source / "Search-R1/eval/data/corpus/hotpotqa_corpus.jsonl"),
        "asset_policy": "references_only; original sources and weights unchanged",
        "excluded_material": "PDF and commercial guide documents are not part of this project",
    }
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / relative, target)
        (destination / "asset-paths.local.json").write_text(
            json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (destination / ".gitignore").open("a", encoding="utf-8") as output:
            output.write("\nasset-paths.local.json\n")
    except BaseException:
        # This directory was created exclusively above; remove a partial copy.
        shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    install(Path(__file__).resolve().parents[1], args.source, args.destination)
    print("New source tree installed; original sources and large assets unchanged.")


if __name__ == "__main__":
    main()
