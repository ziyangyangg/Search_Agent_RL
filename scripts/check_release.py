#!/usr/bin/env python3
"""Verify release files and preserved research sources without installing GPU tools.

Run from any directory: python /path/to/scripts/check_release.py.
The release manifest covers final repository files; SOURCE_MANIFEST additionally
checks the original baseline bytes and explicitly records README relocation.
"""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(root: Path, expected: dict[str, str], relocations=None) -> list[str]:
    errors = []
    for name, digest in expected.items():
        path = root / (relocations or {}).get(name, name)
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(root)}")
        elif sha256(path) != digest:
            errors.append(f"changed: {path.relative_to(root)}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    release = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    errors = check_files(root, release["files"])
    if not (root / "legacy/SOURCE_MANIFEST.json").is_file():
        print("This check requires the full source checkout; application-only installs exclude legacy.")
        return 2
    source = json.loads((root / "legacy/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    errors.extend(check_files(root / "legacy/Search-R1", source["original_files_sha256"],
                              source.get("relocated_original_files", {})))
    errors.extend(check_files(root / "legacy/autodl-scripts", source["autodl_script_files_sha256"]))
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"Release {release['release']}: {len(release['files'])} file hashes verified.")
    print("Original baseline and AutoDL source hashes verified; README relocation accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
