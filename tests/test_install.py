import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("install_on_drive", Path(__file__).resolve().parents[1] / "scripts/install_on_drive.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def prepare(self, root):
        source, bundle, destination = root / "original", root / "bundle", root / "new"
        sentinel = source / "Search-R1/search_r1/llm_agent/generation.py"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("original")
        bundle.mkdir()
        (bundle / ".gitignore").write_text("runs/\n")
        (bundle / "README.md").write_text("new project")
        return source, bundle, destination

    def test_assets_are_referenced_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, destination = root / "original", root / "bundle", root / "new"
            sentinel = source / "Search-R1/search_r1/llm_agent/generation.py"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("original")
            bundle.mkdir()
            (bundle / ".gitignore").write_text("runs/\n")
            (bundle / "README.md").write_text("new project")
            installer.install(bundle, source, destination)
            self.assertEqual(sentinel.read_text(), "original")
            self.assertFalse((destination / "Search-R1").exists())
            self.assertTrue((destination / "asset-paths.local.json").exists())
            with self.assertRaises(FileExistsError):
                installer.install(bundle, source, destination)

    def test_developed_checkout_copies_only_release_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source, bundle, destination = self.prepare(Path(directory))
            (bundle / "src").mkdir()
            (bundle / "src/main.py").write_text("pass")
            (bundle / "src/.env.secret").write_text("secret")
            (bundle / "src/model.safetensors").write_text("weight")
            for name in (".env", "local-notes.txt"):
                (bundle / name).write_text("private")
            for name in ("assets", "build", "dist", "legacy", "runs", ".git"):
                (bundle / name).mkdir()
                (bundle / name / "private.txt").write_text("private")
            installer.install(bundle, source, destination)
            self.assertTrue((destination / "src/main.py").exists())
            for relative in (".env", "local-notes.txt", "src/.env.secret", "src/model.safetensors",
                             "assets", "build", "dist", "legacy", "runs", ".git"):
                self.assertFalse((destination / relative).exists(), relative)

    def test_release_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, destination = self.prepare(root)
            alias = root / "alias"
            alias.symlink_to(bundle, target_is_directory=True)
            with self.assertRaises(PermissionError):
                installer.install(alias, source, destination)
            self.assertFalse(destination.exists())

    def test_selected_directory_symlink_is_rejected_before_any_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, destination = self.prepare(root)
            outside = root / "private"
            outside.mkdir()
            (outside / "private.txt").write_text("secret")
            (bundle / "docs").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(PermissionError):
                installer.install(bundle, source, destination)
            self.assertFalse(destination.exists())

    def test_selected_file_symlink_is_rejected_before_any_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, destination = self.prepare(root)
            (bundle / "src").mkdir()
            outside = root / "private.txt"
            outside.write_text("secret")
            (bundle / "src/private.txt").symlink_to(outside)
            with self.assertRaises(PermissionError):
                installer.install(bundle, source, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
