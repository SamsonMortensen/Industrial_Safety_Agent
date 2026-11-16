"""Check checkout selection before any environment installation or model work."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import check_public_checkout as checkout


class PublicCheckoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "--quiet")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.com")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", str(self.root / "empty-hooks"))
        self.git("remote", "add", "origin", "https://github.com/" + checkout.REPOSITORY + ".git")
        for name in checkout.REQUIRED_FILES:
            self.write(name, "fixture\n")
        self.write("tests/test_example.py", "# fixture\n")
        self.write(".gitignore", ".venv/\nprivate/\ncode/ignored*.py\ntests/ignored*.py\n")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Fixture")

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.root), *arguments], stderr=subprocess.PIPE)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_clean_checkout_passes_without_fetch_or_install(self):
        before = self.git("status", "--porcelain")
        result = checkout.check(self.root)
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["private_resource_contents_inspected"])
        self.assertEqual(self.git("status", "--porcelain"), before)

    def test_nested_directory_and_non_repository_are_rejected(self):
        self.assertFalse(checkout.check(self.root / "code")["ok"])
        with tempfile.TemporaryDirectory() as unrelated:
            self.assertFalse(checkout.check(Path(unrelated))["ok"])

    def test_modified_public_input_is_rejected_without_repair(self):
        path = self.write("json/regulations.json", "changed fixture\n")
        before = path.read_bytes()
        result = checkout.check(self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["changed_source_or_input_count"], 1)
        self.assertEqual(path.read_bytes(), before)

    def test_staged_source_change_is_rejected_even_if_worktree_was_restored(self):
        path = self.write("code/showcase.py", "changed fixture\n")
        self.git("add", "code/showcase.py")
        path.write_text("fixture\n", encoding="utf-8")
        result = checkout.check(self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["changed_source_or_input_count"], 1)

    def test_ignored_and_untracked_python_modules_are_rejected_without_execution(self):
        for name in ("tests/ignored_private.py", "code/ignored_private.py", "tests/private_helper.py"):
            with self.subTest(name=name):
                path = self.write(name, "raise RuntimeError('must never execute')\n")
                result = checkout.check(self.root)
                self.assertFalse(result["ok"])
                self.assertEqual(result["extra_python_module_count"], 1)
                self.assertNotIn(name, json.dumps(result))
                path.unlink()

    def test_missing_required_file_is_rejected(self):
        (self.root / "main.ipynb").unlink()
        self.assertFalse(checkout.check(self.root)["ok"])

    def test_notebook_edits_and_private_resources_are_left_alone(self):
        notebook = self.write("main.ipynb", "edited notebook fixture\n")
        private = self.write("private/labels.py", "private fixture\n")
        environment = self.write(".venv/extra.py", "environment fixture\n")
        with patch.object(checkout, "python_modules", wraps=checkout.python_modules) as inventory:
            self.assertTrue(checkout.check(self.root)["ok"])
            inventory.assert_called_once_with(self.root.resolve())
        self.assertEqual(notebook.read_text(), "edited notebook fixture\n")
        self.assertEqual(private.read_text(), "private fixture\n")
        self.assertEqual(environment.read_text(), "environment fixture\n")

    def test_official_origins_and_credential_rejection(self):
        for remote in (
            "https://github.com/" + checkout.REPOSITORY,
            "https://github.com/" + checkout.REPOSITORY + ".git",
            "@".join(("git", "github.com:" + checkout.REPOSITORY + ".git")),
        ):
            with self.subTest(remote=remote):
                self.git("remote", "set-url", "origin", remote)
                self.assertTrue(checkout.check(self.root)["ok"])
        secret = "placeholder_credential"
        for remote in (
            "https://" + secret + "@github.com/" + checkout.REPOSITORY + ".git",
            "https://github.com.invalid/" + checkout.REPOSITORY,
            "https://github.com/" + checkout.REPOSITORY + "?credential=" + secret,
            "https://github.com/unrelated/project.git",
        ):
            with self.subTest(remote=remote):
                self.git("remote", "set-url", "origin", remote)
                result = checkout.check(self.root)
                self.assertFalse(result["ok"])
                self.assertNotIn(remote, json.dumps(result))
                self.assertNotIn(secret, json.dumps(result))

    def test_source_symlink_is_rejected_without_following_it(self):
        target = self.root / "private_target.py"
        target.write_text("private fixture\n", encoding="utf-8")
        link = self.root / "tests/linked.py"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("This account cannot create symbolic links")
        self.assertFalse(checkout.check(self.root)["ok"])
        self.assertEqual(target.read_text(), "private fixture\n")

    def test_linked_input_parent_is_rejected_before_git_reads_file_content(self):
        real_link_check = checkout.is_link
        real_git = checkout.git
        calls = []

        def record_git(root, *arguments):
            calls.append(arguments)
            return real_git(root, *arguments)

        def linked_json(path):
            return path == self.root / "json" or real_link_check(path)

        with patch.object(checkout, "is_link", side_effect=linked_json), patch.object(checkout, "git", side_effect=record_git):
            result = checkout.check(self.root)
        self.assertFalse(result["ok"])
        self.assertFalse(any(arguments[0] == "diff" for arguments in calls))

    def test_tracked_private_tree_is_rejected_from_names_before_diff(self):
        private = self.write("data/vision_private/labels.json", "private fixture\n")
        self.git("add", "data/vision_private/labels.json")
        self.git("commit", "--quiet", "-m", "Private fixture tree")
        private.write_text("changed private fixture\n", encoding="utf-8")
        real_git = checkout.git
        calls = []

        def record_git(root, *arguments):
            calls.append(arguments)
            return real_git(root, *arguments)

        with patch.object(checkout, "git", side_effect=record_git), patch.object(checkout, "ordinary_path") as inspect:
            result = checkout.check(self.root)
        self.assertFalse(result["ok"])
        self.assertFalse(any(arguments[0] == "diff" for arguments in calls))
        inspect.assert_not_called()
        self.assertNotIn("labels.json", json.dumps(result))
        self.assertEqual(private.read_text(), "changed private fixture\n")

    def test_cli_reports_failure_with_exit_two_without_printing_file_content(self):
        private_text = "DO_NOT_PRINT_PRIVATE_MODULE_CONTENT"
        self.write("tests/ignored_private.py", private_text)
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["check", "--root", str(self.root)]), contextlib.redirect_stdout(stdout):
            self.assertEqual(checkout.main(), 2)
        self.assertFalse(json.loads(stdout.getvalue())["ok"])
        self.assertNotIn(private_text, stdout.getvalue())
        self.assertNotIn(str(self.root), stdout.getvalue())

    def test_newly_staged_private_tree_is_rejected_before_diff(self):
        private = self.write("data/vision_private/labels.json", "private fixture\n")
        self.git("add", "data/vision_private/labels.json")
        real_git = checkout.git
        calls = []

        def record_git(root, *arguments):
            calls.append(arguments)
            return real_git(root, *arguments)

        with patch.object(checkout, "git", side_effect=record_git), patch.object(checkout, "ordinary_path") as inspect:
            result = checkout.check(self.root)
        self.assertFalse(result["ok"])
        self.assertFalse(any(arguments[0] == "diff" for arguments in calls))
        inspect.assert_not_called()
        self.assertNotIn("labels.json", json.dumps(result))
        self.assertEqual(private.read_text(), "private fixture\n")

    def test_newly_staged_input_link_is_rejected_before_diff(self):
        added = self.write("data/new_fixture/input.json", "fixture\n")
        self.git("add", "data/new_fixture/input.json")
        real_link_check = checkout.is_link
        real_git = checkout.git
        calls = []

        def record_git(root, *arguments):
            calls.append(arguments)
            return real_git(root, *arguments)

        def linked_new_directory(path):
            return path == added.parent or real_link_check(path)

        with patch.object(checkout, "is_link", side_effect=linked_new_directory), patch.object(checkout, "git", side_effect=record_git):
            result = checkout.check(self.root)
        self.assertFalse(result["ok"])
        self.assertFalse(any(arguments[0] == "diff" for arguments in calls))
        self.assertEqual(added.read_text(), "fixture\n")

    def test_cli_keeps_selected_checkout_when_checker_is_a_link(self):
        target = self.root / "code/check_public_checkout.py"
        target.write_bytes(Path(checkout.__file__).read_bytes())
        self.git("add", "code/check_public_checkout.py")
        self.git("commit", "--quiet", "-m", "Runnable checkout guard fixture")
        with tempfile.TemporaryDirectory() as temporary:
            selected = Path(temporary) / "selected_checkout"
            shutil.copytree(self.root, selected)
            link = selected / "code/check_public_checkout.py"
            link.unlink()
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("This account cannot create symbolic links")
            result = subprocess.run(
                [sys.executable, "-B", str(link)], cwd=selected,
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(json.loads(result.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
