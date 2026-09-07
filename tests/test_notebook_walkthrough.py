"""Check the guided notebook against the backend it presents."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NotebookWalkthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        cls.code = [
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]

    def test_notebook_is_unexecuted_and_uses_project_kernel(self):
        self.assertEqual(
            self.notebook["metadata"]["kernelspec"]["name"], "industrial-safety"
        )
        for cell in self.notebook["cells"]:
            self.assertNotIn("\u2014", "".join(cell["source"]))
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_cells_run_offline_and_repeat_without_changing_reference_files(self):
        from IPython import display as ipython_display
        import requests

        paths = [ROOT / "main.ipynb", *(ROOT / "json").glob("*.json")]
        before = {path: path.read_bytes() for path in paths}
        namespace = {"__name__": "__main__"}
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with patch.object(
                socket.socket, "connect", side_effect=AssertionError("network")
            ), patch.object(
                requests.sessions.Session, "request", side_effect=AssertionError("HTTP")
            ), patch.object(
                ipython_display, "display"
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                for _ in range(2):
                    for index, source in enumerate(self.code):
                        exec(
                            compile(source, f"main.ipynb:cell-{index}", "exec"),
                            namespace,
                        )
        finally:
            os.chdir(previous)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertEqual(namespace["my_result"]["status"], "REVIEW")
        self.assertEqual(len(namespace["results"]), 6)
        self.assertEqual(namespace["grounding_check"]["status"], "REVIEW")
        self.assertEqual(namespace["benchmark"]["cases"], 16)
        self.assertEqual(
            [row["trusted"] for row in namespace["learning_checks"]],
            [False, True, True, False],
        )
        self.assertTrue(
            all(not row["Computed again here"] for row in namespace["saved_rows"])
        )
        runtime = namespace["report"]["runtime"]
        self.assertEqual(runtime["model_requests"], 0)
        self.assertFalse(runtime["training_performed"])
        self.assertFalse(runtime["learning_memory_modified"])

    def test_setup_explains_missing_packages(self):
        find_spec = importlib.util.find_spec
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with patch.object(
                importlib.util,
                "find_spec",
                side_effect=lambda name: None if name == "pandas" else find_spec(name),
            ), self.assertRaisesRegex(RuntimeError, "Install requirements.txt"):
                exec(self.code[0], {})
        finally:
            os.chdir(previous)

    def test_setup_explains_wrong_directory(self):
        import tempfile

        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with self.assertRaisesRegex(RuntimeError, "repository directory"):
                    exec(self.code[0], {})
            finally:
                os.chdir(previous)

    def test_readme_pairs_notebook_installation_with_explicit_kernel(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertLess(
            readme.index("## Start with the notebook"),
            readme.index("## Prefer the terminal?"),
        )
        self.assertIn("-m pip install -r requirements.txt", readme)
        self.assertIn(
            "-m ipykernel install --sys-prefix --name industrial-safety", readme
        )
        self.assertIn("-m notebook main.ipynb --ip=127.0.0.1", readme)
        self.assertIn("code/validate_notebook.py", readme)
        self.assertNotIn("-m json.tool main.ipynb", readme)
        requirements = (
            (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        )
        for package in ("notebook", "ipykernel", "nbclient", "nbformat"):
            self.assertIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
