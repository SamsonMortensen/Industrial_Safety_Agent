"""Exercise real kernel cleanup and Python process exit, including cell errors."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
from pathlib import Path
import sys
import psutil
sys.path.insert(0, sys.argv[1])
from validate_notebook import validate_notebook
try:
    count = validate_notebook(Path(sys.argv[2]))
    assert count == 1, count
finally:
    children = psutil.Process().children(recursive=True)
    assert not children, f'Kernel processes survived validation: {children}'
    print('Kernel cleanup completed', flush=True)
"""


class NotebookShutdownTests(unittest.TestCase):
    def run_notebook(self, source):
        import nbformat

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_code_cell(source)]
            )
            original = nbformat.writes(notebook).encode('utf-8')
            path = root / 'main.ipynb'
            path.write_bytes(original)
            env = os.environ.copy()
            for key, name in {
                'JUPYTER_RUNTIME_DIR': 'runtime',
                'JUPYTER_CONFIG_DIR': 'config',
                'IPYTHONDIR': 'ipython',
                'TEMP': 'temp',
                'TMP': 'temp',
            }.items():
                folder = root / name
                folder.mkdir(exist_ok=True)
                env[key] = str(folder)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            result = subprocess.run(
                [sys.executable, '-B', '-X', 'faulthandler', '-c', CHILD,
                 str(ROOT / 'code'), str(root)],
                cwd=root, env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list((root / 'runtime').glob('kernel-*.json')), [])
            self.assertIn('Kernel cleanup completed', result.stdout, result.stderr)
            return result

    def test_success_exits_cleanly_and_reaps_kernel(self):
        result = self.run_notebook('assert 2 + 2 == 4')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cell_error_remains_failure_and_reaps_kernel(self):
        result = self.run_notebook("raise ValueError('intentional shutdown regression')")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('CellExecutionError', result.stderr)
        self.assertIn('intentional shutdown regression', result.stderr)

    def test_network_guard_remains_failure_and_reaps_kernel(self):
        result = self.run_notebook("import requests; requests.get('https://example.invalid')")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('The beginner notebook attempted a network request', result.stderr)


if __name__ == '__main__':
    unittest.main()
