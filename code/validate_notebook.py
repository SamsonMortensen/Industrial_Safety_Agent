"""Run the beginner notebook in a fresh kernel without saving its outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# This runs after the kernel starts. Local kernel messaging remains available.
OFFLINE_GUARD = """
import socket
from unittest.mock import patch
import requests

def _blocked_request(*args, **kwargs):
    raise AssertionError("The beginner notebook attempted a network request")

_socket_guard = patch.object(socket.socket, "connect", side_effect=_blocked_request)
_http_guard = patch.object(requests.sessions.Session, "request", side_effect=_blocked_request)
_socket_guard.start()
_http_guard.start()
"""


def validate_notebook(root: Path = ROOT) -> int:
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    path = root / "main.ipynb"
    original = path.read_bytes()
    notebook = nbformat.reads(original.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(OFFLINE_GUARD))

    # Use this interpreter even if another Python kernel is registered globally.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    manager.kernel_spec.env = {}
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=60,
        startup_timeout=60,
        allow_errors=False,
        resources={"metadata": {"path": str(root)}},
    )
    # We supplied the manager, so explicitly own its cleanup on success or error.
    with client.setup_kernel(cleanup_kc=True):
        client.execute()
    if path.read_bytes() != original:
        raise RuntimeError("The source notebook changed during validation.")
    return code_cells


def main() -> int:
    try:
        count = validate_notebook()
    except ModuleNotFoundError as error:
        print(
            f"Missing dependency: {error.name}. Install requirements.txt "
            "with this Python interpreter.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Notebook verified: {count} code cells ran in a fresh offline-checked kernel."
    )
    print("The source notebook was not changed. No executed copy was saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
