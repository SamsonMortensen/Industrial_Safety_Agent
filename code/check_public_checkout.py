"""Check that the CPU walkthrough is using a separate, unchanged public checkout.

This uses Python's standard library and local Git metadata. It does not install
packages, fetch a remote, inspect private resources, or change the checkout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
from urllib.parse import urlsplit

# Keep the invoked checkout path: resolving a linked script could validate a
# different build while the caller goes on to install into the original folder.
ROOT = Path(os.path.abspath(__file__)).parents[1]
REPOSITORY = "SamsonMortensen/Industrial_Safety_Agent"
REQUIRED_FILES = (
    "README.md", "main.ipynb", "requirements.txt", "requirements-showcase.txt",
    "code/showcase.py", "code/check_public_checkout.py", "code/validate_notebook.py",
    "json/regulations.json", "json/statutes.json", "json/valid_sections.json",
    "data/example_camera_observation.json", "daily_yard_log.csv",
)
SOURCE_PATHS = ("code", "tests", "json", "data", "requirements*.txt", ".gitattributes", "daily_yard_log.csv")
PRIVATE_PREFIXES = (
    "data/external/", "data/vision_raw/", "data/vision_private/",
    "data/independent_evaluation/", "data/calibration_reserve/",
    "independent_evaluation/", "calibration_reserve/", "private/", "backups/",
    "models/", "vision_models/", "vision_runs/", "vision_manifests/",
    "training_runs/", "benchmark_runs/", "hf_cache/", ".venv/", ".venv-train/", ".venv-vision/",
)


class CheckoutError(RuntimeError):
    """A local checkout check could not be completed."""


def git(root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ["git", "--no-pager", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckoutError("Git could not inspect this folder. Check that Git is installed and select the public repository root.") from error
    if result.returncode:
        # Git errors may contain account paths or a credential-bearing remote.
        raise CheckoutError("Git could not verify this checkout. Select a complete public clone with a valid local commit.")
    return result.stdout


def is_link(path: Path) -> bool:
    metadata = path.lstat()
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def official_origin(value: str) -> bool:
    """Accept the ordinary GitHub clone URLs without accepting embedded secrets."""
    value = value.strip()
    ssh_user, separator, ssh_target = value.partition("@")
    if separator and ssh_user == "git" and ssh_target.startswith("github.com:"):
        path = ssh_target.removeprefix("github.com:")
    else:
        try:
            parsed = urlsplit(value)
            if (parsed.scheme != "https" or parsed.hostname != "github.com"
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port is not None or parsed.query or parsed.fragment):
                return False
            path = parsed.path.removeprefix("/")
        except ValueError:
            return False
    return path.removesuffix(".git").lower() == REPOSITORY.lower()


def python_modules(root: Path) -> set[str]:
    """Inspect only source directories, never environment or dataset trees."""
    modules = set()
    for name in ("code", "tests"):
        directory = root / name
        if not directory.is_dir() or is_link(directory):
            raise CheckoutError("The source and test folders must be ordinary directories in this checkout.")
        pending = [directory]
        while pending:
            folder = pending.pop()
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.name == "__pycache__":
                        continue
                    if is_link(path):
                        raise CheckoutError("Linked source or test entries are not part of the public walkthrough. Use a separate public clone.")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False) and path.suffix.lower() == ".py":
                        modules.add(path.relative_to(root).as_posix())
    return modules


def ordinary_path(root: Path, name: str) -> bool:
    """Check each component without resolving a link into another build."""
    path = root
    for component in Path(name).parts:
        path = path / component
        try:
            if is_link(path):
                return False
        except FileNotFoundError:
            return False
    return True


def source_or_input(name: str) -> bool:
    path = Path(name)
    return (path.parts[0] in {"code", "tests", "json", "data"}
            or name in {".gitattributes", "daily_yard_log.csv"}
            or (len(path.parts) == 1 and name.startswith("requirements") and name.endswith(".txt")))


def check(root: Path = ROOT) -> dict:
    report = {
        "schema": "public_checkout_check_v1", "ok": False, "errors": [],
        "checked_against": "local HEAD; remote freshness is not checked",
        "private_resource_contents_inspected": False,
    }
    try:
        requested = Path(os.path.abspath(root))
        if is_link(requested):
            raise CheckoutError("Select the public checkout itself, not a linked directory.")
        root = requested.resolve()
        top = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
        if root != top:
            raise CheckoutError("This is not the repository root. Select the folder containing README.md, code and tests.")
        origin = git(root, "config", "--get", "remote.origin.url").decode("utf-8", "strict").strip()
        if not official_origin(origin):
            raise CheckoutError("The origin is not the expected public GitHub repository or contains embedded credentials. Select a normal public clone; no remote URL was printed.")
        head = git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
        tracked = {os.fsdecode(name) for name in git(root, "ls-tree", "-r", "--name-only", "-z", "HEAD").split(b"\0") if name}
        indexed = {os.fsdecode(name) for name in git(root, "ls-files", "--cached", "-z").split(b"\0") if name}
        if any(name.lower().startswith(PRIVATE_PREFIXES) for name in tracked | indexed):
            raise CheckoutError("This commit or its staged changes include local resource or private evaluation paths. Preserve this build and choose a separate public clone. Those file contents were not inspected.")
        missing = [name for name in REQUIRED_FILES if name not in tracked]
        if missing:
            raise CheckoutError("Required public files are missing or not committed. Use a fresh public checkout before installing packages.")
        for name in tracked | indexed:
            if name in REQUIRED_FILES or source_or_input(name):
                if not ordinary_path(root, name) or not (root / name).is_file():
                    raise CheckoutError("A public source or input file is missing or linked to another build. Use a complete, separate public clone.")
        modules = python_modules(root)
        changes = set()
        for staged in (False, True):
            arguments = ["diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z"]
            if staged:
                arguments.append("--cached")
            changes.update(name for name in git(root, *arguments, "HEAD", "--", *SOURCE_PATHS).split(b"\0") if name)
        if changes:
            report["changed_source_or_input_count"] = len(changes)
            raise CheckoutError("Source, tests or public inputs differ from this checkout's commit. Preserve those development changes and use a separate public clone for the README checks.")
        extra = modules - tracked
        if extra:
            report["extra_python_module_count"] = len(extra)
            raise CheckoutError("Extra Python files were found in code or tests, including any ignored files. This is a mixed development checkout. Keep it and use a separate public clone.")
        report.update(ok=True, commit=head, python_module_count=len(modules))
    except CheckoutError as error:
        report["errors"].append(str(error))
    except (OSError, UnicodeError, ValueError):
        report["errors"].append("The public checkout could not be inspected safely. No file contents were printed or changed.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Public repository root to inspect")
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
