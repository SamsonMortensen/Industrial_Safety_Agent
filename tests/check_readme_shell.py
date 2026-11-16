"""Check the README's POSIX setup flow with mocked Git and Python commands.

Run on Linux/macOS: python tests/check_readme_shell.py
This does not install packages or execute the project. CI tests those separately.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_shell(bash, script, folder, destination="", *, syntax_only=False):
    command = [bash, "--noprofile", "--norc"]
    if syntax_only:
        command.append("-n")
    command.append(str(script))
    # Every native command in the recipe is mocked. An empty PATH also prevents
    # a newly added, unmocked program from running during this check.
    return subprocess.run(
        command,
        input=destination + "\n",
        capture_output=True,
        text=True,
        timeout=20,
        cwd=folder,
        env={"PATH": str(folder / "empty_path"), "HOME": str(folder), "LC_ALL": "C"},
    )


def mocked_recipe(recipe, failure_stage):
    # Preserve the actual control flow, including the filesystem checks. Only
    # replace project-Python commands, then supply shell functions for programs.
    replaced = re.sub(r"(?m)^(\s*)\.venv/bin/python(?=\s)", r"\1mock_python", recipe)
    require(replaced != recipe, "README recipe has no project-Python commands to mock")
    return (
        "git() { printf 'MOCK git %s\\n' \"$*\"; "
        f"if [ '{failure_stage}' = clone ] && [ \"$1\" = clone ]; then return 9; fi; return 0; }}\n"
        "python3() { printf 'MOCK python3 %s\\n' \"$*\"; return 0; }\n"
        "mock_python() { printf 'MOCK project-python %s\\n' \"$*\"; "
        f"if [ '{failure_stage}' = install ] && [ \"$*\" = '-m pip install -r requirements.txt' ]; then return 9; fi; return 0; }}\n"
        + replaced
    )


def main():
    require(os.name == "posix", "Run this shell check on Linux or macOS")
    bash = shutil.which("bash")
    require(bash is not None, "bash is required to check the README shell recipe")
    readme = ROOT / "README.md"
    blocks = re.findall(r"(?ms)^```sh\r?\n(.*?)^```", readme.read_text(encoding="utf-8"))
    require(len(blocks) == 1, "Expected exactly one POSIX README setup block")
    recipe = blocks[0]
    checks = []
    with tempfile.TemporaryDirectory(prefix="readme-shell-check-") as directory:
        folder = Path(directory)
        (folder / "empty_path").mkdir()
        script = folder / "quickstart.sh"
        script.write_text(recipe, encoding="utf-8", newline="\n")
        parsed = run_shell(bash, script, folder, syntax_only=True)
        require(parsed.returncode == 0, "README shell syntax failed: " + parsed.stderr)
        checks.append("POSIX recipe parses with bash -n")
        fixture = folder / "existing checkout with spaces"
        (fixture / ".git").mkdir(parents=True)
        (fixture / "code").mkdir()
        (fixture / "requirements.txt").touch()
        (fixture / "code/showcase.py").touch()
        for name, stage, destination, expected_code in (
            ("failed_clone", "clone", str(folder / "absent checkout"), 9),
            ("failed_dependencies", "install", str(fixture), 9),
            ("success", "none", str(fixture), 0),
            ("relative_path", "none", "relative/path", 1),
            ("non_project_directory", "none", str(fixture / "code"), 1),
        ):
            script.write_text(mocked_recipe(recipe, stage), encoding="utf-8", newline="\n")
            # Execute the block directly, not inside an if/|| wrapper: such a
            # wrapper can suppress errexit and conceal the behavior being tested.
            result = run_shell(bash, script, folder, destination)
            require(
                result.returncode == expected_code,
                f"{name}: expected exit {expected_code}, got {result.returncode}; "
                + result.stdout + result.stderr,
            )
            if name in {"failed_clone", "relative_path", "non_project_directory"}:
                require("MOCK project-python" not in result.stdout, name + ": Python ran after a rejected checkout")
                require("MOCK python3 -m venv" not in result.stdout, name + ": environment creation ran")
            if name == "failed_clone":
                require("MOCK git clone" in result.stdout, "Clone failure path did not reach the mock")
                require(not (folder / "absent checkout").exists(), "Clone mock created a checkout")
            if name == "failed_dependencies":
                require("MOCK project-python -m pip install" in result.stdout, "Install failure path did not reach the mock")
                require("MOCK project-python -m pip check" not in result.stdout, "Dependency checks ran after failed installation")
                require("MOCK project-python -m notebook" not in result.stdout, "Notebook ran after failed installation")
            if name == "success":
                require("MOCK git clone" not in result.stdout, "Existing checkout was cloned again")
                for expected in (
                    "MOCK project-python -m pip check",
                    "MOCK project-python code/check_setup.py",
                    "MOCK project-python -m ipykernel install",
                    "MOCK project-python -m notebook",
                ):
                    require(expected in result.stdout, "Successful flow omitted " + expected)
            require(not (fixture / ".venv").exists(), "A real environment was created")
            checks.append(name)
    print(json.dumps({
        "schema": "readme_shell_mock_checks_v1",
        "status": "passed",
        "readme_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "checks_passed": len(checks),
        "checks": checks,
        "project_runtime_test": False,
        "real_clone_or_install_performed": False,
        "model_or_training_performed": False,
        "evaluation_data_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
