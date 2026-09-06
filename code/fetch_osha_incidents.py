"""Download the official OSHA Severe Injury Report archive to ignored storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_URL = "https://www.osha.gov/sites/default/files/January2015toNovember2025.zip"
DEFAULT_DIRECTORY = ROOT / "data" / "external" / "osha_sir"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_csv_member(names: list[str]) -> str:
    csv_names = []
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {name}")
        if path.suffix.lower() == ".csv":
            csv_names.append(name)
    if len(csv_names) != 1:
        raise ValueError(
            f"Expected exactly one CSV in OSHA archive, found {len(csv_names)}"
        )
    return csv_names[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / Path(args.url).name
    if args.force or not archive_path.exists():
        temporary = archive_path.with_suffix(archive_path.suffix + ".partial")
        with requests.get(args.url, stream=True, timeout=120) as response:
            if response.status_code == 403:
                parser.exit(
                    2,
                    "OSHA refused the automated download (HTTP 403). Download the archive manually from https://www.osha.gov/severe-injury-reports, "
                    f"save the matching archive as {archive_path}, and rerun without --force. "
                    "If using a different release, supply its official --url and pass the extracted CSV to osha_incident_pipeline.py --input.\n",
                )
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(archive_path)

    archive_hash = sha256(archive_path)
    if args.expected_sha256 and archive_hash.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {args.expected_sha256}, got {archive_hash}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        member = safe_csv_member(archive.namelist())
        csv_path = directory / Path(member).name
        with archive.open(member) as source, csv_path.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)

    manifest = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "official_url": args.url,
        "archive": archive_path.name,
        "archive_sha256": archive_hash,
        "csv": csv_path.name,
        "csv_sha256": sha256(csv_path),
    }
    manifest_path = directory / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
