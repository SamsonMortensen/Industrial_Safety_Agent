"""Block common credentials and private artifacts before publication.

This local, standard-library gate complements a full secret scanner and human
review. It cannot identify every person's name, private incident, encoded secret,
or sensitive image. Findings contain locations and categories, never match text.
Staged mode reads Git's index blobs, including force-added ignored files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zlib

MAX_FILE_BYTES = 8 * 1024 * 1024
PRIVATE_ROOTS = (
    'models/', 'hf_cache/', 'training_runs/', 'benchmark_runs/', 'outputs/',
    'private/', 'backups/', '.config/gcloud/',
    'vision_manifests/', 'vision_models/', 'vision_runs/',
    'data/external/', 'data/vision_raw/', 'data/vision_private/',
    'independent_evaluation/', 'calibration_reserve/',
)
PRIVATE_COMPONENTS = {'.aws', '.ssh', '.azure', '.gcloud', '.venv', '.venv-vision', '.venv-train'}
SECRET_EXTENSIONS = {'.pem', '.key', '.p12', '.pfx', '.keystore', '.jks', '.bundle', '.patch', '.zip'}
PUBLIC_REGULATIONS = {'json/regulations.json', 'json/statutes.json', 'json/valid_sections.json'}
GENERATED_CI_ARTIFACTS = (
    'benchmark_runs/ci_environment.json', 'benchmark_runs/showcase/ci_showcase.json',
    'benchmark_runs/retrieval_ci.json',
)
CI_ARTIFACTS = GENERATED_CI_ARTIFACTS + (
    'docs/RETRIEVAL_IMPROVEMENT.md', 'docs/RESULTS.md',
    'docs/results/audit_tradeoffs.png', 'docs/results/calibration_diagnostics.png',
    'docs/results/person_development.png', 'docs/results/original_nine_misses.png',
    'docs/results/retrieval_before_after.png', 'docs/results/retrieval_rank_cutoffs.png',
    'docs/results/group_split_before_after.png', 'docs/results/manifest.json',
    'docs/results/metrics.json', 'docs/results/retrieval_comparison.json',
    'docs/results/retrieval_manifest.json', 'docs/results/group_split_before_after.json',
    'docs/results/retrieval_cases.csv', 'data/retrieval_comparison/after.json',
    'data/retrieval_comparison/before.json', 'data/retrieval_comparison/candidate_freeze.json',
    'data/retrieval_comparison/exploration.json', 'data/retrieval_comparison/protocol.json',
)
GENERIC_USERS = {'user', 'username', 'yourname', 'your-user', 'example', '$user', '$username', '%username%'}
TOKEN_PATTERNS = (
    re.compile(r'\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,}'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'\bhf_[A-Za-z0-9]{20,}'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'\bAIza[0-9A-Za-z_-]{30,}'),
    re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{16,}'),
    re.compile(r'\b(?:sk|rk)_live_[A-Za-z0-9]{16,}'),
    re.compile(r'\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}'),
)
PRIVATE_KEY = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----')
ASSIGNED_SECRET = re.compile(
    r'''(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|secret(?:_key)?)\b["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']''')
EMAIL = re.compile(r"(?i)(?<![\w.+-])[A-Z0-9][A-Z0-9.!#$%&'*+/=?^_`{|}~-]*@([A-Z0-9-]+(?:\.[A-Z0-9-]+)+)")
HOME_PATHS = (
    re.compile(r'''(?i)\b[A-Z]:[\\/]+Users[\\/]+([^\\/\s"'<>,;:)\]}]+)'''),
    re.compile(r'''/(?:home|Users)/([^/\s"'<>,;:)\]}]+)'''),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int | None
    category: str


class ScanError(RuntimeError):
    """A fail-closed error whose message never contains Git output or file data."""


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
    if result.returncode:
        raise ScanError('Git could not provide the publication selection')
    return result.stdout


def private_path(name: str) -> bool:
    lower = name.lower()
    path = PurePosixPath(lower)
    filename = path.name
    if lower.startswith(PRIVATE_ROOTS) or any(p in PRIVATE_COMPONENTS for p in path.parts):
        return True
    if filename.startswith('.env') and filename != '.env.example':
        return True
    if filename in {'env.txt', 'secrets.toml', 'credentials', 'credentials.json', 'credentials.yaml',
                    'credentials.yml', 'token.json', 'tokens.json', 'id_rsa', 'id_ed25519',
                    'id_ecdsa', 'id_dsa', '.netrc', '.npmrc', '.pypirc'}:
        return True
    if re.fullmatch(r'env[^/]*\.json', filename) or path.suffix in SECRET_EXTENSIONS:
        return True
    if re.fullmatch(r'(?:credentials|secrets|tokens?|client_secret)[^/]*\.(?:json|ya?ml|toml|ini|txt|csv)', filename):
        return True
    if path.suffix in {'.json', '.jsonl', '.csv', '.tsv', '.xlsx', '.zip'}:
        if re.search(r'(?:^|_)(?:label_reviews|reviewed_labels|review_receipts|review_store|label_validation|labels|approvals|approval_ledger)(?:\.|_)', filename):
            return True
    return False


def placeholder(value: str) -> bool:
    lower = value.lower().strip()
    return (lower in {'redacted', 'changeme', 'placeholder', 'example', 'not-a-secret'}
            or lower.startswith(('your_', 'your-', '${', '<', 'example_', 'placeholder_')))


def text_findings(name: str, text: str, line_numbers: bool = True) -> set[Finding]:
    found = set()
    for number, line in enumerate(text.splitlines(), 1):
        where = number if line_numbers else None
        if PRIVATE_KEY.search(line):
            found.add(Finding(name, where, 'private-key'))
        if any(pattern.search(line) for pattern in TOKEN_PATTERNS):
            found.add(Finding(name, where, 'credential-token'))
        if any(not placeholder(match.group(1)) for match in ASSIGNED_SECRET.finditer(line)):
            found.add(Finding(name, where, 'literal-credential'))
        for pattern in HOME_PATHS:
            if any(m.group(1).lower() not in GENERIC_USERS for m in pattern.finditer(line)):
                found.add(Finding(name, where, 'personal-home-path'))
        for match in EMAIL.finditer(line):
            domain = match.group(1).lower()
            public = domain in {'users.noreply.github.com', 'noreply.github.com', 'example.com', 'example.org', 'example.net'}
            public = public or match.group(0).lower() == 'noreply@github.com'
            public = public or (name in PUBLIC_REGULATIONS and domain.endswith('.gov'))
            if not public:
                found.add(Finding(name, where, 'personal-contact'))
    return found


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)


def png_findings(name: str, content: bytes) -> set[Finding]:
    found = set()
    if not name.startswith('docs/results/') or not content.startswith(b'\x89PNG\r\n\x1a\n'):
        return {Finding(name, None, 'unreviewed-binary')}
    offset, seen_header, ended = 8, False, False
    while offset < len(content):
        if offset + 12 > len(content):
            return {Finding(name, None, 'invalid-image')}
        length = int.from_bytes(content[offset:offset+4], 'big')
        kind = content[offset+4:offset+8]
        end = offset + 12 + length
        if end > len(content):
            return {Finding(name, None, 'invalid-image')}
        payload = content[offset+8:end-4]
        expected_crc = int.from_bytes(content[end-4:end], 'big')
        if zlib.crc32(kind + payload) != expected_crc:
            return {Finding(name, None, 'invalid-image')}
        if not seen_header:
            if kind != b'IHDR' or length != 13:
                return {Finding(name, None, 'invalid-image')}
            seen_header = True
        if kind == b'tEXt':
            if b'\0' not in payload or payload.split(b'\0', 1)[0] != b'Software':
                found.add(Finding(name, None, 'image-text-metadata'))
            found.update(text_findings(name, payload.decode('latin-1'), False))
        elif kind in {b'zTXt', b'iTXt', b'eXIf'}:
            found.add(Finding(name, None, 'image-text-metadata'))
        offset = end
        if kind == b'IEND':
            ended = length == 0 and offset == len(content)
            break
    if not ended:
        found.add(Finding(name, None, 'invalid-image'))
    return found


def scan_bytes(name: str, content: bytes, *, ci_artifact: bool = False) -> set[Finding]:
    if private_path(name) and not (ci_artifact and name in GENERATED_CI_ARTIFACTS):
        return {Finding(name, None, 'private-artifact-path')}
    if len(content) > MAX_FILE_BYTES:
        return {Finding(name, None, 'file-requires-manual-review')}
    if name.lower().endswith('.png'):
        return png_findings(name, content)
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        return {Finding(name, None, 'unreviewed-binary')}
    if '\0' in text:
        return {Finding(name, None, 'unreviewed-binary')}
    found = text_findings(name, text)
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in {'.json', '.ipynb', '.jsonl'}:
        try:
            data = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if suffix == '.jsonl' else json.loads(text))
        except (ValueError, RecursionError):
            found.add(Finding(name, None, 'invalid-structured-file'))
            return found
        for decoded in strings(data):
            # Decode JSON escapes so a secret/path cannot hide behind \u escapes.
            for finding in text_findings(name, decoded, False):
                if not any(f.category == finding.category for f in found):
                    found.add(finding)
        if suffix == '.ipynb':
            if not isinstance(data, dict) or not isinstance(data.get('cells'), list):
                found.add(Finding(name, None, 'invalid-notebook'))
            else:
                for cell in data['cells']:
                    if not isinstance(cell, dict):
                        found.add(Finding(name, None, 'invalid-notebook'))
                        continue
                    if cell.get('attachments'):
                        found.add(Finding(name, None, 'notebook-attachment'))
                    if cell.get('cell_type') == 'code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                        found.add(Finding(name, None, 'notebook-executed-output'))
                if isinstance(data.get('metadata'), dict) and data['metadata'].get('widgets'):
                    found.add(Finding(name, None, 'notebook-widget-state'))
    return found


def linked_entry(path: Path, root: Path) -> bool:
    """Do not follow symlinks or Windows directory junctions in selected paths."""
    if not path.resolve().is_relative_to(root):
        return True
    for part in (path, *path.parents):
        if part == root:
            break
        metadata = part.lstat()
        if part.is_symlink() or getattr(metadata, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
            return True
    return False


def filename_findings(name: str) -> set[Finding]:
    found = text_findings(name, name, False)
    # Filename prefixes such as backup_ must not hide a complete token.
    if any(re.search(pattern.pattern.removeprefix(r'\b'), name) for pattern in TOKEN_PATTERNS):
        found.add(Finding(name, None, 'credential-token'))
    # A hidden-file dot is a filename prefix, not part of an email boundary.
    for component in PurePosixPath(name).parts:
        if component.startswith('.'):
            found.update(text_findings(name, component.lstrip('.'), False))
    return found


def result(mode: str, selected: list | tuple, findings: set[Finding]) -> dict:
    redacted = []
    for finding in sorted(findings, key=lambda f: (f.path, f.line or 0, f.category)):
        row = asdict(finding)
        if filename_findings(finding.path):
            identifier = hashlib.sha256(finding.path.encode('utf-8')).hexdigest()[:12]
            row['path'] = '[redacted filename:' + identifier + ']'
        redacted.append(row)
    return {'status': 'blocked' if findings else 'passed', 'mode': mode,
            'files_checked': len(selected), 'findings': redacted}


def scan_ci_artifacts(repo: Path) -> dict:
    """Scan the fixed workflow allowlist; unknown siblings are never selected."""
    root = Path(git(repo, 'rev-parse', '--show-toplevel').decode('utf-8').strip()).resolve()
    findings = set()
    for name in CI_ARTIFACTS:
        name_findings = filename_findings(name)
        if name_findings:
            findings.update(name_findings)
            continue
        path = root / name
        try:
            if linked_entry(path, root):
                findings.add(Finding(name, None, 'unsupported-git-entry'))
                continue
            with path.open('rb') as source:
                content = source.read(MAX_FILE_BYTES + 1)
        except OSError:
            findings.add(Finding(name, None, 'unreadable-file'))
            continue
        findings.update(scan_bytes(name, content, ci_artifact=True))
    return result('ci-artifacts', CI_ARTIFACTS, findings)


def scan_repository(repo: Path, staged: bool) -> dict:
    root = Path(git(repo, 'rev-parse', '--show-toplevel').decode('utf-8').strip()).resolve()
    index = {}
    for entry in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        header, path = entry.split(b'\t', 1)
        mode, oid, stage = header.decode('ascii').split()
        name = path.decode('utf-8')
        if stage != '0' or name in index:
            raise ScanError('Resolve index conflicts before publication')
        index[name] = (mode, oid)
    selected = ([p.decode('utf-8') for p in git(root, 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').split(b'\0') if p]
                if staged else sorted(index))
    findings = set()
    for name in selected:
        name_findings = filename_findings(name)
        if name_findings:
            findings.update(name_findings)
            continue
        if name not in index:
            raise ScanError('Publication selection changed during the scan')
        mode, oid = index[name]
        if mode not in {'100644', '100755'}:
            findings.add(Finding(name, None, 'unsupported-git-entry'))
            continue
        if private_path(name):
            findings.add(Finding(name, None, 'private-artifact-path'))
            continue
        try:
            if staged:
                if int(git(root, 'cat-file', '-s', oid)) > MAX_FILE_BYTES:
                    findings.add(Finding(name, None, 'file-requires-manual-review'))
                    continue
                content = git(root, 'cat-file', 'blob', oid)
            else:
                path = root / name
                if linked_entry(path, root):
                    findings.add(Finding(name, None, 'unsupported-git-entry'))
                    continue
                with path.open('rb') as source:
                    content = source.read(MAX_FILE_BYTES + 1)
        except OSError:
            findings.add(Finding(name, None, 'unreadable-file'))
            continue
        findings.update(scan_bytes(name, content))
    return result('staged' if staged else 'tracked', selected, findings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--staged', action='store_true')
    modes.add_argument('--tracked', action='store_true')
    modes.add_argument('--ci-artifacts', action='store_true')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        report = scan_ci_artifacts(args.repo) if args.ci_artifacts else scan_repository(args.repo, args.staged)
    except (ScanError, UnicodeError, ValueError, RecursionError, OSError):
        print(json.dumps({'status': 'error', 'message': 'Unable to complete the publication privacy scan; no file contents were printed.'}))
        return 2
    print(json.dumps(report, indent=2))
    return 1 if report['findings'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
