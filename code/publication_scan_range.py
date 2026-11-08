"""Choose a credential scan range, including pushes after a history rewrite."""
import os
from pathlib import Path
import re
import subprocess


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)


def commit(repo, value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{40}', value) is None:
        raise ValueError('Expected a complete commit identifier')
    if git(repo, 'cat-file', '-e', value + '^{commit}').returncode:
        raise ValueError('Required commit is unavailable')
    return value


def scan_range(repo, environment):
    head = commit(repo, environment.get('GITHUB_SHA'))
    event = environment.get('EVENT_NAME')
    if event == 'pull_request':
        base = commit(repo, environment.get('PR_BASE_SHA'))
        tip = commit(repo, environment.get('PR_HEAD_SHA'))
        return base + '..' + tip
    if event != 'push':
        raise ValueError('Unsupported publication event')
    before = environment.get('BEFORE_SHA', '')
    if before and re.fullmatch(r'[0-9a-f]{40}', before) is None:
        raise ValueError('Invalid previous commit identifier')
    if before and before != '0' * 40:
        # A coordinated rewrite can remove the previous commit from a fresh
        # checkout. Scan all new history instead of failing or skipping it.
        if git(repo, 'cat-file', '-e', before + '^{commit}').returncode:
            return head
        return before + '..' + head
    default = environment.get('DEFAULT_BRANCH', '')
    if not default:
        return head
    default_ref = 'refs/remotes/origin/' + default
    if git(repo, 'check-ref-format', default_ref).returncode:
        raise ValueError('Invalid default branch reference')
    if environment.get('GITHUB_REF') == 'refs/heads/' + default:
        return head
    base = git(repo, 'merge-base', head, default_ref)
    if base.returncode:
        return head
    return commit(repo, base.stdout.strip()) + '..' + head


def main():
    try:
        selected = scan_range(Path(__file__).resolve().parent.parent, os.environ)
    except (ValueError, OSError):
        raise SystemExit('Cannot determine the publication history scan range') from None
    print(selected)


if __name__ == '__main__':
    main()
