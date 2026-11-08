"""Verify scan coverage with real synthetic Git histories and no network."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
from publication_scan_range import scan_range


class PublicationHistoryRangeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.com')
        self.base = self.make_commit('baseline')
        self.git('update-ref', 'refs/remotes/origin/main', self.base)
        self.middle = self.make_commit('intermediate')
        self.head = self.make_commit('candidate')
        self.env = {'GITHUB_SHA': self.head, 'EVENT_NAME': 'push', 'BEFORE_SHA': self.base,
                    'DEFAULT_BRANCH': 'main', 'GITHUB_REF': 'refs/heads/improvement/fixture'}

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.PIPE).decode().strip()

    def make_commit(self, text):
        (self.repo / 'fixture.txt').write_text(text)
        self.git('add', 'fixture.txt')
        self.git('commit', '-q', '-m', text)
        return self.git('rev-parse', 'HEAD')

    def selected(self):
        return scan_range(self.repo, self.env)

    def test_existing_push_covers_every_incoming_commit(self):
        self.assertEqual(self.selected(), self.base + '..' + self.head)
        self.assertEqual(self.git('rev-list', self.selected()).splitlines(), [self.head, self.middle])

    def test_missing_previous_commit_after_rewrite_scans_all_new_history(self):
        self.env['BEFORE_SHA'] = 'f' * 40
        self.assertEqual(self.selected(), self.head)
        self.assertEqual(set(self.git('rev-list', self.selected()).splitlines()), {self.base, self.middle, self.head})

    def test_new_branch_covers_intermediate_commits(self):
        self.env['BEFORE_SHA'] = '0' * 40
        self.assertEqual(self.selected(), self.base + '..' + self.head)
        self.assertIn(self.middle, self.git('rev-list', self.selected()).splitlines())

    def test_missing_before_uses_new_branch_selection(self):
        del self.env['BEFORE_SHA']
        self.assertEqual(self.selected(), self.base + '..' + self.head)

    def test_initial_default_branch_scans_full_history(self):
        self.env.update(BEFORE_SHA='0' * 40, GITHUB_REF='refs/heads/main')
        self.assertEqual(self.selected(), self.head)

    def test_missing_default_branch_falls_back_to_full_history(self):
        self.env.update(BEFORE_SHA='0' * 40, DEFAULT_BRANCH='absent')
        self.assertEqual(self.selected(), self.head)
        self.env['DEFAULT_BRANCH'] = ''
        self.assertEqual(self.selected(), self.head)

    def test_pull_request_uses_exact_base_and_head(self):
        self.env.update(EVENT_NAME='pull_request', PR_BASE_SHA=self.base, PR_HEAD_SHA=self.head)
        self.assertEqual(self.selected(), self.base + '..' + self.head)

    def test_unavailable_pull_request_base_fails_closed(self):
        self.env.update(EVENT_NAME='pull_request', PR_BASE_SHA='f' * 40, PR_HEAD_SHA=self.head)
        with self.assertRaises(ValueError):
            self.selected()

    def test_invalid_commit_or_event_fails_closed(self):
        for field, value in (('BEFORE_SHA', '--all'), ('GITHUB_SHA', 'short'), ('EVENT_NAME', 'workflow_dispatch')):
            with self.subTest(field=field):
                altered = dict(self.env, **{field: value})
                with self.assertRaises(ValueError):
                    scan_range(self.repo, altered)

    def test_invalid_default_ref_fails_closed(self):
        self.env.update(BEFORE_SHA='0' * 40, DEFAULT_BRANCH='../bad ref')
        with self.assertRaises(ValueError):
            self.selected()


if __name__ == '__main__':
    unittest.main()
