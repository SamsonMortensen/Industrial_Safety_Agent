"""Exercise publication boundaries with generated dummy data and temporary Git repos."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import check_publication_privacy as guard


def dummy_token():
    return 's' + 'k-' + 'proj-' + 'a1B2c3D4' * 8


class PublicationPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.git('init', '--quiet')

    def git(self, *args, content=None):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], input=content,
                                       stderr=subprocess.PIPE)

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def stage(self, name, content):
        path = self.write(name, content)
        self.git('add', '--', name)
        return path

    def categories(self, result):
        return {f['category'] for f in result['findings']}

    def test_staged_blob_remains_blocked_after_working_copy_is_cleaned(self):
        path = self.stage('config.py', 'value = "' + dummy_token() + '"\n')
        path.write_text('value = None\n')
        self.assertIn('credential-token', self.categories(guard.scan_repository(self.repo, True)))
        self.assertEqual(guard.scan_repository(self.repo, False)['status'], 'passed')

    def test_force_added_ignored_secret_file_is_blocked(self):
        self.stage('.gitignore', '.env*\n')
        self.write('.env.production', 'SAFE_PLACEHOLDER=example\n')
        self.git('add', '--force', '--', '.env.production')
        self.assertIn('private-artifact-path', self.categories(guard.scan_repository(self.repo, True)))

    def test_cli_never_prints_credential_or_source_line(self):
        secret = dummy_token()
        source = 'DO_NOT_DISCLOSE_THIS_LINE = "' + secret + '"\n'
        self.stage('settings.py', source)
        result = subprocess.run([sys.executable, '-B', str(ROOT/'code/check_publication_privacy.py'),
                                 '--staged', '--repo', str(self.repo)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn(secret, result.stdout + result.stderr)
        self.assertNotIn('DO_NOT_DISCLOSE_THIS_LINE', result.stdout + result.stderr)
        self.assertEqual(set(json.loads(result.stdout)['findings'][0]), {'path', 'line', 'category'})
        self.assertEqual(json.loads(result.stdout)['findings'][0]['path'], 'settings.py')

    def test_token_in_filename_is_blocked_without_echoing_the_filename(self):
        secret = dummy_token()
        filename = 'backup_' + secret + '.txt'
        self.stage(filename, 'ordinary harmless content\n')
        result = subprocess.run([sys.executable, '-B', str(ROOT/'code/check_publication_privacy.py'),
                                 '--staged', '--repo', str(self.repo)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn(secret, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertIn('credential-token', self.categories(report))
        self.assertTrue(all(f['path'].startswith('[redacted filename:') for f in report['findings']))

    def test_private_filename_contact_is_redacted_for_every_finding_category(self):
        contact = 'fixture.person' + '@' + 'private-mail.invalid'
        filename = '.env.' + contact
        self.stage(filename, 'ordinary harmless content\n')
        result = subprocess.run([sys.executable, '-B', str(ROOT/'code/check_publication_privacy.py'),
                                 '--staged', '--repo', str(self.repo)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn(contact, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertIn('personal-contact', self.categories(report))
        self.assertTrue(all(f['path'].startswith('[redacted filename:') for f in report['findings']))
        # Redaction also applies to path-only findings produced by any caller.
        direct = guard.result('staged', [filename], {guard.Finding(filename, None, 'private-artifact-path')})
        self.assertNotIn(contact, json.dumps(direct))
        self.assertTrue(direct['findings'][0]['path'].startswith('[redacted filename:'))

    def test_sensitive_paths_are_blocked_even_without_secret_text(self):
        paths = ('nested/env_clean.json', '.streamlit/secrets.toml', 'nested/client.pem',
                 'credentials.json', 'models/weights.bin', 'vision_runs/run.json',
                 'data/vision_private/review.csv', 'development_label_reviews.jsonl',
                 'label_validation.json', 'reviewed_labels.csv', 'private_approval_ledger.json',
                 'private/report.txt', 'backups/report.txt', 'snapshot.bundle', 'local.patch',
                 'archive.zip', 'secrets.json', 'secrets.yaml', 'secrets.yml', '.config/gcloud/config')
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual({f.category for f in guard.scan_bytes(path, b'{}')}, {'private-artifact-path'})

    def test_env_example_is_allowed_but_real_credential_is_not(self):
        self.assertFalse(guard.scan_bytes('.env.example', b'API_KEY="your_api_key_here"\n'))
        self.assertIn('credential-token', {f.category for f in guard.scan_bytes('.env.example', dummy_token().encode())})

    def test_personal_home_paths_are_blocked_and_generic_examples_pass(self):
        windows = 'C:' + '\\Users\\' + 'private_fixture_person' + '\\data.csv'
        unix = '/home/' + 'private_fixture_person' + '/data.csv'
        mac = '/Users/' + 'private_fixture_person' + '/data.csv'
        for text in (windows, unix, mac):
            with self.subTest(style=text[:3]):
                self.assertIn('personal-home-path', {f.category for f in guard.scan_bytes('notes.txt', text.encode())})
        self.assertFalse(guard.scan_bytes('README.md', b'/home/user/project and /Users/username/project'))

    def test_json_escaped_tokens_and_paths_are_detected(self):
        raw = json.dumps({'value': dummy_token()}).replace('sk-', '\\u0073k-')
        self.assertIn('credential-token', {f.category for f in guard.scan_bytes('snapshot.json', raw.encode())})
        path = 'C:' + '\\Users\\' + 'private_fixture_person' + '\\data'
        self.assertIn('personal-home-path', {f.category for f in guard.scan_bytes('snapshot.json', json.dumps({'path':path}).encode())})

    def test_private_key_and_literal_password_are_blocked(self):
        private = '-----BEGIN ' + 'PRIVATE KEY-----\nfixture\n'
        assigned = 'pass' + 'word = "fixture-value-long-enough"\n'
        self.assertIn('private-key', {f.category for f in guard.scan_bytes('settings.txt', private.encode())})
        self.assertIn('literal-credential', {f.category for f in guard.scan_bytes('settings.py', assigned.encode())})

    def test_contacts_allow_only_noreply_examples_and_official_regulation_source(self):
        public = '150225358+' + 'fixture' + '@users.noreply.github.com'
        official = 'office' + '@agency.gov'
        private = 'fixture.person' + '@private-mail.invalid'
        self.assertFalse(guard.scan_bytes('README.md', public.encode()))
        self.assertFalse(guard.scan_bytes('README.md', b'person@example.com'))
        self.assertFalse(guard.scan_bytes('json/regulations.json', json.dumps({'text':official}).encode()))
        self.assertIn('personal-contact', {f.category for f in guard.scan_bytes('notes.txt', official.encode())})
        self.assertIn('personal-contact', {f.category for f in guard.scan_bytes('notes.txt', private.encode())})

    def test_notebook_must_have_no_outputs_execution_counts_or_attachments(self):
        clean = {'cells':[{'cell_type':'code','source':['print(1)'],'execution_count':None,'outputs':[]}], 'metadata':{}}
        self.assertFalse(guard.scan_bytes('main.ipynb', json.dumps(clean).encode()))
        for field, value, category in [('outputs',[{'output_type':'stream','text':'fixture'}],'notebook-executed-output'),
                                       ('execution_count',1,'notebook-executed-output'),
                                       ('attachments',{'fixture.png':{'image/png':'fixture'}},'notebook-attachment')]:
            altered=json.loads(json.dumps(clean));altered['cells'][0][field]=value
            with self.subTest(field=field):
                self.assertIn(category,{f.category for f in guard.scan_bytes('main.ipynb',json.dumps(altered).encode())})

    def test_invalid_notebook_and_unreadable_tracked_file_fail_closed(self):
        self.assertTrue(guard.scan_bytes('main.ipynb', b'not JSON'))
        path=self.stage('README.md','ordinary text\n');path.unlink()
        self.assertIn('unreadable-file', self.categories(guard.scan_repository(self.repo,False)))

    def test_staged_symlink_entry_is_rejected_without_following_it(self):
        oid=self.git('hash-object','-w','--stdin',content=b'outside-secret.txt').decode().strip()
        self.git('update-index','--add','--cacheinfo','120000,'+oid+',link')
        self.assertIn('unsupported-git-entry',self.categories(guard.scan_repository(self.repo,True)))

    def test_binary_files_require_explicit_image_location(self):
        self.assertIn('unreviewed-binary',{f.category for f in guard.scan_bytes('capture.bin',b'\0\xff')})
        public_png=(ROOT/'docs/results/group_split_before_after.png').read_bytes()
        self.assertFalse(guard.scan_bytes('docs/results/fixture.png',public_png))
        self.assertIn('unreviewed-binary',{f.category for f in guard.scan_bytes('private-frame.png',public_png)})

    def test_image_metadata_and_appended_data_cannot_hide_text(self):
        image=(ROOT/'docs/results/group_split_before_after.png').read_bytes()
        payload=b'Comment\0fixture metadata'
        chunk=b'tEXt'+payload
        encoded=len(payload).to_bytes(4,'big')+chunk+zlib.crc32(chunk).to_bytes(4,'big')
        altered=image[:-12]+encoded+image[-12:]
        self.assertIn('image-text-metadata',{f.category for f in guard.scan_bytes('docs/results/fixture.png',altered)})
        self.assertIn('invalid-image',{f.category for f in guard.scan_bytes('docs/results/fixture.png',image+b'trailing')})

    def test_clean_staged_repository_passes(self):
        self.stage('README.md','Use environment variables. Never publish private data.\n')
        self.stage('.env.example','API_KEY="your_api_key_here"\n')
        result=guard.scan_repository(self.repo,True)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['files_checked'],2)

    def make_ci_artifacts(self):
        for name in guard.CI_ARTIFACTS:
            content = (ROOT/name).read_bytes() if name not in guard.GENERATED_CI_ARTIFACTS else b'{}'
            self.write(name, content)

    def test_ci_artifact_scan_blocks_tokens_in_an_allowed_generated_file(self):
        self.make_ci_artifacts()
        self.write('benchmark_runs/ci_environment.json',json.dumps({'marker':dummy_token()}))
        report=guard.scan_ci_artifacts(self.repo)
        self.assertEqual(report['files_checked'],23)
        self.assertIn('credential-token',self.categories(report))
        self.assertNotIn(dummy_token(),json.dumps(report))

    def test_ci_artifact_scan_does_not_select_unknown_sibling_files(self):
        self.make_ci_artifacts()
        self.write('docs/results/private_extra.json',json.dumps({'marker':dummy_token()}))
        report=guard.scan_ci_artifacts(self.repo)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['files_checked'],23)
        self.assertNotIn('docs/results/private_extra.json',guard.CI_ARTIFACTS)
        # The exception is limited to the exact three named generated outputs.
        self.assertIn('private-artifact-path',{f.category for f in guard.scan_bytes('benchmark_runs/other.json',b'{}',ci_artifact=True)})

    def test_ci_artifact_scan_requires_every_expected_file(self):
        self.make_ci_artifacts()
        (self.repo/'docs/results/group_split_before_after.png').unlink()
        self.assertIn('unreadable-file',self.categories(guard.scan_ci_artifacts(self.repo)))

    def test_workflow_upload_list_exactly_matches_guard_without_globs(self):
        lines=(ROOT/'.github/workflows/validate.yml').read_text().splitlines()
        blocks=[]
        for i,line in enumerate(lines):
            if line.strip()=='path: |':
                indent=len(line)-len(line.lstrip());paths=[]
                for following in lines[i+1:]:
                    if not following.strip():continue
                    if len(following)-len(following.lstrip())<=indent:break
                    paths.append(following.strip())
                blocks.append(paths)
        self.assertEqual(len(blocks),1)
        self.assertEqual(len(blocks[0]),23)
        self.assertEqual(set(blocks[0]),set(guard.CI_ARTIFACTS))
        self.assertFalse(any('*' in name or '?' in name for name in blocks[0]))


if __name__=='__main__':
    unittest.main()
