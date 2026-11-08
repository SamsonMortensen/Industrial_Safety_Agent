# Keep secrets and private data out of publications

Run `python -B code/check_publication_privacy.py --staged` before committing.
The staged check reads the Git index, so cleaning a working file after staging
does not conceal the version that would be committed. CI also checks the tracked
checkout and scans the current source plus incoming commit ranges with Gitleaks.

Keep API credentials in environment variables or a secret manager. Never paste
them into notebooks, incident text, generated JSON, examples, reports, or logs.
Local environment files, private reviews, media, model packages, evaluation
bundles, and backups do not belong in the public repository. Ignore rules help
prevent accidental additions; the publication check also rejects prohibited
files that were force-added.

Before publishing a notebook, clear outputs, execution counts, and attachments.
Before publishing a report, review its narrative and figures for names, faces,
addresses, identifiers, and operational details. Automated checks can detect
common credentials, home paths, email patterns, and unexpected file types, but
cannot establish that arbitrary text or images contain no personal data.

Development exports use an explicit field contract. They retain anonymous case
and group aliases, known model names, validated numeric measurements, UTC
completion times, and source hashes. Additional source metadata does not widen
the export. The CI artifact list uses exact filenames; adding another file to a
results folder does not automatically publish it.

The current-file and incoming-commit checks do not certify all historical Git
objects. If a secret was ever published, revoke or rotate it at its provider
promptly. Deleting the latest file does not remove earlier commits, forks,
downloaded archives, or other people's clones. A coordinated history cleanup is
separate from normal code changes because it changes commit identifiers and can
invalidate saved provenance links. Never upload repository-history backups.

For repository-level protection, enable GitHub secret scanning and push
protection in the repository security settings when available. Local hooks and
CI supplement those controls; they cannot revoke an exposed key or guarantee
that all private information is recognized.

Independent evaluation, calibration reserves, training approval, and model
promotion remain separate. Privacy hardening does not authorize running them or
changing their frozen records.
