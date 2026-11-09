"""Main metadata must use its own source revision and never start compilation."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_workflow_uses_main_producer_and_existing_publisher():
    config = yaml.safe_load((ROOT / '.github/workflows/model-main-catalog.yml').read_text())
    assert set(config.get('on', config.get(True))) == {'workflow_dispatch'}
    assert "github.ref == 'refs/heads/develop'" in config['jobs']['resolve']['if']
    assert config['jobs']['resolve']['with']['vulcan_env'] == 'production'
    publish = config['jobs']['publish']
    assert publish['environment'] == 'production'
    checkouts = [s['with'] for s in publish['steps'] if 'actions/checkout@' in s.get('uses', '')]
    assert checkouts == [
        {'ref': '${{ github.sha }}', 'persist-credentials': False},
        {'ref': 'main', 'path': '_main', 'persist-credentials': False}]
    assert any(s.get('run') == 'python -m pip install ./_main' for s in publish['steps'])
    credentials = next(s for s in publish['steps'] if 'configure-aws-credentials' in s.get('uses', ''))
    assert 'PUBLISHER_ROLE_ARN' in credentials['with']['role-to-assume']
    assert all('compile' not in s.get('run', '') for s in publish['steps'])


def test_publication_pins_main_checkout_not_workflow_revision(tmp_path):
    source = tmp_path / 'main'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / 'README.md').write_text('Main definitions\n')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'Fixture Main source'], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    commands = tmp_path / 'bin'
    commands.mkdir()
    command = commands / 'model-registry'
    command.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    command.chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'], VULCAN_ENV='production',
               VULCAN_ARTIFACT_REGISTRY_API_URL='https://writer.example.com',
               VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL='https://reader.example.com',
               VULCAN_ARTIFACT_REGISTRY_BUCKET='fixture-bucket', AWS_REGION='us-west-2',
               GITHUB_REPOSITORY='sima-neat/models', GITHUB_RUN_ID='123', GITHUB_SHA='a' * 40,
               GITHUB_STEP_SUMMARY=str(tmp_path / 'summary'))
    script = ROOT / 'scripts/publish_main_catalog_metadata.sh'
    subprocess.run(['bash', str(script), str(source)], cwd=tmp_path, env=env, check=True)
    args = json.loads((tmp_path / '_main-catalog/receipt.json').read_text())
    assert args[0] == 'publish-catalog'
    assert args[args.index('--git-sha') + 1] == revision
    assert args[args.index('--branch') + 1] == 'main'
    assert args[args.index('--base-branches') + 1] == 'main'
    assert '--compiled-dir' not in args and '--package-dir' not in args
    (source / 'README.md').write_text('Uncommitted source mutation\n')
    result = subprocess.run(['bash', str(script), str(source)], cwd=tmp_path, env=env, check=False)
    assert result.returncode != 0
    env['VULCAN_ENV'] = 'staging'
    result = subprocess.run(['bash', str(script), str(source)], cwd=tmp_path, env=env, check=False)
    assert result.returncode != 0
