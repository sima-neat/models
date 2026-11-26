#!/usr/bin/env python3
"""Read-only replay of the deployed validator and exact production MPK download.

Requires boto3, botocore, psycopg and the repository's registry dependencies.
This does not invoke a Lambda handler, approve a candidate, or change a catalog.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import ssl
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import boto3
from botocore.httpsession import get_cert_path
from model_registry.registry_api.transport import SigV4RegistryClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(Path(__file__).with_name('mpk.json').read_text())
    candidate = json.loads(args.candidate.read_text())
    assert candidate['deployment_id'] == fixture['deployment_id']
    assert candidate['manifest_digest'] == fixture['manifest_digest']
    assert candidate['state'] == 'finalized'
    assert boto3.client('sts').get_caller_identity()['Account'] == os.environ['PRODUCTION_AWS_ACCOUNT_ID']
    requirements = fixture['temporary_target_patch']['build_requirements'][fixture['profile_id']]
    candidate['id'] = candidate['deployment_id']
    context = ssl.create_default_context(cafile=get_cert_path(True))
    deployed = boto3.client('lambda', region_name='us-west-2').get_function(
        FunctionName='vulcan-production-artifact-registry-writer')
    with urllib.request.urlopen(deployed['Code']['Location'], context=context, timeout=60) as stream:
        payload = stream.read()
    code_hash = base64.b64encode(hashlib.sha256(payload).digest()).decode()
    assert code_hash == deployed['Configuration']['CodeSha256']
    # Extract only deployed Python source. Use local native psycopg rather than
    # the production ARM64 binary; never load unrelated code from the archive.
    with tempfile.TemporaryDirectory(prefix='models-574-validator-') as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                path = Path(name)
                if path.parts[:1] == ('registry_api',) and len(path.parts) == 2 and path.suffix == '.py':
                    target = root / path
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(archive.read(name))
        sys.path.insert(0, str(root))
        from registry_api import promotion_contract  # noqa: PLC0415
        selected = promotion_contract.selection(candidate, requirements, 'sima-neat/models')
    package = next(a for a in selected['artifacts'] if a['artifact_name'] == 'package')
    assert (package['layout'], package['format']) == ('archive', 'mpk')
    client = SigV4RegistryClient('https://artifacts-registry.neat.sima.ai', 'us-west-2')
    artifact = client.request('GET', f"/v1/runs/{package['run_id']}/artifacts/package", None, '')['artifact']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'yolo26x_pose_w8a16_mpk.tar.gz'
    digest, size = hashlib.sha256(), 0
    with (
        urllib.request.urlopen(artifact['download_url'], context=context, timeout=60) as source,
        output.open('wb') as sink,
    ):
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            sink.write(chunk)
    assert (digest.hexdigest(), size) == (artifact['sha256'], artifact['size_bytes'])
    with tarfile.open(output, 'r:gz') as archive:
        members = archive.getmembers()
        assert members
        assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    result = {
        'mode': 'local replay of verified deployed Python source against real production objects',
        'deployment_id': candidate['deployment_id'], 'manifest_digest': candidate['manifest_digest'],
        'deployed_code_sha256_base64': code_hash, 'selection': selected,
        'download': {'run_id': package['run_id'], 'artifact_name': 'package', 'sha256': digest.hexdigest(),
                     'size_bytes': size, 'tar_members': len(members),
                     'via': 'production run-artifact API and exact-version presigned URL'},
        'catalog_changed': False, 'live_protected_promotion': False,
    }
    (args.output_dir / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
