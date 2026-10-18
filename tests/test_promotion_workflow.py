"""Security and durability boundaries of the GitHub promotion workflow."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return yaml.safe_load((ROOT / '.github/workflows/model-promotion.yml').read_text())


def test_workflow_is_manual_and_uses_trusted_code_with_separate_credentials():
    config = workflow()
    assert set(config.get('on', config.get(True))) == {'workflow_dispatch'}
    jobs = config['jobs']
    assert "github.ref == 'refs/heads/develop'" in jobs['configure']['if']
    assert jobs['configure']['environment'] == '${{ vars.VULCAN_ENV }}'
    assert 'needs.configure.outputs.vulcan_environment' in jobs['approve']['environment']['name']
    assert 'issues' not in jobs['approve']['permissions']
    for name, job in jobs.items():
        checkout = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/checkout@'))
        assert checkout['with']['ref'] == '${{ github.sha }}'
        assert checkout['with']['persist-credentials'] is False
        if name == 'configure':
            assert not any('configure-aws-credentials' in s.get('uses', '') for s in job['steps'])
            continue
        assert job['env']['VULCAN_ENV'] == '${{ needs.configure.outputs.vulcan_environment }}'
        assert job['env']['VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL'] == '${{ needs.configure.outputs.read_api_url }}'
        credentials = next(s for s in job['steps'] if 'configure-aws-credentials' in s.get('uses', ''))
        expected = 'PROMOTION' if name == 'approve' else 'PUBLISHER'
        assert f'REGISTRY_{expected}_ROLE_ARN' in credentials['with']['role-to-assume']
        assert all('compile' not in s.get('run', '') for s in job['steps'])


def test_request_plan_is_durable_before_mutation_and_gate_waits_for_summary():
    jobs = workflow()['jobs']
    assert jobs['request']['needs'] == ['configure', 'prepare']
    assert set(jobs['approve']['needs']) == {'configure', 'prepare', 'request'}
    upload = next(s for s in jobs['prepare']['steps'] if 'upload-artifact' in s.get('uses', ''))
    assert upload['with']['name'] == 'promotion-plan'
    assert 'overwrite' not in upload['with']
    assert jobs['report']['if'] == "always() && needs.request.result == 'success'"


def test_workflow_inputs_are_environment_data_not_shell_code():
    for job in workflow()['jobs'].values():
        for step in job['steps']:
            assert '${{ inputs.' not in step.get('run', '')


def orchestration(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('model_promotion_workflow', ROOT / 'scripts/model_promotion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    for key, value in {
        'GITHUB_REPOSITORY': 'sima-neat/models', 'GITHUB_REF': 'refs/heads/develop',
        'VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL': 'https://reader.example.com',
        'VULCAN_ENV': 'staging', 'VULCAN_ARTIFACT_REGISTRY_API_URL': 'https://writer.staging.example.com',
        'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
        'GITHUB_OUTPUT': str(tmp_path / 'outputs'), 'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary'),
    }.items():
        monkeypatch.setenv(key, value)
    return module


def test_missing_plan_on_rerun_cannot_refresh_request(monkeypatch, tmp_path):
    import pytest
    module = orchestration(monkeypatch, tmp_path)
    monkeypatch.setattr(module.sys, 'argv', ['model_promotion.py', 'prepare'])
    monkeypatch.setattr(module, 'gh', lambda *args, **kwargs: [{'artifacts': []}])
    with pytest.raises(module.p.ValidationError, match='Saved plan is missing'):
        module.main()


def test_issue_recovery_uses_rest_marker_and_does_not_duplicate(monkeypatch, tmp_path):
    module = orchestration(monkeypatch, tmp_path)
    calls = []

    def gh(*args, **kwargs):
        calls.append(args)
        return [{'body': '<!-- models-promotion-run:123 -->', 'user': {'login': 'github-actions[bot]'},
                 'html_url': 'https://github.com/sima-neat/models/issues/999'}]

    monkeypatch.setattr(module, 'gh', gh)
    assert module.issue('Preparing') == 'https://github.com/sima-neat/models/issues/999'
    assert len(calls) == 1 and '--method' not in calls[0]


def test_request_receipt_survives_issue_update_failure(monkeypatch, tmp_path):
    import pytest
    module = orchestration(monkeypatch, tmp_path)
    module.p.save(module.PLAN, {**module.p.configuration(), 'operation': 'promote', 'body': {}, 'key': 'stable'})
    record = {'promotion_id': '22222222-2222-4222-8222-222222222222', 'state': 'pending'}
    monkeypatch.setattr(module.sys, 'argv', ['model_promotion.py', 'request'])
    monkeypatch.setattr(module.p, 'request', lambda *args: record)

    def fail_issue(*args):
        raise RuntimeError('GitHub unavailable')

    monkeypatch.setattr(module, 'update_issue', fail_issue)
    with pytest.raises(RuntimeError, match='GitHub unavailable'):
        module.main()
    assert module.p.read(module.RECORD) == record
    assert 'promotion-record-2' in (tmp_path / 'outputs').read_text()


def test_untrusted_ref_fails_before_api_access(monkeypatch, tmp_path):
    import pytest
    module = orchestration(monkeypatch, tmp_path)
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/candidate-branch')
    with pytest.raises(module.p.ValidationError, match='Develop'):
        module.main()


def test_environment_change_rejected_before_workflow_api_access(monkeypatch, tmp_path):
    import pytest
    module = orchestration(monkeypatch, tmp_path)
    module.p.save(module.PLAN, {**module.p.configuration(), 'operation': 'promote'})
    monkeypatch.setenv('VULCAN_ENV', 'production')
    monkeypatch.setattr(module.sys, 'argv', ['model_promotion.py', 'request'])
    with pytest.raises(module.p.ValidationError, match='original configuration'):
        module.main()


def test_model_inputs_explain_metadata_location():
    inputs = workflow().get('on', workflow().get(True))['workflow_dispatch']['inputs']
    assert 'deployment_id' not in inputs
    assert 'model.yaml' in inputs['model_id']['description']
    assert 'build_profiles' in inputs['build_profile']['description']


def test_candidate_receipt_selects_newest_match_across_pages(monkeypatch, tmp_path):
    import io
    import json
    import zipfile
    from types import SimpleNamespace
    module = orchestration(monkeypatch, tmp_path)
    older = {'id': 1, 'name': 'candidate-qwen3_5_0_8b-w4a16-1',
             'created_at': '2026-10-16T00:00:00Z', 'expired': False}
    newer = {**older, 'id': 2, 'name': 'candidate-qwen3_5_0_8b-w4a16-2',
             'created_at': '2026-10-17T00:00:00Z'}
    unrelated = {**newer, 'id': 3, 'name': 'candidate-qwen3_5_0_8b-w8a16-1'}
    monkeypatch.setattr(module, 'gh', lambda *args: [{'artifacts': [unrelated, older]},
                                                   {'artifacts': [newer]}])
    receipt = {'model_id': 'qwen3_5_0_8b', 'profile_id': 'w4a16', 'deployment_id': 'uuid'}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('candidate.json', json.dumps(receipt))
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=archive.getvalue())

    monkeypatch.setattr(module.subprocess, 'run', download)
    assert module.candidate_receipt('qwen3_5_0_8b', 'w4a16') == receipt
    assert calls[0][-1].endswith('/2/zip')
    newer['expired'] = True
    import pytest
    with pytest.raises(module.p.ValidationError, match='expired'):
        module.candidate_receipt('qwen3_5_0_8b', 'w4a16')
    assert len(calls) == 1
    newer['expired'] = False
    with pytest.raises(module.p.ValidationError, match='does not match'):
        module.candidate_receipt('qwen3_5_0_8b', 'w8a16')


def test_candidate_receipt_missing_or_invalid_inputs(monkeypatch, tmp_path):
    import pytest
    module = orchestration(monkeypatch, tmp_path)
    monkeypatch.setattr(module, 'gh', lambda *args: [{'artifacts': []}])
    with pytest.raises(module.p.ValidationError, match='No candidate'):
        module.candidate_receipt('qwen3_5_0_8b', 'w4a16')
    with pytest.raises(module.p.ValidationError, match='model.yaml'):
        module.candidate_receipt('', 'w4a16')


def test_acceptance_branch_is_staging_only(monkeypatch, tmp_path):
    import pytest
    module = orchestration(monkeypatch, tmp_path)
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/codex/promotion-catalog-reader')
    monkeypatch.setattr(module.sys, 'argv', ['model_promotion.py', 'configure'])
    module.main()
    assert 'vulcan_environment=staging' in (tmp_path / 'outputs').read_text()
    monkeypatch.setenv('VULCAN_ENV', 'production')
    with pytest.raises(module.p.ValidationError, match='Develop'):
        module.main()
    monkeypatch.setenv('VULCAN_ENV', 'staging')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'someone/models')
    with pytest.raises(module.p.ValidationError, match='Develop'):
        module.main()
    guard = workflow()['jobs']['configure']['if']
    assert "github.ref == 'refs/heads/codex/promotion-catalog-reader' && vars.VULCAN_ENV == 'staging'" in guard


def test_review_notification_mentions_once_and_links_pinned_request(monkeypatch, tmp_path):
    import json
    module = orchestration(monkeypatch, tmp_path)
    monkeypatch.setenv('SLACK_MODEL_PROMOTION_CHANNEL_ID', 'C123')
    comments = []
    calls = []

    def github(*args, **kwargs):
        calls.append((args, kwargs))
        if '--method' in args:
            comments.append({**kwargs['body'], 'user': {'login': 'github-actions[bot]'}})
            return {}
        return [comments]

    monkeypatch.setattr(module, 'gh', github)
    plan = {'issue_url': 'https://github.com/sima-neat/models/issues/330', 'identity': '123'}
    record = {'promotion_id': 'pinned-id', 'source': {'model_id': 'qwen3_5_0_8b', 'profile_id': 'w4a16'}}
    module.request_review(plan, record)
    module.request_review(plan, record)
    assert len(comments) == 1
    assert '@sima-neat/owners-models' in comments[0]['body']
    message = json.loads((tmp_path / '_promotion/slack-review.json').read_text())
    assert message['channel'] == 'C123'
    assert message['mrkdwn'] is False and message['parse'] == 'none'
    assert 'qwen3_5_0_8b / w4a16' in message['text']
    assert 'pinned-id' in message['text']
    assert plan['issue_url'] in message['text']
    assert 'actions/runs/123' in message['text']


def test_slack_notification_is_before_gate_and_failure_is_nonblocking():
    steps = workflow()['jobs']['request']['steps']
    notify = next(s for s in steps if s.get('name') == 'Notify model owners in Slack')
    assert notify['if'] == "steps.request.outputs.review_ready == 'true'"
    assert notify['continue-on-error'] is True
    assert notify['with']['token'] == '${{ secrets.SLACK_BOT_TOKEN }}'
    assert notify['with']['payload-file-path'] == '_promotion/slack-review.json'
