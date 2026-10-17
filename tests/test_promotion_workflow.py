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
    assert "github.ref == 'refs/heads/develop'" in jobs['prepare']['if']
    assert jobs['approve']['environment']['name'] == 'staging-promote-develop'
    assert 'issues' not in jobs['approve']['permissions']
    for name, job in jobs.items():
        checkout = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/checkout@'))
        assert checkout['with']['ref'] == '${{ github.sha }}'
        assert checkout['with']['persist-credentials'] is False
        credentials = next(s for s in job['steps'] if 'configure-aws-credentials' in s.get('uses', ''))
        expected = 'PROMOTION' if name == 'approve' else 'PUBLISHER'
        assert f'REGISTRY_{expected}_ROLE_ARN' in credentials['with']['role-to-assume']
        assert all('compile' not in s.get('run', '') for s in job['steps'])


def test_request_plan_is_durable_before_mutation_and_gate_waits_for_summary():
    jobs = workflow()['jobs']
    assert jobs['request']['needs'] == 'prepare'
    assert set(jobs['approve']['needs']) == {'prepare', 'request'}
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
    module.p.save(module.PLAN, {'operation': 'promote', 'body': {}, 'key': 'stable'})
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
