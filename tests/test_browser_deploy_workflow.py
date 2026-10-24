import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.load((ROOT / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader)


def test_manual_frontend_deployment_is_independent_and_gated():
    doc = workflow('deploy-model-browser.yml')
    assert set(doc['on']) == {'workflow_dispatch'}
    inputs = doc['on']['workflow_dispatch']['inputs']
    assert inputs['environment']['options'] == ['staging', 'production']
    assert inputs['audience']['options'] == ['developer', 'customer']
    jobs = doc['jobs']
    assert set(jobs) == {'authorize', 'resolve-vulcan-config', 'build', 'publish'}
    assert jobs['authorize']['environment'] == 'model-browser-deploy'
    assert 'refs/heads/develop' in jobs['authorize']['if']
    assert 'refs/heads/main' in jobs['authorize']['if']
    assert 'id-token' not in jobs['authorize']['permissions']
    checkout, authorize = jobs['authorize']['steps']
    assert checkout['uses'] == 'actions/checkout@v6'
    assert checkout['with'] == {'ref': '${{ github.sha }}', 'persist-credentials': 'false'}
    assert authorize['with']['script'].strip() == "await require('./scripts/authorize-model-browser-deployment.cjs')({github, context, core});"
    assert jobs['build']['needs'] == jobs['resolve-vulcan-config']['needs'] == 'authorize'
    assert jobs['publish']['needs'] == ['authorize', 'resolve-vulcan-config', 'build']
    assert jobs['build']['with']['environment'] == '${{ inputs.environment }}'
    assert jobs['resolve-vulcan-config']['with']['vulcan_env'] == '${{ inputs.environment }}'
    publish = jobs['publish']['with']
    assert publish['s3_prefix'] == 'models'
    assert publish['cloudfront_paths'] == '/models /models/*'
    assert publish['delete_removed'] == 'false'
    assert jobs['publish']['concurrency']['group'] == '${{ github.repository }}-model-browser-${{ inputs.environment }}'


@pytest.mark.parametrize('data, allowed', [
    ({'can_admins_bypass': False, 'protection_rules': [{'type': 'required_reviewers', 'reviewers': [{'type': 'Team', 'reviewer': {'id': 20041032}}]}]}, True),
    ({'protection_rules': []}, False),
    ({'can_admins_bypass': True, 'protection_rules': [{'type': 'required_reviewers', 'reviewers': [{'type': 'Team', 'reviewer': {'id': 20041032}}]}]}, False),
    ({'can_admins_bypass': False, 'protection_rules': [{'type': 'required_reviewers', 'reviewers': [{'type': 'Team', 'reviewer': {'id': 1}}]}]}, False),
])
def test_approval_configuration_guard_fails_closed(data, allowed):
    script = f'await (await import({json.dumps((ROOT / "scripts/authorize-model-browser-deployment.cjs").as_uri())})).default({{github, context, core}});'
    harness = 'process.env.DEPLOY_ENV="staging"; process.env.DEPLOY_AUDIENCE="developer"; process.env.GITHUB_TRIGGERING_ACTOR="owner"; process.env.GITHUB_RUN_ATTEMPT="1"; const context={repo:{},sha:"fixture",runAttempt:1,actor:"owner"}; const core={notice:()=>{}}; const github={request:async()=>({data:[{state:"approved",user:{login:"owner"},environments:[{name:"model-browser-deploy"}]}]}),rest:{repos:{getEnvironment:async()=>({data:' + json.dumps(data) + '})}}};\n'
    result = subprocess.run(['node', '--input-type=module', '-e', harness + script], capture_output=True)
    assert (result.returncode == 0) == allowed


def test_reusable_frontend_build_applies_environment_for_build_and_smoke():
    doc = workflow('developer-docs.yml')
    assert doc['on']['workflow_call']['inputs']['environment']['default'] == 'staging'
    assert doc['jobs']['build']['env']['MODEL_BROWSER_ENVIRONMENT'] == "${{ inputs.environment || 'staging' }}"
    assert doc['jobs']['build']['env']['MODEL_BROWSER_UI_MODE'] == '${{ inputs.ui_mode }}'


@pytest.mark.parametrize('approver,attempt', [('someone-else',1), ('owner',2), ('',1)])
def test_non_owner_approval_or_rerun_cannot_deploy(approver, attempt):
    script = f'await (await import({json.dumps((ROOT / "scripts/authorize-model-browser-deployment.cjs").as_uri())})).default({{github, context, core}});'
    data = {'can_admins_bypass': False, 'protection_rules': [{'type': 'required_reviewers', 'reviewers': [{'type': 'Team', 'reviewer': {'id': 20041032}}]}]}
    approvals = [{'state':'approved','user':{'login':approver},'environments':[{'name':'model-browser-deploy'}]}]
    harness = 'process.env.DEPLOY_ENV="staging"; process.env.DEPLOY_AUDIENCE="developer"; process.env.GITHUB_TRIGGERING_ACTOR="owner"; process.env.GITHUB_RUN_ATTEMPT="1"; const context={repo:{},sha:"fixture",actor:"owner",runAttempt:' + str(attempt) + '}; process.env.GITHUB_RUN_ATTEMPT=String(context.runAttempt); const core={notice:()=>{}}; const github={request:async()=>({data:' + json.dumps(approvals) + '}),rest:{repos:{getEnvironment:async()=>({data:' + json.dumps(data) + '})}}};'
    result = subprocess.run(['node','--input-type=module','-e',harness+script],capture_output=True)
    assert result.returncode != 0
