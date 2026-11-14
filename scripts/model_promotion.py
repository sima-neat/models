#!/usr/bin/env python3
"""Usage: model_promotion.py configure|prepare|request|approve|benchmark-source|report

GitHub orchestration only. Inputs come from environment variables; registry
semantics live in model_registry.promotions. Run only from trusted Develop code.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from model_registry import promotions as p

DIRECTORY = Path('_promotion')
PLAN = DIRECTORY / 'plan.json'
RECORD = DIRECTORY / 'record.json'


def gh(*args: str, body=None):
    command = ['gh', *args]
    if body is not None:
        command += ['--input', '-']
    result = subprocess.run(command, input=json.dumps(body) if body is not None else None,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        stream.write(f'{name}={value}\n')


def show(record):
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
        stream.write(p.summary({**record, **p.configuration()}))


def issue(body):
    repository, run_id = os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_RUN_ID']
    marker = f'<!-- models-promotion-run:{run_id} -->'
    # REST listing avoids eventual search-index consistency after a lost create response.
    page = 1
    while True:
        records = gh('api', f'repos/{repository}/issues?state=all&per_page=100&page={page}')
        for record in records:
            if (marker in (record.get('body') or '') and 'pull_request' not in record
                    and record['user']['login'] == 'github-actions[bot]'):
                return record['html_url']
        if len(records) < 100:
            break
        page += 1
    record = gh('api', '--method', 'POST', f'repos/{repository}/issues', body={
        'title': f"{p.configuration()['target_branch'].capitalize()} promotion request (workflow {run_id})",
        'body': marker + '\n\n' + body,
    })
    return record['html_url']


def candidate_receipt(model_id: str, profile_id: str) -> dict:
    """Resolve the newest matching build receipt; never fall back past an expired one."""
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (model_id, profile_id)):
        raise p.ValidationError("Provide model_id and build_profile from models/<model_id>/model.yaml")
    if p.configuration()['vulcan_environment'] != 'staging':
        raise p.ValidationError(
            "Automatic lookup is staging-only. Provide an exact candidate UUID from the selected environment.")
    pages = gh('api', '--paginate', '--slurp',
               'repos/sima-neat/models/actions/artifacts?per_page=100')
    pattern = re.compile(rf"candidate-{re.escape(model_id)}-{re.escape(profile_id)}-[0-9]+")
    matches = [a for page in pages for a in page['artifacts'] if pattern.fullmatch(a['name'])]
    if not matches:
        raise p.ValidationError(
            "No candidate receipt found. Run Vulcan Model Orchestrator for this model/profile first.")
    artifact = max(matches, key=lambda a: (a['created_at'], a['id']))
    if artifact['expired']:
        raise p.ValidationError("Newest candidate receipt expired. Run Vulcan Model Orchestrator again.")
    archive = subprocess.run(
        ['gh', 'api', f"repos/sima-neat/models/actions/artifacts/{int(artifact['id'])}/zip"],
        check=True, capture_output=True,
    ).stdout
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            member = bundle.getinfo('candidate.json')
            if member.file_size > 16 * 1024 * 1024:
                raise ValueError("candidate receipt exceeds 16 MiB")
            receipt = json.loads(bundle.read(member))
        if not isinstance(receipt, dict):
            raise ValueError("expected a candidate object")
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise p.ValidationError(f"Invalid candidate receipt: {exc}") from exc
    if (receipt.get('model_id'), receipt.get('profile_id')) != (model_id, profile_id):
        raise p.ValidationError("Candidate receipt does not match the requested model/profile")
    return receipt


def main():
    if os.environ.get('GITHUB_REPOSITORY') != 'sima-neat/models' or os.environ.get('GITHUB_REF') != p.TARGET:
        raise p.ValidationError('Promotion workflow must execute from sima-neat/models Develop')
    phase = sys.argv[1]
    config = p.configuration()
    if phase == 'configure':
        for key, value in config.items():
            output(key, value)
        return
    client = p.SigV4RegistryClient(config['api_url'], 'us-west-2')
    if phase != 'prepare':
        p.check_configuration(p.read(PLAN), config)
    run_id = os.environ['GITHUB_RUN_ID']
    run_url = f'https://github.com/sima-neat/models/actions/runs/{run_id}'
    if phase == 'prepare':
        artifacts = gh('api', '--paginate', '--slurp',
                       f'repos/sima-neat/models/actions/runs/{run_id}/artifacts?per_page=100')
        existing = [a for page in artifacts for a in page['artifacts'] if a['name'] == 'promotion-plan']
        if existing:
            if existing[0]['expired']:
                raise p.ValidationError('Saved plan expired; start a new workflow and review')
            subprocess.run(['gh', 'run', 'download', run_id, '--name', 'promotion-plan',
                            '--dir', str(DIRECTORY)], check=True)
            output('upload_plan', 'false')
        else:
            if os.environ['GITHUB_RUN_ATTEMPT'] != '1':
                raise p.ValidationError('Saved plan is missing; start a new workflow and review')
            operation = os.environ['PROMOTION_OPERATION']
            reason = os.environ['PROMOTION_REASON']
            if not reason.strip() or len(reason) > 2000:
                raise p.ValidationError('Provide a reason of 1–2000 characters')
            if operation == 'promote':
                model_id, profile_id = os.environ.get('MODEL_ID', ''), os.environ.get('BUILD_PROFILE', '')
                if not model_id or not profile_id:
                    raise p.ValidationError('Provide model_id and build_profile from models/<model_id>/model.yaml')
                deployment_id = os.environ.get('DEPLOYMENT_ID', '').strip()
                if not deployment_id:
                    deployment_id = candidate_receipt(model_id, profile_id).get('deployment_id')
                plan = p.prepare(client, deployment_id=deployment_id, reason=reason,
                                 workflow={'run_id': run_id, 'run_url': run_url}, identity=run_id,
                                 read_api_url=config['read_api_url'],
                                 expected_model_id=model_id, expected_profile_id=profile_id)
                plan['selection'] = {'model_id': model_id, 'profile_id': profile_id}
                issue_url = issue(p.summary({**config, 'selection': plan['selection'],
                                            'planned_request': plan['body'], 'workflow': run_url,
                                            'state': 'preparing; backend hold not yet confirmed'}))
                plan['body']['workflow']['issue_url'] = issue_url
                plan['key'] = 'promotion-request-' + p.canonical_sha256([run_id, plan['body']])
            else:
                plan = {'record': p.inspect(client, os.environ['PROMOTION_ID'])}
                issue_url = ''
            plan.update(**config, operation=operation, decision_reason=reason, issue_url=issue_url, identity=run_id)
            p.save(PLAN, plan)
            output('upload_plan', 'true')
        plan = p.read(PLAN)
        p.check_configuration(plan, config)
        output('operation', plan['operation'])
        show(plan)
    elif phase == 'request':
        plan = p.read(PLAN)
        operation = plan['operation']
        if operation == 'promote':
            record = p.request(client, plan)
        elif operation == 'cancel':
            record = p.decide(client, plan['record'], operation='cancel',
                              reason=plan['decision_reason'], identity=plan['identity'])
        else:
            record = p.inspect(client, plan['record']['promotion_id'])
            p.check_request(record, plan['record'])
        p.save(RECORD, record)
        output('record_artifact', f"promotion-record-{os.environ['GITHUB_RUN_ATTEMPT']}")
        show(record)
        # Make review details visible before the protected job is scheduled.
        update_issue(plan, record)
        if operation == 'promote' and record.get('state') == 'pending':
            request_review(plan, record)
    elif phase == 'approve':
        plan, record = p.read(PLAN), p.read(RECORD)
        operation = 'approve' if plan['operation'] == 'promote' else 'reject'
        if plan['operation'] not in {'promote', 'reject', 'withdraw'}:
            raise p.ValidationError('This operation does not use approval credentials')
        if operation == 'approve':
            p.check_request(record, plan['body'])
        else:
            p.check_request(record, plan['record'])
        if plan['operation'] == 'withdraw':
            result = p.withdraw(client, record, reason=plan['decision_reason'], identity=plan['identity'])
        else:
            result = p.decide(client, record, operation=operation,
                              reason=plan['decision_reason'], identity=plan['identity'])
        p.save(DIRECTORY / 'decision.json', result)
        show(result)
    elif phase == 'benchmark-source':
        current = p.inspect(client, p.read(RECORD)['promotion_id'])
        catalog = p.benchmark_catalog(current, read_api_url=config['read_api_url'])
        if catalog is not None:
            output('revision', catalog['catalog_revision'])
    elif phase == 'report':
        plan, record = p.read(PLAN), p.read(RECORD)
        current = None
        try:
            current = p.inspect(client, record['promotion_id'])
            revision = os.environ.get('BENCHMARK_CATALOG_REVISION')
            if revision:
                p.refresh_benchmarks(client, current, repository=Path('_benchmark-source'), revision=revision,
                                     read_api_url=config['read_api_url'])
        finally:
            if plan['operation'] not in {'status', 'withdraw'}:
                completion_notice(plan, current or record, verified=current is not None)
            if current is not None:
                update_issue(plan, current)
        show(current)
    else:
        raise p.ValidationError('Unknown workflow phase')



def request_review(plan, record):
    """Mention the owning team once per request; each retry can send a Slack reminder."""
    number = int(plan['issue_url'].rsplit('/', 1)[1])
    marker = f"<!-- models-promotion-review:{record['promotion_id']} -->"
    pages = gh('api', '--paginate', '--slurp', f'repos/sima-neat/models/issues/{number}/comments?per_page=100')
    if not any(marker in comment.get('body', '') and comment.get('user', {}).get('login') == 'github-actions[bot]'
               for page in pages for comment in page):
        gh('api', '--method', 'POST', f'repos/sima-neat/models/issues/{number}/comments', body={
            'body': marker + '\n\n@sima-neat/owners-models: this promotion is ready for review. '
                    f"Review the candidate and evidence in this issue, then approve or reject at "
                    f"https://github.com/sima-neat/models/actions/runs/{plan['identity']}.",
        })
    source = record.get('source', {})
    text = (
        f"Model promotion approval requested ({p.configuration()['vulcan_environment']})\n"
        f"Model: {source.get('model_id', 'unknown')} / {source.get('profile_id', 'unknown')}\n"
        "Reviewers: sima-neat/owners-models\n"
        f"Promotion: {record['promotion_id']}\n"
        f"Review evidence: {plan['issue_url']}\n"
        f"Approve or reject: https://github.com/sima-neat/models/actions/runs/{plan['identity']}\n"
        "Review the pinned candidate and validation evidence before approving."
    )
    slack_payload('slack-review.json', text)
    output('review_ready', 'true')


def completion_notice(plan, record, *, verified):
    state = record.get('state', 'unknown') if verified else 'unknown'
    if state == 'promoted' and record.get('result_snapshot'):
        outcome = 'Model promotion completed'
    elif state in {'rejected', 'cancelled', 'expired'}:
        outcome = f'Model promotion {state}'
    else:
        outcome = 'Model promotion not confirmed'
    source = record.get('source', {})
    issue_url = plan.get('issue_url') or record.get('workflow', {}).get('issue_url', '')
    text = (
        f"{outcome} ({p.configuration()['vulcan_environment']})\n"
        f"Model: {source.get('model_id', 'unknown')} / {source.get('profile_id', 'unknown')}\n"
        f"Backend state: {state}\n"
        f"Approval job: {os.environ.get('APPROVAL_RESULT', 'unknown')}\n"
        f"Promotion: {record['promotion_id']}\n"
        f"Workflow: https://github.com/sima-neat/models/actions/runs/{plan['identity']}"
    )
    if issue_url:
        text += f"\nEvidence: {issue_url}"
    if state == 'promoted' and record.get('result_snapshot'):
        text += f"\nPublished snapshot: {record['result_snapshot']}"
    elif state == 'pending':
        text += "\nThe request is still pending; no completed promotion was confirmed."
    elif not verified:
        text += "\nCould not read backend state; inspect the workflow before retrying."
    slack_payload('slack-completion.json', text)
    output('completion_ready', 'true')


def slack_payload(filename, text):
    # Plain text prevents model metadata from injecting Slack mentions or formatting.
    p.save(DIRECTORY / filename, {
        'channel': os.environ.get('SLACK_VULCAN_EVENT_CHANNEL_ID', ''),
        'text': text, 'mrkdwn': False, 'parse': 'none',
        'unfurl_links': False, 'unfurl_media': False,
    })


def update_issue(plan, record):
    workflow = record.get('workflow', {})
    issue_url = plan['issue_url'] or workflow.get('issue_url')
    if not issue_url:
        return
    number = int(issue_url.rsplit('/', 1)[1])
    marker = f"<!-- models-promotion-run:{workflow.get('run_id', plan['identity'])} -->"
    body = {
        'body': marker + '\n\n' + p.summary({**record, **p.configuration()}),
    }
    if record.get('state') == 'promoted' and record.get('result_snapshot'):
        body.update(state='closed', state_reason='completed')
    elif record.get('state') in {'rejected', 'cancelled', 'expired'}:
        body.update(state='closed', state_reason='not_planned')
    gh('api', '--method', 'PATCH', f'repos/sima-neat/models/issues/{number}', body=body)


if __name__ == '__main__':
    try:
        main()
    except (p.ValidationError, subprocess.CalledProcessError) as exc:
        print(f'Promotion workflow failed: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
