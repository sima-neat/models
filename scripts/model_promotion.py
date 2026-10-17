#!/usr/bin/env python3
"""Usage: model_promotion.py configure|prepare|request|approve|report

GitHub orchestration only. Inputs come from environment variables; registry
semantics live in model_registry.promotions. Run only from trusted Develop code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
        'title': f'Develop promotion request (workflow {run_id})', 'body': marker + '\n\n' + body,
    })
    return record['html_url']


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
                plan = p.prepare(client, deployment_id=os.environ['DEPLOYMENT_ID'], reason=reason,
                                 workflow={'run_id': run_id, 'run_url': run_url}, identity=run_id)
                issue_url = issue(p.summary({**config, 'planned_request': plan['body'], 'workflow': run_url,
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
    elif phase == 'approve':
        plan, record = p.read(PLAN), p.read(RECORD)
        operation = 'approve' if plan['operation'] == 'promote' else 'reject'
        if plan['operation'] not in {'promote', 'reject'}:
            raise p.ValidationError('This operation does not use approval credentials')
        if operation == 'approve':
            p.check_request(record, plan['body'])
        else:
            p.check_request(record, plan['record'])
        result = p.decide(client, record, operation=operation,
                          reason=plan['decision_reason'], identity=plan['identity'])
        p.save(DIRECTORY / 'decision.json', result)
        show(result)
    elif phase == 'report':
        plan, record = p.read(PLAN), p.read(RECORD)
        current = p.inspect(client, record['promotion_id'])
        show(current)
        update_issue(plan, current)
    else:
        raise p.ValidationError('Unknown workflow phase')


def update_issue(plan, record):
    if not plan['issue_url']:
        return
    number = int(plan['issue_url'].rsplit('/', 1)[1])
    marker = f"<!-- models-promotion-run:{plan['identity']} -->"
    gh('api', '--method', 'PATCH', f'repos/sima-neat/models/issues/{number}', body={
        'body': marker + '\n\n' + p.summary({**record, **p.configuration()}),
    })


if __name__ == '__main__':
    try:
        main()
    except (p.ValidationError, subprocess.CalledProcessError) as exc:
        print(f'Promotion workflow failed: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
