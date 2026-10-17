"""Promotion retries must never change what the reviewer approved."""
from __future__ import annotations

import copy
import json
from argparse import Namespace

import pytest
from model_registry import promotions as p
from model_registry.errors import ValidationError
from model_registry.registry import RegistryPublicationError

CANDIDATE = '11111111-1111-4111-8111-111111111111'
PROMOTION = '22222222-2222-4222-8222-222222222222'


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("VULCAN_ENV", "staging")
    monkeypatch.setenv("VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL", "https://reader.example.com")
    monkeypatch.setenv("VULCAN_ARTIFACT_REGISTRY_API_URL", "https://writer.staging.example.com")


class Backend:
    def __init__(self):
        self.calls = []
        self.snapshot = 'a' * 64
        self.record = None
        self.lost_create = False
        self.lost_approve = False

    def request(self, method, path, body, key):
        self.calls.append((method, path, copy.deepcopy(body), key))
        if path.endswith('/deployments/' + CANDIDATE):
            return {'state': 'finalized', 'manifest_digest': 'b' * 64}
        if '/catalogs/current?' in path:
            raise AssertionError('Catalog must not use the signed writer client')
        if path.endswith('/promotions'):
            if self.record is None:
                self.record = dict(body, promotion_id=PROMOTION, state='pending', pending_protection=True,
                                   source={'commit_sha': 'c' * 40}, selection={'artifacts': []},
                                   previous_selection=None)
            if self.lost_create:
                self.lost_create = False
                raise RegistryPublicationError('lost response after commit')
            return dict(self.record, state='pending')  # historical idempotency receipt
        if method == 'GET':
            return copy.deepcopy(self.record)
        operation = path.rsplit('/', 1)[1]
        if operation == 'approve' and self.snapshot != self.record['expected_target_snapshot']:
            raise RegistryPublicationError('409 catalog_snapshot_conflict')
        self.record['state'] = {'approve': 'promoted', 'reject': 'rejected', 'cancel': 'cancelled'}[operation]
        self.record['pending_protection'] = False
        if operation == 'approve':
            self.record['result_snapshot'] = 'd' * 64
        if self.lost_approve:
            self.lost_approve = False
            raise RegistryPublicationError('lost response after decision')
        return copy.deepcopy(self.record)


def plan(backend):
    return p.prepare(backend, deployment_id=CANDIDATE, reason='Validated fixture; evidence URL',
                     workflow={'run_id': '123', 'run_url': 'https://github.com/sima-neat/models/actions/runs/123'},
                     identity='123', read_api_url='https://reader.example.com',
                     read_request=lambda url: {'catalog': {'canonical_sha256': backend.snapshot}})


def test_prepare_does_not_mutate_and_pins_canonical_snapshot():
    backend = Backend()
    saved = plan(backend)
    assert saved['body']['expected_target_snapshot'] == 'a' * 64
    assert saved['body']['target_git_ref'] == p.TARGET
    assert all(call[0] == 'GET' for call in backend.calls)


def test_lost_request_response_replays_same_body_key_even_after_target_changes():
    backend = Backend()
    saved = plan(backend)
    backend.lost_create = True
    with pytest.raises(RegistryPublicationError):
        p.request(backend, saved)
    backend.snapshot = 'e' * 64
    result = p.request(backend, saved)
    writes = [c for c in backend.calls if c[0] == 'POST']
    assert len(writes) == 2 and writes[0] == writes[1]
    assert result['expected_target_snapshot'] == 'a' * 64
    assert result['pending_protection']


def test_stale_approval_never_rebases_or_creates_request():
    backend = Backend()
    record = p.request(backend, plan(backend))
    backend.snapshot = 'e' * 64
    backend.calls.clear()
    with pytest.raises(RegistryPublicationError, match='catalog_snapshot_conflict'):
        p.decide(backend, record, operation='approve', reason='Reviewed', identity='123')
    assert backend.record['state'] == 'pending'
    assert all('/catalogs/' not in c[1] and not c[1].endswith('/promotions') for c in backend.calls)


def test_lost_approval_response_recovers_authoritative_success_without_second_write():
    backend = Backend()
    record = p.request(backend, plan(backend))
    backend.lost_approve = True
    with pytest.raises(RegistryPublicationError):
        p.decide(backend, record, operation='approve', reason='Reviewed', identity='123')
    result = p.decide(backend, record, operation='approve', reason='Reviewed', identity='123')
    assert result['state'] == 'promoted' and result['result_snapshot']
    assert len([c for c in backend.calls if c[1].endswith('/approve')]) == 1


@pytest.mark.parametrize('field,value', [('deployment_id', PROMOTION), ('manifest_digest', 'e' * 64),
                                        ('expected_target_snapshot', 'f' * 64), ('selection', {'changed': True}),
                                        ('source', {'commit_sha': 'f' * 40})])
def test_approval_rejects_changed_review_content(field, value):
    backend = Backend()
    record = p.request(backend, plan(backend))
    backend.record[field] = value
    with pytest.raises(ValidationError):
        p.decide(backend, record, operation='approve', reason='Reviewed', identity='123')
    assert not any(c[1].endswith('/approve') for c in backend.calls)


@pytest.mark.parametrize('state', ['expired', 'rejected', 'cancelled'])
def test_terminal_request_cannot_be_approved(state):
    backend = Backend()
    record = p.request(backend, plan(backend))
    backend.record['state'] = state
    with pytest.raises(ValidationError, match=state):
        p.decide(backend, record, operation='approve', reason='Reviewed', identity='123')


@pytest.mark.parametrize('operation,state', [('cancel', 'cancelled'), ('reject', 'rejected')])
def test_explicit_closure_is_idempotent(operation, state):
    backend = Backend()
    record = p.request(backend, plan(backend))
    assert p.decide(backend, record, operation=operation, reason='No longer wanted', identity='123')['state'] == state
    assert p.decide(backend, record, operation=operation, reason='No longer wanted', identity='123')['state'] == state
    assert len([c for c in backend.calls if c[1].endswith('/' + operation)]) == 1


def test_cancellation_cannot_undo_promotion():
    backend = Backend()
    record = p.request(backend, plan(backend))
    p.decide(backend, record, operation='approve', reason='Reviewed', identity='123')
    with pytest.raises(ValidationError, match='promoted'):
        p.decide(backend, record, operation='cancel', reason='Stop', identity='124')


def test_request_checks_current_state_not_historical_receipt():
    backend = Backend()
    saved = plan(backend)
    p.request(backend, saved)
    backend.record['state'] = 'expired'
    with pytest.raises(ValidationError, match='expired'):
        p.request(backend, saved)


def test_request_requires_confirmed_pending_hold():
    backend = Backend()
    saved = plan(backend)
    p.request(backend, saved)
    backend.record['pending_protection'] = False
    with pytest.raises(ValidationError, match='protection'):
        p.request(backend, saved)


def test_main_target_rejected_before_mutation():
    backend = Backend()
    saved = plan(backend)
    saved['body']['target_git_ref'] = 'refs/heads/main'
    with pytest.raises(ValidationError):
        p.request(backend, saved)
    assert all(c[0] == 'GET' for c in backend.calls)


def test_summary_cannot_inject_markdown():
    rendered = p.summary({'reason': '```\n# Forged review\n<script>'})
    assert rendered.count('```') == 2
    assert '<script>' not in rendered


def test_cli_inspect_writes_record(monkeypatch, tmp_path, capsys):
    from model_registry.cli import main
    backend = Backend()
    record = p.request(backend, plan(backend))
    monkeypatch.setattr(p, 'SigV4RegistryClient', lambda *args: backend)
    destination = tmp_path / 'record.json'
    assert main(['promotion', 'inspect', '--record', str(destination), '--promotion-id', PROMOTION]) == 0
    assert json.loads(destination.read_text()) == {**record, **p.configuration()}
    assert json.loads(capsys.readouterr().out)['promotion_id'] == PROMOTION


def test_saved_plan_cannot_be_overwritten(monkeypatch, tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(p, 'SigV4RegistryClient', lambda *args: Backend())
    with pytest.raises(ValidationError, match='already exists'):
        p.run(Namespace(action='prepare', record=path, identity='123', run_url='url', aws_region='us-west-2'))


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_configuration_uses_selected_environment_endpoint(monkeypatch, environment):
    monkeypatch.setenv("VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL", "https://reader.example.com")
    monkeypatch.setenv("VULCAN_ENV", environment)
    monkeypatch.setenv("VULCAN_ARTIFACT_REGISTRY_API_URL", f"https://writer.{environment}.example.com/")
    assert p.configuration() == {"vulcan_environment": environment,
                                 "api_url": f"https://writer.{environment}.example.com",
                                 "read_api_url": "https://reader.example.com"}


@pytest.mark.parametrize("environment", ["", "prod", "stg", "unknown"])
def test_configuration_has_no_implicit_default(monkeypatch, environment):
    monkeypatch.setenv("VULCAN_ENV", environment)
    with pytest.raises(ValidationError, match="VULCAN_ENV"):
        p.configuration()


@pytest.mark.parametrize("url", ["", "http://writer.example.com", "https://user:password@writer.example.com",
                                 "https://writer.example.com/path", "https://writer.example.com?query=1"])
def test_configuration_rejects_invalid_writer_origins(monkeypatch, url):
    monkeypatch.setenv("VULCAN_ARTIFACT_REGISTRY_API_URL", url)
    with pytest.raises(ValidationError, match="HTTPS"):
        p.configuration()


@pytest.mark.parametrize("changed", [{"vulcan_environment": "production"}, {"api_url": "https://other.example.com"},
                                     {"read_api_url": "https://other-reader.example.com"}])
def test_retry_cannot_switch_registry(monkeypatch, tmp_path, changed):
    from model_registry.cli import main
    record = tmp_path / "plan.json"
    p.save(record, {**p.configuration(), **changed})
    backend = Backend()
    monkeypatch.setattr(p, "SigV4RegistryClient", lambda *args: backend)
    assert main(["promotion", "request", "--record", str(record)]) == 2
    assert backend.calls == []


def test_prepare_reads_catalog_publicly_and_candidate_privately():
    backend = Backend()
    urls = []

    def public_read(url):
        urls.append(url)
        return {"catalog": {"canonical_sha256": "e" * 64}}

    saved = p.prepare(backend, deployment_id=CANDIDATE, reason="Validated", workflow={}, identity="123",
                      read_api_url="https://reader.example.com", read_request=public_read)
    assert urls == ["https://reader.example.com" + p.ROOT + "/catalogs/current?git_ref=refs%2Fheads%2Fdevelop"]
    assert len(backend.calls) == 1
    assert backend.calls[0][1] == p.ROOT + "/deployments/" + CANDIDATE
    assert saved["body"]["expected_target_snapshot"] == "e" * 64


@pytest.mark.parametrize("url", ["", "http://reader.example.com", "https://reader.example.com/path"])
def test_configuration_rejects_invalid_reader_origins(monkeypatch, url):
    monkeypatch.setenv("VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL", url)
    with pytest.raises(ValidationError, match="VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL"):
        p.configuration()
