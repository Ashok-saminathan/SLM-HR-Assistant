import json
from types import SimpleNamespace

import pytest

from src.api.service import AssistantService


class FakeEngine:
    """Stands in for the SLM so the pipeline can be tested without model weights."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(text=self.reply(messages) if callable(self.reply) else self.reply)


def echo_policy(messages):
    pid = messages[1]["content"].split("[id: ")[1].split("]")[0]
    return json.dumps({"answer": "ok", "source": pid, "answerable": True})


def test_service_happy_path_and_log(tmp_path):
    log = tmp_path / "req.jsonl"
    svc = AssistantService(FakeEngine(echo_policy), str(log))
    r = svc.ask("How many annual leave days do I get?")
    assert r.retrieved_policy_id == "leave-annual" and r.source == "leave-annual" and not r.degraded
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["valid"] and rec["retrieved"] == "leave-annual"
    assert svc.metrics.snapshot()["requests"] == 1


@pytest.mark.parametrize("bad", ["I think it is 18 days", "{not json", json.dumps({"answer": "x"})])
def test_service_falls_back_on_invalid_output(bad):
    svc = AssistantService(FakeEngine(bad))
    r = svc.ask("How many annual leave days do I get?")
    assert r.degraded and not r.answerable and r.source is None
    assert svc.metrics.snapshot()["degraded"] == 1


def test_service_rejects_source_not_in_context():
    svc = AssistantService(FakeEngine(json.dumps({"answer": "x", "source": "leave-sick", "answerable": True})))
    assert svc.ask("How many annual leave days do I get?").degraded


def test_api_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    client = TestClient(create_app(AssistantService(FakeEngine(echo_policy))))
    assert client.get("/health").json()["status"] == "ok"
    ok = client.post("/ask", json={"question": "  How many   sick days   do I get? "})
    assert ok.status_code == 200 and ok.json()["retrieved_policy_id"] == "leave-sick"
    # edge cases: too short, wrong type, missing field, no letters, too long
    for payload in ({"question": "hi"}, {"question": 123}, {}, {"question": "12345 ?!"}, {"question": "x" * 501}):
        r = client.post("/ask", json=payload)
        assert r.status_code == 422 and r.json()["error"] == "invalid_request"
    assert client.get("/metrics").json()["requests"] == 1


def test_api_engine_crash_returns_500_without_leaking_details():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    def boom(_):
        raise RuntimeError("secret internal path /root/x")

    client = TestClient(create_app(AssistantService(FakeEngine(boom))), raise_server_exceptions=False)
    r = client.post("/ask", json={"question": "How many sick days do I get?"})
    assert r.status_code == 500 and "secret" not in r.text and r.json()["error"] == "internal_error"
