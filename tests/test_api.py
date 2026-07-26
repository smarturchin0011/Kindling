"""API 层测试：HTTP 契约 + 状态机约束。零网络（注入 FakeLLM）。"""
import json

import pytest
from fastapi.testclient import TestClient

from kindling import api
from kindling.store import Store
from kdfakes import FakeLLM
from kdresponses import ACTION_GAP, ANSWERABLE_GAP, FRAMES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "state.json"))
    api.set_llm(None)
    with TestClient(api.app) as c:
        yield c
    api.set_llm(None)


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "state.json"


def use_llm(responses):
    llm = FakeLLM(responses)
    api.set_llm(llm)
    return llm


def seed(client, items):
    for text, t in items:
        r = client.post("/api/entries", json={"text": text, "type": t})
        assert r.status_code == 200, r.text


# ---------- 采集 ----------


def test_add_entry_returns_snapshot(client):
    r = client.post("/api/entries", json={"text": "想教AI产品", "type": "intent"})
    assert r.status_code == 200
    body = r.json()
    assert body["entry"]["type"] == "intent"
    assert body["gate"]["open"] is False
    assert body["gate"]["percent"] >= 0


def test_add_entry_rejects_too_long(client):
    r = client.post("/api/entries", json={"text": "x" * 300, "type": "fact"})
    assert r.status_code == 400
    assert "280" in r.json()["detail"]


def test_add_entry_rejects_empty(client):
    assert client.post("/api/entries", json={"text": "  ", "type": "fact"}).status_code == 400


def test_delete_entry(client):
    eid = client.post("/api/entries", json={"text": "临时", "type": "fact"}).json()["entry"]["id"]
    assert client.delete(f"/api/entries/{eid}").status_code == 200
    assert client.delete(f"/api/entries/{eid}").status_code == 404


def test_topic_persists(client):
    client.post("/api/topic", json={"topic": "如何教AI产品"})
    assert client.get("/api/state").json()["topic"] == "如何教AI产品"


# ---------- ask：两条分支 ----------


def test_ask_requires_some_context(client):
    use_llm([ANSWERABLE_GAP])
    r = client.post("/api/ask")
    assert r.status_code == 400
    assert "扔一两条" in r.json()["detail"]


def test_ask_answerable_returns_gap_without_move(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ANSWERABLE_GAP])
    body = client.post("/api/ask").json()
    assert body["gap"]["answerable_from_memory"] is True
    assert "move" not in body
    assert body["open_move"] is None


def test_ask_unanswerable_creates_move(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP])
    body = client.post("/api/ask").json()
    assert body["gap"]["answerable_from_memory"] is False
    assert body["move"]["retrieves_type"] == "evidence"
    assert body["move"]["est_minutes"] == 4
    assert body["open_move"]["id"] == body["move"]["id"]


def test_ask_blocked_while_move_open(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP, ACTION_GAP])
    client.post("/api/ask")
    r = client.post("/api/ask")
    assert r.status_code == 409
    assert "先完成" in r.json()["detail"]


def test_answer_records_typed_entry(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ANSWERABLE_GAP])
    gap = client.post("/api/ask").json()["gap"]
    body = client.post(
        "/api/answer",
        json={
            "question": gap["question"],
            "answer": "主要是产品经理",
            "target_type": gap["target_type"],
        },
    ).json()
    assert body["entry"]["type"] == "constraint"
    assert body["entry"]["text"] == "主要是产品经理"


# ---------- 飞轮闭合 ----------


def test_done_refluxes_and_raises_completeness(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP])
    client.post("/api/ask")
    before = client.get("/api/state").json()["gate"]["percent"]

    body = client.post(
        "/api/done", json={"artifact": "PM 要求 100% 准确率，项目卡了三周"}
    ).json()

    assert body["after_percent"] > before
    assert body["created"][0]["type"] == "evidence"
    assert body["created"][0]["source"] == "action"
    assert body["open_move"] is None
    assert body["cycles"] == 1


def test_done_without_open_move_is_409(client):
    assert client.post("/api/done", json={"artifact": "x"}).status_code == 409


def test_done_rejects_empty_artifact(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP])
    client.post("/api/ask")
    assert client.post("/api/done", json={"artifact": "   "}).status_code == 400


def test_drop_frees_the_slot(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP, ACTION_GAP])
    client.post("/api/ask")
    assert client.post("/api/drop").json()["open_move"] is None
    assert client.post("/api/ask").json()["move"] is not None


# ---------- 闸门 ----------


def test_synth_gate_returns_423_with_missing_list(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([FRAMES])
    r = client.post("/api/synth", json={"force": False})
    assert r.status_code == 423
    body = r.json()
    assert body["kind"] == "gate"
    assert body["gate"]["open"] is False
    assert any("证据" in m for m in body["gate"]["missing"])


def test_synth_force_bypasses_gate_and_flags(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([FRAMES])
    body = client.post("/api/synth", json={"force": True}).json()
    assert len(body["frames"]) == 2
    assert all(f["forced"] for f in body["frames"])


def test_synth_passes_with_rich_context(client):
    seed(
        client,
        [("事故1", "evidence"), ("事故2", "evidence"), ("受众是PM", "constraint"),
         ("想教AI产品", "intent"), ("时间有限", "constraint")],
    )
    use_llm([FRAMES])
    r = client.post("/api/synth", json={"force": False})
    assert r.status_code == 200
    body = r.json()
    assert body["gate"]["open"] is True
    assert len(body["candidate_frames"]) == 2
    assert all(not f["forced"] for f in body["frames"])


def test_resynth_supersedes_old_candidates(client):
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, FRAMES])
    client.post("/api/synth", json={"force": False})
    body = client.post("/api/synth", json={"force": False}).json()
    assert len(body["candidate_frames"]) == 2   # 只剩新的一批


# ---------- pick 必须落到动作 ----------


def test_pick_forces_a_move(client):
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, ACTION_GAP])
    fid = client.post("/api/synth", json={"force": False}).json()["frames"][0]["id"]

    body = client.post(f"/api/pick/{fid}").json()
    assert body["frame"]["status"] == "picked"
    assert body["move"]["frame_id"] == fid
    assert body["open_move"] is not None
    assert body["candidate_frames"] == []       # 其余降级


def test_pick_records_sacrifice_as_constraint(client):
    """选择本身是一条上下文：它记录了你自愿承担的代价。"""
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, ACTION_GAP])
    frames = client.post("/api/synth", json={"force": False}).json()["frames"]
    body = client.post(f"/api/pick/{frames[0]['id']}").json()
    notes = [e for e in body["entries"] if "自愿牺牲" in e["text"]]
    assert len(notes) == 1
    assert notes[0]["type"] == "constraint"


def test_pick_converts_answerable_gap_into_move(client):
    """模型在 pick 后仍给可答问题时，必须强制转成动作 —— 看框架不能是终点。"""
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, ANSWERABLE_GAP])
    fid = client.post("/api/synth", json={"force": False}).json()["frames"][0]["id"]
    body = client.post(f"/api/pick/{fid}").json()
    assert body["open_move"] is not None
    assert body["move"]["est_minutes"] <= 5


def test_pick_unknown_frame_404(client):
    assert client.post("/api/pick/frm_nope").status_code == 404


def test_pick_blocked_while_move_open(client):
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, ACTION_GAP])
    fid = client.post("/api/synth", json={"force": False}).json()["frames"][0]["id"]
    client.post(f"/api/pick/{fid}")
    assert client.post(f"/api/pick/{fid}").status_code == 409


# ---------- 可观测性 ----------


def test_logs_capture_llm_prompt_and_response(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP])
    client.post("/api/ask")
    logs = client.get("/api/logs").json()["logs"]
    stages = {l["stage"] for l in logs}
    assert "capture" in stages and "gap" in stages
    gap_logs = [l for l in logs if l["stage"] == "gap"]
    assert any("答不上来" in l["message"] or "缺口" in l["message"] for l in gap_logs)


def test_logs_since_cursor_is_incremental(client):
    seed(client, [("a", "intent")])
    first = client.get("/api/logs").json()["logs"]
    last_seq = first[-1]["seq"]
    seed(client, [("b", "intent")])
    delta = client.get(f"/api/logs?since={last_seq}").json()["logs"]
    assert delta and all(l["seq"] > last_seq for l in delta)


def test_llm_error_surfaces_as_502(client):
    seed(client, [("想教AI产品", "intent")])
    use_llm(["这不是 JSON"])
    r = client.post("/api/ask")
    assert r.status_code == 502
    assert r.json()["kind"] == "llm"


def test_reset_clears_everything(client):
    seed(client, [("a", "intent"), ("b", "evidence")])
    body = client.post("/api/reset").json()
    assert body["entries"] == []
    assert body["gate"]["percent"] == 0


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.headers["content-type"]
