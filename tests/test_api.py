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


def test_add_entry_without_type_autoclassifies(client):
    """输入时零结构决策 —— 类型是内部计价单位，不该暴露成用户决策。"""
    use_llm(['{"type":"constraint"}'])
    r = client.post("/api/entries", json={"text": "受众只能是产品经理"})
    assert r.status_code == 200
    assert r.json()["entry"]["type"] == "constraint"


def test_add_entry_without_llm_still_works(client):
    """没配 key 也要能录入 —— 录入不该被 LLM 可用性阻塞。"""
    r = client.post("/api/entries", json={"text": "随便扔一个想法"})
    assert r.status_code == 200
    assert r.json()["entry"]["type"] == "intent"


def test_add_entry_with_explicit_type_still_works(client):
    r = client.post("/api/entries", json={"text": "一条约束", "type": "constraint"})
    assert r.json()["entry"]["type"] == "constraint"


def test_add_entry_rejects_unknown_explicit_type(client):
    r = client.post("/api/entries", json={"text": "x", "type": "bogus"})
    assert r.status_code == 400


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


# ---------- 议题模式 ----------


def test_mode_defaults_to_explore_and_can_switch(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    assert r.json()["mode"] == "explore"

    r = client.post("/api/mode", json={"mode": "validate"})
    assert r.status_code == 200
    assert r.json()["mode"] == "validate"

    assert client.get("/api/state").json()["mode"] == "validate"


def test_mode_rejects_unknown_value(client):
    r = client.post("/api/mode", json={"mode": "nonsense"})
    assert r.status_code == 400


def test_ask_in_explore_mode_degrades_external_action(client):
    """端到端：构思模式下 LLM 要求跑实验，必须降级为追问，不生成 Move。"""
    seed(client, [("我想梳理一个还没上线的功能方案", "intent")])
    use_llm([json.dumps({
        "question": "去跑一遍线上流程，记录系统反应",
        "target_type": "evidence",
        "answerable_from_memory": False,
        "why_critical": "决定方案骨架",
        "suggested_action": "跑一遍完整流程并记录",
        "est_minutes": 5,
        "action_kind": "external",
    }, ensure_ascii=False)])

    body = client.post("/api/ask").json()
    assert body["gap"]["answerable_from_memory"] is True
    assert "move" not in body
    assert body["open_move"] is None, "构思阶段不该被外部实验锁住"


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
    assert body["created"][0]["type"] == "constraint"
    assert body["created"][0]["text"] == "主要是产品经理"


def test_answer_persists_question_on_entry(client):
    """question 必须落盘，不能只在 log 里 —— 否则账本是一堆孤立答案。"""
    r = client.post(
        "/api/answer",
        json={
            "question": "你的受众到底是谁？",
            "answer": "主要是产品经理",
            "target_type": "constraint",
        },
    )
    assert r.status_code == 200
    assert r.json()["created"][0]["question"] == "你的受众到底是谁？"

    hit = [
        e
        for e in client.get("/api/state").json()["entries"]
        if e["text"] == "主要是产品经理"
    ][0]
    assert hit["question"] == "你的受众到底是谁？"


def test_answer_longer_than_280_is_accepted_and_split(client):
    """认真回答一个尖锐问题时超长是正常的，不该被 400 惩罚。"""
    long_answer = "答" * 300
    r = client.post(
        "/api/answer",
        json={"question": "q？", "answer": long_answer, "target_type": "constraint"},
    )
    assert r.status_code == 200
    created = r.json()["created"]
    assert len(created) == 2
    assert all(e["question"] == "q？" for e in created)


# ---------- 纠偏通道 ----------


def test_correct_adds_directive_without_score_change(client):
    seed(client, [("事故1", "evidence")])
    before = client.get("/api/state").json()["gate"]["percent"]
    r = client.post("/api/correct", json={"text": "你误解了，我在构思不是在验证"})
    assert r.status_code == 200
    body = r.json()
    assert body["entry"]["type"] == "directive"
    assert body["entry"]["weight"] == 0.0
    assert body["gate"]["percent"] == before, "纠偏不该改变完整度"


def test_retype_entry_fixes_miscategorized_context(client):
    """修正历史错类数据：实测有一条元指令被存成了 evidence 权重 5.0。"""
    r = client.post(
        "/api/entries",
        json={"text": "不要再问上线后才能验证的事", "type": "evidence"},
    )
    eid = r.json()["entry"]["id"]
    assert r.json()["gate"]["percent"] > 0

    r = client.patch(f"/api/entries/{eid}", json={"type": "directive"})
    assert r.status_code == 200
    assert r.json()["gate"]["percent"] == 0

    hit = [e for e in r.json()["entries"] if e["id"] == eid][0]
    assert hit["type"] == "directive"


def test_retype_rejects_unknown_type(client):
    eid = client.post(
        "/api/entries", json={"text": "随便一条", "type": "intent"}
    ).json()["entry"]["id"]
    assert (
        client.patch(f"/api/entries/{eid}", json={"type": "bogus"}).status_code == 400
    )


def test_retype_unknown_entry_404(client):
    assert client.patch("/api/entries/ctx_nope", json={"type": "fact"}).status_code == 404


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


# ---------- 搁置：现在做不了，但问题要留着 ----------


def test_shelve_endpoint_unblocks_ask(client):
    """实测缺陷：外部实验 Move 把用户锁在产品外 6 小时。"""
    seed(client, [("想教AI产品", "intent")])
    use_llm([ACTION_GAP, ACTION_GAP])
    client.post("/api/ask")

    r = client.post("/api/shelve")
    assert r.status_code == 200
    body = r.json()
    assert body["open_move"] is None
    assert any(e["type"] == "unknown" for e in body["entries"])
    assert body["open_questions"], "snapshot 必须携带未决清单"
    assert body["cycles"] == 0, "搁置不算飞轮圈数"

    # 搁置后立刻可以继续
    assert client.post("/api/ask").json()["move"] is not None


def test_shelve_does_not_change_completeness(client):
    seed(client, [("事故1", "evidence")])
    use_llm([ACTION_GAP])
    client.post("/api/ask")
    before = client.get("/api/state").json()["gate"]["percent"]
    assert client.post("/api/shelve").json()["gate"]["percent"] == before


def test_shelve_without_open_move_409(client):
    assert client.post("/api/shelve").status_code == 409


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


# ---------- L7 产出层 ----------


BRIEF = json.dumps(
    {
        "title": "决策简报",
        "verdict": "选定反直觉清单方向。",
        "rationale": "有真实事故支撑（证据：事故1）",
        "risks": "不成体系",
        "next_probes": ["下一步验证 A", "下一步验证 B"],
    },
    ensure_ascii=False,
)


def test_export_returns_markdown_pack(client):
    """导出永远可用 —— 零 LLM，不依赖闸门，不依赖 key。"""
    client.post("/api/topic", json={"topic": "测试议题"})
    r = client.get("/api/export")
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert "测试议题" in r.text
    assert "使用要求" in r.text


def test_export_carries_qa_pairs(client):
    client.post(
        "/api/answer",
        json={"question": "受众是谁？", "answer": "主要是PM", "target_type": "constraint"},
    )
    text = client.get("/api/export").text
    assert "受众是谁？" in text
    assert "主要是PM" in text


def test_brief_blocked_when_gate_closed(client):
    """闸门未开时不产出简报 —— 那只会是漂亮空话（麻醉剂 2.0）。"""
    seed(client, [("想教AI产品", "intent")])
    r = client.post("/api/brief")
    assert r.status_code == 423
    assert "闸门" in r.json()["error"] or "完整度" in r.json()["error"]


def test_brief_blocked_without_picked_frame(client):
    """简报是「决定的记录」，没有决定就没有简报。"""
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    r = client.post("/api/brief")
    assert r.status_code == 423
    assert "框架" in r.json()["error"]


def test_brief_endpoint_persists_result(client):
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, ACTION_GAP, BRIEF])
    fid = client.post("/api/synth", json={"force": False}).json()["frames"][0]["id"]
    client.post(f"/api/pick/{fid}")

    r = client.post("/api/brief")
    assert r.status_code == 200
    body = r.json()
    assert body["brief"]["verdict"] == "选定反直觉清单方向。"
    assert "markdown" in body
    assert "决策简报" in body["markdown"]
    assert "证据清单" in body["markdown"], "简报必须带确定性附录"

    # 刷新后简报还在
    assert client.get("/api/state").json()["brief"] is not None


def test_reset_clears_brief(client):
    seed(client, [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")])
    use_llm([FRAMES, ACTION_GAP, BRIEF])
    fid = client.post("/api/synth", json={"force": False}).json()["frames"][0]["id"]
    client.post(f"/api/pick/{fid}")
    client.post("/api/brief")
    assert client.post("/api/reset").json()["brief"] is None


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
    """两次都给坏 JSON（complete_json 会带纠错提示重试一次）才算真失败。"""
    seed(client, [("想教AI产品", "intent")])
    use_llm(["这不是 JSON", "这依然不是 JSON"])
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
