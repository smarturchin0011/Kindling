"""流程层测试：缺口检测 → Move → 回流 → 闸门 → 综合。"""
import json

import pytest

from kindling.completeness import score
from kindling.context import ContextEntry, EntryType
from kindling.gap import Gap, detect_gap
from kindling.llm import LLMError, parse_json, strip_fence
from kindling.moves import gap_to_move
from kindling.reflux import reflux
from kindling.store import MoveAlreadyOpen, Store
from kindling.synth import GateClosed, expire_stale_frames, synthesize
from kdfakes import FakeLLM
from kdresponses import ACTION_GAP, ANSWERABLE_GAP, FRAMES


def _ctx():
    return [ContextEntry.new("想教别人 AI 产品知识", EntryType.INTENT)]


def _rich():
    return (
        [ContextEntry.new(f"真实事故 {i}", EntryType.EVIDENCE) for i in range(3)]
        + [ContextEntry.new(f"约束 {i}", EntryType.CONSTRAINT) for i in range(2)]
        + [ContextEntry.new("想教 AI 产品", EntryType.INTENT)]
    )


def _action_gap(est=4):
    return Gap(
        question="PM 做错了什么决定？",
        target_type=EntryType.EVIDENCE,
        answerable_from_memory=False,
        why_critical="没证据框架就是抽象的",
        suggested_action="翻记录找 1 个具体决定，写 2 句话",
        est_minutes=est,
    )


# ---------- JSON 解析健壮性 ----------


def test_strip_fence_handles_json_block():
    assert strip_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_fence('```\n{"a":1}\n```') == '{"a":1}'
    assert strip_fence('{"a":1}') == '{"a":1}'


def test_parse_json_recovers_from_prose_wrapper():
    """模型爱在 JSON 前后加话。必须能救回来。"""
    assert parse_json('好的，这是结果：\n{"question":"x"}\n希望有帮助') == {"question": "x"}


def test_parse_json_raises_on_garbage():
    with pytest.raises(LLMError):
        parse_json("完全不是 JSON 的一段话")


def test_complete_json_retries_once_on_bad_json():
    """实测崩溃：模型在 JSON 字符串内写了中文直引号，兜底正则抓到的仍是坏 JSON。

    修复策略是 prompt 层 + 一次带纠错提示的重试，不做正则修补
    （正则修补会引入静默的错误解析）。
    """
    from kindling.llm import complete_json

    bad = '{"question":"那些"不生效"的案例","target_type":"fact"}'
    good = '{"question":"那些「不生效」的案例","target_type":"fact"}'
    llm = FakeLLM([bad, good])

    out = complete_json(llm, system="s", user="u", stage="gap")
    assert out["question"] == "那些「不生效」的案例"
    assert len(llm.calls) == 2
    assert "不要使用双引号" in llm.calls[1]["user"]


def test_complete_json_raises_after_second_failure():
    from kindling.llm import complete_json

    llm = FakeLLM(["还是不是 JSON", "依然不是 JSON"])
    with pytest.raises(LLMError):
        complete_json(llm, system="s", user="u", stage="gap")
    assert len(llm.calls) == 2


# ---------- 缺口检测 ----------


def test_detect_answerable_gap():
    g = detect_gap("教AI产品", _ctx(), FakeLLM([ANSWERABLE_GAP]))
    assert g.answerable_from_memory is True
    assert g.target_type is EntryType.CONSTRAINT
    assert g.suggested_action == ""


def test_detect_action_gap_carries_action():
    g = detect_gap("教AI产品", _ctx(), FakeLLM([ACTION_GAP]))
    assert g.answerable_from_memory is False
    assert g.target_type is EntryType.EVIDENCE
    assert "翻最近的工作记录" in g.suggested_action
    assert g.est_minutes == 4


def test_unanswerable_without_action_degrades_gracefully():
    """答不上来却不给动作 = 死路。降级为可答，而不是崩溃。"""
    bad = json.dumps(
        {
            "question": "q",
            "target_type": "evidence",
            "answerable_from_memory": False,
            "why_critical": "w",
        },
        ensure_ascii=False,
    )
    g = detect_gap("t", _ctx(), FakeLLM([bad]))
    assert g.answerable_from_memory is True


def test_oversized_estimate_is_clamped():
    big = json.dumps(
        {
            "question": "q",
            "target_type": "evidence",
            "answerable_from_memory": False,
            "why_critical": "w",
            "suggested_action": "做点大事",
            "est_minutes": 45,
        },
        ensure_ascii=False,
    )
    g = detect_gap("t", _ctx(), FakeLLM([big]))
    assert g.est_minutes == 5


def test_unknown_target_type_falls_back():
    weird = json.dumps(
        {
            "question": "q",
            "target_type": "vibes",
            "answerable_from_memory": True,
            "why_critical": "w",
        },
        ensure_ascii=False,
    )
    assert detect_gap("t", _ctx(), FakeLLM([weird])).target_type is EntryType.FACT


def test_prompt_includes_typed_context_and_evidence_count():
    llm = FakeLLM([ANSWERABLE_GAP])
    detect_gap("教AI产品", _ctx(), llm)
    sent = llm.calls[0]["user"]
    assert "intent" in sent
    assert "想教别人 AI 产品知识" in sent
    assert "evidence 0 条" in sent


def test_gate_missing_is_fed_into_prompt():
    """闸门联动：不告诉检测器还缺什么类型，它会一直问同一类，
    用户看着 85% 却打不开闸门，不知道该补什么。"""
    from kindling.completeness import gate_status

    ctx = [ContextEntry.new(f"事故{i}", EntryType.EVIDENCE) for i in range(4)]
    gate = gate_status(ctx)
    assert not gate["open"]          # 缺约束
    llm = FakeLLM([ANSWERABLE_GAP])
    detect_gap("t", ctx, llm, gate=gate)
    sent = llm.calls[0]["user"]
    assert "闸门还缺" in sent
    assert "约束" in sent


def test_open_gate_adds_no_pressure_block():
    from kindling.completeness import gate_status

    ctx = _rich()
    llm = FakeLLM([ANSWERABLE_GAP])
    detect_gap("t", ctx, llm, gate=gate_status(ctx))
    assert "闸门还缺" not in llm.calls[0]["user"]


# ---------- Move ----------


def test_move_records_what_it_retrieves():
    m = gap_to_move(_action_gap())
    assert m.retrieves_type is EntryType.EVIDENCE
    assert "没证据" in m.retrieves_why
    assert m.est_minutes == 4
    assert m.status == "open"
    assert m.retrieves_label_zh == "证据"


def test_answerable_gap_cannot_become_move():
    g = Gap("受众是谁", EntryType.CONSTRAINT, True, "w")
    with pytest.raises(ValueError, match="能凭记忆回答"):
        gap_to_move(g)


def test_oversized_gap_rejected_at_move():
    with pytest.raises(ValueError, match="上限 5"):
        gap_to_move(_action_gap(est=30))


def test_move_roundtrip():
    m = gap_to_move(_action_gap())
    back = type(m).from_dict(m.to_dict())
    assert back.retrieves_type is EntryType.EVIDENCE
    assert back.description == m.description


# ---------- 回流（飞轮） ----------


def test_reflux_creates_typed_entry_linked_to_move():
    m = gap_to_move(_action_gap())
    ctx: list[ContextEntry] = []
    reflux(m, "PM 要求 100% 准确率，项目卡了三周", ctx)
    assert m.status == "done"
    assert len(ctx) == 1
    assert ctx[0].type is EntryType.EVIDENCE
    assert ctx[0].source == "action"
    assert ctx[0].move_id == m.id


def test_reflux_raises_completeness():
    """飞轮的核心断言：行动让上下文变厚。"""
    ctx = [ContextEntry.new("想教AI产品", EntryType.INTENT)]
    before = score(ctx)
    reflux(gap_to_move(_action_gap()), "一条真实证据", ctx)
    assert score(ctx) > before


def test_reflux_chunks_long_artifact():
    ctx: list[ContextEntry] = []
    reflux(gap_to_move(_action_gap()), "x" * 700, ctx)
    assert len(ctx) == 3
    assert all(len(e.text) <= 280 for e in ctx)


def test_reflux_rejects_empty_artifact():
    with pytest.raises(ValueError):
        reflux(gap_to_move(_action_gap()), "   ", [])


def test_reflux_rejects_closed_move():
    m = gap_to_move(_action_gap())
    m.status = "done"
    with pytest.raises(RuntimeError):
        reflux(m, "something", [])


# ---------- 综合 + 闸门 ----------


def test_gate_blocks_thin_context():
    thin = [ContextEntry.new("想教AI产品", EntryType.INTENT)]
    with pytest.raises(GateClosed) as exc:
        synthesize("教AI产品", thin, FakeLLM([FRAMES]))
    assert "证据" in str(exc.value) or "证据" in str(exc.value.gate["missing"])
    assert exc.value.gate["open"] is False


def test_force_bypasses_gate_but_is_flagged():
    thin = [ContextEntry.new("想教AI产品", EntryType.INTENT)]
    frames = synthesize("教AI产品", thin, FakeLLM([FRAMES]), force=True)
    assert len(frames) == 2
    assert all(f.forced is True for f in frames)


def test_rich_context_passes_gate():
    frames = synthesize("教AI产品", _rich(), FakeLLM([FRAMES]))
    assert len(frames) == 2
    assert all(f.sacrifices for f in frames)
    assert all(f.forced is False for f in frames)


def test_single_frame_rejected():
    """只给一个框架会破坏择优机制。"""
    one = json.dumps({"frames": [json.loads(FRAMES)["frames"][0]]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="至少 2"):
        synthesize("t", _rich(), FakeLLM([one]))


def test_frame_missing_sacrifices_rejected():
    """没有 tradeoff 的框架是假框架。"""
    bad = json.loads(FRAMES)
    bad["frames"][0]["sacrifices"] = ""
    with pytest.raises(ValueError, match="sacrifices"):
        synthesize("t", _rich(), FakeLLM([json.dumps(bad, ensure_ascii=False)]))


def test_hallucinated_grounding_ids_filtered():
    """模型编造的条目 id 必须被丢掉，否则"基于证据 N 条"会撒谎。"""
    spec = json.loads(FRAMES)
    ctx = _rich()
    spec["frames"][0]["grounded_in_entries"] = [ctx[0].id, "ctx_fake123"]
    frames = synthesize(
        "t", ctx, FakeLLM([json.dumps(spec, ensure_ascii=False)])
    )
    assert frames[0].grounded_in_entries == [ctx[0].id]
    assert frames[0].grounded_count == 1


def test_synth_prompt_includes_entry_ids():
    ctx = _rich()
    llm = FakeLLM([FRAMES])
    synthesize("t", ctx, llm)
    assert ctx[0].id in llm.calls[0]["user"]


def test_expire_stale_frames():
    from datetime import datetime, timedelta, timezone

    frames = synthesize("t", _rich(), FakeLLM([FRAMES]))
    frames[0].created_at = (
        datetime.now(timezone.utc) - timedelta(hours=100)
    ).isoformat()
    assert expire_stale_frames(frames) == 1
    assert frames[0].status == "expired"
    assert frames[1].status == "candidate"


def test_picked_frames_never_expire():
    from datetime import datetime, timedelta, timezone

    frames = synthesize("t", _rich(), FakeLLM([FRAMES]))
    frames[0].status = "picked"
    frames[0].created_at = (
        datetime.now(timezone.utc) - timedelta(hours=500)
    ).isoformat()
    assert expire_stale_frames(frames) == 0


# ---------- Store ----------


def test_store_roundtrip_preserves_types(tmp_path):
    st = Store(tmp_path / "s.json", topic="教AI产品")
    st.add_entry(ContextEntry.new("受众是PM", EntryType.CONSTRAINT))
    st.add_move(gap_to_move(_action_gap()))
    st.frames.extend(synthesize("t", _rich(), FakeLLM([FRAMES])))
    st.save()

    st2 = Store(tmp_path / "s.json").load()
    assert st2.topic == "教AI产品"
    assert st2.entries[0].type is EntryType.CONSTRAINT
    assert st2.moves[0].retrieves_type is EntryType.EVIDENCE
    assert st2.open_move() is not None
    assert len(st2.candidate_frames()) == 2


def test_single_open_move_enforced(tmp_path):
    st = Store(tmp_path / "s.json")
    st.add_move(gap_to_move(_action_gap()))
    with pytest.raises(MoveAlreadyOpen):
        st.add_move(gap_to_move(_action_gap()))


def test_new_move_allowed_after_done(tmp_path):
    st = Store(tmp_path / "s.json")
    m = st.add_move(gap_to_move(_action_gap()))
    reflux(m, "拿到了证据", st.entries)
    st.add_move(gap_to_move(_action_gap()))
    assert len(st.moves) == 2
    assert len(st.done_moves()) == 1


def test_store_missing_file_is_empty(tmp_path):
    st = Store(tmp_path / "nope.json").load()
    assert st.entries == [] and st.moves == [] and st.frames == []


def test_store_survives_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json at all", encoding="utf-8")
    st = Store(p).load()
    assert st.entries == []


def test_snapshot_shape(tmp_path):
    st = Store(tmp_path / "s.json")
    snap = st.snapshot()
    for key in ("gate", "entries", "open_move", "candidate_frames", "cycles"):
        assert key in snap
    assert snap["gate"]["open"] is False


# ---------- 议题模式：构思阶段不该被逼做实验 ----------


def test_gap_carries_action_kind():
    """动作性质必须是结构化字段，后端才能确定性地约束它。"""
    from kindling.gap import ActionKind

    g = Gap(
        question="q",
        target_type=EntryType.EVIDENCE,
        answerable_from_memory=False,
        why_critical="w",
        suggested_action="a",
        est_minutes=4,
        action_kind=ActionKind.EXTERNAL,
    )
    assert g.action_kind is ActionKind.EXTERNAL
    assert g.to_dict()["action_kind"] == "external"


def test_action_kind_defaults_to_recall():
    from kindling.gap import ActionKind

    g = Gap(
        question="q",
        target_type=EntryType.FACT,
        answerable_from_memory=True,
        why_critical="w",
    )
    assert g.action_kind is ActionKind.RECALL


def test_explore_mode_rejects_external_action():
    """构思阶段最严重的产品缺陷：功能还没上线，却被要求去跑一遍验证。

    explore 模式下 external 动作必须被降级，而不是生成一个做不到的 Move。
    """
    llm = FakeLLM([json.dumps({
        "question": "拿一个失败组合走一遍完整流程，记录 agent 有没有报错",
        "target_type": "evidence",
        "answerable_from_memory": False,
        "why_critical": "决定方案骨架",
        "suggested_action": "去跑一遍线上流程并记录系统反应",
        "est_minutes": 5,
        "action_kind": "external",
    }, ensure_ascii=False)])

    gap = detect_gap("测试议题", _ctx(), llm, mode="explore")
    assert gap.answerable_from_memory is True, "explore 模式不该产出外部依赖的 Move"
    assert gap.suggested_action == ""
    assert gap.est_minutes == 0


def test_validate_mode_keeps_external_action():
    """验证模式下 external 是合法的 —— 那才是真的要去做实验。"""
    from kindling.gap import ActionKind

    llm = FakeLLM([json.dumps({
        "question": "跑一次真实调用，结果如何？",
        "target_type": "evidence",
        "answerable_from_memory": False,
        "why_critical": "没跑过就是空中楼阁",
        "suggested_action": "调用一次并把结果写成 3 行",
        "est_minutes": 5,
        "action_kind": "external",
    }, ensure_ascii=False)])

    gap = detect_gap("测试议题", _ctx(), llm, mode="validate")
    assert gap.answerable_from_memory is False
    assert gap.action_kind is ActionKind.EXTERNAL
    assert gap.suggested_action != ""


def test_explore_mode_keeps_internal_action():
    """explore 模式并非禁止一切动作 —— 认知动作（回忆/判断/书写）必须保留。"""
    from kindling.gap import ActionKind

    llm = FakeLLM([json.dumps({
        "question": "你过去见过的最接近这个场景的一次失败是什么？",
        "target_type": "evidence",
        "answerable_from_memory": False,
        "why_critical": "需要一个具体事例而不是抽象判断",
        "suggested_action": "写下那一次的具体经过，3 行",
        "est_minutes": 4,
        "action_kind": "recall",
    }, ensure_ascii=False)])

    gap = detect_gap("测试议题", _ctx(), llm, mode="explore")
    assert gap.answerable_from_memory is False
    assert gap.action_kind is ActionKind.RECALL
    assert gap.suggested_action != ""


def test_mode_prompt_is_injected():
    """模式必须真的进到 system prompt 里，否则约束只有后端一半。"""
    payload = json.dumps({
        "question": "q", "target_type": "fact",
        "answerable_from_memory": True, "why_critical": "w",
        "suggested_action": "", "est_minutes": 0, "action_kind": "recall",
    }, ensure_ascii=False)

    llm = FakeLLM([payload])
    detect_gap("t", _ctx(), llm, mode="explore")
    assert "构思模式" in llm.calls[0]["system"]
    assert "绝对禁止" in llm.calls[0]["system"]

    llm2 = FakeLLM([payload])
    detect_gap("t", _ctx(), llm2, mode="validate")
    assert "验证模式" in llm2.calls[0]["system"]


# ---------- question 贯穿 gap → move → reflux ----------


def test_move_carries_question_into_evidence():
    """行动回流的证据必须带上它当初要回答的问题，否则账本上是孤立答案。"""
    from kindling.gap import ActionKind

    gap = Gap(
        question="上传音色和人设差异大时系统怎么反应？",
        target_type=EntryType.EVIDENCE,
        answerable_from_memory=False,
        why_critical="决定方案骨架",
        suggested_action="写下 3 行观察",
        est_minutes=4,
        action_kind=ActionKind.RECALL,
    )
    m = gap_to_move(gap)
    assert m.question == "上传音色和人设差异大时系统怎么反应？"

    entries: list[ContextEntry] = []
    created = reflux(m, "完全不生效，而且没有任何提示", entries)
    assert created[0].question == "上传音色和人设差异大时系统怎么反应？"


def test_move_question_backward_compatible():
    from kindling.moves import Move

    old = {
        "id": "mov_x",
        "description": "d",
        "est_minutes": 4,
        "retrieves_type": "evidence",
        "retrieves_why": "w",
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "done",
        "artifact": "a",
        "frame_id": "",
    }
    assert Move.from_dict(old).question == ""


# ---------- 搁置：单 Move 硬锁的泄压阀 ----------


def test_shelve_move_unblocks_and_registers_unknown(tmp_path):
    """实测缺陷：外部实验 Move 把用户锁在产品外 6 小时。

    搁置 = 状态转 shelved + 问题转为 UNKNOWN 条目（权重 0）。
    问题被记住（未决清单），但不再挡路。
    """
    from kindling.moves import Move
    from kindling.shelve import shelve_move

    st = Store(tmp_path / "s.json")
    m = Move(
        id="mov_x",
        description="去跑一遍线上流程",
        est_minutes=5,
        retrieves_type=EntryType.EVIDENCE,
        retrieves_why="w",
        created_at="2026-01-01T00:00:00+00:00",
        question="系统对失败组合怎么反应？",
    )
    st.moves.append(m)
    assert st.open_move() is not None

    entry = shelve_move(m, st.entries)

    assert m.status == "shelved"
    assert st.open_move() is None, "搁置后必须解除硬锁"
    assert entry.type is EntryType.UNKNOWN
    assert entry.weight == 0.0, "未决问题不能计分"
    assert entry.question == "系统对失败组合怎么反应？"
    assert entry.move_id == "mov_x"


def test_shelve_requires_open_move():
    from kindling.moves import Move
    from kindling.shelve import shelve_move

    m = Move(
        id="mov_y",
        description="d",
        est_minutes=5,
        retrieves_type=EntryType.EVIDENCE,
        retrieves_why="w",
        created_at="2026-01-01T00:00:00+00:00",
        status="done",
    )
    with pytest.raises(RuntimeError):
        shelve_move(m, [])


def test_shelved_move_frees_slot_for_new_move(tmp_path):
    """搁置后必须能加新 Move —— 这才是泄压阀的意义。"""
    from kindling.shelve import shelve_move

    st = Store(tmp_path / "s.json")
    m = st.add_move(gap_to_move(_action_gap()))
    shelve_move(m, st.entries)
    st.add_move(gap_to_move(_action_gap()))
    assert len(st.moves) == 2
    assert len(st.done_moves()) == 0, "搁置不是完成，不该算进飞轮圈数"


# ---------- 自动分类：输入时零结构决策 ----------


def test_classify_returns_entry_type():
    from kindling.classify import classify_entry

    llm = FakeLLM(['{"type":"constraint"}'])
    assert classify_entry("受众只能是产品经理", llm) is EntryType.CONSTRAINT


def test_classify_detects_directive():
    from kindling.classify import classify_entry

    llm = FakeLLM(['{"type":"directive"}'])
    assert classify_entry("你理解错了，别再问这个", llm) is EntryType.DIRECTIVE


def test_classify_falls_back_to_intent_not_evidence():
    """失败必须回落到最便宜的类型，否则自动分类会变成刷分漏洞。"""
    from kindling.classify import classify_entry

    llm = FakeLLM(["这不是 JSON", "还是不是 JSON"])
    assert classify_entry("随便一条", llm) is EntryType.INTENT


def test_classify_never_returns_evidence_for_user_typed_text():
    """用户手输的东西不可能是证据 —— 证据只能由行动回流产生。

    否则完整度可以靠打字刷出来，闸门失效。
    """
    from kindling.classify import classify_entry

    llm = FakeLLM(['{"type":"evidence"}'])
    got = classify_entry("我觉得应该这样", llm)
    assert got is not EntryType.EVIDENCE
    assert got is EntryType.INTENT
