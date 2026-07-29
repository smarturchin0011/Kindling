"""L7 产出层测试：上下文包导出（确定性、零 LLM）+ 决策简报（双闸）。"""
import json as _json

import pytest

from kindling.context import ContextEntry, EntryType
from kindling.export import export_pack
from kindling.store import Store
from kindling.synth import Frame
from kdfakes import FakeLLM


def _store(tmp_path) -> Store:
    st = Store(tmp_path / "s.json", topic="音色参考方案")
    st.mode = "explore"
    st.entries = [
        ContextEntry.new("不要再问上线后才能验证的事", EntryType.DIRECTIVE),
        ContextEntry.new(
            "上传音色与人设差异大则完全不生效",
            EntryType.EVIDENCE,
            source="action",
            question="差异大时系统怎么反应？",
        ),
        ContextEntry.new("没有人设匹配度检测", EntryType.CONSTRAINT),
        ContextEntry.new(
            "未决：文案怎么写",
            EntryType.UNKNOWN,
            question="三个选项的文案各写一句是什么？",
        ),
    ]
    st.frames = [
        Frame(
            id="f1",
            name="默认生成兜底",
            thesis="生成设为默认",
            optimizes_for="成功率",
            sacrifices="上传仍可能失效",
            created_at="2026-01-01T00:00:00+00:00",
            status="picked",
        ),
        Frame(
            id="f2",
            name="风险拦截优先",
            thesis="加检测环节",
            optimizes_for="一致性",
            sacrifices="流程变长",
            created_at="2026-01-01T00:00:00+00:00",
            status="superseded",
        ),
    ]
    return st


# ---------- 上下文包：确定性导出 ----------


def test_pack_contains_all_sections_in_value_order(tmp_path):
    out = export_pack(_store(tmp_path))
    # 议题与阶段
    assert "音色参考方案" in out
    assert "构思" in out
    # 纠偏必须先于证据出现（注意力顺序）
    assert out.index("不要再问上线后才能验证的事") < out.index("上传音色与人设差异大")
    # 证据必须带问答对
    assert "差异大时系统怎么反应？" in out
    # 已选方向带代价
    assert "默认生成兜底" in out and "上传仍可能失效" in out
    # 落选框架也要在（探索过的路也是信息）
    assert "风险拦截优先" in out
    # 未决问题清单
    assert "三个选项的文案各写一句是什么？" in out
    # 给下游 LLM 的使用要求
    assert "使用要求" in out


def test_pack_works_on_empty_store(tmp_path):
    out = export_pack(Store(tmp_path / "empty.json"))
    assert "议题" in out  # 不崩，给出最小可用输出


def test_pack_is_deterministic(tmp_path):
    """零 LLM 意味着同样的输入必然给同样的输出。"""
    st = _store(tmp_path)
    assert export_pack(st) == export_pack(st)


def test_pack_includes_open_move_as_pending(tmp_path):
    from kindling.moves import Move

    st = _store(tmp_path)
    st.moves.append(
        Move(
            id="mov_o",
            description="写三个选项的文案",
            est_minutes=5,
            retrieves_type=EntryType.FACT,
            retrieves_why="w",
            created_at="2026-01-01T00:00:00+00:00",
            question="按钮文案怎么措辞？",
        )
    )
    out = export_pack(st)
    assert "按钮文案怎么措辞？" in out
    assert "进行中" in out


# ---------- 决策简报：LLM 收束，双闸 ----------


def _brief_json() -> str:
    return _json.dumps(
        {
            "title": "音色参考方案简报",
            "verdict": "以「默认生成兜底」为方向：描述生成设为默认。",
            "rationale": "描述生成一致性良好（证据：上传音色与人设差…）",
            "risks": "上传选项对人设差异大的用户仍无保护",
            "next_probes": ["文案如何措辞", "默认值的选中率"],
        },
        ensure_ascii=False,
    )


def _rich_store(tmp_path):
    """闸门开 + 已选框架的 store。"""
    st = _store(tmp_path)
    for i in range(4):
        st.entries.append(
            ContextEntry.new(f"补充证据 {i}", EntryType.EVIDENCE, source="action")
        )
    return st


def test_brief_requires_picked_frame(tmp_path, monkeypatch):
    from kindling.harvest import BriefBlocked, compose_brief

    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "s.json"))
    st = _rich_store(tmp_path)
    for f in st.frames:
        f.status = "superseded"
    with pytest.raises(BriefBlocked, match="框架"):
        compose_brief(st, FakeLLM([_brief_json()]))


def test_brief_requires_open_gate(tmp_path, monkeypatch):
    from kindling.harvest import BriefBlocked, compose_brief

    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "s.json"))
    st = _store(tmp_path)
    st.entries = [e for e in st.entries if e.type is not EntryType.EVIDENCE]
    with pytest.raises(BriefBlocked, match="闸门|完整度"):
        compose_brief(st, FakeLLM([_brief_json()]))


def test_brief_composes_and_renders_markdown(tmp_path, monkeypatch):
    from kindling.harvest import compose_brief, render_brief

    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "s.json"))
    st = _rich_store(tmp_path)
    brief = compose_brief(st, FakeLLM([_brief_json()]))
    assert brief["title"] == "音色参考方案简报"
    assert brief["frame_id"] == "f1"

    md = render_brief(brief, st)
    assert "默认生成兜底" in md          # verdict / 框架
    assert "证据：" in md                # 出处标注保留
    assert "未决" in md                  # 确定性附录：未决问题
    assert "文案如何措辞" in md          # next_probes


def test_brief_prompt_carries_full_context(tmp_path, monkeypatch):
    """简报必须建立在全部上下文之上，包括纠偏指令。"""
    from kindling.harvest import compose_brief

    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "s.json"))
    st = _rich_store(tmp_path)
    llm = FakeLLM([_brief_json()])
    compose_brief(st, llm)
    sent = llm.calls[0]["user"]
    assert "默认生成兜底" in sent
    assert "不要再问上线后才能验证的事" in sent


def test_brief_rejects_empty_verdict(tmp_path, monkeypatch):
    from kindling.harvest import compose_brief

    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "s.json"))
    st = _rich_store(tmp_path)
    bad = _json.dumps({"title": "t", "verdict": "", "rationale": "r"}, ensure_ascii=False)
    with pytest.raises(ValueError, match="verdict"):
        compose_brief(st, FakeLLM([bad]))
