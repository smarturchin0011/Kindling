"""域模型测试：类型、权重、完整度闸门。"""
import pytest

from kindling.completeness import GATE_THRESHOLD, gate_status, score
from kindling.context import (
    ContextEntry,
    EntryTooLong,
    EntryType,
    WEIGHTS,
    render_context,
)


def _e(t: EntryType, n: int = 1):
    return [ContextEntry.new(f"内容 {t.value} {i}", t) for i in range(n)]


# ---------- ContextEntry ----------


def test_entry_carries_type():
    e = ContextEntry.new("受众主要是产品经理", EntryType.CONSTRAINT)
    assert e.type is EntryType.CONSTRAINT
    assert e.id.startswith("ctx_")
    assert e.label_zh == "约束"


def test_evidence_outweighs_everything():
    """证据最贵，因为它是唯一不能靠想产生的类型。"""
    assert WEIGHTS[EntryType.EVIDENCE] > WEIGHTS[EntryType.CONSTRAINT]
    assert WEIGHTS[EntryType.CONSTRAINT] > WEIGHTS[EntryType.FACT]
    assert WEIGHTS[EntryType.FACT] > WEIGHTS[EntryType.INTENT]
    assert WEIGHTS[EntryType.INTENT] > WEIGHTS[EntryType.PREFERENCE]


def test_entry_enforces_280_limit():
    with pytest.raises(EntryTooLong) as exc:
        ContextEntry.new("x" * 281, EntryType.FACT)
    assert "281" in str(exc.value)


def test_entry_accepts_exactly_280():
    assert len(ContextEntry.new("x" * 280, EntryType.FACT).text) == 280


def test_entry_rejects_empty():
    with pytest.raises(ValueError):
        ContextEntry.new("   ", EntryType.FACT)


def test_unknown_entry_marks_gap():
    assert ContextEntry.new("我怎么建立判断力的", EntryType.UNKNOWN).is_gap is True
    assert ContextEntry.new("受众是PM", EntryType.CONSTRAINT).is_gap is False


def test_entry_roundtrip():
    e = ContextEntry.new("测试", EntryType.EVIDENCE, source="action", move_id="mov_1")
    back = ContextEntry.from_dict(e.to_dict())
    assert back.type is EntryType.EVIDENCE
    assert back.source == "action"
    assert back.move_id == "mov_1"


def test_render_groups_by_type():
    out = render_context(_e(EntryType.EVIDENCE, 2) + _e(EntryType.INTENT, 1))
    assert "evidence" in out and "intent" in out
    assert out.index("evidence") < out.index("intent")  # 权重高的在前


def test_render_empty():
    assert "空" in render_context([])


# ---------- 完整度 ----------


def test_empty_context_scores_zero():
    assert score([]) == 0.0


def test_intent_only_stays_low():
    """只有意图没有证据 —— 这正是用户卡住的状态，必须评低分。"""
    assert score(_e(EntryType.INTENT, 5)) < 0.3


def test_evidence_dominates():
    assert score(_e(EntryType.EVIDENCE, 3)) > score(_e(EntryType.INTENT, 8))


def test_score_saturates_at_one():
    assert score(_e(EntryType.EVIDENCE, 50)) == 1.0


def test_gate_opens_with_evidence_and_constraints():
    ctx = (
        _e(EntryType.EVIDENCE, 2)
        + _e(EntryType.CONSTRAINT, 2)
        + _e(EntryType.INTENT, 2)
    )
    g = gate_status(ctx)
    assert g["score"] >= GATE_THRESHOLD
    assert g["open"] is True
    assert g["missing"] == []


def test_gate_reports_what_is_missing():
    g = gate_status(_e(EntryType.INTENT, 3))
    assert g["open"] is False
    assert any("证据" in m for m in g["missing"])
    assert any("约束" in m for m in g["missing"])


def test_gate_blocks_high_score_without_evidence():
    """分数够但没证据 —— 必须仍然关闸。这是防止空转的关键。"""
    ctx = _e(EntryType.CONSTRAINT, 10)
    g = gate_status(ctx)
    assert g["score"] == 1.0
    assert g["open"] is False
    assert any("证据" in m for m in g["missing"])


def test_gate_criteria_are_consistent():
    """不变量：满足结构最小值时，分数必须自动过阈值。

    否则闸门会自相矛盾 —— 提示"缺 2 条证据"，用户补齐了却还是打不开，
    完全摧毁信任。这条测试锁住 SATURATION 与 MIN_* 的对齐关系。
    """
    from kindling.completeness import MIN_CONSTRAINTS, MIN_EVIDENCE

    minimal = _e(EntryType.EVIDENCE, MIN_EVIDENCE) + _e(
        EntryType.CONSTRAINT, MIN_CONSTRAINTS
    )
    g = gate_status(minimal)
    assert g["score"] >= GATE_THRESHOLD, (
        f"结构最小值只得 {g['score']:.0%}，低于闸门 {GATE_THRESHOLD:.0%} —— "
        f"闸门条件自相矛盾，请调 SATURATION"
    )
    assert g["open"] is True
    assert g["missing"] == []
