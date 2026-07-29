"""L2 上下文层 —— 产品的真正资产。

带类型的上下文条目。类型决定权重，权重决定完整度。
证据（evidence）权重最高，因为它是唯一不能靠"想"产生的类型。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

MAX_ENTRY_CHARS = 280


class EntryType(str, Enum):
    FACT = "fact"                # 世界的可验证状态
    CONSTRAINT = "constraint"    # 什么限制了解空间
    INTENT = "intent"            # 你想要什么、为什么
    PREFERENCE = "preference"    # 品味、风格
    EVIDENCE = "evidence"        # 由行动产生 ★ 价值最高
    UNKNOWN = "unknown"          # 显式登记的缺口
    DEBT = "debt"                # 故意的简化 + 触发条件
    DIRECTIVE = "directive"      # 用户对 LLM 理解的纠偏。权重 0，但渲染置顶


# 证据最贵：它是唯一不能靠"想"产生的类型。
WEIGHTS: dict[EntryType, float] = {
    EntryType.EVIDENCE: 5.0,
    EntryType.CONSTRAINT: 3.0,
    EntryType.FACT: 2.0,
    EntryType.INTENT: 1.0,
    EntryType.PREFERENCE: 0.5,
    EntryType.UNKNOWN: 0.0,
    EntryType.DEBT: 0.0,
    EntryType.DIRECTIVE: 0.0,
}

TYPE_LABELS_ZH: dict[EntryType, str] = {
    EntryType.EVIDENCE: "证据",
    EntryType.CONSTRAINT: "约束",
    EntryType.FACT: "事实",
    EntryType.INTENT: "意图",
    EntryType.PREFERENCE: "偏好",
    EntryType.UNKNOWN: "未知",
    EntryType.DEBT: "债",
    EntryType.DIRECTIVE: "纠偏",
}


class EntryTooLong(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class ContextEntry:
    id: str
    text: str
    type: EntryType
    created_at: str
    source: str = "user"      # user | action | file
    move_id: str = ""         # 由哪个 Move 产生（evidence 专用）
    question: str = ""        # 这条上下文回答的是哪个问题（Q&A 配对）

    @classmethod
    def new(
        cls,
        text: str,
        type: EntryType | str,
        source: str = "user",
        move_id: str = "",
        question: str = "",
    ) -> "ContextEntry":
        text = text.strip()
        if not text:
            raise ValueError("上下文条目不能为空")
        if len(text) > MAX_ENTRY_CHARS:
            raise EntryTooLong(
                f"这条有 {len(text)} 字，上限 {MAX_ENTRY_CHARS}。拆成两条。"
            )
        return cls(
            id=new_id("ctx"),
            text=text,
            type=EntryType(type),
            created_at=now_iso(),
            source=source,
            move_id=move_id,
            question=question.strip(),
        )

    @property
    def is_gap(self) -> bool:
        return self.type is EntryType.UNKNOWN

    @property
    def weight(self) -> float:
        return WEIGHTS[self.type]

    @property
    def label_zh(self) -> str:
        return TYPE_LABELS_ZH[self.type]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type.value,
            "label_zh": self.label_zh,
            "created_at": self.created_at,
            "source": self.source,
            "move_id": self.move_id,
            "question": self.question,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContextEntry":
        return cls(
            id=d["id"],
            text=d["text"],
            type=EntryType(d["type"]),
            created_at=d["created_at"],
            source=d.get("source", "user"),
            move_id=d.get("move_id", ""),
            question=d.get("question", ""),
        )


# 渲染顺序 = 权重降序。让 LLM 先看到最贵的证据，而不是枚举声明顺序。
RENDER_ORDER: list[EntryType] = sorted(
    EntryType, key=lambda t: WEIGHTS[t], reverse=True
)


def _render_line(e: ContextEntry) -> str:
    """带 question 的条目渲染成问答配对，否则 LLM 看不到自己问过什么。"""
    if e.question:
        return f"  - 问：{e.question}\n    答：{e.text}"
    return f"  - {e.text}"


def render_context(entries: list[ContextEntry]) -> str:
    """按类型分组渲染给 LLM 看。分组是刻意的 —— 让模型看见 evidence 的稀缺。

    两个刻意的设计：
    1. directive 特判置顶：它权重为 0（不计分），但必须最先被看到。
       计价体系和注意力体系在这里刻意解耦。
    2. 带 question 的条目渲染成问答配对 —— 否则 LLM 看不到自己当初
       问了什么，会重复提问。
    """
    if not entries:
        return "(空 —— 什么上下文都还没有)"

    lines: list[str] = []

    directives = [e for e in entries if e.type is EntryType.DIRECTIVE]
    if directives:
        lines.append("[directive / 纠偏 —— 用户的纠正，最高优先级，必须遵守]")
        lines.extend(_render_line(e) for e in directives)
        lines.append("")

    for t in RENDER_ORDER:
        if t is EntryType.DIRECTIVE:
            continue
        group = [e for e in entries if e.type is t]
        if group:
            lines.append(f"[{t.value} / {TYPE_LABELS_ZH[t]}]")
            lines.extend(_render_line(e) for e in group)
    return "\n".join(lines)


def chunked_entries(
    text: str,
    type: EntryType | str,
    source: str = "user",
    move_id: str = "",
    question: str = "",
) -> list[ContextEntry]:
    """把任意长度文本切成 ≤280 字的条目序列。

    280 上限是对「采集」的降级承诺（防止第一步就写设计文档），
    不是对「回答」的惩罚 —— 认真回答一个尖锐问题时超长是正常的。
    """
    text = text.strip()
    if not text:
        raise ValueError("内容不能为空")
    return [
        ContextEntry.new(
            text[i : i + MAX_ENTRY_CHARS],
            type,
            source=source,
            move_id=move_id,
            question=question,
        )
        for i in range(0, len(text), MAX_ENTRY_CHARS)
    ]
