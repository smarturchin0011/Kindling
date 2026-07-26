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


# 证据最贵：它是唯一不能靠"想"产生的类型。
WEIGHTS: dict[EntryType, float] = {
    EntryType.EVIDENCE: 5.0,
    EntryType.CONSTRAINT: 3.0,
    EntryType.FACT: 2.0,
    EntryType.INTENT: 1.0,
    EntryType.PREFERENCE: 0.5,
    EntryType.UNKNOWN: 0.0,
    EntryType.DEBT: 0.0,
}

TYPE_LABELS_ZH: dict[EntryType, str] = {
    EntryType.EVIDENCE: "证据",
    EntryType.CONSTRAINT: "约束",
    EntryType.FACT: "事实",
    EntryType.INTENT: "意图",
    EntryType.PREFERENCE: "偏好",
    EntryType.UNKNOWN: "未知",
    EntryType.DEBT: "债",
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

    @classmethod
    def new(
        cls,
        text: str,
        type: EntryType | str,
        source: str = "user",
        move_id: str = "",
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
        )


# 渲染顺序 = 权重降序。让 LLM 先看到最贵的证据，而不是枚举声明顺序。
RENDER_ORDER: list[EntryType] = sorted(
    EntryType, key=lambda t: WEIGHTS[t], reverse=True
)


def render_context(entries: list[ContextEntry]) -> str:
    """按类型分组渲染给 LLM 看。分组是刻意的 —— 让模型看见 evidence 的稀缺。"""
    if not entries:
        return "(空 —— 什么上下文都还没有)"
    lines: list[str] = []
    for t in RENDER_ORDER:
        group = [e for e in entries if e.type is t]
        if group:
            lines.append(f"[{t.value} / {TYPE_LABELS_ZH[t]}]")
            lines.extend(f"  - {e.text}" for e in group)
    return "\n".join(lines)
