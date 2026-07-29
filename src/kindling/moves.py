"""L5 收窄层 —— Move。

Move ≠ 任务。Move = 一个只能通过行动才能填补的上下文缺口。
它携带 retrieves_type / retrieves_why，让用户看见这个动作服务于什么。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .context import EntryType, TYPE_LABELS_ZH, new_id, now_iso
from .gap import Gap

MAX_MOVE_MINUTES = 5


@dataclass
class Move:
    id: str
    description: str
    est_minutes: int
    retrieves_type: EntryType
    retrieves_why: str
    created_at: str
    status: str = "open"      # open | done | abandoned | shelved
    artifact: str = ""
    frame_id: str = ""
    question: str = ""        # 这个动作要回答的问题（进入回流的证据）

    @property
    def retrieves_label_zh(self) -> str:
        return TYPE_LABELS_ZH[self.retrieves_type]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieves_type"] = self.retrieves_type.value
        d["retrieves_label_zh"] = self.retrieves_label_zh
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Move":
        return cls(
            id=d["id"],
            description=d["description"],
            est_minutes=d["est_minutes"],
            retrieves_type=EntryType(d["retrieves_type"]),
            retrieves_why=d["retrieves_why"],
            created_at=d["created_at"],
            status=d.get("status", "open"),
            artifact=d.get("artifact", ""),
            frame_id=d.get("frame_id", ""),
            question=d.get("question", ""),
        )


def gap_to_move(gap: Gap, frame_id: str = "") -> Move:
    if gap.answerable_from_memory:
        raise ValueError("这个缺口能凭记忆回答，直接问就行，不要生成 Move")
    if not gap.suggested_action:
        raise ValueError("缺口没有 suggested_action，无法生成 Move")
    est = gap.est_minutes or MAX_MOVE_MINUTES
    if est > MAX_MOVE_MINUTES:
        raise ValueError(f"动作预估 {est} 分钟，上限 {MAX_MOVE_MINUTES}。切小。")
    return Move(
        id=new_id("mov"),
        description=gap.suggested_action,
        est_minutes=est,
        retrieves_type=gap.target_type,
        retrieves_why=gap.why_critical,
        created_at=now_iso(),
        frame_id=frame_id,
        question=gap.question,
    )
