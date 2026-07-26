"""L6 回流层 —— 飞轮的闭合点。

行动产出物 → 带类型的上下文条目（几乎总是 evidence）。
这是"上下文复利"的物理实现：每转一圈，上下文变厚一层。
"""
from __future__ import annotations

from .context import ContextEntry, MAX_ENTRY_CHARS
from .moves import Move
from .observability import log


def reflux(
    move: Move,
    artifact: str,
    entries: list[ContextEntry],
) -> list[ContextEntry]:
    if move.status != "open":
        raise RuntimeError(f"这个 Move 状态是 {move.status}，不是 open")
    text = artifact.strip()
    if not text:
        raise ValueError("产出物不能为空。真的什么都没拿到就用「放弃」。")

    move.status = "done"
    move.artifact = text

    created: list[ContextEntry] = []
    for i in range(0, len(text), MAX_ENTRY_CHARS):
        e = ContextEntry.new(
            text[i : i + MAX_ENTRY_CHARS],
            move.retrieves_type,
            source="action",
            move_id=move.id,
        )
        entries.append(e)
        created.append(e)

    log(
        "reflux",
        f"回流 {len(created)} 条 [{move.retrieves_type.value}] 进上下文",
        detail={"move_id": move.id, "artifact": text[:500]},
    )
    return created
