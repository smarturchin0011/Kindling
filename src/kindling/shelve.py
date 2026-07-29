"""搁置 —— 单 Move 硬锁的泄压阀。

单 open Move 约束假设动作都能 5 分钟做完。但 validate 模式下的
外部动作可能需要离开电脑几小时 —— 期间用户被 409 锁在产品外面，
"收窄"变成了"监禁"（实测：一个要求实测的 Move 造成 6 小时断档，
期间 /api/ask 全部 409，用户无法推进任何事）。

搁置不是放弃：问题转成 UNKNOWN 条目（权重 0，不计分）留在账本，
会出现在导出的「未决问题」清单里。信息保留，道路解锁。

与 drop（放弃）的语义区别：
  drop    —— 这个问题不重要，不必再想
  shelve  —— 这个问题重要，但现在做不了
"""
from __future__ import annotations

from .context import ContextEntry, EntryType
from .moves import Move
from .observability import log


def shelve_move(move: Move, entries: list[ContextEntry]) -> ContextEntry:
    """把一个 open Move 搁置，并在账本登记一条未决问题。"""
    if move.status != "open":
        raise RuntimeError(f"这个 Move 状态是 {move.status}，不是 open，无法搁置")

    move.status = "shelved"
    e = ContextEntry.new(
        f"未决：{move.description}"[:280],
        EntryType.UNKNOWN,
        source="action",
        move_id=move.id,
        question=move.question,
    )
    entries.append(e)
    log(
        "narrow",
        f"搁置动作，登记为未决问题：{move.description[:60]}",
        detail={"move_id": move.id, "question": move.question},
    )
    return e
