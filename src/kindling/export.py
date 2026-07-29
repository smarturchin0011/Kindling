"""L7 产出层 —— 上下文包导出。

账本是资产，资产不等于交付物。这个模块把一轮飞轮的全部所得
变成能带走的东西：一个任何 LLM 都能直接消费的上下文包。

刻意零 LLM：导出是对已有事实的重排，不是生成。
确定性 = 可测、零成本、永远可用、绝不编造。

产品论证的闭环：用户最初的判断是「强 LLM + 足够上下文就能整理出
结构」—— 这个包就是那个「足够上下文」的物理形态。Kindling 不和
下游 LLM 竞争回答，它制造让回答「只对你成立」的原料。
"""
from __future__ import annotations

from .context import ContextEntry, EntryType, TYPE_LABELS_ZH

MODE_LABELS = {
    "explore": "构思（方案尚未实现，证据来自记忆与判断）",
    "validate": "验证（已可运行，证据来自真实反馈）",
}

FOOTER = """---
## 给阅读此包的 LLM 的使用要求

1. 回答必须只建立在上述上下文之上，给出「只对这个用户成立」的内容，
   不要输出对所有人都成立的泛泛框架。
2. 「未决问题」是已知的知识缺口：不要擅自编造答案填补；
   涉及时明确标注「此项未决」。
3. 「纠偏指令」是用户对 AI 工作方式的要求，优先级最高，必须遵守。
4. 「已放弃的方向」是用户探索后主动排除的 —— 除非有新证据，不要重新提议。"""


def _qa_line(e: ContextEntry) -> str:
    if e.question:
        return f"- 问：{e.question}\n  答：{e.text}"
    return f"- {e.text}"


def _section(title: str, lines: list[str]) -> list[str]:
    return [f"## {title}", *lines, ""] if lines else []


def export_pack(store) -> str:
    """把整个 Store 渲染成 markdown 上下文包。价值降序排列。"""
    by_type: dict[EntryType, list[ContextEntry]] = {}
    for e in store.entries:
        by_type.setdefault(e.type, []).append(e)

    out: list[str] = [
        f"# 议题：{store.topic or '未命名议题'}",
        f"阶段：{MODE_LABELS.get(getattr(store, 'mode', 'explore'), MODE_LABELS['explore'])}",
        "",
    ]

    # 1. 纠偏指令 —— 注意力最高，必须最先被看到
    out += _section(
        "纠偏指令（用户对 AI 的要求，最高优先级）",
        [f"- {e.text}" for e in by_type.get(EntryType.DIRECTIVE, [])],
    )

    # 2. 已选方向及其代价
    picked = store.picked_frame()
    if picked:
        out += _section(
            "已选定的方向",
            [
                f"「{picked.name}」：{picked.thesis}",
                f"- 优化：{picked.optimizes_for}",
                f"- 自愿承担的代价：{picked.sacrifices}",
            ],
        )

    # 3. 证据（带问答对） → 约束 → 事实 → 意图 → 偏好
    for t in (
        EntryType.EVIDENCE,
        EntryType.CONSTRAINT,
        EntryType.FACT,
        EntryType.INTENT,
        EntryType.PREFERENCE,
    ):
        group = by_type.get(t, [])
        out += _section(
            f"{TYPE_LABELS_ZH[t]}（{len(group)} 条）",
            [_qa_line(e) for e in group],
        )

    # 4. 已放弃的方向（探索过的路也是信息）
    rejected = [f for f in store.frames if f.status in ("superseded", "expired")]
    out += _section(
        "已放弃的方向",
        [f"- 「{f.name}」：{f.thesis}（代价：{f.sacrifices}）" for f in rejected],
    )

    # 5. 未决问题：UNKNOWN 条目 + 仍 open 的 Move
    open_qs = [f"- {e.question or e.text}" for e in by_type.get(EntryType.UNKNOWN, [])]
    om = store.open_move()
    if om is not None:
        open_qs.append(f"- （进行中）{om.question or om.description}")
    out += _section("未决问题（已知缺口，不要擅自填补）", open_qs)

    out.append(FOOTER)
    return "\n".join(out)
