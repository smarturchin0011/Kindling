"""L7 产出层 —— 决策简报。

上下文包（export.py）是原料，简报是判断。两者刻意分开：
原料确定性导出，判断由 LLM 收束但有两道闸：
  闸 1：闸门必须已开（证据不足时的简报 = 漂亮空话 = 麻醉剂）
  闸 2：必须已选框架（简报是「决定的记录」，没有决定就没有简报）

这两道闸在机制上保证简报只出现在环的末端 —— 它是出口凭证，
不是行动替代品。这也是它区别于「起步阶段的漂亮框架」的关键。
"""
from __future__ import annotations

from .context import EntryType, render_context
from .llm import LLMClient, complete_json
from .observability import log

SYSTEM = """你把用户一轮已经完成的上下文积累收束成一份「决策简报」。
你不是在创造新内容 —— 你只能使用给你的上下文，像一个严格的书记员。

铁律：
1. verdict 必须是一段能独立成立的结论：选了什么方向、为什么、代价是什么。
2. rationale 里每一个论断都必须能对应到某条具体证据或约束，
   用（证据：原文前10字…）的内联形式标注出处。无出处的论断不允许出现。
3. risks 必须如实：已知失效场景、未验证的假设、选框架时自愿牺牲的东西。
4. next_probes 是 2-3 个将来值得验证的具体问题（是问题，不是行动指派）。
5. 不要鼓励，不要客套。简报是给未来的自己和其他 AI 看的。
6. 用中文。JSON 字符串内不要使用双引号，强调用「」。

只输出 JSON，不要任何其他文字：
{"title":"","verdict":"","rationale":"","risks":"","next_probes":["",""]}"""


class BriefBlocked(RuntimeError):
    """双闸未过。message 会直接展示给用户。"""


def check_brief_gates(store) -> None:
    """双闸检查。独立成函数：API 层要在取 LLM 之前先查，
    否则没配 key 时 502 会盖掉「你还缺什么」这个有用信息。
    """
    gate = store.completeness()
    if not gate.get("open"):
        raise BriefBlocked(
            "闸门未开，完整度不足以支撑简报 —— 现在收束只会得到漂亮空话。还缺："
            + "；".join(gate.get("missing", []))
        )
    if store.picked_frame() is None:
        raise BriefBlocked("还没有选定框架。简报是「决定的记录」，先做选择。")


def compose_brief(store, llm: LLMClient) -> dict:
    """收束成简报。双闸：闸门已开 + 已选框架。"""
    check_brief_gates(store)
    picked = store.picked_frame()

    user = (
        f"议题：{store.topic or '未命名议题'}\n"
        f"选定的框架：「{picked.name}」：{picked.thesis}\n"
        f"它优化：{picked.optimizes_for}\n"
        f"自愿牺牲：{picked.sacrifices}\n\n"
        f"全部上下文：\n{render_context(store.entries)}\n\n"
        "收束成简报。"
    )
    spec = complete_json(llm, system=SYSTEM, user=user, stage="harvest")

    brief = {
        "title": str(spec.get("title", "")).strip() or (store.topic or "决策简报"),
        "verdict": str(spec.get("verdict", "")).strip(),
        "rationale": str(spec.get("rationale", "")).strip(),
        "risks": str(spec.get("risks", "")).strip(),
        "next_probes": [
            str(p).strip() for p in spec.get("next_probes", []) if str(p).strip()
        ],
        "frame_id": picked.id,
    }
    if not brief["verdict"]:
        raise ValueError("模型没有给出 verdict")
    log("harvest", f"简报已生成：{brief['title']}", detail=brief)
    return brief


def render_brief(brief: dict, store) -> str:
    """简报 markdown = LLM 的判断 + 确定性附录（证据清单、未决问题）。

    附录不经 LLM：它直接来自账本，保证简报里的事实部分不可能被编造。
    """
    picked = store.picked_frame()
    lines = [
        f"# {brief['title']}",
        "",
        "## 结论",
        brief["verdict"],
        "",
        "## 依据",
        brief["rationale"],
        "",
        "## 风险与代价",
        brief["risks"],
    ]
    if picked:
        lines += ["", f"（选定框架「{picked.name}」，自愿牺牲：{picked.sacrifices}）"]
    if brief["next_probes"]:
        lines += ["", "## 下一步值得验证的问题"]
        lines += [f"- {p}" for p in brief["next_probes"]]

    # 确定性附录 —— 不经 LLM，直接来自账本
    ev = [e for e in store.entries if e.type is EntryType.EVIDENCE]
    if ev:
        lines += ["", "---", "## 附：证据清单"]
        for e in ev:
            if e.question:
                lines.append(f"- 问：{e.question}\n  答：{e.text}")
            else:
                lines.append(f"- {e.text}")

    unknowns = [e for e in store.entries if e.type is EntryType.UNKNOWN]
    if unknowns:
        lines += ["", "## 附：未决问题"]
        lines += [f"- {e.question or e.text}" for e in unknowns]
    return "\n".join(lines)
