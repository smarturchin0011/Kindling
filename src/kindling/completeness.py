"""完整度评分 + 闸门。

刻意用可解释的加权饱和函数，而不是让 LLM 打分：
  1. 确定性 —— 可测试
  2. 可解释 —— 能告诉用户"还缺 2 条证据"
  3. LLM 打分会漂移，无法做闸门
"""
from __future__ import annotations

from collections import Counter

from .context import ContextEntry, EntryType

GATE_THRESHOLD = 0.60
MIN_EVIDENCE = 2
MIN_CONSTRAINTS = 1

# 饱和值刻意与结构最小值对齐：
#   MIN_EVIDENCE * 5.0 + MIN_CONSTRAINTS * 3.0 = 13 分
#   13 / 20 = 65% > GATE_THRESHOLD(60%)
# 也就是"满足结构最小值"和"分数过阈值"这两个条件必须同时成立，
# 否则闸门会给出自相矛盾的提示（说你缺证据，补齐了却还是过不了）。
# tests/test_domain.py::test_gate_criteria_are_consistent 锁住这个不变量。
SATURATION = 20.0


def score(entries: list[ContextEntry]) -> float:
    """加权饱和评分。证据主导，因为它是唯一靠想不出来的类型。"""
    total = sum(e.weight for e in entries)
    return min(total / SATURATION, 1.0)


def counts(entries: list[ContextEntry]) -> Counter:
    return Counter(e.type for e in entries)


def gate_status(
    entries: list[ContextEntry],
    threshold: float | None = None,
    min_evidence: int | None = None,
    min_constraints: int | None = None,
) -> dict:
    """闸门状态 + 可向用户解释的缺口清单。

    阈值可由运行时设置覆盖（settings.json）。不传则用模块默认值，
    这样纯域层测试不必依赖设置文件。
    """
    thr = GATE_THRESHOLD if threshold is None else float(threshold)
    need_ev = MIN_EVIDENCE if min_evidence is None else int(min_evidence)
    need_con = MIN_CONSTRAINTS if min_constraints is None else int(min_constraints)

    c = counts(entries)
    s = score(entries)
    missing: list[str] = []

    n_ev = c[EntryType.EVIDENCE]
    if n_ev < need_ev:
        missing.append(f"至少 {need_ev} 条证据（现在 {n_ev} 条）")

    n_con = c[EntryType.CONSTRAINT]
    if n_con < need_con:
        missing.append(f"至少 {need_con} 条约束（现在 {n_con} 条）")

    if s < thr:
        missing.append(f"完整度需达到 {thr:.0%}（现在 {s:.0%}）")

    return {
        "score": round(s, 4),
        "percent": round(s * 100),
        "open": not missing,
        "missing": missing,
        "threshold": thr,
        "counts": {k.value: v for k, v in c.items()},
        "total_entries": len(entries),
        # 分数已封顶且闸门已开 —— 继续追问不会再有增益。
        # UI 应把主行动引导到「生成框架」而不是「再问一个缺口」。
        # 实测缺陷：打到 100% 后仍被连问 5 轮，用户只能手动喊停。
        "saturated": (not missing) and s >= 1.0,
    }


def bar(s: float, width: int = 10) -> str:
    filled = round(s * width)
    return "▓" * filled + "░" * (width - filled)
