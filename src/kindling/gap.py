"""L3 缺口检测层 —— 产品的护城河。

交互方向被反转：不是用户提问 LLM 回答，而是 LLM 提问用户回答。
理由：用户不知道自己缺什么上下文，但 LLM 可以计算出来。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .context import ContextEntry, EntryType, render_context
from .llm import LLMClient, parse_json
from .observability import log

SYSTEM = """你是一个上下文缺口检测器。你服务的用户擅长宏观思考，
但因为追求完备的顶层设计而无法启动。

你的任务**不是**回答他、给建议、或输出框架。你的唯一任务是找出：
要给出一个「只对他成立、而不是对所有人都成立」的答案，你最缺哪一条信息。

铁律：
1. 只输出一个问题。不要列清单，不要一次问多个。
2. 这个问题必须是「不知道这条，任何框架都会是泛泛而谈」的级别。
   不要问客套问题，不要问他能自己查到的东西，不要问他已经说过的东西。
3. 判断他能否凭记忆一句话答出：
   - 能 → answerable_from_memory: true
   - 不能（需要去查、去试、去问别人、去看真实反应）→ false，
     并且**必须**给出一个 5 分钟内能做完、有具体产出物的动作。
4. 最有价值的缺口通常是 evidence 类型 —— 真实世界的反馈、具体的事故、
   真人的反应。这类几乎总是 answerable_from_memory: false。
   如果他的上下文里 evidence 数量为 0，优先找 evidence 缺口。
5. suggested_action 必须有可见的产出物（几行字、一个清单、一句结论）。
   "思考一下…"、"研究…"、"梳理…" 都不合格。
6. 不要鼓励，不要铺垫，不要总结他已经说过的话。直接问。
7. 用中文。question 要口语化、直接，像一个尖锐的朋友在问，不要客套。

只输出 JSON，不要任何其他文字：
{"question":"","target_type":"fact|constraint|intent|preference|evidence",
 "answerable_from_memory":true,"why_critical":"",
 "suggested_action":"","est_minutes":4}"""

MAX_MOVE_MINUTES = 5


@dataclass
class Gap:
    question: str
    target_type: EntryType
    answerable_from_memory: bool
    why_critical: str
    suggested_action: str = ""
    est_minutes: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_type"] = self.target_type.value
        return d


def detect_gap(
    topic: str,
    entries: list[ContextEntry],
    llm: LLMClient,
    extra_instruction: str = "",
    gate: dict | None = None,
) -> Gap:
    n_ev = sum(1 for e in entries if e.type is EntryType.EVIDENCE)
    user = (
        f"议题：{topic}\n\n"
        f"我目前掌握的上下文（按类型分组）：\n{render_context(entries)}\n\n"
        f"统计：共 {len(entries)} 条，其中 evidence {n_ev} 条。\n"
    )
    # 闸门联动：把"还差什么类型"告诉检测器，否则它会一直问同一类，
    # 用户看着 85% 却打不开闸门，完全不知道该补什么。
    if gate and not gate.get("open") and gate.get("missing"):
        user += (
            "\n【当前闸门还缺】\n"
            + "\n".join(f"  - {m}" for m in gate["missing"])
            + "\n请优先针对上面缺的类型提问。\n"
        )
    if extra_instruction:
        user += f"\n额外要求：{extra_instruction}\n"
    user += "\n找出最致命的那一个缺口。"

    raw = llm.complete(system=SYSTEM, user=user, stage="gap")
    spec = parse_json(raw, stage="gap")

    answerable = bool(spec.get("answerable_from_memory", True))
    action = str(spec.get("suggested_action") or "").strip()

    if not answerable and not action:
        log(
            "gap",
            "模型说答不上来却没给动作 —— 死路，降级为可答缺口",
            level="warn",
            detail={"spec": spec},
        )
        answerable = True

    est = int(spec.get("est_minutes") or 0)
    if not answerable and est > MAX_MOVE_MINUTES:
        log(
            "gap",
            f"模型给了 {est} 分钟的动作，夹到 {MAX_MOVE_MINUTES} 分钟",
            level="warn",
        )
        est = MAX_MOVE_MINUTES

    try:
        target = EntryType(spec.get("target_type", "fact"))
    except ValueError:
        log("gap", f"未知 target_type: {spec.get('target_type')}，回落到 fact", level="warn")
        target = EntryType.FACT

    gap = Gap(
        question=str(spec.get("question", "")).strip(),
        target_type=target,
        answerable_from_memory=answerable,
        why_critical=str(spec.get("why_critical", "")).strip(),
        suggested_action=action if not answerable else "",
        est_minutes=est if not answerable else 0,
    )
    if not gap.question:
        raise ValueError("模型没有给出问题")

    log(
        "gap",
        f"缺口: [{target.value}] {'可凭记忆答' if answerable else '需要行动'}",
        detail=gap.to_dict(),
    )
    return gap
