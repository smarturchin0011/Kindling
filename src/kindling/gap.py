"""L3 缺口检测层 —— 产品的护城河。

交互方向被反转：不是用户提问 LLM 回答，而是 LLM 提问用户回答。
理由：用户不知道自己缺什么上下文，但 LLM 可以计算出来。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

from .context import ContextEntry, EntryType, render_context
from .llm import LLMClient, parse_json
from .observability import log

SYSTEM_BASE = """你是一个上下文缺口检测器。你服务的用户擅长宏观思考，
但因为追求完备的顶层设计而无法启动。

你的任务**不是**回答他、给建议、或输出框架。你的唯一任务是找出：
要给出一个「只对他成立、而不是对所有人都成立」的答案，你最缺哪一条信息。

铁律：
1. 只输出一个问题。不要列清单，不要一次问多个。
2. 这个问题必须是「不知道这条，任何框架都会是泛泛而谈」的级别。
   不要问客套问题，不要问他能自己查到的东西，不要问他已经说过的东西。
3. 判断他能否凭记忆一句话答出：
   - 能 → answerable_from_memory: true
   - 不能（需要动手产出点什么）→ false，并且**必须**给出一个 5 分钟内
     能做完、有具体产出物的动作。
4. 你必须声明这个动作的性质 action_kind：
   - "recall"   回忆一个已经发生过的具体事例（不需要新做什么）
   - "judge"    做一次取舍/排序/判断，产出一句结论
   - "write"    把模糊的东西写成具体文字（文案、清单、反例）
   - "external" 需要外部世界配合：调接口、跑系统、问别人、等功能上线
   answerable_from_memory 为 true 时，action_kind 填 "recall"。
5. suggested_action 必须有可见的产出物（几行字、一个清单、一句结论）。
   "思考一下…"、"研究…"、"梳理…" 都不合格。
6. 不要鼓励，不要铺垫，不要总结他已经说过的话。直接问。
7. 用中文。question 要口语化、直接，像一个尖锐的朋友在问，不要客套。
8. **JSON 字符串内绝对不要使用双引号**。需要强调或引用时用「」。

只输出 JSON，不要任何其他文字：
{"question":"","target_type":"fact|constraint|intent|preference|evidence",
 "answerable_from_memory":true,"why_critical":"",
 "suggested_action":"","est_minutes":4,
 "action_kind":"recall|judge|write|external"}"""

MODE_EXPLORE = """
【当前是构思模式】
他正在设计一个**还不存在**的东西。方案没上线、系统没实现、没有真实用户。

因此：
- **绝对禁止** action_kind 为 "external" 的动作。不要让他去跑流程、
  调接口、看日志、问技术、等上线后验证。他现在做不到，这只会让他卡住。
- 证据依然是最贵的，但在这个阶段证据**来自他的记忆，不是来自新实验**。
  他脑子里有大量没被榨出来的具体事例 —— 做过的类似项目、见过的失败、
  用户说过的原话。你的工作是把这些挖出来，而不是让他去创造新的。
- 优先用这三招逐步引导他的思路：
  (a) 要一个**具体事例**替换他的抽象判断（"最难的是没有直觉" → 哪一次？谁？）
  (b) 逼他做一次**取舍**（两个都想要的时候，哪个先牺牲？）
  (c) 要一个**反例或边界**（什么情况下你这个想法会失效？）
- 允许并鼓励 answerable_from_memory: true。构思阶段大部分缺口都该是追问。
"""

MODE_VALIDATE = """
【当前是验证模式】
这个东西已经能跑了，或者已经有真实用户在用。

因此：
- 最有价值的缺口是 evidence —— 真实世界的反馈、具体的事故、真人的反应。
  这类几乎总是 answerable_from_memory: false，action_kind 为 "external"。
- 如果他的上下文里 evidence 数量为 0，优先找 evidence 缺口。
- 不要问他能凭想象回答的问题。这个阶段的价值全在真实反馈里。
"""

MODE_PROMPTS = {"explore": MODE_EXPLORE, "validate": MODE_VALIDATE}


class ActionKind(str, Enum):
    """动作性质。决定 explore 模式下这个动作是否可被接受。

    recall  — 回忆一个已经发生过的具体事例（向内挖证据）
    judge   — 做一次取舍/判断/排序，产出一句结论
    write   — 把模糊表述写成具体的文字（文案、清单、反例）
    external— 需要外部世界配合：调接口、跑系统、问别人、等上线
    """

    RECALL = "recall"
    JUDGE = "judge"
    WRITE = "write"
    EXTERNAL = "external"


# explore（构思）模式下允许的动作性质 —— 全部不依赖外部世界
INTERNAL_KINDS = frozenset({ActionKind.RECALL, ActionKind.JUDGE, ActionKind.WRITE})

MAX_MOVE_MINUTES = 5


@dataclass
class Gap:
    question: str
    target_type: EntryType
    answerable_from_memory: bool
    why_critical: str
    suggested_action: str = ""
    est_minutes: int = 0
    action_kind: ActionKind = ActionKind.RECALL

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_type"] = self.target_type.value
        d["action_kind"] = self.action_kind.value
        return d


def detect_gap(
    topic: str,
    entries: list[ContextEntry],
    llm: LLMClient,
    extra_instruction: str = "",
    gate: dict | None = None,
    mode: str = "explore",
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

    system = SYSTEM_BASE + MODE_PROMPTS.get(mode, MODE_EXPLORE)
    raw = llm.complete(system=system, user=user, stage="gap")
    spec = parse_json(raw, stage="gap")

    answerable = bool(spec.get("answerable_from_memory", True))
    action = str(spec.get("suggested_action") or "").strip()

    try:
        kind = ActionKind(spec.get("action_kind", "recall"))
    except ValueError:
        log(
            "gap",
            f"未知 action_kind: {spec.get('action_kind')}，回落到 recall",
            level="warn",
        )
        kind = ActionKind.RECALL

    if not answerable and not action:
        log(
            "gap",
            "模型说答不上来却没给动作 —— 死路，降级为可答缺口",
            level="warn",
            detail={"spec": spec},
        )
        answerable = True

    # 模式约束：构思阶段的议题不该被要求去验证一个还不存在的东西。
    # 这里刻意用结构化的 action_kind 而不是关键词黑名单 —— 可测、不漏。
    if mode == "explore" and not answerable and kind not in INTERNAL_KINDS:
        log(
            "gap",
            f"explore 模式拒绝 {kind.value} 动作，降级为追问",
            level="warn",
            detail={"rejected_action": action, "question": spec.get("question")},
        )
        answerable = True
        action = ""
        kind = ActionKind.RECALL

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
        action_kind=kind,
    )
    if not gap.question:
        raise ValueError("模型没有给出问题")

    log(
        "gap",
        f"缺口: [{target.value}] {'可凭记忆答' if answerable else '需要行动'}",
        detail=gap.to_dict(),
    )
    return gap
