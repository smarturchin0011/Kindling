"""输入自动分类。

存在的理由：README 的第一个硬机制是「输入时零分类决策」，
但前端曾经摆着一个五选一的下拉框 —— 代码和设计文档自相矛盾。
类型是内部计价单位，不该暴露成用户决策。

刻意的约束：手输文本永远不会被判为 evidence。
证据是"唯一不能靠想产生的类型"，只能由行动回流（reflux）产生。
如果手输能被判成 evidence，完整度就可以靠打字刷出来，闸门失效。
"""
from __future__ import annotations

from .context import EntryType
from .llm import LLMClient, LLMError, complete_json
from .observability import log

# 手输可用的类型。刻意排除 EVIDENCE（只能由行动产生）与 UNKNOWN/DEBT（内部用）
USER_TYPES = (
    EntryType.CONSTRAINT,
    EntryType.FACT,
    EntryType.INTENT,
    EntryType.PREFERENCE,
    EntryType.DIRECTIVE,
)

# 回落到最便宜的类型（权重 1.0）。绝不回落 evidence（5.0），
# 否则网络抖动即得高分，自动分类会变成刷分漏洞。
FALLBACK = EntryType.INTENT

SYSTEM = """给一条上下文碎片分类。只输出 JSON，不要任何其他文字。

类型定义：
- constraint 限制解空间的东西：资源、期限、必须兼容什么、不能做什么
- fact       世界的可验证状态：现状是什么、系统当前有什么、数据是多少
- intent     他想要什么、为什么要：目标、动机、假设、想法
- preference 品味和风格倾向：喜欢哪种、倾向于什么
- directive  对你（AI）的纠正或元指令：别问这个、你理解错了、换个方向

判断要点：
1. 如果这句话是在纠正你的理解或指示你怎么工作 → directive
2. 如果它描述"不能/必须/只能/上限" → constraint
3. 如果它描述客观现状且可核实 → fact
4. 如果它表达愿望、目标或猜测 → intent
5. 只在明显是审美/风格偏好时用 preference

输出：{"type":"constraint|fact|intent|preference|directive"}"""


def classify_entry(text: str, llm: LLMClient) -> EntryType:
    """给手输文本定类型。任何失败都回落到 intent（最便宜，不可刷分）。"""
    try:
        spec = complete_json(llm, system=SYSTEM, user=text, stage="classify")
        t = EntryType(str(spec.get("type", "")).strip())
    except (LLMError, ValueError, KeyError) as e:
        log("classify", f"分类失败，回落 {FALLBACK.value}: {e}", level="warn")
        return FALLBACK

    if t not in USER_TYPES:
        log(
            "classify",
            f"模型给了不允许手输的类型 {t.value}，回落 {FALLBACK.value}",
            level="warn",
        )
        return FALLBACK

    log("classify", f"自动分类 → {t.value}", detail={"text": text[:100]})
    return t
