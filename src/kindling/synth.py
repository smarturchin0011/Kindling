"""L4 综合层 —— 带闸门的框架生成。

闸门是防止产品退化成 ChatGPT 的第一道机制：
完整度不足时拒绝生成框架，因为那时只能产出"对任何人都成立"的空框架，
而空框架会带来认知闭合的满足感，从而替代行动。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone

from .completeness import gate_status
from .context import ContextEntry, new_id, now_iso, render_context
from .llm import LLMClient, parse_json
from .observability import log

DEFAULT_TTL_HOURS = 72

SYSTEM = """你为一个擅长宏观思考、但因追求完备顶层设计而无法启动的用户
生成 2-3 个**互相竞争**的组织框架。

铁律：
1. 必须 2-3 个，绝不只给 1 个。他的强项是批判择优，弱项是从零生成。
   给一个方案会引发"这个不够好我要改"的无底洞。
2. 每个框架必须显式声明它优化什么、牺牲什么。没有 tradeoff 的框架是假框架。
   sacrifices 必须是真实的代价，不能写"几乎没有缺点"这类废话。
3. 框架之间必须是**不同的切分维度**，不是同一维度的措辞变体。
4. 每个框架必须标注它建立在哪些具体上下文条目的 id 上
   （grounded_in_entries）。优先建立在 evidence 类型的条目上。
   如果某个框架其实没有证据支撑，如实给空数组，不要编造 id。
5. 不要追求完整覆盖所有上下文。允许一个框架只解释一部分。
6. 不要鼓励，不要总结，不要提建议。只输出框架。
7. 用中文。name 要短（≤10字）且有辨识度。

只输出 JSON，不要任何其他文字：
{"frames":[{"name":"","thesis":"","optimizes_for":"","sacrifices":"",
            "grounded_in_entries":[]}]}"""


class GateClosed(RuntimeError):
    def __init__(self, message: str, gate: dict):
        super().__init__(message)
        self.gate = gate


@dataclass
class Frame:
    id: str
    name: str
    thesis: str
    optimizes_for: str
    sacrifices: str
    grounded_in_entries: list[str] = field(default_factory=list)
    created_at: str = ""
    status: str = "candidate"   # candidate | picked | superseded | expired
    ttl_hours: int = DEFAULT_TTL_HOURS
    forced: bool = False

    @property
    def grounded_count(self) -> int:
        return len(self.grounded_in_entries)

    @property
    def expires_at(self) -> str:
        base = datetime.fromisoformat(self.created_at)
        return (base + timedelta(hours=self.ttl_hours)).isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grounded_count"] = self.grounded_count
        d["expires_at"] = self.expires_at
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            id=d["id"],
            name=d["name"],
            thesis=d["thesis"],
            optimizes_for=d["optimizes_for"],
            sacrifices=d["sacrifices"],
            grounded_in_entries=list(d.get("grounded_in_entries", [])),
            created_at=d.get("created_at", now_iso()),
            status=d.get("status", "candidate"),
            ttl_hours=d.get("ttl_hours", DEFAULT_TTL_HOURS),
            forced=bool(d.get("forced", False)),
        )


def synthesize(
    topic: str,
    entries: list[ContextEntry],
    llm: LLMClient,
    force: bool = False,
    gate: dict | None = None,
) -> list[Frame]:
    # gate 由调用方传入（携带运行时阈值）；缺省时回落到模块默认，
    # 保证纯域层测试无需设置文件。
    if gate is None:
        gate = gate_status(entries)
    if not gate["open"] and not force:
        log("synth", f"闸门拒绝：完整度 {gate['percent']}%", level="warn", detail=gate)
        raise GateClosed(
            f"完整度 {gate['percent']}%，低于 {gate['threshold']:.0%} 闸门。"
            f"现在生成框架，你会拿到一个对任何人都成立的空框架。",
            gate,
        )

    was_forced = bool(force and not gate["open"])
    if was_forced:
        log("synth", "用户强行越过闸门", level="warn", detail=gate)

    valid_ids = {e.id for e in entries}
    user = (
        f"议题：{topic}\n\n"
        f"上下文条目（带 id，evidence 类型最可信）：\n"
        + "\n".join(
            f"  [{e.id}] ({e.type.value}) {e.text}" for e in entries
        )
        + f"\n\n分组视图：\n{render_context(entries)}\n\n生成 2-3 个竞争框架。"
    )

    raw = llm.complete(system=SYSTEM, user=user, stage="synth")
    payload = parse_json(raw, stage="synth")
    specs = payload.get("frames", [])

    if len(specs) < 2:
        log("synth", f"模型只给了 {len(specs)} 个框架，拒绝", level="error")
        raise ValueError("需要至少 2 个竞争框架供你比较，模型只给了 1 个。请重新生成。")

    created_at = now_iso()
    out: list[Frame] = []
    for s in specs[:3]:
        for label in ("name", "thesis", "optimizes_for", "sacrifices"):
            if not str(s.get(label, "")).strip():
                raise ValueError(f"框架缺少必填字段 {label}（没有 tradeoff 的框架是假框架）")
        # 过滤模型编造的 id
        grounded = [i for i in s.get("grounded_in_entries", []) if i in valid_ids]
        out.append(
            Frame(
                id=new_id("frm"),
                name=str(s["name"]).strip(),
                thesis=str(s["thesis"]).strip(),
                optimizes_for=str(s["optimizes_for"]).strip(),
                sacrifices=str(s["sacrifices"]).strip(),
                grounded_in_entries=grounded,
                created_at=created_at,
                forced=was_forced,
            )
        )

    log(
        "synth",
        f"生成 {len(out)} 个竞争框架",
        detail={"frames": [f.name for f in out], "forced": was_forced},
    )
    return out


def expire_stale_frames(frames: list[Frame]) -> int:
    """候选框架过 TTL 自动过期 —— 禁止长期维护一个"总架构"。"""
    now = datetime.now(timezone.utc)
    n = 0
    for f in frames:
        if f.status != "candidate":
            continue
        try:
            created = datetime.fromisoformat(f.created_at)
        except (ValueError, TypeError):
            continue
        if now - created > timedelta(hours=f.ttl_hours):
            f.status = "expired"
            n += 1
    if n:
        log("synth", f"{n} 个候选框架超过 {DEFAULT_TTL_HOURS}h 已过期")
    return n
