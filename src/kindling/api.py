"""FastAPI 后端。前端是单文件 HTML，由 / 直接返回。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .completeness import GATE_THRESHOLD
from .context import ContextEntry, EntryTooLong, EntryType
from .gap import Gap, detect_gap
from .llm import LLMClient, LLMError, OpenRouterClient
from .moves import gap_to_move
from .observability import clear_logs, get_logs, log
from .reflux import reflux
from .store import MoveAlreadyOpen, Store
from .synth import GateClosed, synthesize

# .env 优先从项目根找，其次 Hermes 的 .env（用户 key 存在那里）
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
if not os.environ.get("OPENROUTER_API_KEY"):
    hermes_env = Path.home() / "AppData" / "Local" / "hermes" / ".env"
    if hermes_env.exists():
        load_dotenv(hermes_env)

WEB_DIR = _ROOT / "web"

app = FastAPI(title="Kindling", version="0.2.0")

# 测试时可注入 FakeLLM
_llm_override: LLMClient | None = None


def set_llm(client: LLMClient | None) -> None:
    global _llm_override
    _llm_override = client


def get_llm() -> LLMClient:
    if _llm_override is not None:
        return _llm_override
    return OpenRouterClient()


def load_store() -> Store:
    return Store().load()


# ---------------- 请求模型 ----------------


class AddEntryReq(BaseModel):
    text: str
    type: str = "intent"


class AnswerReq(BaseModel):
    question: str
    answer: str
    target_type: str = "fact"


class DoneReq(BaseModel):
    artifact: str


class TopicReq(BaseModel):
    topic: str = Field(default="", max_length=200)


class SynthReq(BaseModel):
    force: bool = False


# ---------------- 错误处理 ----------------


@app.exception_handler(LLMError)
async def _llm_error_handler(request, exc: LLMError):
    return JSONResponse(status_code=502, content={"error": str(exc), "kind": "llm"})


# ---------------- 路由 ----------------


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
def api_state():
    st = load_store()
    snap = st.snapshot()
    snap["model"] = os.environ.get("KINDLING_MODEL", "anthropic/claude-sonnet-4.5")
    snap["has_key"] = bool(os.environ.get("OPENROUTER_API_KEY"))
    return snap


@app.post("/api/topic")
def api_topic(req: TopicReq):
    st = load_store()
    st.topic = req.topic.strip()
    st.save()
    log("capture", f"议题设为：{st.topic or '(空)'}")
    return st.snapshot()


@app.post("/api/entries")
def api_add_entry(req: AddEntryReq):
    st = load_store()
    try:
        e = ContextEntry.new(req.text, EntryType(req.type))
    except EntryTooLong as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    st.add_entry(e)
    st.save()
    log("capture", f"收下一条 [{e.type.value}]：{e.text[:60]}")
    return {"entry": e.to_dict(), **st.snapshot()}


@app.delete("/api/entries/{entry_id}")
def api_delete_entry(entry_id: str):
    st = load_store()
    before = len(st.entries)
    st.entries = [e for e in st.entries if e.id != entry_id]
    if len(st.entries) == before:
        raise HTTPException(status_code=404, detail="找不到这条上下文")
    st.save()
    log("capture", f"删除条目 {entry_id}")
    return st.snapshot()


@app.post("/api/ask")
def api_ask():
    """L3 核心入口：让系统问你一个最致命的缺口。"""
    st = load_store()
    if st.open_move():
        raise HTTPException(
            status_code=409,
            detail=f"先完成手上的动作：{st.open_move().description}",
        )
    if not st.entries:
        raise HTTPException(
            status_code=400, detail="先扔一两条想法进来，系统才知道该问什么。"
        )

    picked = st.picked_frame()
    topic = st.topic or "未命名议题"
    if picked:
        topic = f"{topic} / 已选框架「{picked.name}」：{picked.thesis}"

    gap = detect_gap(topic, st.entries, get_llm(), gate=st.completeness())

    payload: dict = {"gap": gap.to_dict()}
    if not gap.answerable_from_memory:
        move = gap_to_move(gap, frame_id=picked.id if picked else "")
        try:
            st.add_move(move)
        except MoveAlreadyOpen as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        st.save()
        payload["move"] = move.to_dict()
    payload.update(st.snapshot())
    return payload


@app.post("/api/answer")
def api_answer(req: AnswerReq):
    """用户直接回答一个可凭记忆回答的缺口。"""
    st = load_store()
    try:
        e = ContextEntry.new(req.answer, EntryType(req.target_type))
    except (EntryTooLong, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    st.add_entry(e)
    st.save()
    log(
        "capture",
        f"回答已记入 [{e.type.value}]",
        detail={"question": req.question, "answer": req.answer},
    )
    return {"entry": e.to_dict(), **st.snapshot()}


@app.post("/api/done")
def api_done(req: DoneReq):
    """L6 回流：产出物作为证据进入上下文。飞轮闭合点。"""
    st = load_store()
    m = st.open_move()
    if m is None:
        raise HTTPException(status_code=409, detail="没有进行中的动作。")
    before = st.completeness()
    try:
        created = reflux(m, req.artifact, st.entries)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    st.save()
    after = st.completeness()
    log(
        "reflux",
        f"完整度 {before['percent']}% → {after['percent']}%",
        detail={"gained": after["score"] - before["score"]},
    )
    return {
        "created": [e.to_dict() for e in created],
        "before_percent": before["percent"],
        "after_percent": after["percent"],
        **st.snapshot(),
    }


@app.post("/api/drop")
def api_drop():
    st = load_store()
    m = st.open_move()
    if m is None:
        raise HTTPException(status_code=409, detail="没有进行中的动作。")
    m.status = "abandoned"
    st.save()
    log("narrow", f"放弃动作：{m.description}")
    return st.snapshot()


@app.post("/api/synth")
def api_synth(req: SynthReq):
    """L4 综合：带闸门的竞争框架生成。"""
    st = load_store()
    try:
        frames = synthesize(
            st.topic or "未命名议题", st.entries, get_llm(), force=req.force
        )
    except GateClosed as exc:
        return JSONResponse(
            status_code=423,
            content={"error": str(exc), "kind": "gate", "gate": exc.gate},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 新一批候选出现时，把上一批未选的候选降级
    for f in st.frames:
        if f.status == "candidate":
            f.status = "superseded"
    st.frames.extend(frames)
    st.save()
    return {"frames": [f.to_dict() for f in frames], **st.snapshot()}


@app.post("/api/pick/{frame_id}")
def api_pick(frame_id: str):
    """选框架 → 强制推导出一个动作。看框架不能是终点。"""
    st = load_store()
    picked = st.find_frame(frame_id)
    if picked is None:
        raise HTTPException(status_code=404, detail="找不到这个框架")
    if st.open_move():
        raise HTTPException(
            status_code=409, detail=f"先完成手上的动作：{st.open_move().description}"
        )

    picked.status = "picked"
    for f in st.frames:
        if f.id != frame_id and f.status == "candidate":
            f.status = "superseded"

    # 选择本身是一条上下文：它记录了你自愿承担的代价
    note = ContextEntry.new(
        f"选定框架「{picked.name}」，自愿牺牲：{picked.sacrifices}"[:280],
        EntryType.CONSTRAINT,
    )
    st.add_entry(note)
    log("pick", f"选定框架「{picked.name}」，其余降级")

    gap = detect_gap(
        f"{st.topic or '未命名议题'} / 已选框架「{picked.name}」：{picked.thesis}",
        st.entries,
        get_llm(),
        extra_instruction=(
            "用户刚刚选定了这个框架。现在必须给出一个需要动手做的动作"
            "（answerable_from_memory 必须为 false），让他立刻产出这个框架下的"
            "第一块真实内容。不要再问他能凭记忆回答的问题。"
        ),
    )

    # 硬保证：选完框架必须落到一个动作上
    if gap.answerable_from_memory:
        log("pick", "模型仍给了可答问题，强制转为动作", level="warn")
        gap = Gap(
            question=gap.question,
            target_type=gap.target_type,
            answerable_from_memory=False,
            why_critical=gap.why_critical or "选定框架后必须立刻产出第一块真实内容",
            suggested_action=f"用 3-5 句话写下你对这个问题的答案：{gap.question}",
            est_minutes=4,
        )

    move = gap_to_move(gap, frame_id=frame_id)
    st.add_move(move)
    st.save()
    return {
        "frame": picked.to_dict(),
        "move": move.to_dict(),
        "gap": gap.to_dict(),
        **st.snapshot(),
    }


@app.get("/api/logs")
def api_logs(since: int = 0):
    return {"logs": get_logs(since)}


@app.delete("/api/logs")
def api_clear_logs():
    clear_logs()
    return {"ok": True}


@app.post("/api/reset")
def api_reset():
    st = load_store()
    st.reset()
    st.save()
    clear_logs()
    return st.snapshot()


@app.get("/api/meta")
def api_meta():
    return {
        "gate_threshold": GATE_THRESHOLD,
        "entry_types": [
            {"value": t.value, "label": t.value} for t in EntryType
        ],
    }
