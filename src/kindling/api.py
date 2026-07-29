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
from .credentials import (
    InvalidKey,
    clear_key,
    has_key,
    set_key,
)
from .credentials import status as key_status
from .gap import ActionKind, Gap, detect_gap
from .llm import API_URL, DEFAULT_MODEL, LLMClient, LLMError, OpenRouterClient, parse_json
from .models_catalog import fetch_models
from .moves import gap_to_move
from .observability import clear_logs, get_logs, log
from .reflux import reflux
from .settings import Settings, load_settings, save_settings
from .store import VALID_MODES, MoveAlreadyOpen, Store
from .synth import GateClosed, synthesize

# 只加载项目根 .env（可选，主要给 CI / 自动化用）。
# 刻意不扫描 home 目录或其他应用的配置：隐式借用别处的凭据会让用户
# 不知道自己在消费哪个账号。日常使用请在设置面板粘贴 key —— 只存内存，不落盘。
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

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


class ModeReq(BaseModel):
    mode: str


class SynthReq(BaseModel):
    force: bool = False


class SettingsReq(BaseModel):
    model: str | None = None
    temperature: float | None = None
    gate_threshold: float | None = None
    min_evidence: int | None = None
    min_constraints: int | None = None


class KeyReq(BaseModel):
    api_key: str


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
    s = load_settings()
    snap["model"] = s.model
    snap["temperature"] = s.temperature
    snap["has_key"] = has_key()
    snap["key"] = key_status()
    return snap


@app.post("/api/topic")
def api_topic(req: TopicReq):
    st = load_store()
    st.topic = req.topic.strip()
    st.save()
    log("capture", f"议题设为：{st.topic or '(空)'}")
    return st.snapshot()


@app.post("/api/mode")
def api_mode(req: ModeReq):
    """切换议题模式。explore=构思(默认) / validate=验证。

    存在的理由：构思阶段的议题不该被要求去验证一个还不存在的东西。
    """
    m = req.mode.strip()
    if m not in VALID_MODES:
        raise HTTPException(
            status_code=400, detail=f"模式只能是 {' / '.join(VALID_MODES)}"
        )
    st = load_store()
    st.mode = m
    st.save()
    log("capture", f"议题模式切换为：{m}")
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

    gap = detect_gap(
        topic, st.entries, get_llm(), gate=st.completeness(), mode=st.mode
    )

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
            st.topic or "未命名议题",
            st.entries,
            get_llm(),
            force=req.force,
            gate=st.completeness(),
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
            + (
                "注意：当前是构思模式，这个动作只能是 write / judge / recall，"
                "绝不能要求他去跑系统或等上线。"
                if st.mode == "explore"
                else ""
            )
        ),
        mode=st.mode,
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
            action_kind=ActionKind.WRITE,
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


# ---------------- 设置 / 凭据 / 模型目录 ----------------


@app.get("/api/settings")
def api_get_settings():
    return {
        "settings": load_settings().to_dict(),
        "provider": "OpenRouter",
        "endpoint": API_URL,
        "default_model": DEFAULT_MODEL,
        "key": key_status(),
    }


@app.post("/api/key")
def api_set_key(req: KeyReq):
    """存入 API key。只进内存，不写任何文件，重启即失效。"""
    try:
        set_key(req.api_key)
    except InvalidKey as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"key": key_status()}


@app.delete("/api/key")
def api_clear_key():
    clear_key()
    return {"key": key_status()}


@app.get("/api/models")
def api_models(refresh: bool = False):
    """OpenRouter 真实模型列表（公开端点，不需要 key）。"""
    return fetch_models(force=refresh)


@app.post("/api/settings")
def api_set_settings(req: SettingsReq):
    cur = load_settings()
    merged = Settings(
        model=req.model if req.model is not None else cur.model,
        temperature=(
            req.temperature if req.temperature is not None else cur.temperature
        ),
        gate_threshold=(
            req.gate_threshold
            if req.gate_threshold is not None
            else cur.gate_threshold
        ),
        min_evidence=(
            req.min_evidence if req.min_evidence is not None else cur.min_evidence
        ),
        min_constraints=(
            req.min_constraints
            if req.min_constraints is not None
            else cur.min_constraints
        ),
    )
    saved = save_settings(merged)
    # 闸门阈值可能变了 —— 回传新快照，前端立即重算完整度环
    st = load_store()
    return {"settings": saved.to_dict(), **st.snapshot()}


@app.post("/api/settings/test")
def api_test_model(req: SettingsReq):
    """用指定模型发一个最小请求，确认它真的可用（而且会返回 JSON）。

    换模型最大的坑是某些模型不遵守"只输出 JSON"，跑到一半才炸。
    这个端点让你在正式用之前 5 秒内验出来。
    """
    model = (req.model or load_settings().model).strip()
    try:
        client = OpenRouterClient(model=model, temperature=0)
        raw = client.complete(
            system='你是一个测试端点。只输出 JSON，不要任何其他文字。',
            user='输出这个 JSON：{"ok":true,"say":"你好"}',
            stage="llm",
        )
    except LLMError as e:
        return JSONResponse(
            status_code=502, content={"ok": False, "model": model, "error": str(e)}
        )
    try:
        parsed = parse_json(raw, stage="llm")
        json_ok = bool(parsed.get("ok"))
    except LLMError:
        parsed, json_ok = None, False
    return {
        "ok": True,
        "model": model,
        "json_compliant": json_ok,
        "raw": raw[:400],
        "note": (
            "可用，且能稳定返回 JSON。"
            if json_ok
            else "能连通，但没有干净返回 JSON —— 用它跑缺口检测可能会不稳定。"
        ),
    }


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
