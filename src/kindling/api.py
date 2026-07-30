"""FastAPI 后端。前端是单文件 HTML，由 / 直接返回。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .classify import FALLBACK, classify_entry
from .completeness import GATE_THRESHOLD
from .context import ContextEntry, EntryTooLong, EntryType, chunked_entries
from .credentials import (
    InvalidKey,
    clear_key,
    has_key,
    set_key,
)
from .credentials import status as key_status
from .export import export_pack
from .gap import ActionKind, Gap, detect_gap
from .harvest import BriefBlocked, check_brief_gates, compose_brief, render_brief
from .llm import API_URL, DEFAULT_MODEL, LLMClient, LLMError, OpenRouterClient, parse_json
from .models_catalog import fetch_models
from .moves import gap_to_move
from .observability import clear_logs, get_logs, log
from .reflux import reflux
from .settings import Settings, load_settings, save_settings
from .shelve import shelve_move
from .store import VALID_MODES, MoveAlreadyOpen, Store
from .synth import GateClosed, synthesize
from .topics import (
    TopicIndex,
    archive_topic,
    create_topic,
    list_topics,
    switch_topic,
    topic_state_path,
)

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


def snapshot_with_brief(st: Store) -> dict:
    """snapshot + 渲染好的简报 markdown。

    任何会切换/改变议题的端点都必须用这个 —— 否则前端拿不到
    brief_markdown，点「查看简报」会白白重新调一次 LLM（实测 16 秒）。
    """
    snap = st.snapshot()
    snap["brief_markdown"] = render_brief(st.brief, st) if st.brief else None
    return snap


# ---------------- 请求模型 ----------------


class AddEntryReq(BaseModel):
    text: str
    type: str | None = None


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


class CreateTopicReq(BaseModel):
    title: str = Field(default="", max_length=200)


class CorrectReq(BaseModel):
    text: str


class RetypeReq(BaseModel):
    type: str


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
    snap = snapshot_with_brief(st)
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


# ---------------- 多议题 ----------------


@app.get("/api/topics")
def api_list_topics(include_archived: bool = False):
    """议题列表 + 每个议题的进度摘要（给主页用）。"""
    idx = TopicIndex.load()
    out = []
    for t in list_topics(include_archived):
        st = Store(topic_state_path(t["id"])).load()
        g = st.completeness()
        out.append({
            **t,
            "mode": st.mode,
            "entries": len(st.entries),
            "percent": g["percent"],
            "gate_open": g["open"],
            "cycles": len(st.done_moves()),
            "has_brief": st.brief is not None,
            "has_frame": st.picked_frame() is not None,
            "current": t["id"] == idx.current_id,
        })
    return {"topics": out, "current_id": idx.current_id}


@app.post("/api/topics")
def api_create_topic(req: CreateTopicReq):
    """新建议题并切换过去。默认构思模式。

    响应键用 created 而不是 topic —— snapshot 里已有一个 topic 键
    （议题标题字符串），同名会被覆盖成字符串。
    """
    t = create_topic(req.title)
    st = load_store()
    # 议题标题同时作为 store.topic —— 它会进 prompt 和导出包的第一行
    if req.title.strip():
        st.topic = req.title.strip()
        st.save()
    return {"created": t, **snapshot_with_brief(st)}


@app.post("/api/topics/{topic_id}/switch")
def api_switch_topic(topic_id: str):
    """切换议题。可逆 —— 随时可以切回来，像切会话。"""
    try:
        switch_topic(topic_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return snapshot_with_brief(load_store())


@app.delete("/api/topics/{topic_id}")
def api_archive_topic(topic_id: str):
    """归档议题（不是删除）。文件永久保留 —— 上下文是资产。"""
    try:
        t = archive_topic(topic_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"archived": t}


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
    """收下一条碎片。不传 type 时自动分类 —— 输入时零结构决策。

    类型是内部计价单位，不该暴露成用户决策（README 第一个硬机制）。
    """
    st = load_store()

    if req.type:
        try:
            etype = EntryType(req.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"未知类型：{req.type}")
    else:
        try:
            etype = classify_entry(req.text, get_llm())
        except LLMError:
            # 没配 key 也要能录入 —— 录入不该被 LLM 可用性阻塞
            etype = FALLBACK
            log("classify", "无可用 LLM，类型回落 intent", level="warn")

    try:
        e = ContextEntry.new(req.text, etype)
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


@app.post("/api/correct")
def api_correct(req: CorrectReq):
    """纠正 LLM 的理解。权重 0，不参与完整度，但在 prompt 里置顶。

    存在的理由：没有这个通道时，用户只能把纠正语料堆进上下文，
    一条元指令会被当成关于世界的知识计分（实测被存成 evidence 权重 5.0）。
    """
    st = load_store()
    try:
        e = ContextEntry.new(req.text, EntryType.DIRECTIVE)
    except (EntryTooLong, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    st.add_entry(e)
    st.save()
    log("capture", f"纠偏：{e.text[:60]}")
    return {"entry": e.to_dict(), **st.snapshot()}


@app.patch("/api/entries/{entry_id}")
def api_retype_entry(entry_id: str, req: RetypeReq):
    """改一条上下文的类型。用于修正自动分类的误判和历史错类数据。"""
    st = load_store()
    e = next((x for x in st.entries if x.id == entry_id), None)
    if e is None:
        raise HTTPException(status_code=404, detail="找不到这条上下文")
    try:
        new_type = EntryType(req.type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"未知类型：{req.type}")
    old = e.type.value
    e.type = new_type
    st.save()
    log("capture", f"条目 {entry_id} 类型 {old} → {new_type.value}")
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
    """用户直接回答一个可凭记忆回答的缺口。长回答自动分块，绝不 400。

    question 落盘（而不是只进 log）：否则账本上是一堆孤立答案，
    LLM 也看不到自己当初问了什么，会重复提问。
    """
    st = load_store()
    try:
        created = chunked_entries(
            req.answer, EntryType(req.target_type), question=req.question
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    for e in created:
        st.add_entry(e)
    st.save()
    log(
        "capture",
        f"回答已记入 [{created[0].type.value}] × {len(created)} 条",
        detail={"question": req.question, "answer": req.answer},
    )
    return {"created": [e.to_dict() for e in created], **st.snapshot()}


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


@app.post("/api/shelve")
def api_shelve():
    """搁置当前动作：现在做不了，但问题要留着。

    与 drop（放弃）的区别：drop 是"这问题不重要"，shelve 是"这问题重要
    但现在做不了"。存在的理由：单 Move 硬锁 + 外部实验会把用户物理锁在
    产品外面（实测 6 小时断档）。
    """
    st = load_store()
    m = st.open_move()
    if m is None:
        raise HTTPException(status_code=409, detail="没有进行中的动作。")
    entry = shelve_move(m, st.entries)
    st.save()
    return {"shelved": entry.to_dict(), **st.snapshot()}


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
    # 旧的 picked 也要降级 —— 否则 picked_frame()（取第一个）会返回旧框架，
    # 且账本会累积多条「选定框架…」约束（实测污染过）。
    # 降级不是丢弃：它会出现在导出包的「已放弃的方向」里，探索过的路也是信息。
    for f in st.frames:
        if f.id != frame_id and f.status in ("candidate", "picked"):
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


# ---------------- L7 产出层 ----------------


@app.get("/api/export")
def api_export():
    """导出上下文包。确定性、零 LLM、任何时候都可用。

    这是产品的最终产出物之一：账本是资产，这个包是交付物。
    可以直接粘给任何 LLM，让它的回答「只对你成立」。
    """
    st = load_store()
    pack = export_pack(st)
    log("harvest", f"导出上下文包（{len(pack)} 字符）")
    return PlainTextResponse(
        pack,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="kindling-pack.md"'},
    )


@app.post("/api/brief")
def api_brief():
    """L7：把这一轮收束成决策简报。双闸：闸门开 + 已选框架。

    刻意在取 LLM 之前先查双闸：否则没配 key 时会返回 502，
    把「你还缺什么」这个有用的信息盖掉。
    """
    st = load_store()
    try:
        check_brief_gates(st)
        brief = compose_brief(st, get_llm())
    except BriefBlocked as exc:
        return JSONResponse(
            status_code=423, content={"error": str(exc), "kind": "brief"}
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    st.brief = brief
    st.save()
    return {
        "brief": brief,
        # 键名与 /api/state 保持一致，前端只认一个键
        "brief_markdown": render_brief(brief, st),
        **st.snapshot(),
    }


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
