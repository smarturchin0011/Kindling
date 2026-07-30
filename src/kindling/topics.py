"""多议题 —— 索引 + 每议题一个 state 文件。

为什么需要：之前全局只有一个 state.json，想讨论另一件事只能清空重来。
上下文是这个产品定义的资产，"换个话题就得毁掉资产"是不可接受的。

设计要点：
- current_id 存在索引里；store.state_path() 通过 current_state_path()
  解析路径 —— 所有既有端点零改动，这是此方案的关键收益。
- 切换 ≠ 归档。切换像切会话，随时可以切回来；只有归档才把议题
  从列表移走，而且**永不物理删除文件**。
- KINDLING_HOME 环境变量重定向根目录（测试隔离用），默认 ~/.kindling。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from .context import now_iso
from .observability import log

_lock = threading.Lock()


def kindling_home() -> Path:
    return Path(os.environ.get("KINDLING_HOME", str(Path.home() / ".kindling")))


def _index_path() -> Path:
    return kindling_home() / "topics.json"


def _topics_dir() -> Path:
    return kindling_home() / "topics"


def topic_state_path(topic_id: str) -> Path:
    return _topics_dir() / f"{topic_id}.json"


class TopicIndex:
    def __init__(self, current_id: str = "", topics: list[dict] | None = None):
        self.current_id = current_id
        self.topics = topics or []

    @classmethod
    def load(cls) -> "TopicIndex":
        p = _index_path()
        if not p.exists():
            return cls()
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log("topics", f"议题索引损坏，从空开始: {e}", level="error")
            return cls()
        return cls(d.get("current_id", ""), d.get("topics", []))

    def save(self) -> None:
        with _lock:
            p = _index_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {"current_id": self.current_id, "topics": self.topics},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(p)

    def find(self, topic_id: str) -> dict:
        t = next((x for x in self.topics if x["id"] == topic_id), None)
        if t is None:
            raise KeyError(f"找不到议题 {topic_id}")
        return t


def create_topic(title: str = "") -> dict:
    idx = TopicIndex.load()
    t = {
        "id": f"top_{uuid.uuid4().hex[:8]}",
        "title": title.strip() or "未命名议题",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "archived": False,
    }
    idx.topics.insert(0, t)
    idx.current_id = t["id"]
    idx.save()
    _topics_dir().mkdir(parents=True, exist_ok=True)
    log("topics", f"新建议题：{t['title']}")
    return t


def list_topics(include_archived: bool = False) -> list[dict]:
    idx = TopicIndex.load()
    return [t for t in idx.topics if include_archived or not t.get("archived")]


def switch_topic(topic_id: str) -> dict:
    """切换当前议题。可逆 —— 随时可以再切回来，像切会话。"""
    idx = TopicIndex.load()
    t = idx.find(topic_id)
    idx.current_id = topic_id
    idx.save()
    log("topics", f"切换到议题：{t['title']}")
    return t


def archive_topic(topic_id: str) -> dict:
    """归档议题。文件永不物理删除 —— 上下文是资产，误删不可恢复是最坏体验。"""
    idx = TopicIndex.load()
    t = idx.find(topic_id)
    t["archived"] = True
    if idx.current_id == topic_id:
        nxt = next((x for x in idx.topics if not x.get("archived")), None)
        idx.current_id = nxt["id"] if nxt else ""
    idx.save()
    log("topics", f"归档议题：{t['title']}（文件保留）")
    return t


def touch_topic(topic_id: str, title: str | None = None) -> None:
    """更新 updated_at / 标题。找不到议题时静默返回（KINDLING_STATE 模式下会发生）。"""
    idx = TopicIndex.load()
    try:
        t = idx.find(topic_id)
    except KeyError:
        return
    t["updated_at"] = now_iso()
    if title is not None and title.strip():
        t["title"] = title.strip()
    idx.save()


def migrate_legacy_state() -> None:
    """把旧的单文件 state.json 迁移为第一个议题。幂等，且无损（重命名而非复制）。"""
    legacy = kindling_home() / "state.json"
    if not legacy.exists() or _index_path().exists():
        return
    try:
        d = json.loads(legacy.read_text(encoding="utf-8"))
        title = d.get("topic") or "旧议题"
    except (json.JSONDecodeError, OSError):
        title = "旧议题"
    t = create_topic(title)
    legacy.replace(topic_state_path(t["id"]))
    log("topics", f"已迁移旧数据为议题「{title}」")


def current_state_path() -> Path:
    """当前议题的 state 文件路径。空环境或 current 失效时自动恢复。"""
    migrate_legacy_state()
    idx = TopicIndex.load()
    valid = any(
        t["id"] == idx.current_id and not t.get("archived") for t in idx.topics
    )
    if not idx.current_id or not valid:
        active = [t for t in idx.topics if not t.get("archived")]
        if active:
            switch_topic(active[0]["id"])
        else:
            create_topic("")
        idx = TopicIndex.load()
    _topics_dir().mkdir(parents=True, exist_ok=True)
    return topic_state_path(idx.current_id)
