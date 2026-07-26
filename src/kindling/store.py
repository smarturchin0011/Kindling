"""持久化。单文件 JSON —— 刻意的技术选型。

不用数据库的理由：这个产品最大的风险是它自己变成一个完美主义陷阱。
30 条上下文用不上任何索引结构。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .completeness import gate_status
from .context import ContextEntry
from .moves import Move
from .observability import log
from .synth import Frame, expire_stale_frames

DEFAULT_PATH = Path.home() / ".kindling" / "state.json"
_write_lock = threading.Lock()


class MoveAlreadyOpen(RuntimeError):
    def __init__(self, current: Move):
        self.current = current
        super().__init__(
            f"已有一个进行中的动作：「{current.description}」\n"
            f"先完成它或放弃它。一次只做一件事。"
        )


def state_path() -> Path:
    return Path(os.environ.get("KINDLING_STATE", str(DEFAULT_PATH)))


class Store:
    def __init__(self, path: Path | str | None = None, topic: str = ""):
        self.path = Path(path) if path else state_path()
        self.topic = topic
        self.entries: list[ContextEntry] = []
        self.moves: list[Move] = []
        self.frames: list[Frame] = []

    # ---------- 持久化 ----------

    def load(self) -> "Store":
        if not self.path.exists():
            return self
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log("store", f"状态文件损坏，从空开始: {e}", level="error")
            return self
        self.topic = d.get("topic", "")
        self.entries = [ContextEntry.from_dict(x) for x in d.get("entries", [])]
        self.moves = [Move.from_dict(x) for x in d.get("moves", [])]
        self.frames = [Frame.from_dict(x) for x in d.get("frames", [])]
        expire_stale_frames(self.frames)
        return self

    def save(self) -> None:
        with _write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "topic": self.topic,
                        "entries": [e.to_dict() for e in self.entries],
                        "moves": [m.to_dict() for m in self.moves],
                        "frames": [f.to_dict() for f in self.frames],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)

    # ---------- 写 ----------

    def add_entry(self, e: ContextEntry) -> ContextEntry:
        self.entries.append(e)
        return e

    def add_move(self, m: Move) -> Move:
        cur = self.open_move()
        if cur is not None:
            raise MoveAlreadyOpen(cur)
        self.moves.append(m)
        return m

    # ---------- 读 ----------

    def open_move(self) -> Move | None:
        return next((m for m in self.moves if m.status == "open"), None)

    def find_move(self, move_id: str) -> Move | None:
        return next((m for m in self.moves if m.id == move_id), None)

    def find_frame(self, frame_id: str) -> Frame | None:
        return next((f for f in self.frames if f.id == frame_id), None)

    def candidate_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.status == "candidate"]

    def picked_frame(self) -> Frame | None:
        return next((f for f in self.frames if f.status == "picked"), None)

    def done_moves(self) -> list[Move]:
        return [m for m in self.moves if m.status == "done"]

    def completeness(self) -> dict:
        return gate_status(self.entries)

    def snapshot(self) -> dict:
        """给前端的完整状态。"""
        om = self.open_move()
        return {
            "topic": self.topic,
            "gate": self.completeness(),
            "entries": [e.to_dict() for e in self.entries],
            "open_move": om.to_dict() if om else None,
            "candidate_frames": [f.to_dict() for f in self.candidate_frames()],
            "picked_frame": (
                self.picked_frame().to_dict() if self.picked_frame() else None
            ),
            "done_moves": [m.to_dict() for m in self.done_moves()],
            "cycles": len(self.done_moves()),
            "state_file": str(self.path),
        }

    def reset(self) -> None:
        self.entries.clear()
        self.moves.clear()
        self.frames.clear()
        log("store", "状态已清空")
