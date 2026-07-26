"""可观测性 —— 用户明确要求能看到产品内部运行的 log。

一个进程内 ring buffer。每条记录带阶段、耗时、以及送给 LLM 的完整
prompt 和原始响应。前端轮询 /api/logs 渲染。

设计理由：这不是调试设施，是产品功能。用户要验证的假设是
"LLM 提问比我自己写 prompt 更能产出上下文" —— 他必须能看见
LLM 到底收到了什么、算出了什么，否则无法判断。
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

MAX_LOGS = 400
_counter = itertools.count(1)
_lock = threading.Lock()
_logs: deque["LogRecord"] = deque(maxlen=MAX_LOGS)


@dataclass
class LogRecord:
    seq: int
    ts: str
    level: str          # info | llm | warn | error
    stage: str          # capture | gap | narrow | reflux | synth | pick | store | http
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def log(
    stage: str,
    message: str,
    level: str = "info",
    detail: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> LogRecord:
    rec = LogRecord(
        seq=next(_counter),
        ts=datetime.now(timezone.utc).isoformat(),
        level=level,
        stage=stage,
        message=message,
        detail=detail or {},
        duration_ms=duration_ms,
    )
    with _lock:
        _logs.append(rec)
    return rec


def get_logs(since: int = 0) -> list[dict]:
    with _lock:
        return [r.to_dict() for r in _logs if r.seq > since]


def clear_logs() -> None:
    with _lock:
        _logs.clear()


class timer:
    """with timer() as t: ...  然后读 t.ms"""

    def __enter__(self) -> "timer":
        self._t0 = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = int((time.perf_counter() - self._t0) * 1000)
