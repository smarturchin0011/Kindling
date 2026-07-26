"""运行时设置 —— 落盘持久化，改完立即生效，不需要重启服务。

为什么独立成一层：验证 prompt 效果时，换模型对比是核心动作。
把模型锁在环境变量里意味着每次对比都要重启，这会直接杀死验证节奏。

闸门阈值也放在这里可调 —— 如果你发现 60% 太严或太松，
调它比改代码快，而且调整记录会进 log。
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

from .observability import log

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

_lock = threading.Lock()


def settings_path() -> Path:
    from .store import state_path

    return state_path().with_name("settings.json")


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    gate_threshold: float = 0.60
    min_evidence: int = 2
    min_constraints: int = 1

    # ---------- 校验 ----------

    def normalized(self) -> "Settings":
        """夹到合法区间。前端可能传任何东西，后端不能信。"""
        model = (self.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return Settings(
            model=model,
            temperature=min(max(float(self.temperature), 0.0), 2.0),
            gate_threshold=min(max(float(self.gate_threshold), 0.05), 0.95),
            min_evidence=min(max(int(self.min_evidence), 0), 20),
            min_constraints=min(max(int(self.min_constraints), 0), 20),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_settings() -> Settings:
    p = settings_path()
    if not p.exists():
        # 首次启动：尊重环境变量，之后由设置文件接管
        return Settings(
            model=os.environ.get("KINDLING_MODEL", DEFAULT_MODEL)
        ).normalized()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log("store", f"设置文件损坏，用默认值: {e}", level="warn")
        return Settings().normalized()
    known = {f for f in Settings().to_dict()}
    return Settings(**{k: v for k, v in raw.items() if k in known}).normalized()


def save_settings(s: Settings) -> Settings:
    s = s.normalized()
    with _lock:
        p = settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(s.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(p)
    log(
        "store",
        f"设置已更新：模型={s.model} temp={s.temperature} 闸门={s.gate_threshold:.0%}",
        detail=s.to_dict(),
    )
    return s
