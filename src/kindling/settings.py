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

# 预设清单：给你一键切换的常用对比组。
# 不是白名单 —— 自定义输入任何 OpenRouter 模型 id 都接受。
MODEL_PRESETS: list[dict[str, str]] = [
    {"id": "anthropic/claude-sonnet-4.5", "label": "Claude Sonnet 4.5", "note": "默认。指令遵循强，JSON 稳"},
    {"id": "anthropic/claude-opus-4.1", "label": "Claude Opus 4.1", "note": "最强推理，缺口问题更尖锐，慢且贵"},
    {"id": "openai/gpt-5", "label": "GPT-5", "note": "对比基线"},
    {"id": "google/gemini-2.5-pro", "label": "Gemini 2.5 Pro", "note": "长上下文便宜"},
    {"id": "deepseek/deepseek-chat-v3.1", "label": "DeepSeek V3.1", "note": "极便宜，中文好，适合刷轮次"},
    {"id": "x-ai/grok-4", "label": "Grok 4", "note": "风格更直接，可能问得更冒犯"},
    {"id": "qwen/qwen3-235b-a22b", "label": "Qwen3 235B", "note": "中文母语级"},
    {"id": "z-ai/glm-4.6", "label": "GLM-4.6", "note": "中文强，便宜"},
]

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
