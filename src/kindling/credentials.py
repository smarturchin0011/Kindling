"""API key 的进程内存保管。

**刻意不落盘。** key 只活在这个进程的内存里：不写 .env、不进 settings.json、
不进 log、不出现在任何 API 响应里。服务一停就没了，下次启动必须重新输入。

为什么这样设计：
- 公开仓库 + 落盘 = 迟早有人把它 commit 上去
- 重启失效是明确的安全边界，不需要用户记得去清理什么
- 代价是每次启动多花 10 秒粘贴一次，换来"绝不可能泄漏到磁盘"

环境变量 OPENROUTER_API_KEY 仍然支持（适合 CI / 自动化），
但优先级低于运行时输入。
"""
from __future__ import annotations

import os
import re
import threading

from .observability import log

# 进程内存，不落盘。模块级私有变量，只能通过下面的函数访问。
_lock = threading.Lock()
_runtime_key: str = ""

KEY_RE = re.compile(r"^sk-or-v1-[A-Za-z0-9_-]{32,}$")

SETUP_HINT = (
    "未配置 API key。点右上角「设置」粘贴你的 OpenRouter key。\n"
    "key 在 https://openrouter.ai/keys 获取。\n"
    "出于安全考虑 key 只存在内存中，不写入任何文件 —— 重启服务需重新输入。"
)


class InvalidKey(ValueError):
    pass


def set_key(raw: str) -> None:
    """存入运行时 key。校验格式但绝不记录内容。"""
    key = (raw or "").strip()
    if not key:
        raise InvalidKey("key 不能为空")
    if not KEY_RE.match(key):
        raise InvalidKey(
            "格式不对。OpenRouter key 形如 sk-or-v1- 开头的长字符串，"
            "在 https://openrouter.ai/keys 获取。"
        )
    global _runtime_key
    with _lock:
        _runtime_key = key
    # 只记录事件与长度，绝不记录内容
    log("auth", f"API key 已设入内存（{len(key)} 字符，不落盘）")


def clear_key() -> None:
    global _runtime_key
    with _lock:
        _runtime_key = ""
    log("auth", "API key 已从内存清除")


def get_key() -> str:
    """运行时输入优先，其次环境变量（CI / 自动化用）。"""
    with _lock:
        if _runtime_key:
            return _runtime_key
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def has_key() -> bool:
    return bool(get_key())


def key_source() -> str:
    with _lock:
        if _runtime_key:
            return "runtime"
    return "env" if os.environ.get("OPENROUTER_API_KEY", "").strip() else "none"


def status() -> dict:
    """给前端的状态。**绝不包含 key 的任何片段。**"""
    src = key_source()
    return {
        "has_key": src != "none",
        "source": src,
        "label": {
            "runtime": "已配置（内存，重启失效）",
            "env": "已配置（环境变量）",
            "none": "未配置",
        }[src],
        "persisted": False,
        "setup_hint": "" if src != "none" else SETUP_HINT,
    }
