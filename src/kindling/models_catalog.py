"""OpenRouter 模型目录。

从 https://openrouter.ai/api/v1/models 拉真实列表，进程内缓存 1 小时。
不需要 key（这个端点是公开的），所以设置面板在还没配 key 时也能浏览模型。
"""
from __future__ import annotations

import threading
import time

import httpx

from .observability import log, timer

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL = 3600.0

_lock = threading.Lock()
_cache: list[dict] = []
_fetched_at: float = 0.0

# 推荐清单：按"验证 Kindling 缺口检测"的用途挑的，不是按名气。
# 命中时在 UI 里标星并给出理由。
RECOMMENDED: dict[str, str] = {
    "anthropic/claude-sonnet-4.5": "默认。指令遵循强，JSON 稳",
    "anthropic/claude-opus-4.1": "缺口问题最尖锐，慢且贵",
    "openai/gpt-5": "对比基线",
    "google/gemini-2.5-pro": "长上下文，便宜",
    "deepseek/deepseek-chat-v3.1": "极便宜、中文好，适合刷轮次做对比",
    "x-ai/grok-4": "风格更直接，可能问得更冒犯",
    "qwen/qwen3-235b-a22b": "中文母语级",
    "z-ai/glm-4.6": "中文强，便宜",
}


def _price_per_mtok(pricing: dict, field: str) -> float | None:
    """OpenRouter 的 pricing 是"每 token 美元"的字符串。转成每百万 token。"""
    try:
        v = float(pricing.get(field, "") or 0)
    except (TypeError, ValueError):
        return None
    return round(v * 1_000_000, 3) if v > 0 else 0.0


def _shape(m: dict) -> dict:
    mid = m.get("id", "")
    pricing = m.get("pricing", {}) or {}
    top = m.get("top_provider", {}) or {}
    return {
        "id": mid,
        "name": m.get("name") or mid,
        "context": m.get("context_length") or top.get("context_length") or 0,
        "prompt_price": _price_per_mtok(pricing, "prompt"),
        "completion_price": _price_per_mtok(pricing, "completion"),
        "recommended": mid in RECOMMENDED,
        "note": RECOMMENDED.get(mid, ""),
    }


def _sort_key(m: dict):
    # 推荐的排最前，其余按 id 字母序，便于扫读
    return (0 if m["recommended"] else 1, m["id"])


def fetch_models(force: bool = False) -> dict:
    """返回 {models, cached, count, error}。失败时降级为推荐清单，不阻塞用户。"""
    global _cache, _fetched_at
    now = time.time()
    with _lock:
        fresh = _cache and (now - _fetched_at) < CACHE_TTL
        if fresh and not force:
            return {"models": _cache, "cached": True, "count": len(_cache), "error": ""}

    try:
        with timer() as t:
            resp = httpx.get(MODELS_URL, timeout=20.0)
            resp.raise_for_status()
            data = resp.json().get("data", [])
    except (httpx.HTTPError, ValueError) as e:
        log("models", f"拉取模型列表失败：{e}", level="warn")
        fallback = sorted(
            (_shape({"id": k, "name": k}) for k in RECOMMENDED), key=_sort_key
        )
        with _lock:
            models = _cache or fallback
        return {
            "models": models,
            "cached": bool(_cache),
            "count": len(models),
            "error": f"无法连接 OpenRouter 模型列表（{e}）。显示的是内置推荐清单。",
        }

    models = sorted((_shape(m) for m in data if m.get("id")), key=_sort_key)
    with _lock:
        _cache, _fetched_at = models, now
    log("models", f"拉取到 {len(models)} 个模型", duration_ms=t.ms)
    return {"models": models, "cached": False, "count": len(models), "error": ""}


def is_known(model_id: str) -> bool:
    with _lock:
        return any(m["id"] == model_id for m in _cache)
