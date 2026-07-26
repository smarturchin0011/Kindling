"""LLM 层。所有调用都记录到 observability，前端可见完整 prompt/响应。"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

import httpx

from .observability import log, timer

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


class LLMClient(Protocol):
    def complete(self, system: str, user: str, stage: str = "llm") -> str: ...


class LLMError(RuntimeError):
    pass


def strip_fence(text: str) -> str:
    """剥掉 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def parse_json(text: str, stage: str = "llm") -> dict:
    """解析 LLM 的 JSON 输出。失败时记录原文，方便你在 Log 面板里看到到底返回了什么。"""
    cleaned = strip_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 兜底：抓第一个 {...} 块
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        log(
            stage,
            "LLM 返回的不是合法 JSON",
            level="error",
            detail={"raw": text[:2000]},
        )
        raise LLMError(f"模型没有返回合法 JSON。原文开头：{cleaned[:200]}")


def _scrub(text: str) -> str:
    """从上游错误里剔掉账号标识，避免它出现在 UI 和 log 里。"""
    text = re.sub(r'"?user_id"?\s*:\s*"[^"]*",?', "", text)
    text = re.sub(r"\buser_[A-Za-z0-9]{8,}\b", "user_***", text)
    return text.strip().rstrip(",").strip()


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        http: httpx.Client | None = None,
        temperature: float | None = None,
    ):
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise LLMError(
                "未配置 API key。在项目根目录创建 .env 文件，写入一行："
                "OPENROUTER_API_KEY=sk-or-v1-你的key（在 https://openrouter.ai/keys 获取），"
                "然后重启服务。可参考 .env.example。"
            )
        self.api_key = key
        # 模型/温度优先取运行时设置（设置面板改完立即生效，无需重启）
        if model is None or temperature is None:
            from .settings import load_settings

            s = load_settings()
            model = model or s.model
            temperature = s.temperature if temperature is None else temperature
        self.model = model
        self.temperature = temperature
        self.http = http or httpx.Client(timeout=120.0)

    def complete(self, system: str, user: str, stage: str = "llm") -> str:
        log(
            stage,
            f"→ 调用 {self.model}",
            level="llm",
            detail={
                "model": self.model,
                "temperature": self.temperature,
                "system_prompt": system,
                "user_prompt": user,
            },
        )
        with timer() as t:
            try:
                resp = self.http.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://localhost/kindling",
                        "X-Title": "Kindling",
                    },
                    json={
                        "model": self.model,
                        "temperature": self.temperature,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                )
            except httpx.HTTPError as e:
                log(stage, f"网络错误: {e}", level="error", duration_ms=t.ms)
                raise LLMError(f"调用 LLM 失败：{e}") from e

        if resp.status_code != 200:
            body = _scrub(resp.text[:800])
            log(
                stage,
                f"HTTP {resp.status_code}",
                level="error",
                detail={"body": body},
                duration_ms=t.ms,
            )
            raise LLMError(f"OpenRouter 返回 {resp.status_code}: {body}")

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            log(stage, "响应结构异常", level="error", detail={"body": data})
            raise LLMError(f"响应结构异常: {data}") from e

        usage = data.get("usage", {}) or {}
        log(
            stage,
            f"← {self.model} 返回 {len(content)} 字符",
            level="llm",
            detail={
                "response": content,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
            duration_ms=t.ms,
        )
        return content
