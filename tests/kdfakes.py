from __future__ import annotations


class FakeLLM:
    """测试替身。零网络。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, stage: str = "llm") -> str:
        self.calls.append({"system": system, "user": user, "stage": stage})
        if not self._responses:
            raise AssertionError("FakeLLM 响应队列已空 —— 测试预期的调用次数不对")
        return self._responses.pop(0)
