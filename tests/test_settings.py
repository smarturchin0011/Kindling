"""设置层测试：持久化、校验夹取、运行时生效（含闸门联动）、密钥不外泄。"""
import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kindling import api
from kindling.completeness import gate_status
from kindling.context import ContextEntry, EntryType
from kindling.settings import (
    DEFAULT_MODEL,
    MODEL_PRESETS,
    Settings,
    load_settings,
    save_settings,
)
from kdfakes import FakeLLM
from kdresponses import FRAMES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-FAKEKEYFORTESTSONLY00")
    api.set_llm(None)
    with TestClient(api.app) as c:
        yield c
    api.set_llm(None)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "state.json"))
    return tmp_path


# ---------- 持久化 ----------


def test_defaults_when_no_file(iso):
    s = load_settings()
    assert s.model == DEFAULT_MODEL
    assert 0.0 <= s.temperature <= 2.0
    assert s.gate_threshold == 0.60


def test_roundtrip(iso):
    save_settings(Settings(model="deepseek/deepseek-chat-v3.1", temperature=0.2))
    s = load_settings()
    assert s.model == "deepseek/deepseek-chat-v3.1"
    assert s.temperature == 0.2


def test_settings_file_sits_next_to_state(iso):
    save_settings(Settings())
    assert (iso / "settings.json").exists()


def test_corrupt_file_falls_back(iso):
    (iso / "settings.json").write_text("{broken", encoding="utf-8")
    assert load_settings().model == DEFAULT_MODEL


def test_unknown_keys_ignored(iso):
    (iso / "settings.json").write_text(
        '{"model":"x/y","bogus_field":123}', encoding="utf-8"
    )
    assert load_settings().model == "x/y"


# ---------- 校验：后端不能信前端 ----------


def test_temperature_clamped():
    assert Settings(temperature=99).normalized().temperature == 2.0
    assert Settings(temperature=-5).normalized().temperature == 0.0


def test_threshold_clamped():
    assert Settings(gate_threshold=5.0).normalized().gate_threshold == 0.95
    assert Settings(gate_threshold=0.0).normalized().gate_threshold == 0.05


def test_empty_model_falls_back_to_default():
    assert Settings(model="   ").normalized().model == DEFAULT_MODEL


def test_negative_minimums_clamped():
    n = Settings(min_evidence=-3, min_constraints=-1).normalized()
    assert n.min_evidence == 0 and n.min_constraints == 0


def test_presets_are_wellformed():
    assert len(MODEL_PRESETS) >= 4
    for p in MODEL_PRESETS:
        assert {"id", "label", "note"} <= set(p)
        assert "/" in p["id"]          # OpenRouter 形如 vendor/model


def test_upstream_errors_scrub_account_id():
    """OpenRouter 的 400 响应里带 user_id，不能泄漏到 UI 和 log。"""
    import httpx

    from kindling.llm import LLMError, OpenRouterClient

    def handler(request):
        return httpx.Response(
            400,
            text='{"error":{"message":"bad model","code":400},'
                 '"user_id":"user_FAKEACCOUNTID000000000000"}',
        )

    c = OpenRouterClient(
        api_key="sk-test",
        model="x/y",
        temperature=0,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError) as exc:
        c.complete(system="s", user="u")
    msg = str(exc.value)
    assert "bad model" in msg                 # 有用信息保留
    assert "user_FAKEACCOUNTID000000000000" not in msg
    assert "user_id" not in msg


# ---------- 闸门阈值运行时生效 ----------


def test_gate_threshold_override_changes_verdict():
    ctx = [ContextEntry.new(f"事故{i}", EntryType.EVIDENCE) for i in range(2)]
    ctx.append(ContextEntry.new("受众PM", EntryType.CONSTRAINT))
    assert gate_status(ctx)["open"] is True
    # 抬高门槛后同样的上下文应被拒
    strict = gate_status(ctx, threshold=0.95, min_evidence=8)
    assert strict["open"] is False
    assert any("8 条证据" in m for m in strict["missing"])


def test_lowering_requirements_opens_gate():
    ctx = [ContextEntry.new("只有一个意图", EntryType.INTENT)]
    assert gate_status(ctx)["open"] is False
    loose = gate_status(ctx, threshold=0.05, min_evidence=0, min_constraints=0)
    assert loose["open"] is True


# ---------- HTTP ----------


def test_get_settings_never_echoes_key_material(client):
    """公开仓库：绝不回显 key 的任何片段，只报告已配置/未配置。"""
    body = client.get("/api/settings").json()
    assert body["provider"] == "OpenRouter"
    assert "openrouter.ai" in body["endpoint"]
    assert body["settings"]["model"]
    assert len(body["presets"]) >= 4
    assert body["has_key"] is True

    blob = json.dumps(body, ensure_ascii=False)
    key = os.environ["OPENROUTER_API_KEY"]
    assert key not in blob                    # 完整 key 不出现
    assert key[:12] not in blob               # 前缀也不出现
    assert key[-6:] not in blob               # 后缀也不出现
    assert body["key_hint"] == "已配置（不显示）"
    assert body["setup_hint"] == ""           # 已配置时无需提示


def test_missing_key_yields_actionable_setup_hint(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    body = client.get("/api/settings").json()
    assert body["has_key"] is False
    assert body["key_hint"] == ""
    hint = body["setup_hint"]
    assert ".env" in hint and "OPENROUTER_API_KEY" in hint
    assert "openrouter.ai/keys" in hint       # 告诉用户去哪拿
    assert body["env_file"].endswith(".env")  # 告诉用户放哪


def test_no_key_fails_loudly_not_silently(client, monkeypatch):
    """没有 key 时必须明确报错，绝不能静默降级或借用别处凭据。"""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    api.set_llm(None)
    client.post("/api/entries", json={"text": "想教AI产品", "type": "intent"})
    r = client.post("/api/ask")
    assert r.status_code == 502
    assert ".env" in r.json()["error"]


def test_source_never_reads_credentials_outside_project(client):
    """回归防护：曾经有过"找不到就读 Hermes 的 .env"的逻辑。

    在公开项目里隐式借用别处凭据是不可接受的 —— 用户不知道自己在消费
    哪个账号，且审计者会合理地判断为可疑行为。
    """
    src = Path(api.__file__).resolve().parent
    for py in src.glob("*.py"):
        low = py.read_text(encoding="utf-8").lower()
        assert "hermes" not in low, f"{py.name} 引用了外部应用的配置"
        # 只允许加载项目根的 .env；不得把 home 目录路径喂给 load_dotenv
        for call in re.findall(r"load_dotenv\(([^)]*)\)", low):
            assert "home()" not in call, f"{py.name} 从 home 目录加载凭据: {call}"


def test_post_settings_persists_and_returns_snapshot(client):
    body = client.post(
        "/api/settings", json={"model": "z-ai/glm-4.6", "temperature": 0.15}
    ).json()
    assert body["settings"]["model"] == "z-ai/glm-4.6"
    assert body["settings"]["temperature"] == 0.15
    assert "gate" in body                      # 快照一并回传
    assert client.get("/api/state").json()["model"] == "z-ai/glm-4.6"


def test_partial_update_preserves_other_fields(client):
    client.post("/api/settings", json={"model": "a/b", "temperature": 0.3})
    client.post("/api/settings", json={"temperature": 0.9})
    s = client.get("/api/settings").json()["settings"]
    assert s["model"] == "a/b"                 # 未提交的字段不被清掉
    assert s["temperature"] == 0.9


def test_post_settings_clamps_hostile_input(client):
    s = client.post(
        "/api/settings", json={"temperature": 999, "gate_threshold": -1}
    ).json()["settings"]
    assert s["temperature"] == 2.0
    assert s["gate_threshold"] == 0.05


def test_threshold_change_immediately_moves_gate(client):
    """改闸门阈值必须立刻反映到完整度判定，不需要重启。"""
    for t, ty in [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")]:
        client.post("/api/entries", json={"text": t, "type": ty})
    assert client.get("/api/state").json()["gate"]["open"] is True

    body = client.post(
        "/api/settings", json={"gate_threshold": 0.95, "min_evidence": 9}
    ).json()
    assert body["gate"]["open"] is False       # 同一批上下文，判定翻转
    assert client.get("/api/state").json()["gate"]["open"] is False


def test_synth_respects_runtime_threshold(client):
    """抬高门槛后，原本能过的上下文必须被 423 拒绝。"""
    for t, ty in [("事故1", "evidence"), ("事故2", "evidence"), ("受众PM", "constraint")]:
        client.post("/api/entries", json={"text": t, "type": ty})
    client.post("/api/settings", json={"gate_threshold": 0.99, "min_evidence": 12})
    api.set_llm(FakeLLM([FRAMES]))
    r = client.post("/api/synth", json={"force": False})
    assert r.status_code == 423
    assert any("12 条证据" in m for m in r.json()["gate"]["missing"])


def test_gate_threshold_surfaces_in_state(client):
    client.post("/api/settings", json={"gate_threshold": 0.42})
    assert client.get("/api/state").json()["gate"]["threshold"] == 0.42
