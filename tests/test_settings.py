"""设置层测试：持久化、校验夹取、运行时生效（含闸门联动）、密钥不外泄。"""
import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kindling import api, credentials
from kindling.completeness import gate_status
from kindling.context import ContextEntry, EntryType
from kindling.settings import (
    DEFAULT_MODEL,
    Settings,
    load_settings,
    save_settings,
)
from kdfakes import FakeLLM
from kdresponses import FRAMES

GOOD_KEY = "sk-or-v1-" + "f" * 64


@pytest.fixture(autouse=True)
def _clean_key():
    """每个测试从"无 key"开始，避免互相污染。"""
    credentials.clear_key()
    yield
    credentials.clear_key()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLING_STATE", str(tmp_path / "state.json"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    api.set_llm(None)
    with TestClient(api.app) as c:
        yield c
    api.set_llm(None)


@pytest.fixture
def keyed_client(client):
    assert client.post("/api/key", json={"api_key": GOOD_KEY}).status_code == 200
    return client


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


def test_recommended_models_are_wellformed():
    from kindling.models_catalog import RECOMMENDED

    assert len(RECOMMENDED) >= 4
    for mid, note in RECOMMENDED.items():
        assert "/" in mid          # OpenRouter 形如 vendor/model
        assert note.strip()


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


def test_get_settings_never_echoes_key_material(keyed_client):
    """公开仓库：绝不回显 key 的任何片段，只报告已配置/未配置。"""
    body = keyed_client.get("/api/settings").json()
    assert body["provider"] == "OpenRouter"
    assert "openrouter.ai" in body["endpoint"]
    assert body["settings"]["model"]
    assert body["key"]["has_key"] is True
    assert body["key"]["source"] == "runtime"
    assert body["key"]["persisted"] is False

    blob = json.dumps(body, ensure_ascii=False)
    assert GOOD_KEY not in blob               # 完整 key 不出现
    assert GOOD_KEY[:12] not in blob          # 前缀也不出现
    assert GOOD_KEY[-8:] not in blob          # 后缀也不出现
    assert body["key"]["setup_hint"] == ""    # 已配置时无需提示


def test_missing_key_yields_actionable_setup_hint(client):
    k = client.get("/api/settings").json()["key"]
    assert k["has_key"] is False
    assert k["source"] == "none"
    hint = k["setup_hint"]
    assert "设置" in hint                      # 告诉用户去哪配
    assert "openrouter.ai/keys" in hint        # 告诉用户去哪拿
    assert "内存" in hint                      # 说明重启失效


def test_no_key_fails_loudly_not_silently(client):
    """没有 key 时必须明确报错，绝不能静默降级或借用别处凭据。"""
    api.set_llm(None)
    client.post("/api/entries", json={"text": "想教AI产品", "type": "intent"})
    r = client.post("/api/ask")
    assert r.status_code == 502
    assert "key" in r.json()["error"].lower()


# ---------- 运行时 key：只进内存 ----------


def test_set_key_then_available(client):
    body = client.post("/api/key", json={"api_key": GOOD_KEY}).json()
    assert body["key"]["has_key"] is True
    assert body["key"]["source"] == "runtime"
    assert credentials.get_key() == GOOD_KEY
    assert client.get("/api/state").json()["has_key"] is True


def test_key_never_written_to_disk(keyed_client, tmp_path):
    """核心安全断言：key 不得出现在任何落盘文件里。"""
    keyed_client.post("/api/settings", json={"temperature": 0.3})
    keyed_client.post("/api/entries", json={"text": "x", "type": "intent"})

    scanned = 0
    for p in list(tmp_path.rglob("*")) + list(Path(api.__file__).resolve().parents[1].rglob("*.json")):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        assert GOOD_KEY not in text, f"key 泄漏到 {p}"
        assert GOOD_KEY[:20] not in text, f"key 前缀泄漏到 {p}"
    assert scanned > 0                         # 确认真的扫到了文件


def test_key_never_appears_in_logs(keyed_client):
    logs = json.dumps(keyed_client.get("/api/logs").json(), ensure_ascii=False)
    assert GOOD_KEY not in logs
    assert GOOD_KEY[:20] not in logs
    assert "auth" in logs                      # 但事件本身有记录


def test_clear_key_removes_it(keyed_client):
    body = keyed_client.delete("/api/key").json()
    assert body["key"]["has_key"] is False
    assert credentials.get_key() == ""


def test_malformed_keys_rejected(client):
    for bad in ["", "   ", "not-a-key", "sk-ant-api03-xxxx", "sk-or-v1-short"]:
        r = client.post("/api/key", json={"api_key": bad})
        assert r.status_code == 400, f"应拒绝: {bad!r}"
    assert credentials.get_key() == ""


def test_env_key_used_as_fallback(client, monkeypatch):
    """环境变量仍支持（CI 用），但运行时输入优先。"""
    env_key = "sk-or-v1-" + "e" * 64
    monkeypatch.setenv("OPENROUTER_API_KEY", env_key)
    assert client.get("/api/settings").json()["key"]["source"] == "env"

    client.post("/api/key", json={"api_key": GOOD_KEY})
    assert credentials.get_key() == GOOD_KEY   # 运行时覆盖 env
    assert client.get("/api/settings").json()["key"]["source"] == "runtime"


# ---------- 模型目录 ----------

# 真实的 OpenRouter /models 响应形状（截取两条），用于离线测试。
FAKE_MODELS_PAYLOAD = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Anthropic: Claude Sonnet 4.5",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "some-vendor/free-model",
            "name": "Some Vendor: Free Model",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


@pytest.fixture
def offline_models(monkeypatch):
    """让模型目录走假响应 —— 测试不依赖网络。"""
    import httpx

    from kindling import models_catalog

    monkeypatch.setattr(models_catalog, "_cache", [])
    monkeypatch.setattr(models_catalog, "_fetched_at", 0.0)
    monkeypatch.setattr(
        models_catalog.httpx,
        "get",
        lambda url, **k: httpx.Response(
            200, json=FAKE_MODELS_PAYLOAD, request=httpx.Request("GET", url)
        ),
    )
    return models_catalog


def test_models_endpoint_shape(client, offline_models):
    body = client.get("/api/models?refresh=true").json()
    assert body["error"] == ""
    assert body["count"] == 2
    m = body["models"][0]
    assert {"id", "name", "context", "prompt_price", "recommended"} <= set(m)
    assert m["recommended"] is True             # 推荐的排最前
    assert m["id"] == "anthropic/claude-sonnet-4.5"
    assert m["prompt_price"] == 3.0             # 每 token 换算为每百万 token
    assert body["models"][1]["prompt_price"] == 0.0   # 免费模型


def test_models_endpoint_needs_no_key(client, offline_models):
    """模型列表是公开端点 —— 没配 key 也能先浏览。"""
    assert client.get("/api/settings").json()["key"]["has_key"] is False
    assert client.get("/api/models?refresh=true").json()["count"] == 2


def test_models_are_cached(client, offline_models):
    assert client.get("/api/models?refresh=true").json()["cached"] is False
    assert client.get("/api/models").json()["cached"] is True


def test_models_falls_back_when_upstream_down(client, monkeypatch):
    import httpx

    from kindling import models_catalog

    monkeypatch.setattr(models_catalog, "_cache", [])
    monkeypatch.setattr(models_catalog, "_fetched_at", 0.0)

    def boom(*a, **k):
        raise httpx.ConnectError("网络不可用")

    monkeypatch.setattr(models_catalog.httpx, "get", boom)
    body = client.get("/api/models?refresh=true").json()
    assert body["error"]                        # 明确告知降级
    assert body["count"] >= 4                   # 但仍给出推荐清单
    assert all(m["recommended"] for m in body["models"])


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


def test_key_never_persisted_by_design(client):
    """credentials 模块不得包含任何写文件的操作。"""
    src = Path(credentials.__file__).read_text(encoding="utf-8")
    for forbidden in ("write_text", "open(", "json.dump", "Path.home"):
        assert forbidden not in src, f"credentials.py 含落盘操作: {forbidden}"


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
