"""多议题：索引、迁移、切换、归档。

关键语义（用户明确要求）：切换 ≠ 归档。
切换像切会话，随时可以切回来；只有归档才把议题从列表里移走（文件仍保留）。
"""
import json
from pathlib import Path

import pytest

from kindling.topics import (
    archive_topic,
    create_topic,
    current_state_path,
    list_topics,
    migrate_legacy_state,
    switch_topic,
    topic_state_path,
    touch_topic,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("KINDLING_STATE", raising=False)
    monkeypatch.setenv("KINDLING_HOME", str(tmp_path))
    return tmp_path


def test_create_topic_and_current_path(home):
    t = create_topic("音色参考方案")
    assert t["id"].startswith("top_")
    assert current_state_path().name == f"{t['id']}.json"
    assert list_topics()[0]["title"] == "音色参考方案"


def test_create_topic_blank_title_gets_placeholder(home):
    t = create_topic("   ")
    assert t["title"] == "未命名议题"


def test_switch_topic_is_reversible(home):
    """切换 ≠ 归档：切走再切回来，两边都还在列表里。"""
    a = create_topic("A")
    b = create_topic("B")
    assert current_state_path().name == f"{b['id']}.json"

    switch_topic(a["id"])
    assert current_state_path().name == f"{a['id']}.json"

    switch_topic(b["id"])
    assert current_state_path().name == f"{b['id']}.json"
    assert {t["title"] for t in list_topics()} == {"A", "B"}


def test_switch_unknown_topic_raises(home):
    create_topic("A")
    with pytest.raises(KeyError):
        switch_topic("top_nope")


def test_archive_hides_but_keeps_file(home):
    a = create_topic("A")
    b = create_topic("B")
    topic_state_path(a["id"]).write_text('{"topic":"A"}', encoding="utf-8")

    archive_topic(a["id"])

    assert [t["id"] for t in list_topics()] == [b["id"]]
    assert topic_state_path(a["id"]).exists(), "归档不能物理删文件：上下文是资产"
    assert a["id"] in [t["id"] for t in list_topics(include_archived=True)]


def test_archive_current_switches_to_next(home):
    a = create_topic("A")
    b = create_topic("B")
    switch_topic(b["id"])
    archive_topic(b["id"])
    assert current_state_path().name == f"{a['id']}.json"


def test_archive_unknown_raises(home):
    with pytest.raises(KeyError):
        archive_topic("top_nope")


def test_touch_topic_updates_title_and_time(home):
    t = create_topic("旧标题")
    before = t["updated_at"]
    touch_topic(t["id"], "新标题")
    got = list_topics()[0]
    assert got["title"] == "新标题"
    assert got["updated_at"] >= before


def test_touch_unknown_topic_is_silent(home):
    touch_topic("top_nope", "x")  # 不抛异常


def test_migrate_legacy_state(home):
    legacy = Path(home) / "state.json"
    legacy.write_text(
        json.dumps(
            {"topic": "旧议题", "mode": "explore", "entries": [], "moves": [], "frames": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    migrate_legacy_state()

    ts = list_topics()
    assert len(ts) == 1 and ts[0]["title"] == "旧议题"
    assert not legacy.exists(), "旧文件应被搬走而不是复制"
    assert topic_state_path(ts[0]["id"]).exists()

    # 幂等：再跑一次不产生第二个议题
    migrate_legacy_state()
    assert len(list_topics()) == 1


def test_migrate_legacy_preserves_content(home):
    legacy = Path(home) / "state.json"
    payload = {
        "topic": "音色方案",
        "mode": "explore",
        "entries": [
            {
                "id": "ctx_1",
                "text": "真实证据",
                "type": "evidence",
                "created_at": "2026-01-01T00:00:00+00:00",
                "source": "action",
                "move_id": "",
            }
        ],
        "moves": [],
        "frames": [],
    }
    legacy.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    migrate_legacy_state()

    tid = list_topics()[0]["id"]
    got = json.loads(topic_state_path(tid).read_text(encoding="utf-8"))
    assert got["entries"][0]["text"] == "真实证据", "迁移必须无损"


def test_migrate_skipped_when_index_exists(home):
    create_topic("已有议题")
    legacy = Path(home) / "state.json"
    legacy.write_text('{"topic":"不该被迁移"}', encoding="utf-8")
    migrate_legacy_state()
    assert len(list_topics()) == 1
    assert list_topics()[0]["title"] == "已有议题"
    assert legacy.exists(), "已有索引时不动旧文件"


def test_no_topics_creates_default_on_current_path(home):
    """空环境下 current_state_path 必须可用（自动建默认议题）。"""
    p = current_state_path()
    assert p.parent.name == "topics"
    assert len(list_topics()) == 1


def test_current_path_recovers_when_current_archived(home):
    a = create_topic("A")
    b = create_topic("B")
    switch_topic(b["id"])
    archive_topic(b["id"])
    # current 已被归档 → 自动回落到还活着的议题
    assert current_state_path().name == f"{a['id']}.json"


def test_corrupt_index_does_not_crash(home):
    (Path(home) / "topics.json").write_text("{not json", encoding="utf-8")
    p = current_state_path()   # 应重建而不是抛异常
    assert p.parent.name == "topics"
