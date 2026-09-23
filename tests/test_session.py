"""会话持久化与崩溃自愈的测试。"""
from __future__ import annotations

import json

import pytest

from baize_agents.session import Session, SessionError, _repair, list_sessions


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """把 BAIZE_HOME 指到临时目录，别碰真实数据。"""
    monkeypatch.setenv("BAIZE_HOME", str(tmp_path))
    return tmp_path


def assistant_tool_call(call_id: str = "c1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "file_read", "arguments": "{}"}}],
    }


def tool_result(call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": "文件内容"}


# ---------------------------------------------------------------- 读写


def test_empty_session_when_file_missing():
    assert Session("new").load() == []


def test_add_then_sync_then_reload():
    s = Session("demo")
    s.load()
    s.add({"role": "user", "content": "你好"})
    s.add({"role": "assistant", "content": "你好呀"})
    s.sync()

    assert Session("demo").load() == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


def test_sync_is_incremental_and_idempotent():
    s = Session("demo")
    s.load()
    s.add({"role": "user", "content": "一"})
    s.sync()
    assert len(s.path.read_text().strip().splitlines()) == 1

    s.sync()  # 再调一次不应重复写
    assert len(s.path.read_text().strip().splitlines()) == 1

    s.add({"role": "user", "content": "二"})
    s.sync()
    assert len(s.path.read_text().strip().splitlines()) == 2


def test_chinese_is_written_readable():
    s = Session("demo")
    s.load()
    s.add({"role": "user", "content": "中文不要变成转义"})
    s.sync()
    assert "中文不要变成转义" in s.path.read_text(encoding="utf-8")


def test_clear_removes_file():
    s = Session("demo")
    s.load()
    s.add({"role": "user", "content": "x"})
    s.sync()
    assert s.path.exists()

    s.clear()
    assert not s.path.exists()
    assert s.messages == []
    assert len(s) == 0


def test_list_sessions_sorted():
    for name in ("b", "a", "c"):
        s = Session(name)
        s.load()
        s.add({"role": "user", "content": name})
        s.sync()
    assert list_sessions() == ["a", "b", "c"]


# ---------------------------------------------------------------- 安全


@pytest.mark.parametrize("bad", ["", ".", "..", "../evil", "a/b", "a\\b"])
def test_illegal_session_names_rejected(bad):
    with pytest.raises(SessionError):
        Session(bad)


# ---------------------------------------------------------------- 崩溃自愈


def test_repair_keeps_complete_turn():
    messages = [
        {"role": "user", "content": "读文件"},
        assistant_tool_call("c1"),
        tool_result("c1"),
        {"role": "assistant", "content": "读完了"},
    ]
    assert _repair(list(messages)) == messages


def test_repair_drops_dangling_tool_call():
    messages = [
        {"role": "user", "content": "读文件"},
        assistant_tool_call("c1"),
    ]
    assert _repair(list(messages)) == [{"role": "user", "content": "读文件"}]


def test_repair_drops_partially_answered_multi_tool_call():
    messages = [
        {"role": "user", "content": "读两个文件"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "file_read", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "file_read", "arguments": "{}"}},
            ],
        },
        tool_result("c1"),  # c2 没回答
    ]
    assert _repair(list(messages)) == [{"role": "user", "content": "读两个文件"}]


def test_repair_leaves_plain_chat_alone():
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]
    assert _repair(list(messages)) == messages


def test_repair_applied_on_load():
    s = Session("demo")
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text(
        "\n".join(
            json.dumps(m, ensure_ascii=False)
            for m in [
                {"role": "user", "content": "读文件"},
                assistant_tool_call("c1"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert Session("demo").load() == [{"role": "user", "content": "读文件"}]


# ---------------------------------------------------------------- 坏数据


def test_bad_json_line_reports_line_number():
    s = Session("demo")
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text('{"role": "user", "content": "ok"}\n这不是 JSON\n', encoding="utf-8")
    with pytest.raises(SessionError, match="第 2 行"):
        s.load()


def test_blank_lines_are_ignored():
    s = Session("demo")
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text('{"role": "user", "content": "ok"}\n\n\n', encoding="utf-8")
    assert len(s.load()) == 1
