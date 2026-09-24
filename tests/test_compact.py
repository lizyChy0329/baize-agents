"""历史压缩（compact）的测试。

重点：
1. 切点绝不能落在 assistant tool_calls 和它的 tool 结果之间
2. 压缩后的历史必须能被 API 接受（不出现孤儿 tool 消息）
3. 压缩要能跨进程持久化（重开也能读到摘要）
"""
from __future__ import annotations

from typing import Any

import pytest

from baize_agents.context.compact import (
    compact,
    find_cut,
    format_history,
    needs_compaction,
)
from baize_agents.context.tokens import messages_tokens
from baize_agents.session import Session


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def tool_exchange(call_id: str, tool_text: str = "结果") -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "file_read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": tool_text},
        assistant("读完了"),
    ]


class FakeProvider:
    """只用来返回一段固定摘要。"""

    def __init__(self, summary: str = "这是摘要") -> None:
        self.summary = summary
        self.seen: list[str] = []

    def chat(self, messages, tools=None):
        self.seen.append(messages[-1]["content"])
        return {"role": "assistant", "content": self.summary}


def heavy(n: int = 1) -> dict[str, Any]:
    """一条很长的消息，用来撑爆阈值。"""
    return user("中" * (200 * n))


# ---------------------------------------------------------------- 切点


def test_no_cut_when_everything_fits():
    messages = [user("短"), assistant("也短")]
    assert find_cut(messages, keep_recent_tokens=10000) == 0


def test_cut_lands_on_user_message():
    """绝不能在 tool 消息处切，否则 assistant 的 tool_calls 会变孤儿。"""
    messages = [
        user("第一轮"),
        *tool_exchange("c1"),
        heavy(3),
        user("第二轮"),
        assistant("好的"),
    ]
    cut = find_cut(messages, keep_recent_tokens=10)

    assert cut > 0
    assert messages[cut]["role"] == "user"


def test_cut_never_splits_tool_pair():
    messages = [user("a"), *tool_exchange("c1"), heavy(5), user("b"), assistant("c")]
    cut = find_cut(messages, keep_recent_tokens=5)
    kept = messages[cut:]

    # 保留下来的第一条必须是 user
    assert kept[0]["role"] == "user"
    # 保留部分里不能出现"没有来源的 tool 消息"
    for i, m in enumerate(kept):
        if m["role"] == "tool":
            assert any(
                k["role"] == "assistant" and k.get("tool_calls") for k in kept[:i]
            ), "出现了孤儿 tool 消息"


def test_cut_returns_zero_when_no_user_message_to_keep():
    """后面全是工具往返、没有 user 消息时，宁可不切。"""
    messages = [heavy(5), *tool_exchange("c1")]
    assert find_cut(messages, keep_recent_tokens=1) == 0


# ---------------------------------------------------------------- 压缩


def test_compact_replaces_old_messages_with_summary():
    messages = [heavy(3), user("最近的问题"), assistant("最近的回答")]
    provider = FakeProvider("用户之前聊了很多")

    result = compact(provider, messages, keep_recent_tokens=50)

    assert result is not None
    summary, dropped = result
    assert summary == "用户之前聊了很多"
    assert dropped == 1
    assert messages[0]["role"] == "system"
    assert "用户之前聊了很多" in messages[0]["content"]
    assert messages[-1]["content"] == "最近的回答"


def test_compact_mutates_in_place():
    """必须是就地修改：session 持有的是同一个列表引用。"""
    messages = [heavy(3), user("q"), assistant("a")]
    original_id = id(messages)
    compact(FakeProvider(), messages, keep_recent_tokens=50)
    assert id(messages) == original_id


def test_compact_returns_none_when_short():
    messages = [user("短"), assistant("也短")]
    assert compact(FakeProvider(), messages, keep_recent_tokens=10000) is None


def test_compact_reduces_tokens():
    messages = [heavy(5), user("q"), assistant("a")]
    before = messages_tokens(messages)
    compact(FakeProvider(), messages, keep_recent_tokens=50)
    assert messages_tokens(messages) < before


def test_needs_compaction():
    assert needs_compaction([heavy(5)], 10) is True
    assert needs_compaction([user("短")], 100000) is False


def test_format_history_includes_tool_call_names():
    text = format_history([user("读文件"), *tool_exchange("c1")])
    assert "file_read" in text
    assert "读文件" in text


# ---------------------------------------------------------------- 持久化


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BAIZE_HOME", str(tmp_path))


def test_compaction_persists_across_reload():
    """回归测试：曾经只写 compact 记录、忘了写保留的消息，导致重启后历史全丢。"""
    session = Session("demo")
    session.load()
    for i in range(3):
        session.add(user(f"第{i}轮问题"))
        session.add(assistant(f"第{i}轮回答"))
    session.sync()

    # 压缩：保留最近的消息
    session.messages[:] = [
        {"role": "system", "content": "旧摘要"},
        user("最近的问题"),
        assistant("最近的回答"),
    ]
    session.record_compaction("旧摘要")

    reloaded = Session("demo").load()
    assert reloaded[0]["role"] == "system"
    assert "旧摘要" in reloaded[0]["content"]
    assert reloaded[-1]["content"] == "最近的回答"
    assert len(reloaded) == 3


def test_after_compaction_new_messages_still_append():
    session = Session("demo")
    session.load()
    session.add(user("老的"))
    session.sync()

    session.messages[:] = [{"role": "system", "content": "摘要"}, user("新的")]
    session.record_compaction("摘要")
    session.sync()

    reloaded = Session("demo").load()
    assert [m["role"] for m in reloaded] == ["system", "user"]
    assert reloaded[1]["content"] == "新的"


def test_multiple_compactions_keep_last_summary():
    session = Session("demo")
    session.load()
    session.add(user("一"))
    session.sync()

    session.messages[:] = [{"role": "system", "content": "摘要A"}, user("二")]
    session.record_compaction("摘要A")

    session.messages[:] = [{"role": "system", "content": "摘要B"}, user("三")]
    session.record_compaction("摘要B")

    reloaded = Session("demo").load()
    assert "摘要B" in reloaded[0]["content"]
    assert reloaded[-1]["content"] == "三"
    assert all("摘要A" not in (m.get("content") or "") for m in reloaded)
