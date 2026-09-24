"""runner 循环的测试（用假 provider，不打真实 API）。"""
from __future__ import annotations

from typing import Any

import pytest

from baize_agents.providers.base import Provider
from baize_agents.runner import RunnerError, run


class FakeProvider(Provider):
    """按脚本依次返回预设的 message，并记录每次收到的 payload。"""

    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        super().__init__(config=None)  # type: ignore[arg-type]  # 测试用不上真实配置
        self.scripted = list(scripted)
        self.seen: list[list[dict[str, Any]]] = []

    def _chat_once(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        if not self.scripted:
            raise AssertionError("脚本用完啦")
        return self.scripted.pop(0)


def test_plain_answer_returns_content():
    provider = FakeProvider([{"role": "assistant", "content": "你好"}])
    messages = [{"role": "user", "content": "hi"}]

    assert run(provider, messages) == "你好"
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好"},
    ]


def test_system_prompt_is_sent_but_not_persisted():
    provider = FakeProvider([{"role": "assistant", "content": "ok"}])
    messages = [{"role": "user", "content": "hi"}]

    run(provider, messages, system_prompt="你是 baize。")

    # 发给 API 的第一条是 system
    assert provider.seen[0][0] == {"role": "system", "content": "你是 baize。"}
    # 磁盘上要保存的 messages 里没有 system
    assert all(m["role"] != "system" for m in messages)


def test_system_prompt_resent_on_every_turn():
    provider = FakeProvider(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "nope", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "收工"},
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    assert run(provider, messages, system_prompt="sys") == "收工"
    assert len(provider.seen) == 2
    for payload in provider.seen:
        assert payload[0]["role"] == "system"


def test_tool_result_is_fed_back():
    provider = FakeProvider(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "nope", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "收到结果了"},
        ]
    )
    messages = [{"role": "user", "content": "hi"}]
    seen_calls: list[tuple[str, str, str]] = []

    reply = run(provider, messages, on_tool_call=lambda n, a, r: seen_calls.append((n, a, r)))

    assert reply == "收到结果了"
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "c1"
    assert tool_messages[0]["content"].startswith("[错误] 未知工具")  # execute 永不抛异常
    assert seen_calls and seen_calls[0][0] == "nope"


def test_bad_json_arguments_are_fed_back_as_error():
    provider = FakeProvider(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "file_read", "arguments": "{坏JSON"}}
                ],
            },
            {"role": "assistant", "content": "我知道错了"},
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    run(provider, messages)
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert "不是合法 JSON" in tool_messages[0]["content"]


def test_gives_up_after_max_turns():
    looping = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "nope", "arguments": "{}"}}
        ],
    }
    provider = FakeProvider([looping] * 5)
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(RunnerError, match="最大轮数"):
        run(provider, messages, max_turns=3)


def test_tool_result_is_truncated_before_feedback(monkeypatch):
    """工具结果过长时要截断后才回喂，否则一次读大文件就撑爆上下文。"""
    import baize_agents.tools as T
    from baize_agents.context.tokens import estimate_tokens
    from baize_agents.tools.base import Tool

    huge = "中" * 50000
    fake = Tool(name="fake_tool", description="", parameters={}, func=lambda: huge)
    monkeypatch.setitem(T.REGISTRY, "fake_tool", fake)

    provider = FakeProvider(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "fake_tool", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    run(provider, messages, max_tool_result_tokens=500)

    tool_message = [m for m in messages if m["role"] == "tool"][0]
    assert "省略" in tool_message["content"]
    assert estimate_tokens(tool_message["content"]) < 2000


def test_short_tool_result_not_truncated(monkeypatch):
    import baize_agents.tools as T
    from baize_agents.tools.base import Tool

    fake = Tool(name="small_tool", description="", parameters={}, func=lambda: "很短")
    monkeypatch.setitem(T.REGISTRY, "small_tool", fake)

    provider = FakeProvider(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "small_tool", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    run(provider, messages, max_tool_result_tokens=4000)
    tool_message = [m for m in messages if m["role"] == "tool"][0]
    assert tool_message["content"] == "很短"
