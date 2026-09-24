"""SSE 流式解析的测试 —— 重点是 tool_calls 分片拼装。"""
from __future__ import annotations

import json

from baize_agents.providers.openai_compat import parse_sse


def sse(*chunks: dict | str) -> list[str]:
    """把若干 chunk 变成 SSE 行。"""
    lines = []
    for c in chunks:
        payload = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        lines.append(f"data: {payload}\n")
    lines.append("\n")
    lines.append("data: [DONE]\n")
    return lines


def delta(d: dict) -> dict:
    return {"choices": [{"index": 0, "delta": d}]}


def collect(lines: list[str]) -> tuple[dict, str]:
    out: list[str] = []
    message = parse_sse(lines, out.append)
    return message, "".join(out)


# ---------------------------------------------------------------- 纯文本


def test_plain_text_streaming():
    lines = sse(
        delta({"role": "assistant", "content": ""}),
        delta({"content": "你好"}),
        delta({"content": "，世界"}),
        delta({"content": "！"}),
    )
    message, streamed = collect(lines)

    assert message["role"] == "assistant"
    assert message["content"] == "你好，世界！"
    assert streamed == "你好，世界！"  # 增量按顺序吐出来
    assert "tool_calls" not in message


def test_empty_content_deltas_are_skipped():
    lines = sse(delta({"role": "assistant"}), delta({"content": ""}), delta({"content": "x"}))
    message, streamed = collect(lines)
    assert message["content"] == "x"
    assert streamed == "x"


# ---------------------------------------------------------------- 工具调用分片


def test_tool_call_arguments_are_concatenated():
    """这是流式最容易翻车的地方：arguments 是一段段来的。"""
    lines = sse(
        delta({"role": "assistant", "content": ""}),
        delta({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "file_read", "arguments": ""}}]}),
        delta({"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]}),
        delta({"tool_calls": [{"index": 0, "function": {"arguments": 'th": "'}}]}),
        delta({"tool_calls": [{"index": 0, "function": {"arguments": 'README.md"}'}}]}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    )
    message, _ = collect(lines)

    assert message["content"] == ""
    calls = message["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "file_read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "README.md"}


def test_fragmented_function_name():
    lines = sse(
        delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "file_"}}]}),
        delta({"tool_calls": [{"index": 0, "function": {"name": "read"}}]}),
    )
    message, _ = collect(lines)
    assert message["tool_calls"][0]["function"]["name"] == "file_read"


def test_multiple_parallel_tool_calls_kept_separate():
    """模型一次要调多个工具时，靠 index 分槽，不能串到一起。"""
    lines = sse(
        delta({"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "file_read"}}]}),
        delta({"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "list_dir"}}]}),
        delta({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":"a"'}}]}),
        delta({"tool_calls": [{"index": 1, "function": {"arguments": '{"path":"b"'}}]}),
        delta({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
        delta({"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]}),
    )
    message, _ = collect(lines)
    calls = message["tool_calls"]

    assert len(calls) == 2
    assert calls[0]["id"] == "c0"
    assert calls[0]["function"]["name"] == "file_read"
    assert calls[0]["function"]["arguments"] == '{"path":"a"}'
    assert calls[1]["id"] == "c1"
    assert calls[1]["function"]["name"] == "list_dir"
    assert calls[1]["function"]["arguments"] == '{"path":"b"}'


# ---------------------------------------------------------------- 健壮性


def test_malformed_lines_are_ignored():
    lines = [
        "data: {这不是 JSON\n",
        ": 这是注释\n",
        "\n",
        "data: " + json.dumps(delta({"content": "ok"}), ensure_ascii=False) + "\n",
        "data: [DONE]\n",
    ]
    message, streamed = collect(lines)
    assert message["content"] == "ok"
    assert streamed == "ok"


def test_stream_without_done_terminator():
    lines = [f"data: {json.dumps(delta({'content': '还是能读完'}))}\n"]
    message, _ = collect(lines)
    assert message["content"] == "还是能读完"


def test_usage_only_trailing_chunk_is_tolerated():
    lines = sse(
        delta({"content": "hi"}),
        {"choices": [], "usage": {"prompt_tokens": 1}},
    )
    message, _ = collect(lines)
    assert message["content"] == "hi"


def test_text_and_tool_call_together():
    lines = sse(
        delta({"content": "让我看看..."}),
        delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "file_read", "arguments": "{}"}}]}),
    )
    message, streamed = collect(lines)
    assert message["content"] == "让我看看..."
    assert streamed == "让我看看..."
    assert message["tool_calls"][0]["function"]["name"] == "file_read"
