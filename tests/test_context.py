"""token 估算与截断的测试。"""
from __future__ import annotations

from baize_agents.context.tokens import (
    estimate_tokens,
    message_tokens,
    messages_tokens,
)
from baize_agents.context.truncate import truncate_text, truncate_tool_result


# ---------------------------------------------------------------- 估算


def test_empty_text_is_zero():
    assert estimate_tokens("") == 0


def test_chinese_is_about_one_token_per_char():
    assert 90 <= estimate_tokens("中" * 100) <= 110


def test_ascii_is_cheaper_than_chinese():
    """同样字符数，英文比中文便宜（词典压缩）。"""
    assert estimate_tokens("a" * 100) < estimate_tokens("中" * 100)


def test_symbols_are_expensive():
    """数字和符号很贵：比字母贵，比中文便宜。"""
    letters = estimate_tokens("a" * 100)
    symbols = estimate_tokens("1" * 100)
    chinese = estimate_tokens("中" * 100)
    assert letters < symbols < chinese


def test_estimate_is_monotonic():
    assert estimate_tokens("x" * 100) < estimate_tokens("x" * 200)


def test_mixed_text_between_the_two_extremes():
    mixed = "hello 你好 world 世界"
    assert estimate_tokens(mixed) < estimate_tokens("中" * len(mixed))


def test_message_includes_overhead():
    bare = {"role": "user", "content": ""}
    assert message_tokens(bare) > 0  # 固定开销


def test_message_counts_tool_calls():
    plain = {"role": "assistant", "content": ""}
    with_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "file_read", "arguments": '{"path":"a"}'}}
        ],
    }
    assert message_tokens(with_calls) > message_tokens(plain)


def test_messages_tokens_sums_up():
    one = {"role": "user", "content": "你好"}
    assert messages_tokens([one, one, one]) == 3 * message_tokens(one)


# ---------------------------------------------------------------- 截断


def test_short_text_untouched():
    text = "很短的内容"
    assert truncate_text(text, 1000) is text


def test_long_text_keeps_head_and_tail():
    text = "开头" + "中" * 5000 + "结尾"
    out = truncate_text(text, 200)

    assert out.startswith("开头")
    assert out.endswith("结尾")
    assert "省略" in out


def test_truncated_result_reports_dropped_amount():
    text = "中" * 5000
    out = truncate_text(text, 100)
    assert "省略" in out
    assert "字" in out


def test_truncation_actually_reduces_tokens():
    text = "中" * 5000
    before = estimate_tokens(text)
    after = estimate_tokens(truncate_text(text, 100))
    assert after < before / 10


def test_tool_result_adds_hint():
    text = "中" * 5000
    out = truncate_tool_result(text, 100)
    assert "已截断" in out


def test_tool_result_hint_not_added_when_short():
    text = "短"
    assert truncate_tool_result(text, 1000) == text


def test_truncate_handles_empty():
    assert truncate_text("", 100) == ""
