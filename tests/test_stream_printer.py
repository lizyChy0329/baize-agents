"""StreamPrinter 区分「铺垫」与「答案」的测试。"""
from __future__ import annotations

import pytest

from baize_agents.stream import StreamPrinter


@pytest.fixture
def printer():
    return StreamPrinter(enabled=True, flush_after=100)


@pytest.fixture(autouse=True)
def force_color(monkeypatch):
    """测试环境里 stdout 不是终端，需要强制打开颜色才能验证样式。"""
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)


def test_short_preamble_is_dimmed(printer, capsys):
    """短文本 + 后面跟了工具调用 → 判定为铺垫，上暗灰色。"""
    printer("先读一下文件。")
    printer.end_turn(is_preamble=True)
    printer.finish()

    out = capsys.readouterr().out
    assert "先读一下文件。" in out
    assert "\033[2m" in out  # 暗灰
    assert printer.started is True


def test_short_answer_is_not_dimmed(printer, capsys):
    """短文本 + 没有工具调用 → 判定为答案，不上色。"""
    printer("好的。")
    printer.end_turn(is_preamble=False)
    printer.finish()

    out = capsys.readouterr().out
    assert "好的。" in out
    assert "\033[2m" not in out


def test_long_text_streams_live_without_waiting_for_turn_end(capsys):
    """超过阈值的文本应立刻实时输出，不等轮结束。"""
    printer = StreamPrinter(enabled=True, flush_after=10)
    printer("这是")  # 4 字，还没到阈值
    assert capsys.readouterr().out == ""  # 攒着，没输出

    printer("一段足够长的正文")  # 累计超过 10
    assert "这是" in capsys.readouterr().out  # 立即吐出


def test_long_text_keeps_streaming_after_threshold(capsys):
    printer = StreamPrinter(enabled=True, flush_after=5)
    printer("一二三四五六")  # 触发 flush
    capsys.readouterr()
    printer("七八九")
    assert "七八九" in capsys.readouterr().out


def test_buffer_resets_between_turns(capsys):
    printer = StreamPrinter(enabled=True, flush_after=100)
    printer("铺垫一")
    printer.end_turn(is_preamble=True)
    first = capsys.readouterr().out
    assert "铺垫一" in first

    printer("答案二")
    printer.end_turn(is_preamble=False)
    second = capsys.readouterr().out
    assert "答案二" in second
    assert "\033[2m" not in second  # 第二轮是答案，不该是灰的


def test_disabled_printer_outputs_nothing(capsys):
    printer = StreamPrinter(enabled=False)
    printer("不该出现")
    printer.end_turn(is_preamble=False)
    printer.finish()
    assert capsys.readouterr().out == ""
    assert printer.started is False


def test_finish_is_idempotent(capsys):
    printer = StreamPrinter(enabled=True)
    printer("x")
    printer.end_turn(is_preamble=False)
    printer.finish()
    printer.finish()
    out = capsys.readouterr().out
    # 开头一个换行 + 结尾一个换行；第二次 finish() 不应再加
    assert out.count("\n") == 2
