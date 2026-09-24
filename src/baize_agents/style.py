"""终端样式：用颜色把「agent 的动作」和「模型的回答」区分开。

规则：
- 工具调用 / 结果 / 提示 / 错误 → 上色，让它们一眼看出是"过程"
- 模型的回答 → 不上色，于是成为屏幕上唯一"素净"的内容，自然突出

自动降级：
- 输出被重定向或管道接走（不是终端）→ 关掉颜色，避免 `| grep` 收到一堆转义码
- 环境变量 NO_COLOR 有值 → 强制关闭（这是通行约定）
- 环境变量 FORCE_COLOR 有值 → 强制打开
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TextIO

_RESET = "\033[0m"


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


@dataclass(frozen=True)
class Style:
    """一组样式方法。disable 状态下全部退化为原样返回。"""

    enabled: bool = False

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}{_RESET}" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)


def stdout_style() -> Style:
    return Style(enabled=_supports_color(sys.stdout))


def stderr_style() -> Style:
    return Style(enabled=_supports_color(sys.stderr))


def tool_call(text: str) -> str:
    """工具调用：agent 主动做的动作，用青色。"""
    return stderr_style().cyan(text)


def tool_result(text: str) -> str:
    """工具结果：细节，压暗。"""
    return stderr_style().dim(text)


def notice(text: str) -> str:
    """提示信息（载入历史、切换 provider 等）。"""
    return stderr_style().dim(text)


def warning(text: str) -> str:
    """重试之类的警告。"""
    return stderr_style().yellow(text)


def error(text: str) -> str:
    """错误。"""
    return stderr_style().red(text)
