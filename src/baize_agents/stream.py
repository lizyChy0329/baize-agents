"""流式输出与工具日志的显示辅助。

两件事：
1. StreamPrinter —— 边收边打增量文本，并管理「当前行」的开合
2. make_tool_tracer —— 打印工具调用日志前，先结束正在输出的那一行

为什么要管「当前行」：模型可能先吐一句话再调工具，如果不管，
工具日志会和正文挤在同一行（"I'll read the file.[工具] file_read(...)"）。
"""
from __future__ import annotations

import sys
from typing import Callable

ToolHook = Callable[[str, str, str], None]


class StreamPrinter:
    """把增量文本打到 stdout，并管理换行。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._line_open = False  # 当前有一行正在输出、还没换行
        self._ever = False  # 整个过程中是否输出过任何文本

    @property
    def started(self) -> bool:
        """是否输出过文本。用来判断最后还要不要打印返回值。"""
        return self._ever

    def __call__(self, chunk: str) -> None:
        if not self.enabled or not chunk:
            return
        if not self._line_open:
            print()  # 让回答从新行开始，不和提示符挤在一起
            self._line_open = True
        self._ever = True
        print(chunk, end="", flush=True)

    def finish(self) -> None:
        """结束当前行。可以重复调用，只有真的开着行时才换行。"""
        if self._line_open:
            print()
            sys.stdout.flush()
            self._line_open = False


def make_tool_tracer(printer: StreamPrinter) -> ToolHook:
    """造一个工具日志回调；打印前会先结束流式输出的当前行。"""

    def trace(name: str, arguments: str, result: str) -> None:
        printer.finish()  # 关键：避免工具日志粘在正文后面
        preview = result.replace("\n", "\\n")
        if len(preview) > 80:
            preview = preview[:80] + "..."
        print(f"[工具] {name}({arguments})", file=sys.stderr)
        print(f"[结果] {preview}", file=sys.stderr)

    return trace
