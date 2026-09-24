"""流式输出与工具日志的显示辅助。

三件事：
1. StreamPrinter  —— 边收边打增量文本，并区分「铺垫」与「答案」
2. make_tool_tracer —— 打印工具日志前，先结束正在输出的那一行
3. 颜色区分：铺垫暗灰、工具青色、答案素净

## 为什么需要缓冲

流式下文本是一个个字到达的，而「这段文字是铺垫还是最终答案」只有等**这一轮结束**
才知道（看它后面跟不跟 tool_calls）。所以：

- 先攒着不打印
- 攒够 PREAMBLE_FLUSH_AFTER 个字符 → 判定为正文（答案）→ 转实时输出
- 这一轮结束时还没攒够 → 按判定结果上色（铺垫灰、答案素净）

铺垫通常只有一句话（十几字），答案通常更长，所以这个阈值很少误判；
代价是答案开头会攒够阈值才一次性冒出（约 0.1~0.2 秒）。
"""
from __future__ import annotations

import sys
from typing import Callable

from . import style

ToolHook = Callable[[str, str, str], None]

# 攒够这么多字符，就认定它是正文（答案），转为实时输出。
#
# 这是启发式，两边都可能误判，实测调优后的取舍：
#   设小 → 实测模型的一句话铺垫可以有 50+ 字，会被误判成正文（该上灰却没上）
#   设大 → 答案要先攒够才出现，首字延迟变大（100 字约多等 0.4s）
# 「铺垫没上灰」比「首字晚一点」更伤体验，所以取值偏大：
PREAMBLE_FLUSH_AFTER = 100


class StreamPrinter:
    """按轮次管理文本输出：区分铺垫（暗灰）与答案（素净）。"""

    def __init__(self, enabled: bool = True, flush_after: int = PREAMBLE_FLUSH_AFTER) -> None:
        self.enabled = enabled
        self.flush_after = flush_after
        self._pending: list[str] = []
        self._size = 0
        self._streaming = False  # 本轮是否已转入实时输出
        self._line_open = False  # 当前有一行开着、还没换行
        self._ever = False  # 整个过程中是否输出过文本

    @property
    def started(self) -> bool:
        """是否输出过文本。用来判断最后还要不要打印返回值。"""
        return self._ever

    def __call__(self, chunk: str) -> None:
        """收到一段文本增量（作为 on_text 传给 runner）。"""
        if not self.enabled or not chunk:
            return
        if self._streaming:
            self._write(chunk, dim=False)
            return

        self._pending.append(chunk)
        self._size += len(chunk)
        if self._size >= self.flush_after:
            # 攒够了，判定为正文，把攒下的一次性吐出来并转为实时
            self._flush(dim=False)

    def end_turn(self, is_preamble: bool) -> None:
        """一轮结束。is_preamble 为真表示这一轮的文本是铺垫（后面跟了工具调用）。"""
        if self._pending:
            self._flush(dim=is_preamble)
        self._streaming = False  # 下一轮重新判断

    def _flush(self, dim: bool) -> None:
        text = "".join(self._pending)
        self._pending.clear()
        self._size = 0
        if not text:
            return
        self._streaming = True
        self._write(text, dim=dim)

    def _write(self, text: str, dim: bool) -> None:
        if not self._line_open:
            print()  # 让内容从新行开始，不和提示符挤在一起
            self._line_open = True
        self._ever = True
        # 注意：内容走 stdout，所以按 stdout 是否终端来决定颜色
        shown = style.stdout_style().dim(text) if dim else text
        print(shown, end="", flush=True)

    def finish(self) -> None:
        """结束当前行。可以重复调用，只有真的开着行时才换行。"""
        if self._line_open:
            print()
            sys.stdout.flush()
            self._line_open = False


def make_tool_tracer(printer: StreamPrinter) -> ToolHook:
    """造一个工具日志回调；打印前会先结束流式输出的当前行。

    用青色的 [工具] 标记 agent 的动作，用暗灰的 [结果] 展示细节，
    这样它们和模型的回答（不上色）在视觉上一眼能分开。
    """

    def trace(name: str, arguments: str, result: str) -> None:
        printer.finish()  # 关键：避免工具日志粘在正文后面
        preview = result.replace("\n", "\\n")
        if len(preview) > 80:
            preview = preview[:80] + "..."
        print(style.tool_call(f"[工具] {name}({arguments})"), file=sys.stderr)
        print(style.tool_result(f"[结果] {preview}"), file=sys.stderr)

    return trace
