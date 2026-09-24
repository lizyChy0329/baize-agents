"""交互式 REPL：像聊天一样连续对话，不必每次敲 -m。

设计要点：
- 一轮 = 你输入一句 → 跑 agent 循环 → 打印回答
- 只有成功的一轮才落盘；中途 Ctrl-C 会把这一轮从内存里回滚，
  免得留下"有提问没回答"的半截历史
- 命令以 / 开头，和发给模型的消息区分开
"""
from __future__ import annotations

import sys
from typing import Any, Callable

from . import tools
from .providers.base import Provider
from .providers.errors import ProviderError
from .runner import RunnerError, run
from .session import Session
from .stream import StreamPrinter, make_tool_tracer
from . import style

HELP = """可用命令：
  /help    显示这份帮助
  /tools   列出可用工具
  /clear   清空当前会话的历史
  /exit    退出（也可以按 Ctrl-D）
"""


def _handle_command(line: str, session: Session) -> bool:
    """处理 / 开头的命令。返回 True 表示要退出。"""
    cmd = line.split()[0].lower()

    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        print(style.notice(HELP), file=sys.stderr)
    elif cmd == "/tools":
        for tool in tools.REGISTRY.values():
            print(style.notice(f"  {tool.name}: {tool.description}"), file=sys.stderr)
    elif cmd == "/clear":
        session.clear()
        print(style.notice(f"已清空会话 {session.name}"), file=sys.stderr)
    else:
        print(style.warning(f"未知命令：{cmd}（试试 /help）"), file=sys.stderr)
    return False


def run_repl(
    provider: Provider,
    session: Session,
    system_prompt: str | None = None,
    stream: bool = True,
) -> int:
    print(
        style.notice(
            f"baize-agents 交互模式（会话 {session.name}）。/help 看命令，/exit 退出。"
        ),
        file=sys.stderr,
    )
    if len(session):
        print(style.notice(f"已载入 {len(session)} 条历史，接着聊吧。"), file=sys.stderr)

    while True:
        try:
            line = input("> ")
        except EOFError:  # Ctrl-D
            print(file=sys.stderr)
            return 0
        except KeyboardInterrupt:  # Ctrl-C 在输入时：清掉这行，继续
            print(file=sys.stderr)
            continue

        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if _handle_command(line, session):
                return 0
            continue

        # 记下这一轮开始前的位置，便于中断时回滚
        before = len(session.messages)
        session.add({"role": "user", "content": line})

        printer = StreamPrinter(enabled=stream)
        try:
            reply = run(
                provider,
                session.messages,
                on_tool_call=make_tool_tracer(printer),
                system_prompt=system_prompt,
                on_text=printer,
                on_turn_end=printer.end_turn,
                stream=stream,
            )
        except (ProviderError, RunnerError) as e:
            printer.finish()
            print(style.error(f"[错误] {e}"), file=sys.stderr)
            del session.messages[before:]  # 回滚这一轮
            continue
        except KeyboardInterrupt:
            printer.finish()
            print(style.warning("\n[已中断]"), file=sys.stderr)
            del session.messages[before:]
            continue

        # 这一轮完整了，才落盘
        session.sync()
        if printer.started:
            printer.finish()  # 已经边收边打了，只需收尾换行
        else:
            print(reply)
