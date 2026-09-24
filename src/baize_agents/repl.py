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
from .config import ContextConfig
from .governance import maybe_compact
from .providers.base import Provider
from .providers.errors import ProviderError
from .runner import RunnerError, run
from .session import Session
from .stream import StreamPrinter, make_tool_tracer
from . import style

HELP = """可用命令：
  /help     显示这份帮助
  /tools    列出可用工具
  /compact  现在就把历史压缩成摘要（腾出上下文空间）
  /tokens   看看当前历史占多少 token
  /clear    清空当前会话的历史
  /exit     退出（也可以按 Ctrl-D）
"""


def _handle_command(
    line: str,
    session: Session,
    provider: Provider | None = None,
    context: ContextConfig | None = None,
) -> bool:
    """处理 / 开头的命令。返回 True 表示要退出。"""
    cmd = line.split()[0].lower()

    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        print(style.notice(HELP), file=sys.stderr)
    elif cmd == "/tools":
        for tool in tools.REGISTRY.values():
            print(style.notice(f"  {tool.name}: {tool.description}"), file=sys.stderr)
    elif cmd == "/tokens":
        from .context.tokens import messages_tokens

        used = messages_tokens(session.messages)
        limit = context.compact_threshold_tokens if context else 24000
        print(
            style.notice(
                f"当前历史 {len(session.messages)} 条，约 {used} tokens"
                f"（超过 {limit} 会自动压缩）"
            ),
            file=sys.stderr,
        )
    elif cmd == "/compact":
        if provider is None or context is None:
            print(style.warning("此模式不支持 /compact"), file=sys.stderr)
        else:
            _do_compact(provider, session, context, force=True)
    elif cmd == "/clear":
        session.clear()
        print(style.notice(f"已清空会话 {session.name}"), file=sys.stderr)
    else:
        print(style.warning(f"未知命令：{cmd}（试试 /help）"), file=sys.stderr)
    return False


def _do_compact(
    provider: Provider,
    session: Session,
    context: ContextConfig,
    *,
    force: bool = False,
) -> None:
    """压缩历史。force=True 时忽略阈值（/compact 手动触发用）。"""
    from .context.compact import compact, needs_compaction
    from .context.tokens import messages_tokens

    if not force and not needs_compaction(session.messages, context.compact_threshold_tokens):
        return
    if len(session.messages) <= 2:
        print(style.notice("历史太短，无需压缩"), file=sys.stderr)
        return

    before = messages_tokens(session.messages)
    try:
        result = compact(provider, session.messages, context.keep_recent_tokens)
    except ProviderError as e:
        print(style.warning(f"压缩失败：{e}"), file=sys.stderr)
        return
    if result is None:
        print(style.notice("历史太短，无需压缩"), file=sys.stderr)
        return

    summary, dropped = result
    session.record_compaction(summary)
    after = messages_tokens(session.messages)
    print(
        style.notice(
            f"[压缩] {dropped} 条消息 → 摘要 {len(summary)} 字；"
            f"约 {before} → {after} tokens"
        ),
        file=sys.stderr,
    )


def run_repl(
    provider: Provider,
    session: Session,
    system_prompt: str | None = None,
    stream: bool = True,
    context: ContextConfig | None = None,
) -> int:
    ctx = context or ContextConfig()
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
            if _handle_command(line, session, provider, ctx):
                return 0
            continue

        # 每轮开始前检查一次：历史太长就先压缩
        _do_compact(provider, session, ctx)

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
                max_tool_result_tokens=ctx.max_tool_result_tokens,
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
