"""命令行入口：baize -m "hello" 或 python -m baize_agents -m "hello"。

里程碑 5：不带 -m 时进入交互模式（REPL），可以像聊天一样连续对话。

用法：
    baize                            # 交互模式（推荐）
    baize -m "hello"                 # 一次性提问
    baize -m "我叫什么？"            # 会记得上文
    baize -s work -m "..."           # 用名为 work 的会话
    baize --no-session -m "..."      # 一次性问答，不读也不写历史
    baize -p ollama -m "..."         # 换一个 provider
    baize --list                     # 列出所有会话
    baize -s work --clear            # 清空某个会话
"""
from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .prompt import build_system_prompt
from .providers.errors import ProviderError
from .providers.openai_compat import OpenAICompatProvider
from .repl import run_repl
from .runner import RunnerError, run
from .session import Session, SessionError, list_sessions
from .stream import StreamPrinter, make_tool_tracer
from . import style


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baize",
        description="最小可用的 agent 骨架：带工具的问答。",
    )
    parser.add_argument("-m", "--message", help="要发给模型的消息")
    parser.add_argument(
        "-s",
        "--session",
        default="default",
        help="会话名，历史存到 ~/.baize-agents/sessions/<名字>.jsonl（默认 default）",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="不读写会话历史，纯一次性问答",
    )
    parser.add_argument("--list", action="store_true", help="列出所有会话后退出")
    parser.add_argument("--clear", action="store_true", help="清空指定会话后退出")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.json）")
    parser.add_argument(
        "--system",
        default=None,
        help="覆盖系统提示词（优先级高于 config.json 的 system_prompt）",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭流式输出（一次性等完整个回答，便于调试/对比）",
    )
    parser.add_argument(
        "--show-system",
        action="store_true",
        help="打印最终的系统提示词后退出（不调模型）",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default=None,
        help="用哪个 provider（config.json 里定义的名字，默认取 default_provider）",
    )
    return parser


def _fail(message: str) -> None:
    """统一的错误输出（红色）。"""
    print(style.error(f"[错误] {message}"))


def _trace_retry(attempt: int, error: Exception, delay: float) -> None:
    """重试时提示一下，否则用户会以为程序卡住了。"""
    print(
        style.warning(f"[重试 {attempt}] {error}")
        + "\n         "
        + style.notice(f"等待 {delay:.1f}s 后重试..."),
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- 不消耗模型的子命令 ----
    if args.list:
        names = list_sessions()
        print("\n".join(names) if names else style.notice("（还没有任何会话）"))
        return 0

    if args.clear:
        try:
            Session(args.session).clear()
        except SessionError as e:
            _fail(str(e))
            return 1
        print(style.notice(f"已清空会话：{args.session}"))
        return 0

    if not args.message and args.no_session:
        _fail("--no-session 只能用于一次性问答；交互模式需要有会话")
        return 1

    # ---- 组装系统提示词 ----
    try:
        config = load_config(args.config)
    except ConfigError as e:
        _fail(str(e))
        return 1

    system_prompt = build_system_prompt(args.system or config.system_prompt)

    if args.show_system:
        print(system_prompt)
        return 0

    # ---- 准备 provider ----
    try:
        provider_config = config.get(args.provider)
        provider = OpenAICompatProvider(
            provider_config, config.retry, on_retry=_trace_retry
        )
    except (ConfigError, ProviderError) as e:
        _fail(str(e))
        return 1

    if len(config.names) > 1:
        used = args.provider or config.default
        print(style.notice(f"[provider] {used} ({provider_config.model})"), file=sys.stderr)

    # ---- 准备消息历史 ----
    try:
        session = None if args.no_session else Session(args.session)
        if session is not None:
            session.load()
    except SessionError as e:
        _fail(str(e))
        return 1

    if session is not None:
        if len(session):
            print(style.notice(f"[会话 {session.name}] 载入 {len(session)} 条历史"), file=sys.stderr)

    # ---- 没有 -m：进交互模式 ----
    if not args.message:
        assert session is not None  # 前面已挡住 --no-session 的情况
        return run_repl(
            provider,
            session,
            system_prompt=system_prompt,
            stream=not args.no_stream,
        )

    # ---- 有 -m：一次性问答 ----
    if session is not None:
        session.add({"role": "user", "content": args.message})
        session.sync()  # 先把用户消息落盘，避免后面崩溃丢失
        messages = session.messages
    else:
        messages = [{"role": "user", "content": args.message}]

    # ---- 跑 agent 循环 ----
    printer = StreamPrinter(enabled=not args.no_stream)
    try:
        reply = run(
            provider,
            messages,
            on_tool_call=make_tool_tracer(printer),
            system_prompt=system_prompt,
            on_text=printer,
            stream=not args.no_stream,
        )
    except (ProviderError, RunnerError) as e:
        printer.finish()
        _fail(str(e))
        return 1
    finally:
        # 即使中途失败，也把已经发生的消息落盘
        if session is not None:
            session.sync()

    if printer.started:
        printer.finish()  # 已经边收边打了，只需收尾换行
    else:
        print(reply)
    return 0
