"""命令行入口：baize -m "hello" 或 python -m baize_agents -m "hello"。

里程碑 2.3：接上 runner 循环，模型可以自己调用工具（目前只有 file_read）。
"""
from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .providers.openai_compat import OpenAICompatProvider, ProviderError
from .runner import RunnerError, run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baize",
        description="最小可用的 agent 骨架：一次问答。",
    )
    parser.add_argument("-m", "--message", required=True, help="要发给模型的消息")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.json）")
    return parser


def _trace_tool_call(name: str, arguments: str, result: str) -> None:
    """把"模型要调什么工具、结果如何"打印到 stderr，方便观察循环在干嘛。

    用 stderr 是为了不污染 stdout 上的最终答案（方便管道处理）。
    """
    preview = result.replace("\n", "\\n")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    print(f"[工具] {name}({arguments})", file=sys.stderr)
    print(f"[结果] {preview}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        provider = OpenAICompatProvider(config.provider)
    except (ConfigError, ProviderError) as e:
        print(f"[错误] {e}")
        return 1

    messages = [{"role": "user", "content": args.message}]

    try:
        reply = run(provider, messages, on_tool_call=_trace_tool_call)
    except (ProviderError, RunnerError) as e:
        print(f"[错误] {e}")
        return 1

    print(reply)
    return 0
