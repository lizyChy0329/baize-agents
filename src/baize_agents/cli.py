"""命令行入口：baize -m "hello" 或 python -m baize_agents -m "hello"。

里程碑 1：一次性问答，无工具、无会话记忆。
"""
from __future__ import annotations

import argparse

from .config import ConfigError, load_config
from .providers.openai_compat import OpenAICompatProvider, ProviderError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baize",
        description="最小可用的 agent 骨架：一次问答。",
    )
    parser.add_argument("-m", "--message", required=True, help="要发给模型的消息")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.json）")
    return parser


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
        reply = provider.chat(messages)
    except ProviderError as e:
        print(f"[错误] {e}")
        return 1

    print(reply)
    return 0
