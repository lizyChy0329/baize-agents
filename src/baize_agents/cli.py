"""命令行入口：baize -m "hello" 或 python -m baize_agents -m "hello"。

里程碑 2.1：provider 已支持 tools，但 CLI 仍是纯问答（runner 循环在 2.3 接入）。
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
        message = provider.chat(messages)
    except ProviderError as e:
        print(f"[错误] {e}")
        return 1

    # provider 现在返回完整 message；这里先只取文本部分。
    print(message.get("content") or "")
    return 0
