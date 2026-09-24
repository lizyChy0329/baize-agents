"""系统提示词：对话最开头给模型的「岗位说明书」。

作用：
1. 告诉模型它是谁、在什么环境里
2. 明确要求它「需要文件内容时主动调用工具，不要猜」
   —— 实测能显著降低「该调工具却不调」的概率
3. 规定回答风格

注意：系统提示词**只在发给 API 时临时插入**，不写进会话历史。
这样改了提示词，所有已有会话立刻生效，磁盘上也不会有重复副本。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import tools

DEFAULT_IDENTITY = """\
你是 baize，一个运行在用户终端里的命令行助手。

工作原则：
- 需要了解文件内容时，主动调用工具去读，不要凭猜测或记忆回答。
- 工具返回的结果是事实，优先采信它。
- 回答简洁直接，不要客套和重复用户的话。
- 用与用户相同的语言回答。"""


def _environment_block() -> str:
    tool_lines = "\n".join(
        f"- {tool.name}: {tool.description}" for tool in tools.REGISTRY.values()
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "# 当前环境\n"
        f"- 工作目录：{Path.cwd()}\n"
        f"- 当前时间：{now}\n"
        "- 可用工具：\n"
        f"{tool_lines}"
    )


def build_system_prompt(custom: str | None = None) -> str:
    """组装最终的系统提示词。

    custom 为 None 时用内置的默认身份描述；
    无论哪种情况都会追加上「当前环境」（工作目录、时间、工具清单），
    因为这些是模型无法自己知道的事实。
    """
    identity = (custom.strip() if custom else "") or DEFAULT_IDENTITY
    return f"{identity}\n\n{_environment_block()}"
