"""Agent 循环：请求模型 → 执行工具 → 回喂结果 → 直到模型给出最终答案。

这是整个项目的"心脏"。里程碑 1 是一条直线（问 → 答），
里程碑 2 变成一个环（问 → 模型要工具 → 执行 → 回喂 → 再问 → ... → 答）。
"""
from __future__ import annotations

import json
from typing import Any, Callable

from . import tools
from .providers.base import Provider

# 一轮 = 一次模型请求（一次可能执行多个工具）
MAX_TURNS = 10

# 回调签名：(工具名, 参数字符串, 执行结果字符串)
ToolCallHook = Callable[[str, str, str], None]


class RunnerError(Exception):
    """循环没能正常结束。"""


def run(
    provider: Provider,
    messages: list[dict[str, Any]],
    max_turns: int = MAX_TURNS,
    on_tool_call: ToolCallHook | None = None,
    system_prompt: str | None = None,
) -> str:
    """跑完一个 agent 循环，返回模型的最终文本回答。

    注意：会**就地修改** messages（把模型回复和工具结果追加进去）。
    这份不断变长的 messages 就是模型"记得刚才干了什么"的原因，
    也是后续做会话持久化的基础。

    system_prompt 只在发给 API 时临时插在最前面，**不会写进 messages**，
    所以改了提示词能立刻对所有已有会话生效，磁盘上也不会重复存。
    """
    schemas = tools.get_schemas()

    for _ in range(max_turns):
        # ① 问模型（带上工具说明书；系统提示词临时插入）
        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}, *messages]
        message = provider.chat(payload, tools=schemas)
        messages.append(message)

        tool_calls = message.get("tool_calls")
        # ② 没要工具 → 这就是最终答案，收工
        if not tool_calls:
            return message.get("content") or ""

        # ③ 逐个执行模型要的工具
        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"] or "{}"

            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                # 模型给了一段不是 JSON 的参数，把错误回喂给它
                result = f"[错误] 工具参数不是合法 JSON：{raw_args}"
            else:
                result = tools.execute(name, arguments)

            if on_tool_call is not None:
                on_tool_call(name, raw_args, result)

            # ④ 把结果记进历史（必须带上 tool_call_id 配对）
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )
        # ⑤ 回到 ①：这次模型能看到工具结果了

    raise RunnerError(f"超过最大轮数 {max_turns}，模型可能陷入了死循环")
