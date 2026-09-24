"""历史压缩（compact）：把久远的对话摘要成一段，腾出上下文空间。

## 什么时候需要

上下文窗口有限。聊到几十轮、或者读过几个大文件之后，
消息历史必然撑满，再问就会拿到 context_length_exceeded。
截断工具结果只治了"单条太大"，这里治的是"聊得太久"。

## 最难的一点：边界不能切断工具调用

OpenAI 格式要求：assistant 的每个 tool_call 后面必须跟一条配对的 tool 消息。
如果切开的位置落在这两者中间：

    [user]      读文件
    [assistant] tool_calls: c1     ← 从这里切开
    [tool]      c1 的结果           ← 变成孤儿，直接 400
    [assistant] 读完了

所以切点必须落在 **user 消息**上 —— 那里天然是"一轮对话"的起点。

## 做法

    ┌───────────── 摘要（新的 system 消息）─────────────┐
    │ 用户的目标、已确认的事实、未完成的任务            │
    └───────────────────────────────────────────────────┘
    [user]      ...        ← 最近的消息原样保留
    [assistant] ...
"""
from __future__ import annotations

from typing import Any

from ..providers.base import Provider
from .tokens import messages_tokens

SUMMARY_PROMPT = """\
请把下面这段对话历史压缩成一份简明摘要，供后续对话继续使用。

必须保留：
- 用户的目标、需求和偏好（例如"我叫小明""用中文回答"）
- 已经确认的事实和结论（例如"这个项目的 version 是 0.1.0"）
- 未完成的任务和下一步计划
- 涉及的关键文件名、路径、标识符

可以丢弃：
- 寒暄和客套
- 工具调用的原始输出（只留结论）
- 重复的内容

用第三人称、条目式写，不要编造原文没有的信息。

对话历史：
---
{history}
---

摘要："""


def find_cut(messages: list[dict[str, Any]], keep_recent_tokens: int) -> int:
    """算出该从哪里切：返回下标 i，表示保留 messages[i:]。返回 0 表示不压缩。

    做法：以 user 消息为「轮次边界」，从最后一轮往前累加，
    累到快超过 keep_recent_tokens 就停。这样天然不会切断工具调用对
    （assistant tool_calls + tool 结果都在同一轮里）。
    """
    starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(starts) < 2:
        # 没有轮次边界，或者只有一轮（切了就没历史了），不动
        return 0

    ends = starts[1:] + [len(messages)]
    total = 0
    cut = starts[-1]  # 至少保留最后一轮

    for k in range(len(starts) - 1, -1, -1):
        turn_tokens = messages_tokens(messages[starts[k] : ends[k]])
        # 至少保留一轮，所以只在 k 不是最后一轮时才考虑停
        if total + turn_tokens > keep_recent_tokens and k < len(starts) - 1:
            break
        total += turn_tokens
        cut = starts[k]

    return cut if 0 < cut < len(messages) else 0


def format_history(messages: list[dict[str, Any]]) -> str:
    """把消息渲染成可读文本，喂给模型做摘要。"""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            names = ", ".join(
                (c.get("function") or {}).get("name", "?") for c in m["tool_calls"]
            )
            lines.append(f"[助手] 请求调用工具：{names}")
            if content:
                lines.append(f"       同时说：{content}")
        elif role == "tool":
            preview = content.replace("\n", " ")[:200]
            lines.append(f"[工具结果] {preview}")
        elif role == "system":
            lines.append(f"[背景] {content}")
        else:
            label = {"user": "用户", "assistant": "助手"}.get(role, role)
            lines.append(f"[{label}] {content}")
    return "\n".join(lines)


def summarize(provider: Provider, messages: list[dict[str, Any]]) -> str:
    """让模型把这段历史摘要成一段文字。"""
    history = format_history(messages)
    prompt = SUMMARY_PROMPT.format(history=history)
    message = provider.chat([{"role": "user", "content": prompt}])
    return (message.get("content") or "").strip()


def compact(
    provider: Provider,
    messages: list[dict[str, Any]],
    keep_recent_tokens: int,
) -> tuple[str, int] | None:
    """就地压缩 messages。返回 (摘要, 被压缩掉的消息条数)；不需要压缩则返回 None。

    压缩后的 messages 变成：[摘要 system 消息] + 最近的消息。
    """
    cut = find_cut(messages, keep_recent_tokens)
    if cut == 0:
        return None

    dropped = messages[:cut]
    kept = messages[cut:]
    summary = summarize(provider, dropped)

    summary_message = {
        "role": "system",
        "content": "以下是本次会话较早内容的摘要（原始消息已被压缩）：\n" + summary,
    }
    # 必须就地修改，这样持有同一个列表引用的 session 也能看到变化
    messages[:] = [summary_message, *kept]
    return summary, len(dropped)


def needs_compaction(messages: list[dict[str, Any]], threshold_tokens: int) -> bool:
    return messages_tokens(messages) > threshold_tokens
