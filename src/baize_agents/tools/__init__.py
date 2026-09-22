"""工具层：注册表 + 统一执行入口。

runner 负责"循环"，tools 负责"有哪些工具、怎么执行"。
要加新工具，只需 import 进来、往 REGISTRY 里加一行。
"""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolError
from .file_read import FILE_READ

# 所有可用工具：名字 → Tool
REGISTRY: dict[str, Tool] = {
    FILE_READ.name: FILE_READ,
}


def get_schemas() -> list[dict[str, Any]]:
    """返回所有工具的说明书，直接塞进请求体的 tools 字段。"""
    return [tool.schema() for tool in REGISTRY.values()]


def execute(name: str, arguments: dict[str, Any]) -> str:
    """按名字执行工具，永远返回字符串。

    工具失败不抛异常，而是返回 "[错误] ..." 文本，交给 runner 回喂给模型，
    让模型自己决定下一步（重试 / 换个路径 / 放弃）。
    这样单个工具出错不会崩掉整个 agent 循环。
    """
    tool = REGISTRY.get(name)
    if tool is None:
        return f"[错误] 未知工具：{name}"

    try:
        return tool.func(**arguments)
    except ToolError as e:
        return f"[错误] {e}"
    except TypeError as e:
        # 模型给的参数不匹配：少传、多传、名字写错
        return f"[错误] 调用 {name} 的参数不对：{e}"
    except Exception as e:  # noqa: BLE001 —— 故意兜底，绝不让工具崩掉 agent 循环
        return f"[错误] {name} 执行失败：{type(e).__name__}: {e}"
