"""工具的通用类型定义。

一个工具 = 一段"说明书"（给模型看）+ 一个函数（给程序执行）。
把两者绑在一个对象里，就不会出现"说明书改了、函数没改"的错位。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ToolError(Exception):
    """工具执行失败（预期内的错误）。会被 tools.execute 转成文本回喂给模型。"""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI 请求体 tools 字段要求的 JSON Schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
