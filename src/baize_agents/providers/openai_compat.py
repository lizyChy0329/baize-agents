"""OpenAI 兼容接口的极简客户端（只依赖标准库）。

适用于 DeepSeek、OpenRouter、Ollama（/v1）、vLLM 等任何 OpenAI 兼容服务。
只负责「发一次请求」；重试与退避由 providers/base.py 负责。
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any, Iterable

from ..config import ProviderConfig
from .base import Provider, RetryHook, TextHook
from .errors import ProviderError, TransientError, classify_http_error
from .retry import RetryPolicy

__all__ = ["OpenAICompatProvider", "ProviderError", "parse_sse"]


def _merge_tool_call(store: dict[int, dict[str, Any]], fragment: dict[str, Any]) -> None:
    """把一段 tool_call 增量合并进累加器。

    流式下 tool_calls 是**分片**到达的，典型情况：
        {"index":0,"id":"call_x","function":{"name":"file_read","arguments":"{\\"pa"}}
        {"index":0,"function":{"arguments":"th\\":\\"RE"}}
        {"index":0,"function":{"arguments":"ADME.md\\"}"}}
    必须按 index 分槽、把 arguments 字符串一段段接起来，拼错工具就废了。
    """
    index = fragment.get("index", 0)
    slot = store.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if fragment.get("id"):
        slot["id"] = fragment["id"]
    if fragment.get("type"):
        slot["type"] = fragment["type"]
    function = fragment.get("function") or {}
    if function.get("name"):
        slot["function"]["name"] += function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]


def parse_sse(lines: Iterable[str], emit: TextHook) -> dict[str, Any]:
    """把 SSE 行流拼装成一个完整的 message。

    纯逻辑，不涉及网络，所以可以直接喂假数据测试。
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    role = "assistant"

    for raw in lines:
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue  # 坏行直接跳过，不要让整个流崩掉

        choices = chunk.get("choices") or []
        if not choices:
            continue  # 例如只带 usage 的收尾块
        delta = choices[0].get("delta") or {}

        if delta.get("role"):
            role = delta["role"]
        text = delta.get("content")
        if text:
            content_parts.append(text)
            emit(text)
        for fragment in delta.get("tool_calls") or []:
            _merge_tool_call(tool_calls, fragment)

    message: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return message


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        config: ProviderConfig,
        policy: RetryPolicy | None = None,
        *,
        on_retry: RetryHook | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, policy, on_retry=on_retry, **kwargs)
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.api_key = config.resolve_api_key()

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """发一次请求，返回模型回复的完整 message 字典。

        返回可能是三种情况之一：
        - 纯文本回复：  {"role": "assistant", "content": "..."}
        - 请求调用工具：{"role": "assistant", "content": "", "tool_calls": [...]}
        - 两者都有
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise classify_http_error(e.code, body, e.headers.get("Retry-After")) from e
        except urllib.error.URLError as e:
            # 连不上 / 超时：多半是瞬时的，值得重试
            raise TransientError(f"无法连接 {url}：{e.reason}") from e
        except (OSError, http.client.HTTPException) as e:
            raise TransientError(f"请求 {url} 失败：{e}") from e

        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"API 响应格式不符合预期：{data}") from e

    def _stream_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        emit: TextHook,
    ) -> dict[str, Any]:
        """流式发一次请求：边收边吐，最后返回拼装好的完整 message。"""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type:
                    # 对方没按流式返回（有些代理/本地服务会这样），退回普通 JSON
                    return self._read_plain(response, emit)
                lines = (raw.decode("utf-8", errors="replace") for raw in response)
                return parse_sse(lines, emit)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise classify_http_error(e.code, body, e.headers.get("Retry-After")) from e
        except urllib.error.URLError as e:
            raise TransientError(f"无法连接 {url}：{e.reason}") from e
        except (OSError, http.client.HTTPException) as e:
            raise TransientError(f"请求 {url} 失败：{e}") from e

    @staticmethod
    def _read_plain(response: Any, emit: TextHook) -> dict[str, Any]:
        data = json.loads(response.read().decode("utf-8"))
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"API 响应格式不符合预期：{data}") from e
        if message.get("content"):
            emit(message["content"])
        return message
