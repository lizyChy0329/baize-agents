"""OpenAI 兼容接口的极简客户端（只依赖标准库）。

适用于 DeepSeek、OpenRouter、Ollama（/v1）、vLLM 等任何 OpenAI 兼容服务。
只负责「发一次请求」；重试与退避由 providers/base.py 负责。
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any

from ..config import ProviderConfig
from .base import Provider, RetryHook
from .errors import ProviderError, TransientError, classify_http_error
from .retry import RetryPolicy

__all__ = ["OpenAICompatProvider", "ProviderError"]


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
