"""OpenAI 兼容接口的极简客户端（只依赖标准库）。

里程碑 1：只支持纯文本问答（POST /chat/completions），暂不支持工具调用。
适用于 DeepSeek、OpenRouter、Ollama（/v1）、vLLM 等任何 OpenAI 兼容服务。
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

from ..config import ProviderConfig


class ProviderError(Exception):
    """调用模型 API 失败。"""


class OpenAICompatProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.api_key = config.resolve_api_key()

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送消息列表，返回模型的文本回复。

        messages 形如 [{"role": "user", "content": "hello"}]，
        role 取值：system / user / assistant。
        """
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages, "stream": False}
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
            detail = e.read().decode("utf-8", errors="replace")
            raise ProviderError(f"API 返回 HTTP {e.code}：{detail}") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"无法连接 {url}：{e.reason}") from e
        except (OSError, http.client.HTTPException) as e:
            raise ProviderError(f"请求 {url} 失败：{e}") from e

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"API 响应格式不符合预期：{data}") from e
