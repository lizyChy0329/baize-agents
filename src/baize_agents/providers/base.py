"""Provider 抽象：把「和模型对话」的接口固定下来。

目前只有 OpenAI 兼容一种实现，但接口先立好：
- 换实现（Anthropic 原生 / 本地模型）不用改上层
- 重试逻辑集中在这里，子类只管「发一次请求」这一件事
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..config import ProviderConfig
from .retry import RetryPolicy, retry_call

# 重试回调：(第几次尝试, 错误, 将等待秒数)
RetryHook = Callable[[int, Exception, float], None]


class Provider(ABC):
    def __init__(
        self,
        config: ProviderConfig,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: RetryHook | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._on_retry = on_retry

    @abstractmethod
    def _chat_once(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """发一次请求，返回完整 message。失败时抛分类好的 ProviderError。

        子类只需要实现这一个方法，重试由基类负责。
        """

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """带重试地请求模型。"""
        return retry_call(
            lambda: self._chat_once(messages, tools),
            self.policy,
            sleep=self._sleep,
            on_retry=self._on_retry,
        )
