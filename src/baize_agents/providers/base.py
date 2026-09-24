"""Provider 抽象：把「和模型对话」的接口固定下来。

目前只有 OpenAI 兼容一种实现，但接口先立好：
- 换实现（Anthropic 原生 / 本地模型）不用改上层
- 重试逻辑集中在这里，子类只管「发一次请求」这一件事
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..config import ProviderConfig
from .retry import RetryPolicy, retry_call

# 重试回调：(第几次尝试, 错误, 将等待秒数)
RetryHook = Callable[[int, Exception, float], None]

# 流式文本回调：收到一段增量文本就叫一次
TextHook = Callable[[str], None]


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

    def _stream_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        emit: TextHook,
    ) -> dict[str, Any]:
        """流式发一次请求。默认实现：不真流式，拿到结果后一次性吐出。

        支持真流式的子类覆盖它，每收到一段文本就调 emit(片段)。
        """
        message = self._chat_once(messages, tools)
        content = message.get("content")
        if content:
            emit(content)
        return message

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """带重试地请求模型（非流式）。"""
        return retry_call(
            lambda: self._chat_once(messages, tools),
            self.policy,
            sleep=self._sleep,
            on_retry=self._on_retry,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: TextHook | None = None,
    ) -> dict[str, Any]:
        """带重试地流式请求模型，返回完整 message。

        重试的限制：**一旦已经开始往外吐字，就不再重试**。
        因为用户已经看到了半截回答，重试会把开头重复一遍，反而更乱。
        """
        attempt = 0
        while True:
            emitted = False

            def emit(chunk: str) -> None:
                nonlocal emitted
                emitted = True
                if on_text is not None:
                    on_text(chunk)

            try:
                return self._stream_once(messages, tools, emit)
            except ProviderError as e:
                attempt += 1
                if emitted or not e.retryable or attempt >= self.policy.max_attempts:
                    raise
                delay = self.policy.delay_for(attempt, e, random.random())
                if self._on_retry is not None:
                    self._on_retry(attempt, e, delay)
                self._sleep(delay)
