"""重试与退避策略。

这里刻意把「计算等多久」（纯函数，好测试）和「真的睡一觉」（IO，测试时替换掉）
分开，这样测退避时序不需要真的等几秒。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .errors import ProviderError, RateLimitError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略。

    max_attempts: 最多尝试几次（含第一次）。1 表示不重试。
    base_delay:   第一次重试前等几秒，之后翻倍。
    max_delay:    单次等待的上限。
    jitter_ratio: 抖动比例。加抖动是为了避免多个客户端「同时醒来」再一起撞服务器。
    """

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter_ratio: float = 0.25

    def delay_for(self, attempt: int, error: Exception, jitter: float = 0.0) -> float:
        """第 attempt 次失败后该等多少秒（attempt 从 1 开始）。

        jitter 是 [0, 1) 的随机数，由调用方传入，方便测试时固定住。
        """
        # 服务器明确说了等多久，就听它的
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            return min(max(0.0, error.retry_after), self.max_delay)

        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return delay + delay * self.jitter_ratio * jitter


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """反复调用 fn，直到成功、或错误不可重试、或用完次数。"""
    attempt = 0
    while True:
        try:
            return fn()
        except ProviderError as e:
            attempt += 1
            if not e.retryable or attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt, e, jitter())
            if on_retry is not None:
                on_retry(attempt, e, delay)
            sleep(delay)
