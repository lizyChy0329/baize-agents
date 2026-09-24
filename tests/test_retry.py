"""重试与退避的测试。

关键：用假的 sleep 和假的 jitter，这样测时序逻辑不用真的等几秒。
"""
from __future__ import annotations

import pytest

from baize_agents.providers.errors import (
    AuthError,
    ProviderError,
    RateLimitError,
    TransientError,
)
from baize_agents.providers.retry import RetryPolicy, retry_call


class FakeClock:
    """记录「睡了几次、每次多久」，但真的不睡。"""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.slept)


# ---------------------------------------------------------------- 退避计算


def test_delay_grows_exponentially():
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter_ratio=0.0)
    delays = [policy.delay_for(n, TransientError("boom")) for n in (1, 2, 3, 4)]
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_delay_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter_ratio=0.0)
    assert policy.delay_for(10, TransientError("boom")) == 5.0


def test_jitter_adds_up_to_ratio():
    policy = RetryPolicy(base_delay=4.0, jitter_ratio=0.25)
    # jitter=0 → 不加；jitter=1 → 加满 25%
    assert policy.delay_for(1, TransientError("x"), jitter=0.0) == 4.0
    assert policy.delay_for(1, TransientError("x"), jitter=1.0) == 5.0


def test_retry_after_wins_over_backoff():
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
    error = RateLimitError("限流", retry_after=7.5)
    assert policy.delay_for(1, error) == 7.5


def test_retry_after_is_capped_by_max_delay():
    policy = RetryPolicy(max_delay=10.0)
    error = RateLimitError("限流", retry_after=999.0)
    assert policy.delay_for(1, error) == 10.0


# ---------------------------------------------------------------- 重试流程


def test_retries_until_success():
    clock = FakeClock()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("网络抖动")
        return "ok"

    result = retry_call(
        flaky,
        RetryPolicy(max_attempts=5, base_delay=1.0, jitter_ratio=0.0),
        sleep=clock.sleep,
        jitter=lambda: 0.0,
    )

    assert result == "ok"
    assert calls["n"] == 3
    assert clock.slept == [1.0, 2.0]  # 第1次失败等1s，第2次失败等2s


def test_non_retryable_error_fails_immediately():
    clock = FakeClock()
    calls = {"n": 0}

    def bad_key():
        calls["n"] += 1
        raise AuthError("密钥错误")

    with pytest.raises(AuthError):
        retry_call(bad_key, RetryPolicy(max_attempts=5), sleep=clock.sleep)

    assert calls["n"] == 1  # 只试了一次
    assert clock.slept == []  # 一次都没等


def test_gives_up_after_max_attempts():
    clock = FakeClock()
    calls = {"n": 0}

    def always_down():
        calls["n"] += 1
        raise TransientError("一直挂")

    with pytest.raises(TransientError):
        retry_call(
            always_down,
            RetryPolicy(max_attempts=3, base_delay=1.0, jitter_ratio=0.0),
            sleep=clock.sleep,
            jitter=lambda: 0.0,
        )

    assert calls["n"] == 3
    assert clock.slept == [1.0, 2.0]


def test_max_attempts_one_means_no_retry():
    calls = {"n": 0}

    def always_down():
        calls["n"] += 1
        raise TransientError("挂")

    with pytest.raises(TransientError):
        retry_call(always_down, RetryPolicy(max_attempts=1), sleep=lambda _: None)

    assert calls["n"] == 1


def test_on_retry_hook_reports_attempt_error_and_delay():
    events: list[tuple[int, Exception, float]] = []

    def flaky():
        if not events:
            raise TransientError("抖一下")
        return "好了"

    retry_call(
        flaky,
        RetryPolicy(max_attempts=3, base_delay=2.0, jitter_ratio=0.0),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
        on_retry=lambda attempt, error, delay: events.append((attempt, error, delay)),
    )

    assert len(events) == 1
    attempt, error, delay = events[0]
    assert attempt == 1
    assert isinstance(error, TransientError)
    assert delay == 2.0


def test_non_provider_exceptions_are_not_swallowed():
    def bug():
        raise ValueError("这是代码 bug，不该被重试逻辑吞掉")

    with pytest.raises(ValueError):
        retry_call(bug, RetryPolicy(max_attempts=3), sleep=lambda _: None)


def test_provider_error_base_is_not_retryable():
    assert ProviderError.retryable is False
    assert TransientError.retryable is True
    assert RateLimitError.retryable is True
    assert AuthError.retryable is False


# ---------------------------------------------------------------- 流式重试


class _StreamProvider:
    """最小的 Provider 子类，用来测 chat_stream 的重试行为。"""

    def __init__(self, script, policy, clock):
        from baize_agents.providers.base import Provider

        class P(Provider):
            def __init__(self):
                super().__init__(config=None, policy=policy, sleep=clock.sleep)  # type: ignore[arg-type]
                self.script = list(script)

            def _chat_once(self, messages, tools=None):  # 测试用不到
                raise AssertionError("不该走非流式路径")

            def _stream_once(self, messages, tools, emit):
                action = self.script.pop(0)
                if isinstance(action, Exception):
                    raise action
                if action == "emit-then-fail":
                    emit("半截")
                    raise TransientError("中途断了")
                emit(action)
                return {"role": "assistant", "content": action}

        self.instance = P()


def test_stream_retries_when_nothing_emitted_yet():
    clock = FakeClock()
    holder = _StreamProvider([TransientError("连不上"), "成功"], RetryPolicy(max_attempts=3), clock)

    chunks: list[str] = []
    msg = holder.instance.chat_stream(
        [{"role": "user", "content": "hi"}], on_text=chunks.append
    )

    assert msg["content"] == "成功"
    assert chunks == ["成功"]
    # 没吐字，可以重试。等待时长 = base_delay(1.0) + 0~25% 抖动
    assert len(clock.slept) == 1
    assert 1.0 <= clock.slept[0] <= 1.25


def test_stream_does_not_retry_after_emitting():
    """关键行为：已经吐了半截就不能重试，否则用户会看到重复的开头。"""
    clock = FakeClock()
    holder = _StreamProvider(["emit-then-fail"], RetryPolicy(max_attempts=3), clock)

    chunks: list[str] = []
    with pytest.raises(TransientError):
        holder.instance.chat_stream([{"role": "user", "content": "hi"}], on_text=chunks.append)

    assert chunks == ["半截"]  # 已经吐出去的字收不回来
    assert clock.slept == []  # 一次都没重试


def test_stream_imports_provider_error():
    """回归测试：base.py 曾经漏导入 ProviderError，导致流式重试直接崩。"""
    from baize_agents.providers import base

    assert hasattr(base, "ProviderError")
