"""HTTP 状态码 → 错误分类的测试。"""
from __future__ import annotations

from baize_agents.providers.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    QuotaError,
    RateLimitError,
    TransientError,
    classify_http_error,
    parse_retry_after,
)


def test_429_is_retryable_rate_limit():
    err = classify_http_error(429, '{"error":"rate limited"}')
    assert isinstance(err, RateLimitError)
    assert err.retryable is True
    assert err.retry_after is None


def test_429_picks_up_retry_after_header():
    err = classify_http_error(429, "slow down", retry_after_header="3")
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 3.0


def test_401_and_403_are_auth_errors():
    for status in (401, 403):
        err = classify_http_error(status, "bad key")
        assert isinstance(err, AuthError)
        assert err.retryable is False


def test_402_is_quota_error():
    err = classify_http_error(402, "Insufficient Balance")
    assert isinstance(err, QuotaError)
    assert err.retryable is False


def test_5xx_is_transient():
    for status in (500, 502, 503, 504):
        err = classify_http_error(status, "upstream error")
        assert isinstance(err, TransientError)
        assert err.retryable is True


def test_context_overflow_is_detected():
    body = '{"error":{"code":"context_length_exceeded","message":"too long"}}'
    err = classify_http_error(400, body)
    assert isinstance(err, ContextOverflowError)
    assert err.retryable is False  # 重试没用，得压缩历史


def test_plain_400_is_generic_error():
    err = classify_http_error(400, '{"error":"bad request"}')
    assert type(err) is ProviderError
    assert err.retryable is False


def test_unknown_status_is_generic_error():
    err = classify_http_error(418, "I'm a teapot")
    assert type(err) is ProviderError


def test_body_is_truncated_in_message():
    err = classify_http_error(500, "x" * 5000)
    assert len(str(err)) < 500


def test_parse_retry_after():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    # HTTP-date 形式不支持 → 返回 None，退回指数退避
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
