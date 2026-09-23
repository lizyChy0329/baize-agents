"""模型调用的错误分类。

为什么要分类？因为不同错误的应对方式完全不同：

- 429 / 5xx / 网络抖动  → 等一会儿重试，可能就好了
- 401 / 403            → 密钥不对，重试一万次也没用
- 402 / 余额不足        → 要充值，重试无意义
- 上下文超长            → 需要压缩历史，原样重试永远失败

分不清的后果：该重试的不重试（一抖动就挂），
不该重试的疯狂重试（烧钱、还被封）。
"""
from __future__ import annotations

# 各家 API 表示"上下文超长"的常见说法
_CONTEXT_HINTS = (
    "context_length_exceeded",
    "context length",
    "maximum context",
    "too many tokens",
    "reduce the length",
)


class ProviderError(Exception):
    """调用模型 API 失败的基类。默认不可重试。"""

    retryable = False


class TransientError(ProviderError):
    """瞬时故障：网络抖动、超时、5xx。重试可能成功。"""

    retryable = True


class RateLimitError(TransientError):
    """429 限流。可能带 retry_after（服务器告诉你等多久）。"""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AuthError(ProviderError):
    """401 / 403：密钥无效或没权限。重试无意义。"""


class QuotaError(ProviderError):
    """402 / 余额不足。重试无意义。"""


class ContextOverflowError(ProviderError):
    """请求超过模型上下文上限。要靠压缩历史解决，不是重试。"""


def _snippet(body: str, limit: int = 300) -> str:
    body = body.strip()
    return body if len(body) <= limit else body[:limit] + "..."


def parse_retry_after(value: str | None) -> float | None:
    """解析 Retry-After 头，只支持「秒数」形式（HTTP-date 形式暂不处理）。"""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def classify_http_error(
    status: int,
    body: str,
    retry_after_header: str | None = None,
) -> ProviderError:
    """把 HTTP 状态码翻译成有意义的错误类型。"""
    detail = _snippet(body)
    lowered = body.lower()

    if status == 429:
        return RateLimitError(
            f"HTTP 429 请求过于频繁（限流）：{detail}",
            retry_after=parse_retry_after(retry_after_header),
        )
    if status in (401, 403):
        return AuthError(f"HTTP {status} 认证失败，请检查 api_key：{detail}")
    if status == 402:
        return QuotaError(f"HTTP 402 余额不足，请充值：{detail}")
    if status == 400 and any(hint in lowered for hint in _CONTEXT_HINTS):
        return ContextOverflowError(f"HTTP 400 上下文超出模型上限：{detail}")
    if 500 <= status < 600:
        return TransientError(f"HTTP {status} 服务端错误：{detail}")
    return ProviderError(f"HTTP {status}：{detail}")
