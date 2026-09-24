"""把「上下文治理」和「会话持久化」串起来的小助手。

单独放一个模块，是为了让 cli 和 repl 共用同一套逻辑，
而 runner 继续保持只关心「循环」这件事。
"""
from __future__ import annotations

from typing import Callable

from .config import ContextConfig
from .context.compact import compact, needs_compaction
from .providers.base import Provider
from .session import Session

# 压缩完成回调：(被压缩掉的消息条数, 摘要)
CompactHook = Callable[[int, str], None]


def maybe_compact(
    provider: Provider,
    session: Session,
    context: ContextConfig,
    on_compact: CompactHook | None = None,
) -> str | None:
    """历史总量超阈值就压缩，并写进会话文件。返回摘要或 None。

    注意：compact() 是**就地修改** session.messages，所以这里不需要重新赋值。
    """
    if not needs_compaction(session.messages, context.compact_threshold_tokens):
        return None

    result = compact(provider, session.messages, context.keep_recent_tokens)
    if result is None:
        return None

    summary, dropped = result
    session.record_compaction(summary)
    if on_compact is not None:
        on_compact(dropped, summary)
    return summary
