"""工具结果截断。

工具结果是上下文里最大的膨胀源：读一个 200KB 的日志文件，
一次就能塞进几万 token，直接把上下文撑爆。

## 为什么保留头尾

- **头部**：通常有标题、结构、关键信息（文件开头、表格表头、命令输出前几行）
- **尾部**：通常有结论、错误信息、汇总（日志末尾的错误、测试失败信息）

只留头部会丢掉"最后报了什么错"，只留尾部会丢掉"这是什么文件"。

## 为什么要写明省略了多少

模型看到 `[...省略 12345 字...]` 才知道自己被截断了，
而不是以为文件就这么短 —— 否则它会基于不完整的信息自信地下结论。
"""
from __future__ import annotations

from .tokens import estimate_tokens

_DEFAULT_HEAD = 0.6  # 头部占比
_MARKER = "\n\n[...中间省略 {n} 字（约 {t} tokens）...]\n\n"


def truncate_text(text: str, max_tokens: int, head_ratio: float = _DEFAULT_HEAD) -> str:
    """把长文本截断到约 max_tokens。没超就原样返回。"""
    if not text or estimate_tokens(text) <= max_tokens:
        return text

    # 按字符比例切（token 与字符近似线性，够用）
    keep_chars = max(1, int(len(text) * max_tokens / max(1, estimate_tokens(text))))
    head_len = int(keep_chars * head_ratio)
    tail_len = keep_chars - head_len
    dropped = len(text) - head_len - tail_len

    marker = _MARKER.format(n=dropped, t=estimate_tokens(text) - max_tokens)
    return text[:head_len] + marker + (text[-tail_len:] if tail_len > 0 else "")


def truncate_tool_result(text: str, max_tokens: int) -> str:
    """截断工具结果，并加一句提示让模型知道还有更多内容。"""
    truncated = truncate_text(text, max_tokens)
    if truncated is text:
        return text
    return truncated + "\n\n（提示：内容过长已截断，如需更多请用更精确的查询，例如指定行范围。）"
