"""会话持久化：把消息历史存成 JSONL（每行一条 JSON）。

为什么用 JSONL 而不是一个大 JSON？
- 追加写：新消息直接 append 到文件尾，不用重写整个文件
- 抗崩溃：即使写一半崩了，前面完整的行还在
- 好调试：cat / grep 都能直接看

存储位置：$BAIZE_HOME/sessions/<会话名>.jsonl
$BAIZE_HOME 默认是 ~/.baize-agents（可用环境变量覆盖，方便测试）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class SessionError(Exception):
    """会话读写失败。"""


def _home() -> Path:
    override = os.environ.get("BAIZE_HOME")
    return Path(override) if override else Path.home() / ".baize-agents"


def sessions_dir() -> Path:
    return _home() / "sessions"


def _safe_name(name: str) -> str:
    """防止会话名里带 / 或 .. 逃出 sessions 目录。"""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SessionError(f"非法会话名：{name!r}（不能为空、含斜杠或为 . / ..）")
    return name


def _repair(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """丢掉"没写完的工具调用"尾巴。

    进程如果在"模型要工具、但结果还没回喂"时崩了，文件里会留下一个
    没有配对 tool 结果的 assistant tool_calls。这种历史直接发给 API 会报错，
    所以载入时把它（及之后）砍掉。
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            needed = {c.get("id") for c in msg["tool_calls"]}
            answered = {
                m.get("tool_call_id")
                for m in messages[i + 1 :]
                if m.get("role") == "tool"
            }
            if not needed <= answered:
                return messages[:i]
            # 最后一组 tool_calls 是完整的，说明前面的也必然完整
            return messages
    return messages


class Session:
    """一个会话 = 一份不断增长的消息历史（对应一个 JSONL 文件）。"""

    def __init__(self, name: str = "default") -> None:
        self.name = _safe_name(name)
        self.path = sessions_dir() / f"{self.name}.jsonl"
        self.messages: list[dict[str, Any]] = []
        self._saved = 0  # 已经落盘的消息条数

    def load(self) -> list[dict[str, Any]]:
        """从文件读出历史。文件不存在则视为空会话。"""
        self.messages = self._read()
        self._saved = len(self.messages)
        return self.messages

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        messages: list[dict[str, Any]] = []
        for lineno, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise SessionError(f"{self.path} 第 {lineno} 行不是合法 JSON：{e}") from e
        return _repair(messages)

    def add(self, message: dict[str, Any]) -> None:
        """只加进内存；要落盘再调 sync()。"""
        self.messages.append(message)

    def sync(self) -> None:
        """把还没落盘的消息追加写入文件。重复调用是安全的。"""
        pending = self.messages[self._saved :]
        if not pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for message in pending:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._saved = len(self.messages)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self.messages = []
        self._saved = 0

    def __len__(self) -> int:
        return len(self.messages)


def list_sessions() -> list[str]:
    d = sessions_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))
