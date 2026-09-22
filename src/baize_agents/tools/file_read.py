"""file_read 工具：读取工作目录内的文本文件。"""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError

# 单文件读取上限，防止把超大文件塞进上下文
MAX_BYTES = 100_000


def file_read(path: str) -> str:
    """读取文件内容。

    安全限制（防止模型读到不该读的东西）：
    1. 只允许工作目录（cwd）内的文件，挡掉 /etc/passwd、~/.ssh/id_rsa 之类
    2. 用 resolve() 解析，符号链接指向目录外也会被挡
    3. 限制文件大小
    """
    root = Path.cwd().resolve()
    raw = Path(path)
    target = (raw if raw.is_absolute() else root / raw).resolve()

    if not target.is_relative_to(root):
        raise ToolError(f"拒绝访问工作目录之外的路径：{path}")
    if not target.exists():
        raise ToolError(f"文件不存在：{path}")
    if not target.is_file():
        raise ToolError(f"不是普通文件：{path}")

    size = target.stat().st_size
    if size > MAX_BYTES:
        raise ToolError(f"文件太大（{size} 字节，上限 {MAX_BYTES} 字节）")

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ToolError(f"不是 UTF-8 文本文件：{path}") from e


FILE_READ = Tool(
    name="file_read",
    description=(
        "读取工作目录内某个文本文件的完整内容。"
        "当你需要查看一个文件里写了什么时使用它。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径，相对工作目录，例如 README.md 或 src/app.py",
            }
        },
        "required": ["path"],
    },
    func=file_read,
)
