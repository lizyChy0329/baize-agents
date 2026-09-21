"""配置加载：从 JSON 读取一个 OpenAI 兼容 provider。

搜索顺序：
1. 命令行 --config 指定的路径
2. 当前目录 ./config.json
3. ~/.config/baize-agents/config.json

API key 支持两种写法（优先用 api_key_env，避免把密钥写进 JSON）：
- api_key_env: 从环境变量读取（自动加载当前目录的 .env）
- api_key:     直接写在 JSON 里（不推荐）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """配置缺失或不合法。"""


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = None

    def resolve_api_key(self) -> str:
        if self.api_key:
            return _validate_api_key(self.api_key, "config.json 的 provider.api_key")
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ConfigError(
                    f"环境变量 {self.api_key_env} 未设置。\n"
                    f"请执行 export {self.api_key_env}=你的密钥，或写入 .env 文件。"
                )
            return _validate_api_key(key, f"环境变量 {self.api_key_env}")
        raise ConfigError("config.json 的 provider 里需要 api_key 或 api_key_env 字段")


@dataclass(frozen=True)
class Config:
    provider: ProviderConfig


def _validate_api_key(key: str, source: str) -> str:
    """拦截最常见的两种错误：没替换占位符、误带空格。"""
    if any(ord(ch) > 127 for ch in key):
        raise ConfigError(
            f"{source} 里的密钥包含非 ASCII 字符（中文等），HTTP 请求头无法携带。\n"
            f"多半是 .env 里的占位符没替换成真实密钥。"
        )
    if key != key.strip() or " " in key:
        raise ConfigError(f"{source} 里的密钥包含空格，请检查是否有多余内容。")
    return key


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载器：只支持 KEY=VALUE 行，不覆盖 shell 里已有的环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_config(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise ConfigError(f"指定的配置文件不存在：{p}")
        return p
    for p in (Path("config.json"), Path.home() / ".config" / "baize-agents" / "config.json"):
        if p.exists():
            return p
    raise ConfigError(
        "找不到配置文件 config.json。\n"
        "请先执行 cp config.example.json config.json，再填写你的 API 信息。"
    )


def load_config(explicit: str | None = None) -> Config:
    _load_dotenv(Path(".env"))
    path = _find_config(explicit)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件 {path} 不是合法 JSON：{e}") from e

    provider_raw = raw.get("provider")
    if not isinstance(provider_raw, dict):
        raise ConfigError(f"配置文件 {path} 缺少 provider 字段")

    provider = ProviderConfig(
        base_url=provider_raw.get("base_url", ""),
        model=provider_raw.get("model", ""),
        api_key_env=provider_raw.get("api_key_env"),
        api_key=provider_raw.get("api_key"),
    )
    if not provider.base_url:
        raise ConfigError(f"配置文件 {path} 的 provider.base_url 不能为空")
    if not provider.model:
        raise ConfigError(f"配置文件 {path} 的 provider.model 不能为空")

    return Config(provider=provider)
