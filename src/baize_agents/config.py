"""配置加载：从 JSON 读取一个或多个 OpenAI 兼容 provider。

搜索顺序：
1. 命令行 --config 指定的路径
2. 当前目录 ./config.json
3. ~/.config/baize-agents/config.json

配置格式（多 provider）：
{
  "default_provider": "deepseek",
  "providers": {
    "deepseek": {"base_url": "...", "model": "...", "api_key_env": "DEEPSEEK_API_KEY"},
    "ollama":   {"base_url": "...", "model": "...", "api_key": "ollama"}
  }
}

也兼容单 provider 的旧格式：{"provider": {...}}

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
    """可能配置了多个 provider，default 指定默认用哪个。"""

    providers: dict[str, ProviderConfig]
    default: str

    @property
    def provider(self) -> ProviderConfig:
        """向后兼容：直接取默认 provider。"""
        return self.get()

    @property
    def names(self) -> list[str]:
        return sorted(self.providers)

    def get(self, name: str | None = None) -> ProviderConfig:
        key = name or self.default
        if key not in self.providers:
            available = ", ".join(self.names)
            raise ConfigError(f"未知 provider：{key!r}。可选：{available}")
        return self.providers[key]


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


def _parse_provider(raw: dict, where: str) -> ProviderConfig:
    provider = ProviderConfig(
        base_url=raw.get("base_url", ""),
        model=raw.get("model", ""),
        api_key_env=raw.get("api_key_env"),
        api_key=raw.get("api_key"),
    )
    if not provider.base_url:
        raise ConfigError(f"{where}.base_url 不能为空")
    if not provider.model:
        raise ConfigError(f"{where}.model 不能为空")
    return provider


def load_config(explicit: str | None = None) -> Config:
    _load_dotenv(Path(".env"))
    path = _find_config(explicit)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件 {path} 不是合法 JSON：{e}") from e

    # 新格式：{"default_provider": "deepseek", "providers": {"deepseek": {...}, ...}}
    if "providers" in raw:
        providers_raw = raw["providers"]
        if not isinstance(providers_raw, dict) or not providers_raw:
            raise ConfigError(f"配置文件 {path} 的 providers 必须是非空对象")
        default = raw.get("default_provider") or next(iter(providers_raw))
        if default not in providers_raw:
            raise ConfigError(
                f"配置文件 {path} 的 default_provider={default!r} 不在 providers 里"
            )
    # 旧格式（里程碑1~3）：{"provider": {...}}
    elif "provider" in raw:
        if not isinstance(raw["provider"], dict):
            raise ConfigError(f"配置文件 {path} 的 provider 必须是对象")
        providers_raw = {"default": raw["provider"]}
        default = "default"
    else:
        raise ConfigError(f"配置文件 {path} 缺少 provider 或 providers 字段")

    providers = {
        name: _parse_provider(item, f"{path} 的 providers.{name}")
        for name, item in providers_raw.items()
        if isinstance(item, dict)
    }
    if len(providers) != len(providers_raw):
        raise ConfigError(f"配置文件 {path} 的 providers 里每一项都必须是对象")

    return Config(providers=providers, default=default)
