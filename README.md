# baize-agents

一个用来**学习如何写 agent** 的最小骨架。当前处于里程碑 1：能调一个 OpenAI 兼容模型，完成一次问答（无工具、无记忆）。

## 目录结构

```
src/baize_agents/
├── __main__.py               # python -m baize_agents 入口
├── cli.py                    # 命令行入口
├── config.py                 # 配置加载（JSON + .env）
└── providers/
    └── openai_compat.py      # OpenAI 兼容接口客户端（仅标准库）
```

## 快速开始

本项目**没有第三方运行时依赖**，只用标准库。

```bash
# 1. 建虚拟环境并安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. 准备配置
cp config.example.json config.json

# 3. 设置 API key（推荐写进 .env，自动被程序读取）
cp .env.example .env
# 编辑 .env，填入你的密钥
```

然后运行：

```bash
baize -m "hello"
# 或
python -m baize_agents -m "hello"
```

## 支持的 provider

改 `config.json` 里的 `base_url` / `model` / `api_key_env` 即可切换，任何 OpenAI 兼容服务都行。

### DeepSeek（默认示例）

```json
{
  "provider": {
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY"
  }
}
```

### OpenRouter

```json
{
  "provider": {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "openai/gpt-4o-mini",
    "api_key_env": "OPENROUTER_API_KEY"
  }
}
```

### 本地 Ollama

```json
{
  "provider": {
    "base_url": "http://localhost:11434/v1",
    "model": "llama3.2",
    "api_key": "ollama"
  }
}
```

（Ollama 不需要真实密钥，随便填一个非空字符串即可。）

## 配置说明

- `base_url`：OpenAI 兼容接口的根地址（会自动拼上 `/chat/completions`）
- `model`：模型名
- `api_key_env`：从哪个环境变量读密钥（**推荐**，避免密钥进 JSON）
- `api_key`：直接写在 JSON 里的密钥（不推荐）

配置文件查找顺序：`--config` 参数 → `./config.json` → `~/.config/baize-agents/config.json`。

## 里程碑路线图

- [x] 里程碑 1：`-m "hello"` 一次性问答（无工具）
- [ ] 里程碑 2：Runner 循环（工具调用分支）
- [ ] 里程碑 3：`file_read` 等工具
- [ ] 里程碑 4：会话持久化（JSONL）与多模型
