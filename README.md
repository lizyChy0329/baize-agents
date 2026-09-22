# baize-agents

一个用来**学习如何写 agent** 的最小骨架。当前处于里程碑 4：模型能自己调用工具（`file_read`）、能记住跨次对话的历史、支持多 provider 切换。

## 目录结构

```
src/baize_agents/
├── __main__.py               # python -m baize_agents 入口
├── cli.py                    # 命令行入口
├── config.py                 # 配置加载（JSON + .env，支持多 provider）
├── session.py                # 会话持久化（JSONL）
├── runner.py                 # agent 循环（模型 ⇄ 工具）
├── providers/
│   └── openai_compat.py      # OpenAI 兼容接口客户端（仅标准库）
└── tools/
    ├── base.py               # Tool 类型（说明书 + 函数）
    ├── file_read.py          # 读文件工具
    └── __init__.py           # 工具注册表
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

## 会话记忆

默认会把历史存进 `~/.baize-agents/sessions/<会话名>.jsonl`，所以**跨次调用能记住上下文**：

```bash
baize -m "我叫小明"       # 第一次
baize -m "我叫什么名字？"  # 会答“小明”
```

```bash
baize -s work -m "..."    # 用名为 work 的独立会话
baize --list               # 列出所有会话
baize -s work --clear      # 清空某个会话
baize --no-session -m "."  # 一次性问答，不读也不写历史
```

存储位置由 `BAIZE_HOME` 控制，默认 `~/.baize-agents`。

## 支持的 provider

`config.json` 里可以定义多个 provider，用 `-p/--provider` 切换：

```json
{
  "default_provider": "deepseek",
  "providers": {
    "deepseek": {
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat",
      "api_key_env": "DEEPSEEK_API_KEY"
    },
    "ollama": {
      "base_url": "http://localhost:11434/v1",
      "model": "llama3.2",
      "api_key": "ollama"
    }
  }
}
```

```bash
baize -m "hello"           # 用 default_provider
baize -p ollama -m "hello" # 换一个
```

也兼容只有一个 provider 的旧写法：`{"provider": { ... }}`。

（Ollama 不需要真实密钥，随便填一个非空字符串即可。）

## 配置说明

- `base_url`：OpenAI 兼容接口的根地址（会自动拼上 `/chat/completions`）
- `model`：模型名
- `api_key_env`：从哪个环境变量读密钥（**推荐**，避免密钥进 JSON）
- `api_key`：直接写在 JSON 里的密钥（不推荐）

## 里程碑路线图

- [x] 里程碑 1：`-m "hello"` 一次性问答（无工具）
- [x] 里程碑 2：Runner 循环 + `file_read` 工具
- [x] 里程碑 4：会话持久化（JSONL）与多 provider
- [ ] 里程碑 3：更多工具（`web_fetch`、`file_write` 等）
- [ ] 后续：交互式 REPL、历史截断/摘要

## 试试 agent 循环

```bash
# 模型会自己调 file_read 读取文件再回答
baize -m "读一下 README.md，用一句话概括它"
baize -m "pyproject.toml 里 name 字段是什么？"
```

工具调用过程会打印到 stderr（`[工具] ...` / `[结果] ...`），最终答案在 stdout。
