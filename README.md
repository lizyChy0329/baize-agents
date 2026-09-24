# baize-agents

一个用来**学习如何写 agent** 的最小骨架。当前处于里程碑 5：模型能自己调用工具（`file_read`）、能记住跨次对话的历史、支持多 provider 切换、可交互式对话。

## 目录结构

```
src/baize_agents/
├── __main__.py               # python -m baize_agents 入口
├── cli.py                    # 命令行入口
├── config.py                 # 配置加载（JSON + .env，支持多 provider）
├── session.py                # 会话持久化（JSONL）
├── runner.py                 # agent 循环（模型 ⇄ 工具）
├── repl.py                   # 交互模式
├── prompt.py                 # 系统提示词组装
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
baize                      # 交互模式，像聊天一样连续对话
baize -m "hello"           # 一次性提问
```

## 交互模式

不带 `-m` 就进入 REPL，可以连续对话，历史会被记住：

```
$ baize
baize-agents 交互模式（会话 default）。/help 看命令，/exit 退出。
> 我叫小明
你好，小明！
> 我叫什么名字？
你叫小明呀！
> /exit
```

内置命令：`/help` `/tools` `/clear` `/exit`（或 Ctrl-D）。

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

## 系统提示词

每次请求前会临时插入一段系统提示词（**不写进会话历史**，改了立即对所有会话生效）：

```bash
baize --show-system               # 看看最终拼出来的提示词
baize --system "你是个严谨的代码助手"  # 临时覆盖
```

也可写进 `config.json`：

```json
{ "system_prompt": "你是个严谨的代码助手。" }
```

自定义内容只替换「身份描述」部分，「当前环境」（工作目录 / 时间 / 可用工具清单）仍会自动附加。

> **为什么需要它**：实测在「先闲聊几轮、再问需要读文件的问题」这种场景下，
> 没有系统提示词时模型只有 **3/6** 会主动调用工具；加上之后是 **6/6**。

## 重试与错误分类

请求失败时，程序会先**分类**再决定怎么办：

| 情况 | 状态码 | 行为 |
|---|---|---|
| 瞬态故障 | 5xx / 连接失败 / 超时 | 重试（指数退避） |
| 限流 | 429 | 重试，优先听服务器的 `Retry-After` |
| 认证失败 | 401 / 403 | **立即失败**，重试无意义 |
| 余额不足 | 402 | **立即失败** |
| 上下文超长 | 400 + 特定错误码 | **立即失败**（该压缩历史，不是重试） |

退避默认 `1s → 2s → 4s`，带抖动与上限。可在 `config.json` 调整：

```json
{
  "retry": {
    "max_attempts": 4,
    "base_delay": 1.0,
    "max_delay": 30.0,
    "jitter_ratio": 0.25
  }
}
```

重试过程会提示到 stderr：

```
[重试 1] HTTP 429 请求过于频繁（限流）：...
         等待 1.0s 后重试...
```

## 配置说明

- `base_url`：OpenAI 兼容接口的根地址（会自动拼上 `/chat/completions`）
- `model`：模型名
- `api_key_env`：从哪个环境变量读密钥（**推荐**，避免密钥进 JSON）
- `api_key`：直接写在 JSON 里的密钥（不推荐）

## 开发

```bash
pip install -e ".[dev]"   # 装测试依赖（pytest）
pytest                     # 跑测试
```

运行时**零第三方依赖**；pytest 只在开发时需要。

## 里程碑路线图

- [x] 里程碑 1：`-m "hello"` 一次性问答（无工具）
- [x] 里程碑 2：Runner 循环 + `file_read` 工具
- [x] 里程碑 4：会话持久化（JSONL）与多 provider
- [x] 里程碑 5：交互式 REPL
- [x] 里程碑 6：错误分类与重试退避 + 测试基建
- [x] 里程碑 6.5：系统提示词（提升工具调用稳定性）
- [ ] 里程碑 7：流式输出（SSE 逐字回显）
- [ ] 里程碑 8：上下文治理（token 计量 / 工具结果截断 / 自动摘要）
- [ ] 里程碑 9：更多工具（`list_dir`、`web_fetch`、`shell`）+ 对应沙箱
- [ ] 里程碑 10：消息总线 + Channel 抽象

## 代码导读

逐文件讲解见 [docs/代码导读.md](docs/代码导读.md)，面向 Python 初学者。

## 试试 agent 循环

```bash
# 模型会自己调 file_read 读取文件再回答
baize -m "读一下 README.md，用一句话概括它"
baize -m "pyproject.toml 里 name 字段是什么？"
```

工具调用过程会打印到 stderr（`[工具] ...` / `[结果] ...`），最终答案在 stdout。
