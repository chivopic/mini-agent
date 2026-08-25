# mini-agent

一个轻量、本地、单进程的终端 AI 编程助手。

`mini-agent` 在指定工作区内运行，通过 OpenAI-compatible Chat Completions API 驱动工具循环，能够理解项目结构、检索与修改代码、运行测试，并保存对话上下文。当前支持 OpenAI、DeepSeek V4、Ollama、通义千问、SiliconFlow、Moonshot、智谱等兼容接口。

## 功能

- 流式 Agent 工具循环与 Rich 终端界面
- Python、JavaScript、TypeScript Repo Map
- 工作区内文件列表、读取、全文搜索、精确编辑与写入
- Python/JSON 写入前语法检查
- Shell 命令分级控制、超时和敏感环境变量剥离
- 项目规则发现：`.agentrules`、`MINI_AGENT.md`、`CLAUDE.md`、`.cursorrules`
- 会话保存、列表、恢复和继续
- Token 用量、人民币费用估算和自定义费率
- Git diff 与确认后提交
- `-p` 单次执行模式
- 100% 离线 FakeLLM 测试

## 环境要求

- Python `>=3.12`
- macOS 或 Linux
- Windows 可尝试使用，但当前 CI 尚未覆盖
- 推荐使用 [uv](https://docs.astral.sh/uv/)
- OpenAI-compatible API Key；Ollama 等本地服务可使用占位 Key

## 安装

直接运行或全局安装 PyPI 包：

```bash
uvx chiv-mini-agent
uv tool install chiv-mini-agent
```

从源码运行：

```bash
git clone https://github.com/chenzh659/mini-agent.git
cd mini-agent
uv sync --all-groups
uv run mini-agent --help
```

## 配置

### OpenAI

```bash
export OPENAI_API_KEY="sk-your-openai-key"
uv run mini-agent
```

默认模型是 `gpt-4o-mini`，可以通过 `--model` 或 `MINI_AGENT_MODEL` 覆盖。

### DeepSeek V4

```bash
export OPENAI_API_KEY="sk-your-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export MINI_AGENT_MODEL="deepseek-v4-flash"
uv run mini-agent
```

可用的官方 API 模型名：

- `deepseek-v4-flash`：默认、速度优先
- `deepseek-v4-pro`：复杂编码与推理

也可以使用 CLI 参数：

```bash
uv run mini-agent \
  --base-url https://api.deepseek.com \
  --model deepseek-v4-pro
```

### Ollama

```bash
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export MINI_AGENT_MODEL="qwen2.5-coder:latest"
uv run mini-agent
```

### 工作区 `.env`

CLI 会读取工作区根目录的 `.env`，但只接受以下应用配置，避免项目文件重定向会话存储或覆盖进程内部设置：

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_API_BASE=...
MINI_AGENT_MODEL=...
MINI_AGENT_PRICING=...
```

## CLI

```text
Usage: mini-agent [OPTIONS]

  -w, --workspace PATH   工作区根目录，默认当前目录
  -p, --prompt TEXT      执行一次任务后退出
  -m, --model TEXT       模型名称
  -b, --base-url TEXT    OpenAI-compatible API Base URL
  -c, --continue         恢复当前工作区最近一次会话
  -s, --session TEXT     恢复指定会话 ID
  -v, --verbose          显示诊断信息
      --help             显示帮助
```

示例：

```bash
# 交互模式
uv run mini-agent --workspace /path/to/project

# 单次模式；若任务需要非只读 Shell，仍会要求人工确认
uv run mini-agent -p "检查测试失败的原因并修复"

# 恢复最近会话
uv run mini-agent --continue
```

### REPL 指令

| 指令 | 功能 |
| --- | --- |
| `/help` | 显示指令和工具说明 |
| `/provider [name]` | 查看或切换 Provider；例如 `deepseek`、`deepseek-pro`、`openai`、`ollama` |
| `/model [name]` | 查看或切换当前模型 |
| `/cost` | 查看当前会话 Token 与费用 |
| `/cost list` | 查看费率表 |
| `/cost set <model> <input> <output>` | 设置人民币/百万 Token 费率 |
| `/diff` | 查看 staged、unstaged 和未跟踪文件状态 |
| `/commit [msg]` | 确认暂存，展示精确 staged diff，再确认提交 |
| `/sessions` | 列出当前工作区的历史会话 |
| `/resume <id>` | 恢复同一工作区的指定会话 |
| `/new` | 开始新会话 |
| `/clear` | 清屏并重新显示 Banner |
| `/exit`、`/quit` | 退出 |

## Agent 工具

| 工具 | 功能 | 约束 |
| --- | --- | --- |
| `get_repo_map` | 提取 Python/JS/TS 类与函数骨架 | 仅工作区内目录；跳过外部符号链接 |
| `search_code` | 文本或正则检索 | 忽略依赖与缓存目录；限制结果数量 |
| `list_files` | 展示目录结构 | 相对路径；深度 1–5；最多 500 项 |
| `read_file` | 读取 UTF-8 文本 | 相对路径；单文件最多 100 KiB |
| `edit_file` | 唯一字符串匹配替换 | 匹配必须唯一；Python/JSON 通过语法检查后写入 |
| `write_file` | 新建或覆盖文件 | 相对路径；Python/JSON 通过语法检查后写入 |
| `run_shell` | 执行终端命令 | 高危模式阻断；Shell 语法、代码执行和非白名单命令需要确认 |

## 安全边界

`mini-agent` 提供应用层护栏，不是容器或 OS 沙箱：

- 文件工具会解析真实路径并拒绝绝对路径、`..` 越界和外部符号链接。
- 少量只读命令以 `shell=False` 自动执行；解释器、测试运行器、删除选项、管道、重定向、变量与通配符展开都需要确认。
- 子进程只继承运行所需的最小环境，不包含 API Key、云凭据、Git Token 或 SSH Agent。
- `edit_file` 和 `write_file` 在模型调用后会直接修改工作区文件，请在 Git 工作区使用并在提交前审查 diff。
- `/commit` 在 `git add --all` 前确认一次，随后展示精确 staged diff 并再次确认提交；若取消提交，为避免破坏既有 index，变更保持 staged。
- 不要在不可信仓库、生产服务器或包含未隔离核心凭据的目录中运行。

## 会话与费用

会话默认保存在 `~/.mini-agent/sessions/`。恢复时会校验 Session ID、存储边界和工作区归属。

自定义费率保存在 `~/.mini-agent/pricing.json`。内置价格仅用于估算，实际账单以服务商为准。

## 开发与发布检查

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
uvx --from twine twine check dist/*
```

## 当前限制

- REPL 输入仍是单行模式，没有命令历史和多行编辑。
- Agent 执行期间尚不支持可靠的当前轮取消。
- 没有重试/backoff、MCP、多 Agent 或浏览器工具。
- `edit_file` 是唯一字符串替换，不支持 patch hunks。
- Shell 直接运行在宿主机，确认后可以执行任意用户命令。

0.3 架构演进方案见 [design-0.3-architecture.md](https://github.com/chenzh659/mini-agent/blob/main/docs/design-0.3-architecture.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。
