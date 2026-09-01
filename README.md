# mini-agent

> 一个轻量级、受控的本地终端 AI 编程助手 CLI。

`mini-agent` 在指定工作区内运行，支持项目结构探索、代码检索、文件读写与编辑、受控 Shell 执行、会话恢复和 Token / 成本统计。项目使用 OpenAI-compatible API，可连接 DeepSeek、OpenAI、Ollama、Qwen 等服务。

> `mini-agent` 不是容器或虚拟机级安全沙箱。它会在你的本机工作区执行工具操作，因此仍应只在你信任的项目和机器上运行。

## 功能

- `search_code`：递归搜索工作区代码，支持普通文本和正则表达式。
- `list_files`：查看有限深度的项目目录结构。
- `read_file`：读取工作区内 UTF-8 文本文件。
- `write_file`：创建或覆盖文本文件，使用临时文件 + `fsync` + 原子替换降低中断导致的文件损坏风险。
- `edit_file`：只在目标片段唯一匹配时进行局部替换，并使用原子写入。
- `run_shell`：执行受控 Shell 命令；高危命令直接阻断，其余非自动放行命令需要用户确认。
- 多轮 Agent Loop、Token 流式输出、历史会话保存与恢复。
- `/provider`、`/model`、`/cost`、`/diff`、`/commit` 等 REPL 指令。
- Linux、macOS、Windows 三平台 CI。

## 环境要求

- Python `>= 3.12`
- 推荐使用 `uv`
- 使用远程模型时，需要对应服务商的 API Key

## 从源码运行

```bash
git clone https://github.com/chivopic/mini-agent.git
cd mini-agent
uv sync --all-groups --locked
uv run mini-agent
```

也可以从 PyPI 安装：

```bash
uvx chiv-mini-agent
# 或
uv tool install chiv-mini-agent
```

注意：`uvx` / `uv tool install` 获取的是 PyPI 上已经发布的版本，可能晚于 GitHub 仓库当前源码。

## 配置模型

### DeepSeek

当前 DeepSeek V4 预设使用：

- `deepseek-v4-flash`：默认低延迟模型
- `deepseek-v4-pro`：复杂编码与推理任务

```bash
export OPENAI_API_KEY="your-deepseek-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export MINI_AGENT_MODEL="deepseek-v4-flash"

uv run mini-agent
```

也可以直接通过 CLI：

```bash
uv run mini-agent \
  --base-url https://api.deepseek.com \
  --model deepseek-v4-flash
```

旧的 `deepseek-chat`、`deepseek-v3`、`deepseek-v4` 会兼容映射到 `deepseek-v4-flash`；旧的 `deepseek-reasoner`、`deepseek-r1`、`deepseek-v4-reasoner` 会映射到 `deepseek-v4-pro`。

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
uv run mini-agent --model gpt-4o-mini
```

### Ollama

使用 `/provider ollama` 或配置兼容 OpenAI API 的本地地址即可。Ollama 本地模型不会产生远程 API 费用。

## CLI 参数

```text
mini-agent [OPTIONS]

-w, --workspace PATH   工作区根目录，默认当前目录
-p, --prompt TEXT      单次非交互执行任务后退出
-m, --model MODEL      模型名称
-b, --base-url URL     OpenAI-compatible API Base URL
-c, --continue         恢复当前工作区最近一次会话
-s, --session ID       恢复指定会话
-v, --verbose          显示更多工具执行诊断信息
--help                 显示帮助
```

常见用法：

```bash
# 当前目录启动
uv run mini-agent

# 分析另一个项目
uv run mini-agent --workspace /path/to/project

# 一次性任务
uv run mini-agent -p "运行测试并解释失败原因"

# 恢复最近会话
uv run mini-agent --continue
```

## REPL 指令

| 指令 | 说明 |
| --- | --- |
| `/help` | 查看快捷指令和工具能力 |
| `/provider [name]` | 查看或切换服务商预设 |
| `/model [name]` | 查看或切换当前模型 |
| `/cost` | 查看当前会话 Token 与预估费用 |
| `/cost list` | 查看本地模型费率表 |
| `/cost set <model> <input> <output>` | 自定义模型费率 |
| `/diff` | 查看当前 Git diff |
| `/commit [msg]` | 生成或执行 Git commit |
| `/sessions` | 查看当前工作区历史会话 |
| `/resume <id>` | 恢复指定会话 |
| `/new` | 新建会话 |
| `/clear` | 清屏 |
| `/exit` / `/quit` | 退出 |

## 工具安全边界

### 工作区路径

文件工具只接受工作区内相对路径。绝对路径、`..` 越界和指向工作区外部的符号链接会被拒绝。

### 敏感文件

Agent 的读取、代码搜索、写入和编辑路径会额外阻止常见凭据文件，例如：

- `.env` 和大多数 `.env.*` 文件
- `.npmrc`、`.pypirc`、`.netrc`
- `id_rsa`、`id_ed25519` 等私钥
- `.pem`、`.key`、`.p12`、`.pfx`
- `.aws/credentials`
- `.docker/config.json`
- Google Application Default Credentials

`.env.example`、`.env.sample`、`.env.template` 和 `.env.dist` 等不含真实密钥的模板文件允许读取和编辑。

这项策略用于降低凭据被发送到远程 LLM 上下文或被 Agent 意外改写的风险，但它不是通用 Secret Scanner，仍不应把真实密钥提交到项目目录中。

### Shell

Shell 安全策略分为三层：

1. 明确的破坏性/高风险模式直接阻断。
2. 只有非常有限的简单只读命令可以自动执行，并且自动放行命令使用 `shell=False`。
3. 管道、重定向、命令替换、换行组合以及其他非白名单命令都需要用户明确确认。

子进程环境还会过滤常见 API Key、云凭据和 Token 环境变量。

即便如此，用户确认后的 Shell 命令仍然运行在宿主机上，因此不要把确认机制理解成 OS 级沙箱。

## 项目规则文件

启动时会查找项目规则文件，例如：

- `.agentrules`
- `MINI_AGENT.md`
- `.mini_agent.md`
- `CLAUDE.md`
- `.cursorrules`

这些文件会影响模型的项目级行为，因此在运行不熟悉的仓库前应先检查其内容。工具层权限检查独立于这些规则文件，不能由项目提示词绕过。

## 测试与 CI

本地质量检查：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

GitHub Actions 会在 Ubuntu、macOS 和 Windows 上分别执行 lint、格式检查、测试和构建，且 `fail-fast: false`，便于直接看到跨平台结果。

CI 使用只读默认 `GITHUB_TOKEN` 权限，关键第三方 Actions 固定到具体 commit SHA。

## Release

仓库包含 tag 驱动的 Release workflow。推送 `v*` tag 后会先重新运行 Linux / macOS / Windows 测试门禁；只有全部通过才进入构建和 GitHub Release 创建步骤。

Release workflow 还会检查：

```text
Git tag vX.Y.Z == pyproject.toml project.version X.Y.Z
```

因此不要在 CI 失败时手动发布正式 Release。

## 项目结构

```text
mini-agent/
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── src/mini_agent/
│   ├── agent.py
│   ├── cli.py
│   ├── context.py
│   ├── cost.py
│   ├── llm.py
│   ├── models.py
│   ├── providers.py
│   ├── rules.py
│   ├── session.py
│   └── tools/
│       ├── filesystem.py
│       ├── secure_filesystem.py
│       └── shell.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 当前限制

- 不是容器、VM 或内核级安全沙箱。
- 项目规则文件属于不可信仓库输入，应在运行陌生仓库前检查。
- 内置成本表是估算值；实际价格会随服务商、缓存命中和计费时段变化，可使用 `/cost set` 覆盖。
- PyPI 发布版本可能晚于 GitHub 源码。

## License

MIT
