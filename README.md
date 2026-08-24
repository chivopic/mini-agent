# mini-agent

> 轻量、本地、单进程的终端 AI 编程助手（0.3）。

`mini-agent` 跑在指定工作区里：读项目结构、检索代码、读写文件、受控执行 Shell，并把会话、权限与费用留在本机。通过 OpenAI-compatible API 驱动工具循环（DeepSeek / OpenAI / Ollama / 通义 / SiliconFlow / Moonshot / 智谱等）。

它 **不是** IDE 插件、多用户平台、常驻 daemon，也 **不是** 容器级 OS 沙箱。

---

## 目录

- [功能与非目标](#功能与非目标)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置](#配置)
- [命令行参数](#命令行参数)
- [REPL 斜杠指令](#repl-斜杠指令)
- [项目规则](#项目规则)
- [内置工具](#内置工具)
- [权限](#权限)
- [会话](#会话)
- [交互与取消](#交互与取消)
- [安全声明](#安全声明)
- [测试与代码质量](#测试与代码质量)
- [项目代码结构](#项目代码结构)
- [已知限制](#已知限制)

---

## 功能与非目标

### 已支持

- **多服务商**：DeepSeek V4 / V3 / R1、OpenAI、Ollama、通义、SiliconFlow、Moonshot、智谱，以及任意 OpenAI-compatible 接口。
- **编码工具**：`get_repo_map`、`search_code`、`list_files`、带 `offset`/`limit` 的 `read_file`、带 `replace_all` 的 `edit_file`、`write_file`、`run_shell`。
- **权限服务**：`allow` / `ask` / `deny`；询问时 `once` / `always` / `reject`。敏感路径即使 `edit=allow` 也始终询问。
- **可取消回合**：流式输出或工具执行中 Ctrl-C 取消当前 turn，回到 `>`；2 秒内再按一次退出进程。
- **REPL**：`prompt_toolkit` 输入（多行粘贴、↑ 历史），Rich 输出；斜杠命令含 `/cost`、`/config`、`/commit` 等。
- **配置文件**：`~/.mini-agent/config.toml` 与工作区 `.mini-agent.toml`；密钥不进 TOML。
- **会话 v2**：磁盘 `schema_version: 2`；可读 0.2 的 v1，写入只写 v2。
- **Git**：`/diff` 展示改动；`/commit` 执行 `git add -u` + `git commit`，**永不** `git add .`。
- **费用**：会话 token / 人民币估算，`/cost`、`/cost list`、`/cost set`。
- **离线测试**：FakeLLM，不访问网络。

### 明确不做（0.3）

- MCP、多 Agent、Skills、浏览器工具。
- `apply_patch` / 批量 `edits[]`（编辑是唯一替换 + 可选 `replace_all`）。
- 容器 / OS 级沙箱；`run_shell` 直接跑在宿主机工作区。
- 展示模型思维链；Textual 全屏 TUI；常驻 daemon / 本地 HTTP。
- 正式承诺 Windows TUI（见下方环境要求）。

---

## 环境要求

- **操作系统**：macOS / Linux 为一等公民。Windows 可安装运行，但 **不保证 TUI**（`prompt_toolkit` 与终端能力因环境而异）。取消/超时在 Windows 上没有 `killpg`，回退 `process.terminate()`，超时后再 `kill()`。CI 为 GitHub Actions **Ubuntu**。
- **Python**：`>= 3.12`
- **包管理器**：`uv`（推荐）
- **API 凭据**：`OPENAI_API_KEY`（OpenAI、DeepSeek 或其它兼容服务商）

---

## 快速开始

### 方式一：一行命令免安装直接运行 (uvx / pipx)

```bash
uvx chiv-mini-agent
uv tool install chiv-mini-agent
```

### 方式二：从源码克隆运行

```bash
git clone https://github.com/chenzh659/mini-agent.git
cd mini-agent
uv sync --all-groups
```

### 配置 API Key

#### 方案 A：OpenAI

```bash
export OPENAI_API_KEY="sk-your-openai-key"
uv run mini-agent
```

#### 方案 B：DeepSeek（推荐）

```bash
export OPENAI_API_KEY="sk-your-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export MINI_AGENT_MODEL="deepseek-v4"

uv run mini-agent
```

也可以写在工作区 `.env`（**不覆盖**已在环境里的键）：

```bash
OPENAI_API_KEY=sk-你的DeepSeekKey
OPENAI_BASE_URL=https://api.deepseek.com
MINI_AGENT_MODEL=deepseek-v4
```

启动时也可直接覆盖：

```bash
uv run mini-agent --base-url https://api.deepseek.com --model deepseek-v4
```

---

## 配置

密钥 **只** 走环境变量或工作区 `.env`。用户 / 项目 TOML 里若出现 `api_key`、`openai_api_key`、`OPENAI_API_KEY`，会被忽略并打印警告，永不采用。

| 层 | 路径 | 作用 |
| :--- | :--- | :--- |
| 内置默认 | 代码 `config.defaults()` | 模型、limit、权限默认值 |
| 用户 | `~/.mini-agent/config.toml` | 服务商、模型、权限、limit |
| 项目 | `<workspace>/.mini-agent.toml` | 覆盖权限与 limit（可进 git，**禁止放密钥**） |
| 环境 | 见下表 | 密钥与运行时覆盖 |
| CLI | `-w -m -b -p -c -s -v -y --config` | 最高优先级 |

**合并顺序（高覆盖低）**：CLI > 环境变量（先加载 `.env` 且不覆盖已有键）> 项目 TOML > 用户 TOML > 代码默认。

`[provider].name` 会展开预设（`base_url` + 默认模型）；同层或更高层的显式 `base_url` / `model` 覆盖预设。

用户配置示例（`~/.mini-agent/config.toml`）：

```toml
[provider]
name = "deepseek"
# model = "deepseek-v4"
# base_url = "https://api.deepseek.com"

[limits]
max_tool_rounds = 32
shell_timeout_seconds = 30

[permission]
read = "allow"
edit = "allow"    # 敏感路径仍 always-ask
shell = "ask"
git = "ask"
```

0.3 **不会**从程序回写 TOML。`/provider` 只改当前会话。

### 环境变量

| 变量 | 用途 |
| :--- | :--- |
| `OPENAI_API_KEY` | API 密钥（必填，除非注入测试用 client） |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | 兼容 Base URL |
| `MINI_AGENT_MODEL` | 默认模型（未设置时为 `gpt-4o-mini`；DeepSeek URL 时默认为 `deepseek-v4`） |
| `MINI_AGENT_SESSIONS_DIR` | 会话目录（默认 `~/.mini-agent/sessions`） |
| `MINI_AGENT_PRICING_FILE` | 费率 JSON（默认 `~/.mini-agent/pricing.json`） |
| `MINI_AGENT_DEBUG=1` | 可选，追加写入 `~/.mini-agent/debug.log` |

完整模板见 [`.env.example`](.env.example)。

---

## 命令行参数

```text
用法: mini-agent [选项]

选项:
  -w, --workspace PATH   目标工作区根目录（缺省为当前目录）
  -p, --prompt TEXT      单次非交互执行：跑完任务后退出
  -m, --model MODEL      覆盖本次会话模型
  -b, --base-url URL     自定义 API Base URL
  -c, --continue         接续当前工作区最近一次会话
  -s, --session ID       恢复指定会话 ID
  -v, --verbose          诊断信息（工具耗时、返回码等）
  -y, --yes              权限询问视为允许（黑名单命令仍拒绝）
  --config PATH          用户配置文件（默认 ~/.mini-agent/config.toml）
  --help                 显示命令行帮助
```

```bash
uv run mini-agent
uv run mini-agent -p "运行单元测试并修复失败用例"
uv run mini-agent --continue
uv run mini-agent --workspace /path/to/my-project --model deepseek-v4
```

---

## REPL 斜杠指令

输入由 `prompt_toolkit` 处理：Enter 提交；终端的 bracketed paste 把多行当作一次提交；↑ 翻 `~/.mini-agent/history`。输出仍是 Rich。

| 指令 | 说明 |
| :--- | :--- |
| **`/help`** | 显示斜杠指令与内置工具 |
| **`/provider [name]`** | 查看或切换服务商预设（失败不会假装成功） |
| **`/cost`** | 当前会话用量与费用 |
| **`/cost list`** | 已配置模型费率表 |
| **`/cost set <模型> <输入> <输出>`** | 自定义费率（元 / 1M tokens） |
| **`/config`** | 只读展示合并后的配置（密钥已打码） |
| **`/cancel`** | 空闲时提示用 Ctrl-C；执行中请按 Ctrl-C 取消当前回合 |
| **`/sessions`** | 列出当前工作区历史会话 |
| **`/resume <id>`** | 恢复指定会话（含用量） |
| **`/new`** | 重置上下文，开启新会话 |
| **`/model [name]`** | 查看或临时切换模型 |
| **`/diff`** | 查看工作区 Git 未暂存改动 |
| **`/commit [msg]`** | 生成或使用给定信息提交：`git add -u` 然后 `git commit`（不会 `git add .`） |
| **`/clear`** | 清屏并重绘 Banner |
| **`/exit`**, **`/quit`** | 退出 |

---

## 项目规则

启动时扫描工作区根目录（优先级：`.agentrules` → `MINI_AGENT.md` → `.mini_agent.md` → `CLAUDE.md` → `.cursorrules`），注入系统提示。

---

## 内置工具

路径一律走工作区相对路径沙箱（拦截绝对路径与 `..`）。写文件路径：校验 → syntax_guard → 原子写（tmp + rename）。

| 工具 | 参数 | 功能 | 约束 |
| :--- | :--- | :--- | :--- |
| **`get_repo_map`** | `path`（默认 `.`） | 提取类 / 函数骨架，生成 repo map | 相对路径沙箱 |
| **`search_code`** | `pattern`, `path`, `is_regex`, `case_sensitive`, `max_results` | 全文检索 | 忽略 `.git` / `.venv` / `node_modules` 等；返回路径、行号、行内容 |
| **`list_files`** | `path`（默认 `.`）, `max_depth`（1–5） | 目录树 | 忽略常见缓存目录；跳过越界符号链接；上限 500 项 |
| **`read_file`** | `path`, `offset`, `limit` | 读 UTF-8 文本 | `offset` 为 1-based 行号；返回 **原文切片，不含 `L001:` 前缀**。无区间且 >100 KiB 拒绝；有区间时最多 400 行或 100 KiB |
| **`edit_file`** | `path`, `target_content`, `replacement_content`, `replace_all` | 字符串替换 | 默认必须唯一匹配；`replace_all=true` 替换全部出现 |
| **`write_file`** | `path`, `content` | 创建或覆盖写入 | 自动创建父目录；语法不合法则不落盘 |
| **`run_shell`** | `command` | 在工作区执行 Shell | **不是 OS 沙箱**。黑名单直接拒绝（含 `git add .` / `git add -A`）；非白名单走权限询问；超时默认 30s 强杀 |

只读工具在全部预检为 allow 时可并行；任一写操作或 ask 则整轮串行。

---

## 权限

配置项 `read` / `edit` / `shell` / `git` 取值为 **`allow` / `ask` / `deny`**。默认：`read=allow`、`edit=allow`、`shell=ask`、`git=ask`。

询问时：

| 输入 | 含义 |
| :--- | :--- |
| `y` | **once**：仅本次 |
| `a` | **always**：本会话记住该 pattern |
| `n`（默认） | **reject**：拒绝 |

敏感写路径即使 `edit=allow` 也 **always-ask**（按解析后的相对 POSIX 路径、大小写敏感）：

- `.env`、`.env.*`（含 `backend/.env`）
- 路径中含 `.git`
- `*.pem` / `*.key`
- `.mini-agent.toml`

`/commit` 走 git 权限，实际执行：

```text
git add -u
git commit -m "<message>"
```

未跟踪文件会列出来但 **不会** 被加入。`run_shell` 与斜杠层都拒绝 `git add .` / `git add -A` / `--all`。

非交互模式（`-p`）遇到询问会失败；需要自动允许时加 `-y`（黑名单仍拒绝）。

---

## 会话

- 目录：`~/.mini-agent/sessions/<id>.json`（可用 `MINI_AGENT_SESSIONS_DIR` 覆盖）。
- **读 v1、写 v2**：0.2 配对完好的会话可用 `--continue` / `--session` / `/resume` 打开；下次保存升为 `schema_version: 2`。
- 配对失败或损坏的文件视为 corrupt，`load_session` 返回空。
- 升级前如需保留 0.2 原件：`cp -r ~/.mini-agent/sessions ~/.mini-agent/sessions.bak`。0.2 客户端读不了 0.3 写出的 v2 文件。

---

## 交互与取消

- **Ctrl-C**：取消当前 turn（已完成的工具结果会落盘，磁盘上不留下未配对的 `tool_call`），回到 `>`。
- **2 秒内第二次 Ctrl-C**：退出进程。
- **Ctrl-D** / `/exit`：正常退出。
- 多行粘贴与 ↑ 历史依赖 `prompt_toolkit`（TTY）；pytest 或非 TTY 回退 Rich Prompt。

---

## 安全声明

> 1. `run_shell` **不是** OS 级沙箱，命令在本机工作区直接执行。黑名单是启发式拦截，可被绕过；真正的闸门是 ASK。
> 2. 不要在不可信目录、存有明文私钥的目录或生产环境直接运行。
> 3. 子进程环境会剥离 `OPENAI_API_KEY` 及常见云 / Git 凭据；仍请在确认提示里审阅命令。
> 4. `/commit` 使用消毒环境与 `git add -u`，不会把未跟踪的 `.env` 加进暂存区。

---

## 测试与代码质量

全部单元测试离线，不需要真实 API Key：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

---

## 项目代码结构

```text
mini-agent/
├── .env.example
├── .github/workflows/ci.yml
├── CLAUDE_CODE_EXECUTION_SPEC.md   # 历史 0.1 规格
├── README.md
├── docs/
│   └── design-0.3-architecture.md  # 0.3 架构（现行设计）
├── pyproject.toml
├── uv.lock
├── src/
│   └── mini_agent/
│       ├── __init__.py             # __version__ = "0.3.0"
│       ├── main.py                 # 控制台入口
│       ├── cli.py                  # Typer 装配、one-shot vs REPL
│       ├── repl.py                 # prompt_toolkit 输入、SIGINT
│       ├── commands.py             # 斜杠命令注册表
│       ├── render.py               # Rich 输出与权限询问
│       ├── agent.py                # 可取消 Agent 循环
│       ├── events.py               # 结构化 Agent 事件
│       ├── messages.py             # 类型化 Message / Part
│       ├── session.py              # 会话 v2（兼容读 v1）
│       ├── config.py               # TOML + env + CLI 合并
│       ├── permission.py           # allow / ask / deny
│       ├── prompt.py               # 系统提示
│       ├── context.py              # compaction：压缩旧工具输出
│       ├── gitutil.py              # git add -u / commit / diff
│       ├── llm.py                  # Chat Completions 适配器
│       ├── models.py               # Pydantic 契约
│       ├── providers.py            # 服务商预设
│       ├── cost.py                 # token 与费用
│       ├── rules.py                # 项目规则发现
│       ├── syntax_guard.py         # 写入前语法检查
│       ├── repomap.py              # AST repo map
│       └── tools/
│           ├── __init__.py         # default_registry()
│           ├── protocol.py         # Tool 协议
│           ├── registry.py         # 工具注册与 dispatch
│           ├── filesystem.py       # 路径沙箱与文件工具
│           └── shell.py            # run_shell
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── test_agent.py
    ├── test_cli.py
    ├── test_commands.py
    ├── test_config.py
    ├── test_context.py
    ├── test_cost.py
    ├── test_filesystem.py
    ├── test_messages.py
    ├── test_permission.py
    ├── test_providers.py
    ├── test_registry.py
    ├── test_repomap.py
    ├── test_rules.py
    ├── test_session.py
    ├── test_shell.py
    ├── test_smoke.py
    └── test_syntax_guard.py
```

---

## 已知限制

- `run_shell` 不是沙箱；regex 黑名单拦不住 `python -c` 一类组合。
- Windows：取消用 `terminate` fallback，**不保证 TUI**；CI 只跑 Ubuntu。
- `context_window_tokens` 不按模型自适应；小窗口（如本地 Ollama 32k）需在 TOML 里自行调小。
- 0.3 不从程序写回 `config.toml`；`/provider`、`/model` 只影响当前会话。
- 无 MCP / subagent / `apply_patch`。
