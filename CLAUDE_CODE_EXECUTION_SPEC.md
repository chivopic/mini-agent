# mini-agent：交给 Claude Code 的分阶段实施规范

> **历史文档（0.1 MVP 规格）。** 本文件描述最初的分阶段实施规范，已不再反映当前产品行为。现行行为以 [README.md](README.md) 与 [docs/design-0.3-architecture.md](docs/design-0.3-architecture.md) 为准。

> 用途：将本文件完整交给 Claude Code，作为从零实现 `mini-agent` 的唯一实施规格。  
> 目标：实现一个可在终端交互、理解当前项目、读取文件、列目录、执行受控 shell 命令的 Python CLI Agent MVP。  
> 约束：**严格按阶段执行。每次只完成一个阶段；在阶段末说明设计原因、验证结果、修改文件，并等待用户确认后才开始下一阶段。不要在第一轮生成所有代码。**

---

## 0. Claude Code 工作指令（必须遵守）

你是一名资深 AI Agent 工程师，负责在一个空目录（或用户指定目录）从零实现本项目。

### 工作方式

1. 先阅读本规范，检查当前目录是否已有文件；绝不覆盖用户已有内容。
2. 以当前执行目录为 `workspace_root`。所有文件工具只能访问此目录及其子目录。
3. 从「阶段 1」开始。每完成一个阶段，必须停止继续编码，并按本模板汇报：

   ```text
   阶段 N 已完成
   - 完成内容：...
   - 设计原因：...
   - 验证命令与结果：...
   - 新增/修改文件：...
   下一步将执行：阶段 N+1（等待确认）
   ```

4. 用户确认后再执行下一阶段。不要预先创建下一阶段的实现文件，也不要一次提交全部代码。
5. 任何不能从规范确定的实现细节，选择最小、清晰、可测试的方案，并在阶段汇报中说明。
6. 源码注释和 README 使用中文；标识符、命令、类型名、函数名使用英文。
7. 使用 `uv`，不创建 `venv`、不使用 Poetry、pipenv、requirements.txt。
8. 所有代码通过 Ruff 检查和 pytest。不得为通过检查而关闭规则、跳过测试或吞掉异常。
9. 不要读取、打印、写入或提交 API Key；`.env` 必须被忽略，只提供 `.env.example`（不放真实密钥）。
10. 遇到危险 shell 操作（删除、覆盖、安装系统软件、网络下载、修改 Git 历史等）必须显式请求用户确认；MVP 内部的 `run_shell` 也必须遵守下文的安全策略。

### 非目标（MVP 不做）

- 不实现 `write_file`、文件编辑、补丁应用、Git 操作、浏览器工具、多 Agent、MCP、持久会话、token 计费、流式输出、并发工具调用。
- 不实现容器沙箱或真正的 OS 级命令隔离；README 必须明确说明 `run_shell` 只适合可信本地项目，不能在不可信目录或生产环境运行。
- 不实现模型“思维链”展示。终端仅显示简短的状态，如“正在调用工具：read_file”。

---

## 1. 成功定义与用户体验

安装完成后，用户能够在任意目标项目根目录执行：

```bash
uv run mini-agent
```

看到欢迎信息和交互提示符：

```text
Mini Agent — workspace: /absolute/path/to/project
输入问题；使用 /help 查看帮助，/exit 退出。

> 帮我看看这个项目的入口文件是什么
```

Agent 应能：

1. 让模型按需调用 `list_files`、`read_file`、`run_shell`。
2. 用 Rich 显示工具开始、完成或失败的状态；不打印敏感环境变量或完整内部堆栈。
3. 把工具结果回传模型；模型完成最终回答后，再显示新的 `>` 提示符。
4. 支持 `/help`、`/exit` 和 Ctrl-C；Ctrl-C 不中断整个程序（仅回到提示符），Ctrl-D 正常退出。
5. 未配置 `OPENAI_API_KEY` 时，给出可操作的中文错误说明并以非零状态退出。
6. 任何路径越界、非法参数、超时、非零退出码都能变成结构化工具结果，而不是导致 Agent 崩溃。

---

## 2. 技术决策（不可替换）

| 领域 | 决策 | 原因 |
| --- | --- | --- |
| Python | `>=3.12` | 使用现代类型语法与标准库能力。 |
| 环境/依赖 | `uv` + `pyproject.toml` | 快速、可复现，锁文件应被提交。 |
| CLI | Typer | 命令入口、参数和错误提示简洁。 |
| 终端 UI | Rich | 欢迎语、状态、异常和 Markdown 最终回答可读。 |
| 数据校验 | Pydantic v2 | 工具输入和输出有清晰契约。 |
| 模型调用 | 官方 `openai` Python SDK + Responses API | 支持原生 function calling；由模型选择自定义工具。 |
| 测试 | pytest | 覆盖 Agent Loop、路径约束、shell 失败及 CLI 行为。 |
| 静态检查 | Ruff | 格式化和基础 lint 一体化。 |

### OpenAI 集成约定

- 使用 `OpenAI()`，使 SDK 从环境变量 `OPENAI_API_KEY` 读取凭据；不要自行打印或传递密钥。
- 默认模型必须来自可配置项：环境变量 `MINI_AGENT_MODEL` 优先，未设置时使用代码内的单一常量。README 要说明默认值和如何覆盖，但不要承诺某个模型永远可用。
- 使用 Responses API 自定义 function tools，工具定义由一个集中函数返回。
- 一次 Agent turn 中，模型每次响应后检查全部 `function_call` 输出项；按返回顺序串行执行。
- 每个工具结果必须作为 `function_call_output`，并携带模型返回的同一个 `call_id`，再继续请求模型。
- Agent 应保留本次会话的上下文；推荐做法是维护 `history`，每轮追加用户输入、完整的 `response.output` 项与工具输出。不得只保存纯文本，否则后续工具调用上下文会丢失。
- `max_tool_rounds` 固定为合理小值（建议 8）。超限后，终止循环并给用户清晰错误，避免模型无限调用。
- SDK / API 失败需转换成领域异常并在 CLI 友好展示；测试中必须通过依赖注入 Fake LLM，而不是发真实网络请求。

OpenAI 官方文档说明 Responses API 支持自定义 function tools，工具调用具备 `call_id`，并要求应用把工具输出作为后续输入返回；实现时以官方 SDK 当前类型定义为准。[OpenAI API 平台概览](https://platform.openai.com/overview?height=3448)

---

## 3. 目标目录与文件职责

最终目录必须为：

```text
mini-agent/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── mini_agent/
│       ├── __init__.py
│       ├── main.py
│       ├── cli.py
│       ├── agent.py
│       ├── llm.py
│       ├── models.py
│       └── tools/
│           ├── __init__.py
│           ├── filesystem.py
│           └── shell.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_agent.py
    ├── test_filesystem.py
    ├── test_shell.py
    └── test_cli.py
```

### 模块边界

| 文件 | 仅负责 | 不应负责 |
| --- | --- | --- |
| `main.py` | 暴露控制台脚本函数并调用 Typer app | Agent、模型或工具逻辑 |
| `cli.py` | Typer app、REPL、Rich 输出、命令处理 | 路径解析、SDK 调用细节 |
| `agent.py` | Agent Loop、会话 history、轮数限制、工具分派 | Rich 输出、`subprocess` 细节、OpenAI SDK 类型细节 |
| `llm.py` | SDK client、系统提示词、工具 JSON Schema、LLM 适配协议 | 执行任何本地工具 |
| `models.py` | Pydantic 输入输出模型、配置、领域异常所需类型 | I/O 或 UI |
| `tools/filesystem.py` | 路径安全解析、文件读取、目录遍历 | shell 命令或 LLM 请求 |
| `tools/shell.py` | shell 参数校验、命令执行、超时与输出截断 | 文件目录遍历或 UI |

禁止循环依赖。推荐依赖方向：`main → cli → agent → (llm, tools, models)`；`tools → models`；`llm → models`。

---

## 4. 数据契约与工具规范

所有外部输入先由 Pydantic 校验。工具执行**不抛出预期运行时错误给 Agent**，而是返回 `ToolResult`。

### 4.1 通用模型

至少定义以下概念（字段名可小幅调整，但语义不得改变）：

| 模型 | 必要字段 | 约束 |
| --- | --- | --- |
| `AgentConfig` | `workspace_root: Path`、`model: str`、`max_tool_rounds: int`、`shell_timeout_seconds: int`、`max_output_chars: int` | root 必须在实例化时绝对化；数值必须为正。 |
| `ToolResult` | `ok: bool`、`content: str`、`error: str | None`、`metadata: dict` | `ok=False` 时必须有可读 `error`；给模型的结果必须是安全文本。 |
| `ReadFileInput` | `path: str` | 相对 workspace 的用户路径；禁止绝对路径。 |
| `ListFilesInput` | `path: str = "."`、`max_depth` | max depth 有合理上限，建议 1–5。 |
| `RunShellInput` | `command: str` | 非空；长度受限，建议不超过 1000 字符。 |

不要求为模型 API 响应创建完整 Pydantic 映射；仅通过 `LLMClient` 协议把 `agent.py` 与 SDK 解耦即可。

### 4.2 路径安全规则（所有文件工具必须统一遵守）

实现一个唯一的私有路径解析函数，供 `read_file` 和 `list_files` 复用。它必须：

1. 接受相对路径（`Path` 或字符串）。
2. 拒绝绝对路径。
3. 基于 `workspace_root / relative_path` 解析规范化路径。
4. 使用 `resolve()` 后验证结果仍是 `workspace_root` 本身或其子路径。
5. 拒绝 `..` 逃逸以及指向工作区外部的符号链接。
6. 将路径不存在、权限不足、不是普通文件/目录等情况转换为 `ToolResult(ok=False, ...)`。
7. 错误信息只显示用户传入的相对路径，不泄漏工作区外的绝对路径。

### 4.3 `read_file`

**模型函数定义：**

```text
name: read_file
input: { path: string }
```

行为：

- 只读取普通 UTF-8 文本文件；遇到无效 UTF-8，返回“不是 UTF-8 文本文件”的安全错误，不尝试二进制解码。
- 限制单文件读取大小（建议 100 KiB）。超出时拒绝并提示用户改用更小文件；MVP 不实现分页/行区间。
- `content` 返回原始文本（可在末尾附上是否因总输出限额截断的元信息），不要加不可靠的 Markdown 转义。
- `metadata` 至少带 `path` 和 `truncated`。

### 4.4 `list_files`

**模型函数定义：**

```text
name: list_files
input: { path?: string, max_depth?: integer }
```

行为：

- `path` 默认 `.`；目标必须是目录。
- 递归遍历最多 `max_depth` 层，默认 2，最大 5。
- 返回稳定排序的相对 POSIX 路径树或列表；目录明确标注（例如后缀 `/`）。
- 跳过 `.git`、`.venv`、`__pycache__`、`node_modules`；不要跟随目录符号链接。
- 限制条目数量（建议 500）和输出长度；达到限制在结果中显式说明。
- 任一无法访问的条目可跳过并计数，不应让整次列目录失败。

### 4.5 `run_shell`

**模型函数定义：**

```text
name: run_shell
input: { command: string }
```

这是最危险的 MVP 工具，必须如下实现：

- 使用 `subprocess.run`，`cwd=workspace_root`，`shell=True`，`text=True`，`capture_output=True`，指定 `timeout`。这里允许 `shell=True`，因为本工具的产品语义就是执行模型选择的 shell 字符串；README 必须明确风险。
- 不传递完整宿主环境。构建最小环境：只保留允许工具正常运行的 `PATH`、`HOME`、`LANG` / `LC_*`，并显式移除 `OPENAI_API_KEY`、常见云凭据、SSH/Git 令牌等敏感变量。
- 第一版采用**阻断名单 + 用户确认**，不要声称它是安全沙箱。至少阻断：递归删除、格式化磁盘/设备、关机/重启、提权工具、下载后直接执行、修改 Git 历史、危险 Git 清理。使用清晰规则匹配；无法可靠判断的高风险命令必须被拒绝，而不是猜测允许。
- 默认只自动执行只读或低风险开发命令，例如 `pwd`、`ls`、`find`、`rg`、`git status`、`git diff`、`python -m pytest`、`uv run pytest`。对其他命令返回 `requires_confirmation=True` 的结果，不执行。
- CLI 看到需确认的工具结果时，必须显示完整命令并提示 `[y/N]`；只有明确 `y` 才能由 Agent 再次执行一个带 `confirmed=True` 的内部调用。用户拒绝时，把“用户拒绝执行”作为工具结果回传模型。
- 不要让模型经 shell 读取 `~`、`/etc`、父目录或环境变量中的密钥。此限制不能只靠 prompt，至少必须阻断明显的路径和 `env` / `printenv` / `echo $VAR` 类泄漏途径。此为教学 MVP，不是生产安全边界，README 要如实声明。
- 成功和失败均返回：退出码、合并或分开的 stdout/stderr、是否截断、是否超时。总输出建议截断到 `max_output_chars=12_000`，保留开头与结尾并说明省略量。
- 超时返回 `ok=False`；处理超时进程，不能遗留子进程（可用新的 session/process group）。

### 4.6 工具注册与分派

- `llm.py` 提供 JSON Schema 的 tool definitions：名称、中文描述、JSON Schema、`strict=True`（如当前 SDK 和模型支持）。禁止从 Pydantic schema 直接暴露不需要字段。
- `agent.py` 使用显式字典将名称映射到内部函数；未知工具名返回结构化失败，不使用 `getattr` 或动态 import。
- 对每一次工具调用，先 JSON 解析 arguments、再 Pydantic 验证、再执行。JSON 非法或字段不合规也要回传可读错误。
- 工具显示给终端用户时，`run_shell` 展示命令；文件工具展示相对路径；不要展示完整文件内容，完整内容只作为模型上下文。

---

## 5. Agent Loop 精确定义

### 系统提示词必须表达的规则

系统提示词应简洁且包含：

1. 你是运行在 `workspace_root` 的本地开发助手。
2. 不了解项目结构时，优先 `list_files`，需要细节时才 `read_file`；不要盲目扫描所有文件。
3. 仅在确有必要时用 `run_shell`，先选择安全只读命令。
4. 所有文件路径都必须相对于工作区；工具失败时解释原因或换一种安全方式，不能编造已读取的内容。
5. 工具执行完后，基于真实结果用中文回答用户，简洁说明做了什么；禁止声称已修改任何文件（MVP 没有写入工具）。

### 伪代码（供实现参考，不是可直接复制的完整代码）

```text
history = [system message]

for each user turn:
    append user message to history

    for round in 1..max_tool_rounds:
        response = llm.create_response(history, tool_definitions)
        append every response output item to history

        function_calls = response 中的 function_call 项
        if function_calls 为空:
            return 从 response 提取的最终文本

        for call in function_calls（串行）:
            解析和校验 call.arguments
            通过显式 registry 执行工具
            向 CLI 发出 tool_started/tool_finished 事件
            append {type: function_call_output, call_id: call.call_id, output: ToolResult 的 JSON 文本}
              到 history

    return “工具调用轮数已达到上限，请缩小任务范围后重试。”
```

实现要求：

- 测试可注入 `FakeLLMClient`，其输出能模拟“工具调用 → 最终文本”。
- CLI 与 Agent 之间使用可选回调/事件对象通知状态，`agent.py` 不直接依赖 `rich.Console`。
- 某一工具失败不是整个回合失败：将失败结果交给模型，让模型决定是否改用其他工具。
- 模型没有最终文本、只返回未知项时，给用户安全的兜底错误；不能无限请求。
- 用户输入为空白时，CLI 不调用模型，只要求重新输入。

---

## 6. CLI 规格

### 命令接口

控制台脚本必须是：

```toml
[project.scripts]
mini-agent = "mini_agent.main:main"
```

MVP 主命令允许可选参数（至少实现前两个）：

```text
mini-agent [--workspace PATH] [--model MODEL] [--verbose]
```

| 参数 | 行为 |
| --- | --- |
| `--workspace PATH` | 指定工作区；缺省为当前工作目录。解析后必须是存在的目录。 |
| `--model MODEL` | 仅覆盖本次会话的模型配置。 |
| `--verbose` | 显示有限的诊断信息，例如工具耗时、返回码；绝不输出 API Key、完整环境或模型原始响应。 |

### REPL 行为

- 启动时验证 workspace、配置、API key，并显示 root（允许显示本机绝对路径）。
- 每轮使用 `Rich Prompt` 或等效输入，提示符固定 `>`。
- `/help`：显示命令和工具能力/限制。
- `/exit`、`/quit`：打印告别语并正常返回。
- Ctrl-C：显示“已取消当前输入”，保留 REPL。
- Ctrl-D / EOF：正常退出。
- 正在请求模型用 `rich.status` 或 `Spinner`；工具调用显示一行可复查状态。
- 结果通过 `Markdown` 或普通 Rich 文本渲染；模型输出异常时有 fallback，不能因 Markdown 解析失败崩溃。

不要把 Typer callback、REPL 逻辑和 Agent 创建混在 `main.py`；`main.py` 只负责调用 app。

---

## 7. 分阶段实施计划（严格按阶段停下）

### 阶段 1：架构确认（只设计，不写产品代码）

**要做：**

1. 检查空目录与已有文件。
2. 输出简洁架构图、模块依赖和关键安全取舍。
3. 确认本规范的需求没有冲突。

**禁止：** 创建 `src`、依赖、实现代码、测试。

**验收：** 用户获得可以审阅的架构设计，并确认进入阶段 2。

### 阶段 2：初始化项目骨架与依赖

**要做：**

1. 使用 `uv init --python 3.12` 或等效命令创建标准 `pyproject.toml`。
2. 配置 `src` layout 和 `mini-agent` console script。
3. 添加运行时依赖：`typer`、`rich`、`openai`、`pydantic`；开发依赖：`pytest`、`ruff`。
4. 创建包目录、空 `__init__.py`、`.gitignore`、`.env.example`。
5. 配置 Ruff 的基础 target version / lint / formatter，配置 pytest 的 `testpaths`。
6. 生成并提交 `uv.lock`。

**禁止：** Agent、工具、LLM 的功能实现；可只放非常小的入口占位符。

**验收命令：**

```bash
uv sync --all-groups
uv run python --version
uv run ruff check .
uv run pytest
```

pytest 可显示“没有测试”，但若可轻易添加一个 import smoke test，更好。

### 阶段 3：数据模型与文件工具

**要做：**

1. 在 `models.py` 实现配置、三种工具输入与统一 `ToolResult`。
2. 在 `tools/filesystem.py` 实现唯一的安全相对路径解析器。
3. 实现 `read_file`、`list_files` 及所有输出/深度/条目限制。
4. 写 `test_filesystem.py`，使用 pytest 的 `tmp_path` 构造独立工作区。

**必须覆盖的测试：**

- 正常读取 UTF-8 文件。
- 文件不存在、读取目录、二进制/非 UTF-8 文件。
- 过大文件拒绝或有明确截断策略。
- 正常列当前目录，顺序稳定，目录有标识。
- `max_depth` 生效，`.git` / `.venv` 等被排除。
- `../outside.txt` 与绝对路径被拒绝。
- 指向工作区外的符号链接被拒绝（平台支持时）。

**验收：** `uv run pytest tests/test_filesystem.py`、`uv run ruff check .`、`uv run ruff format --check .` 全通过。

### 阶段 4：受控 shell 工具

**要做：**

1. 在 `tools/shell.py` 实现 `run_shell`、输出截断、超时处理、环境脱敏、阻断名单和确认判定。
2. 将“是否需要确认”和“实际执行已确认命令”作为可测试的显式状态；不要在工具层调用 input。
3. 写 `test_shell.py`。尽量使用跨平台 Python 自身启动的短命令，而不是依赖 shell 专用语法。

**必须覆盖的测试：**

- 成功命令包含 stdout 与退出码。
- 非零退出包含 stderr、`ok=False` 与返回码。
- 输出被截断时有明确标记。
- 超时被处理且有可读错误。
- 危险模式被拒绝、不会真正执行。
- 非白名单命令标记为需要确认，确认前不执行。
- `OPENAI_API_KEY` 等敏感环境变量不被子进程继承（可使用可控的假环境验证）。

**验收：** 上述测试全部通过；展示一条安全命令和一条被拒绝命令的结果。

### 阶段 5：LLM 适配层与 Agent Loop

**要做：**

1. 定义一个最小 `LLMClient` Protocol，明确 Agent 需要什么，而非让测试 Mock OpenAI SDK 内部实现。
2. 在 `llm.py` 实现 OpenAI Responses API 适配器、系统提示词、工具 JSON Schema 与 response 的提取/标准化。
3. 在 `agent.py` 实现历史、工具注册、function call 输出回传、轮数限制、事件回调。
4. 为 SDK 连接错误、非法工具 JSON、未知工具、无最终文本写可读异常路径。
5. 写 `test_agent.py`；使用 Fake LLM 模拟至少一个多步链路。

**必须覆盖的测试：**

- “模型请求 `list_files` → 收到结果 → 返回最终回答”。
- 一轮多个工具调用按顺序执行并全部回传。
- 工具失败仍被回传并能继续。
- 非法 JSON / 不合规参数回传结构化失败。
- 未知工具名不崩溃。
- 达到 `max_tool_rounds` 退出。
- 无 function call 的纯文本回答直接结束。

**验收：** 全部 Agent 测试脱离网络通过；不得要求真实 API Key。

### 阶段 6：Typer + Rich 交互 CLI

**要做：**

1. 实现 `cli.py` REPL、帮助、退出、参数校验、Rich 状态渲染和用户确认流程。
2. 实现 `main.py` 与 console script 接口。
3. 配置不存在时有干净中文报错；无 API Key 时退出码非零。
4. 写 `test_cli.py`，用 Typer `CliRunner` 和依赖注入的假 Agent 测试关键输出。

**必须覆盖的测试：**

- `--help` 成功且说明 workspace/model。
- 无效 workspace 失败且信息可理解。
- 空输入不请求 Agent。
- `/help`、`/exit` 生效。
- Agent 事件在终端有状态展示。
- 需要确认的命令在拒绝和接受时分别回传正确结果。

**验收：**

```bash
uv run mini-agent --help
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### 阶段 7：README、端到端手工验证与收尾

**要做：**

1. 完成 README（见下一节）。
2. 执行静态检查与完整单测。
3. 执行一次不需要真实 API Key 的 CLI 验证；如果用户已明确提供测试 key，才能在其同意下做一次真实交互冒烟测试。
4. 检查 `git diff`，确保没有密钥、`.venv`、缓存和无关文件。
5. 汇总最终项目树、命令、测试结果、已知限制和下一步建议。

**验收：** 所有自动化检查通过；README 命令在干净环境中可复现。

---

## 8. `pyproject.toml` 与依赖标准

Claude Code 应在阶段 2 创建完整、现代的 `pyproject.toml`。它至少应包含：

- build backend（推荐 Hatchling）。
- `[project]`：`name = "mini-agent"`、版本、描述、`requires-python = ">=3.12"`、README、MIT 或用户指定许可证。
- 运行时依赖不加过紧版本锁；让 `uv.lock` 锁定可复现版本。
- `[project.scripts]` 的 `mini-agent` 入口。
- `src` layout 对应的 package discovery 配置。
- 合理的 ruff 和 pytest 配置。
- `dev` 依赖组而非生产依赖中混入 pytest/Ruff。

建议的最小依赖集合：

```text
runtime: typer, rich, openai, pydantic
dev: pytest, ruff
```

只有当实现确实需要时才加 `python-dotenv`；若添加，README 必须说明加载优先级，且 `.env` 永远不提交。优先直接要求用户导出 `OPENAI_API_KEY`，减少隐式行为。

---

## 9. README 完整要求

README 至少包含：

1. 项目一句话说明与 MVP 能做 / 不能做的事。
2. 环境要求：Python 3.12+、uv、OpenAI API Key。
3. 从克隆到启动的精确命令：`uv sync`、导出 API key、`uv run mini-agent`。
4. `--workspace` 和 `--model` 示例。
5. 交互示例（至少包含查询项目结构、读取文件、执行测试的场景）。
6. 三个工具的输入、用途、限制。
7. 安全声明：`run_shell` 可以在本机工作区执行命令；它不是安全沙箱；避免在不可信项目或含敏感凭据目录运行；敏感环境变量会尽力剥离但不能视作绝对隔离。
8. 测试与质量检查命令。
9. 项目结构图。
10. 已知限制与可扩展方向：写文件工具、流式输出、命令沙箱/allowlist、上下文裁剪、持久历史。

不要在 README 中放真实 token、机器特有路径或无法验证的性能承诺。

---

## 10. 最终质量门槛

阶段 7 完成前，Claude Code 必须逐项确认：

- [ ] `uv sync --all-groups` 在干净环境可运行。
- [ ] `uv run ruff check .` 通过。
- [ ] `uv run ruff format --check .` 通过。
- [ ] `uv run pytest` 通过，且不需要网络/API Key。
- [ ] `uv run mini-agent --help` 返回 0。
- [ ] 未设置 `OPENAI_API_KEY` 时有可操作的错误，且不会输出 traceback。
- [ ] 路径越界、符号链接越界、文件大小限制、非 UTF-8 输入都有测试。
- [ ] shell 的非零返回、超时、输出截断、危险命令、确认流程、敏感环境剥离都有测试。
- [ ] Agent 测试验证真实的「模型调用工具 → 工具输出用原 call_id 回传 → 模型最终回答」流程。
- [ ] 没有 API Key、`.env`、`.venv`、`__pycache__`、构建产物或测试缓存被提交。
- [ ] 代码没有空的 `except Exception: pass`、全局可变会话状态或未经验证的动态工具调用。

---

## 11. 可直接发送给 Claude Code 的启动提示词

复制以下文本与本文件内容一起发送：

```text
请严格执行仓库中的 CLAUDE_CODE_EXECUTION_SPEC.md，开发 mini-agent。

当前还未开始实施。先只执行“阶段 1：架构确认”，不要创建任何产品代码或依赖文件。
完成后按规范的阶段汇报模板说明设计与验收，并停止等待我的确认。

必须优先保证：模块边界、工作区路径限制、工具输出限制、受控 shell、OpenAI Responses API function call 的 call_id 回传，以及无真实 API Key 的单元测试。
```

后续每阶段仅发送：

```text
确认，请执行阶段 N+1。仍然严格遵循实施规范，阶段完成后停止并汇报。
```

