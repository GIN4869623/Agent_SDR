# Agent SDR — Claude Code Architecture

基于 Claude Code 的 MCP-centric 架构重构。核心理念：**MCP Server 是唯一的工具接口，
Agent Loop 围绕 MCP 工具进行推理-执行-观察循环，Skills 按需注入提示词。**

## 架构概览

```
外部 Agent (Claude Code)
        │
        │ MCP/stdio          ┌──────────────┐
        ▼                    │   Web UI     │
┌──────────────┐             │  (SSE + HTTP)│
│  MCP Server  │◄────────────┤              │
│ (stdio模式)  │             └──────┬───────┘
└──────────────┘                    │
        │                           │ MCP/SSE
        │                    ┌──────┴───────┐
        │                    │  MCP Server  │
        └────────────────────┤  (SSE模式)   │
                             └──────┬───────┘
                                    │
                             ┌──────┴───────┐
                             │  Agent Loop  │
                             │ think→act→   │
                             │ observe→loop │
                             └──────────────┘
```

### 语义路由与任务完成判定

Web 客户端仍统一调用聊天接口；服务端收到 `instruction` 后先做内部语义路由：

```text
instruction
  ├─ chat       → 不提供工具，直接回答
  ├─ knowledge  → 本地知识检索，未命中时才开放联网搜索
  └─ hardware   → 匹配 Skill，并按工作流阶段过滤可用工具
```

Skill 可以是单工具原子能力，也可以是多工具工作流。Agent 每一阶段只向模型暴露
当前允许的工具集合。工具结果会转换为 `completed_actions`，只有覆盖任务的
`required_actions` 且生成最终答复后，任务才返回 `success`；部分完成返回
`incomplete`，没有有效结果返回 `failed`。

### Agent 工作日志

每次请求会按 `run_id` 写入一条结构化 JSONL 日志，默认位置为
`logs/agent_runs.jsonl`。每条日志按“请求理解 → 任务计划 → LLM 决策 → 工具执行
→ 循环控制 → 完成判断”记录：原始请求和使用的短时记忆数量、路由理由与业务意图、
Skill 工作流和必需动作、每轮 LLM 的用途/候选工具/响应耗时、工具参数校验与三态业务
效果、动作完成证据、重复调用、缺失动作、停止原因和故障归因。大型数组和长文本会
自动截断，不会完整保存波形数据。

`metrics.timing` 提供路由、计划、上下文、LLM 决策、知识检索、硬件执行和最终回复的
分段耗时；`metrics.counts` 提供 LLM 调用、工具提议/执行/拒绝及重复调用次数。若模型
服务返回 token usage，每轮 `metrics.llm_calls` 还会记录输入/输出 token 和生成速度。

```text
GET /api/agent/runs?limit=20       # 最近运行记录
GET /api/agent/runs/{run_id}       # 某次请求的完整轨迹
```

日志采用滚动保留策略，默认最多保存最近 1000 次请求；写入新记录超过上限时会批量
删除最旧的约 10%（单次最多 100 条），避免每次请求都重写整个日志文件。可通过
`AGENT_TRACE_MAX_RECORDS` 修改数量，通过
`AGENT_TRACE_ENABLED=false` 关闭，或用 `AGENT_TRACE_PATH` 修改保存位置。
诊断字段 `stage` 会把问题定位到路由/Skill、参数校验、工具执行、循环控制、完成条件
或最终回复阶段，并给出建议检查的文件。

## 项目结构

```text
Agent_SDR/
├── main.py                     # 向后兼容入口 → src/server.py
├── src/                        # ★ 新架构代码
│   ├── server.py               # FastAPI + SSE + MCP 入口
│   ├── cli.py                  # MCP stdio 入口（给外部 Agent）
│   ├── config.py               # 统一配置
│   ├── prompts.py              # 系统提示词
│   ├── mcp/
│   │   └── __init__.py         # MCP Server + MCPToolRegistry（唯一工具源）
│   ├── agent/
│   │   ├── loop.py             # Agent 循环（think → act → observe）
│   │   ├── router.py           # 语义路由、Skill 工作流和任务动作判定
│   │   ├── trace.py            # 结构化工作日志与自动诊断
│   │   └── context.py          # 上下文组装
│   └── skills/
│       └── registry.py         # 技能加载（从 .md 文件）
├── hardware/                   # 硬件控制层（USRP, 视频流）
├── core/                       # 领域层（记忆, RAG 向量检索）
├── skill/                      # 技能定义（.md 文件 + 匹配引擎）
├── scripts/
│   └── evaluate_model_compatibility.py # 真实模型 + 模拟工具兼容性评测
├── static/                     # 前端
├── data/                       # 知识库文档
└── .env.example
```

## 目录和脚本说明

### 根目录

- `main.py`
  功能：FastAPI 入口，负责创建应用、注册路由、初始化 Agent、知识库、工具注册表和硬件控制器。
  用法：项目启动时由 `uvicorn main:app` 加载；如果要新增 HTTP API，一般从这里加路由。

- `.env.example`
  功能：环境变量模板，集中放模型地址、API Key、USRP IP 等参数示例。
  用法：复制为 `.env` 后再修改实际参数，`src/config.py` 会在启动时自动读取。

- `.gitignore`
  功能：忽略虚拟环境、缓存文件和本地敏感配置。
  用法：通常不需要改，只有当你新增了需要忽略的本地文件类型时才调整。

- `requirements.txt`
  功能：基础运行依赖，包含 FastAPI、OpenAI SDK、Pydantic、NumPy、Matplotlib、faiss-cpu、dashscope 等。
  用法：部署或初始化开发环境时执行 `pip install -r requirements.txt`。

- `requirements-optional.txt`
  功能：PDF 知识库导入脚本使用的可选依赖。
  用法：需要运行 `scripts/import_pdf_to_kb.py` 时安装。

- `README.md`
  功能：项目说明文档。
  用法：开发前先读这里，确认目录职责、启动方式和改动入口。

### `src/`

- `src/config.py`
  功能：当前唯一配置入口，读取 `.env` 并导出模型、Agent、知识库和日志配置。
  用法：修改模型地址、模型名、默认温度、会话长度或日志路径时改这里。

- `src/prompts.py`
  功能：当前 Agent Loop 使用的全局系统提示词和 Skill 注入模板。
  用法：优化全局角色、工具使用原则和回答风格时改这里；单项业务规则改对应 Skill。

- `src/mcp/__init__.py`
  功能：MCP 工具 Schema 与执行器的唯一来源，同时服务内部 Agent Loop 和外部 MCP 客户端。
  用法：新增或修改可执行工具时改这里。

### `core/`

- `core/`
  功能：领域服务层，处理会话记忆、本地知识检索、embedding 和联网搜索。
  用法：Agent 编排已经迁移到 `src/agent/`，这里不再包含 Agent 主循环。

- `core/memory.py`
  功能：管理多会话上下文，按 `session_id` 存取历史消息。
  用法：如果你要做更长会话、持久化会话、数据库存储会话，可以先改这里。

- `core/rag.py`
  功能：本地向量知识库。把 `data/` 下的 `.md`/`.txt` 分块后，用云端 embedding 生成向量，
        写入本地 FAISS 索引（`data/.index`），查询时按余弦相似度检索并返回原文片段；
        无 embedding 可用时自动降级为关键词匹配。
  用法：调整分块大小/重叠、检索逻辑或替换向量库时改这里。

- `core/embeddings.py`
  功能：云端 embedding 封装（百炼 DashScope，模型 `qwen3.7-text-embedding`，1024 维）。
        文档与查询必须使用同一模型，这里集中管理该选择。
  用法：更换 embedding 模型或服务商时改这里。

- `core/__init__.py`
  功能：包初始化文件。
  用法：一般不需要改。

### `hardware/`

- `hardware/`
  功能：硬件控制层，只负责 USRP/UHD 相关动作，不处理大模型推理。
  用法：如果你要改扫频、FSK、AMC、Tone、UHD 参数，应该改这里，而不是改 `main.py` 或提示词。

- `hardware/sdr_controller.py`
  功能：核心硬件控制器，封装 USRP 连接、硬件状态、扫频、文本收发、自适应调制、最优频点搜索、Tone 可视化等逻辑。
  用法：新增物理层实验时先实现底层函数，再到 `src/mcp/` 暴露给 Agent。

- `hardware/__init__.py`
  功能：包初始化文件。
  用法：一般不需要改。

### `skill/`

- `skill/skill_spec.py`
  功能：定义纯 Skill 元数据和匹配器，不包含工具处理函数。
  用法：调整匹配评分机制时改这里。

- `skill/skills/*.md`
  功能：定义意图触发规则、任务提示词、工作流和完成动作。
  用法：修改某项业务的参数提取与执行指南时改对应 Markdown 文件。

### `data/`

- `data/`
  功能：本地知识库的**唯一资料源**（single source of truth），存放教材、设备手册、实验文档。
  用法：把 `.md`/`.txt` 直接丢进来（PDF 需先用 `scripts/import_pdf_to_kb.py` 转成 markdown 分块），
        然后运行 `python scripts/build_index.py` 构建索引；服务启动后首次查询也会自动增量同步。

- `data/.index/`
  功能：本地 FAISS 向量索引（`vectors.npy` + `records.json` + `_manifest.json`），由构建脚本生成。
  用法：已加入 `.gitignore`；重建用 `python scripts/build_index.py --force`。

- `data/README.md`
  功能：说明知识库目录应该放什么类型的内容。
  用法：团队成员扩展知识库前可以先看这个文件。

- `data/sdr_basics.md`
  功能：示例知识文档，演示 BPSK、QPSK、16-QAM、AMC、底噪等基础概念的写法。
  用法：可以继续往里面补内容，也可以复制这个格式新增更多知识文档。

### `static/`

- `static/`
  功能：前端静态资源目录，负责浏览器界面。
  用法：前端页面、样式、交互脚本都放这里，FastAPI 会把这个目录直接挂载为站点根目录。

- `static/index.html`
  功能：页面骨架，定义布局、按钮、聊天区、硬件态势区和诊断区。
  用法：改页面结构、增减模块、调整组件层级时改这个文件。

- `static/css/app.css`
  功能：全局样式，控制颜色、排版、布局、响应式效果和状态样式。
  用法：改视觉风格、移动端表现、卡片样式、按钮样式时改这个文件。

- `static/js/app.js`
  功能：前端交互逻辑，负责切换模式、发送消息、轮询硬件状态、刷新诊断面板和渲染消息流。
  用法：改前端 API 调用、会话行为、状态渲染或按钮事件时改这个文件。

## 常见开发入口

- 想改模型地址、模型名、USRP IP：改 `src/config.py` 或 `.env`。
- 想改 Agent 的角色、全局工具规则、回答风格：改 `src/prompts.py`。
- 想改 Agent 编排与循环：改 `src/agent/loop.py`、`src/agent/router.py`。
- 想调整知识库分块 / embedding / 向量检索：改 `core/rag.py`、`core/embeddings.py`。
- 想新增一个硬件实验能力：先改 `hardware/sdr_controller.py`，再到 `src/mcp/` 暴露成工具，并补充 Skill。
- 想新增一个知识型工具：改 `src/mcp/__init__.py`，必要时配合 `core/rag.py`。
- 想评测本地模型兼容性：运行 `scripts/evaluate_model_compatibility.py`，评测使用模拟工具，不操作硬件。
- 想改网页布局和交互：改 `static/index.html`、`static/css/app.css`、`static/js/app.js`。

## 运行方式

### 0. 构建本地知识索引（首次或资料更新后）

```bash
cd ~/Desktop/Agent_SDR
source .venv/bin/activate
python scripts/build_index.py            # 增量构建；--force 全量重建
```

索引写入 `data/.index/`；服务启动后的首次查询也会自动增量同步 `data/` 的变化。

### 1. Web UI 模式（浏览器访问）

```bash
cd ~/Desktop/Agent_SDR
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
# 浏览器访问 http://0.0.0.0:8000
```

### 2. MCP stdio 模式（给 Claude Code 或其他 Agent 调用）

```bash
cd ~/Desktop/Agent_SDR
source .venv/bin/activate
python -m src.cli
```

**Claude Code 配置** — 在 `~/.claude/claude_desktop_config.json` 或项目的 `.claude/mcp.json` 中添加：

```json
{
  "mcpServers": {
    "sdr": {
      "command": "python",
      "args": ["-m", "src.cli"],
      "cwd": "/home/wenchaoxia/Desktop/Agent_SDR"
    }
  }
}
```

配置后，Claude Code 会自动发现并调用 SDR 工具：

```
可用的 MCP 工具:
  - perform_physical_scan    (扫频/测底噪)
  - tone_loopback_visualize  (正弦波可视化)
  - text_fsk_send_and_receive (文本收发)
  - adaptive_modulation_transmit (自适应调制)
  - auto_optimal_transmit   (认知选频)
  - search_sdr_knowledge    (知识检索)
  - start_video_stream / stop_video_stream (视频流)
  - ...
```

## 开发入口（新架构）

- 想新增 MCP 工具：改 `src/mcp/__init__.py` → `_register()` 装饰器
- 想改 Agent 循环逻辑：改 `src/agent/loop.py`
- 想改系统提示词：改 `src/prompts.py`
- 想新增技能匹配：在 `skill/skills/` 创建 `.md` 文件
- 想调整知识库分块 / embedding / 向量检索：改 `core/rag.py`、`core/embeddings.py`
- 想改硬件控制：改 `hardware/sdr_controller.py`（不变）

## 注意

1. `uhd` 一般不是通过 `pip` 安装，需要按 USRP/UHD 官方流程单独安装 Python 绑定。
2. 知识库采用「本地 FAISS 向量检索 + 云端 embedding（百炼 `qwen3.7-text-embedding`）」方案：embedding 只在构建索引/资料变更时调用（一次性按 token 计费），查询时检索全程本地、零云端检索开销。
3. 当前 Agent 默认仍走 OpenAI 兼容接口，`OLLAMA_BASE_URL` 保持了你原项目的兼容模式。
4. 项目只保留当前 Agent Loop 架构；Agent 与工具相关的新功能请在 `src/` 和 `skill/` 下开发。

## 演示视频

[查看视频 1](./15a9f62542c0ccde0243d3774e631b2b.mp4)

[查看视频 2](./dbb3bbbb0396fc32b3b7eebeafd0ed33.mp4)
