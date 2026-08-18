# Bluecode Web 控制台 设计方案（设计稿 v1）

> 状态：**设计稿，未实现**。定位：全功能交互控制台（发起需求 + 实时过程 + 逐条审批 + 会话管理），
> 零构建轻前端，默认仅本机 localhost 访问。读者：后续实现者（人或 agent）。
> 姊妹文档：`design.md`（v0.1 核心设计稿，历史文档）。

---

## 1. 背景与目标

CLI 是目前唯一交互面，四个真实痛点 Web 化后能显著改善：

| 痛点（CLI 现状） | Web 化收益 |
|---|---|
| 审批靠终端 `input()`，diff 详情靠 rich 分页，看不见历史决策 | 审批卡：diff 高亮、逐条勾选、附注意见，体验对齐现代 code review |
| 过程播报是节点级 print，滚过即丢 | 事件流持久滚动 + 图拓扑可视化（7 节点 + 并行扇出实时高亮） |
| `/resume` `/undo` `/retry` `/history` 是斜杠命令，靠记忆使用 | 会话侧栏 + 按钮化，断点/回退一目了然 |
| token 播报一闪而过，审计日志是 jsonl 裸文件 | 状态面板常驻：token 计量、上下文占用、审计尾部 |

**非目标**（明确不做，防范围蔓延）：
- 多用户协作 / 公网 SaaS——审批 = 放行写文件/执行命令，绝不公网裸奔；
- token 级流式输出——核心尚无 `astream_events`（roadmap 已列），Web 先接受节点级粒度；
- 改动核心图逻辑——全部集成走既有扩展点（见 §2），本方案对 `agent.py` 的净改动 ≈ 3 行。

---

## 2. 现状盘点：可直接复用的扩展点

读码确认（v0.7 阶段二后已模块化：`agent.py` 是 facade，CLI 逻辑在 `cli.py`），以下机制**原样可用**：

| 扩展点 | 位置 | Web 用法 | 是否需改核心 |
|---|---|---|---|
| step 回调 `register_step_callback(fn(node_name, output))` | `cli.py:180`，回调列表、异常不阻断 | 注册 Web 桥接回调，节点输出 → SSE 事件；与 `_print_node`（服务端控制台镜像）、`_file_log_callback`（文件日志）**共存** | ❌ 零改动 |
| **drain 注入**：`_run_graph_core(graph, sess, request, *, banner, drain)` | `agent.py:1360`；`run_round_auto` 已示范注入 `_auto_drain` | 注入 `web_drain`：审批交互从 `input()` 换成「SSE 推卡片 + REST 收决策」 | ❌ 零改动 |
| interrupt 协议 | `agent.py:680` guard：`interrupt({"changes": [...], "question": ...})`；resume 值 `{"action": "approve"/"reject"/"modify", "indices": [0基], "note": str}` | 审批卡数据源与决策回传格式**照抄**，guard 零感知 | ❌ 零改动 |
| `resume_pending(graph, sess)`（/retry 断点续跑） | `agent.py:1290`，末尾硬编码调 `_drain` | Web 版 `/retry` 复用它，但需注入 web_drain | ⚠️ 加一个带默认值的 `drain=_drain` 参数（~3 行，向后兼容） |
| `Session` / `list_sessions()` / `_save_session_meta` | `session.py` | 会话列表/切换 API 直接包一层 | ❌ 零改动 |
| `_undo_latest(thread_id)`（快照回退） | `agent.py:1132` | `/undo` 按钮直调 | ❌ 零改动 |
| `_audit_log(thread_id, decision, changes)` | `agent.py:1178`，jsonl | web_drain 决策后照记（decision 里加 `"source": "web"` 区分来源） | ❌ 零改动 |
| `_token_usage_snapshot()` / `_finish_round_usage` | `agent.py:277/1212` | 轮末 usage 事件数据源 | ❌ 零改动 |
| 审批预览规则 `_preview_lines` / `_shown_change` | `cli.py:267-300`：命令全文、write_file 前 5 行、patch old/new 前 3 行、python 前 10 行 | Web 审批卡预览规则**逐条对齐**（安全底线语义不因换皮而弱化） | ❌ 零改动 |
| 模型注册表 `list_models` / `set_active_model` / `active_context_window` | `models.py`（`/model` 命令同源） | 状态栏显示当前模型/窗口；M4 可做切换器 | ❌ 零改动 |
| pyproject extras 先例 | `pyproject.toml`：`fancy` / `graph` | 新增 `web` extras，核心依赖零变化 | ⚠️ 打包配置（§9） |

**结论：架构上核心已经为外部 UI 预留了全部挂点（step 回调的设计初衷即此），Web 层是纯增量。**

---

## 3. 总体架构

```
┌──────────────────────── 浏览器（零构建单页） ────────────────────────┐
│  会话侧栏   │   对话流（请求/计划/过程/审批卡/验证/评审/报告）   │ 状态栏  │
└──────▲──────────────────────▲───────────────────────────▲─────────┘
       │ SSE（只读推送，断线重连）│ REST（发需求/审批决策/命令）      │ 静态文件
┌──────┴──────────────────────┴───────────────────────────┴─────────┐
│                    FastAPI 网关（web/server.py）                    │
│                                                                    │
│  SessionRouter（REST）        ExecutorRegistry（每活跃会话一个）     │
│  ├ /api/sessions*             ├ 工作线程：跑 _run_graph_core        │
│  ├ /api/.../approvals         │   （注入 web_drain，阻塞等决策）    │
│  └ /api/.../events (SSE)      ├ StepBridge：_emit_step → 环形缓冲   │
│                               │   → asyncio → SSE 推送              │
│                               └ 审批等待：queue.get() ← REST 决策    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ 同进程直接 import（非子进程）
┌──────────────────────────────┴─────────────────────────────────────┐
│  核心层（不动）：build_graph / 7 节点 / guard interrupt / SqliteSaver │
│  Session / _undo_latest / resume_pending / _audit_log / models      │
│  数据全在 ~/.blue/（checkpoints.sqlite、audit.jsonl、backups/）      │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键决策与理由

**D1：同进程 import 核心，不包 CLI 子进程 / pty。**
- interrupt 恢复要求向**同一 stream 循环**发 `Command(resume=...)`，跨进程做不了（除非重造 drain 协议）；
- step 回调只在执行进程触发，子进程方案还得自建 IPC 转发；
- pty + 屏幕刮取方案脆弱且拿不到结构化数据（审批预览、diff、usage 全要靠解析 ANSI 文本）。
- 子进程方案唯一优势是崩溃隔离——但 checkpoint 本就落盘（`/retry` 断点续跑已覆盖进程重启场景），且 `resume_pending` 的熔断超时机制（防 stream 永不返回占满 CPU）可以直接沿用其思路。

**D2：阻塞式 `graph.stream` 跑在每会话一个的工作线程里。**
LangGraph 此处是同步 API，不硬塞 async。工作线程 → `queue.Queue`/`threading.Event` → asyncio 侧用 `loop.call_soon_threadsafe` 桥接（标准做法，无新依赖）。`resume_pending` 已有「后台线程 + join 超时」先例（`agent.py:1334`），线程模型与其一致。

**D3：默认全局串行执行（同一时刻最多一个图在跑），可配置放开。**
保护 sqlite checkpoint 写入（benchmark 的 `--workers` 多进程证明可行，但 Web 场景串行足够——单人使用，审批本来就会阻塞）。同一会话并发发需求直接 409 拒绝（不做排队，v1 从简）。

**D4：Web 与 CLI 平级共存，共享 `~/.blue` 全部数据。**
同一会话**不允许** CLI 和 Web 同时操作（无跨进程锁，v1 靠约定 + 文档说明；若实测冲突频繁，v2 在 sessions 表加 owner/lease 列）。不同会话各自用没有问题。

---

## 4. 通信协议

协议先行：前端只是协议的渲染器，将来换 React 重写前端不动后端。

### 4.1 REST

| 方法 & 路径 | 作用 | 实现 |
|---|---|---|
| `GET /` | 静态单页 | `web/static/index.html` |
| `GET /api/health` | 启动自检信息 | 模型名 / base_url 域名（**掩码**）/ key 是否已配 / 生效权限三元组 |
| `GET /api/sessions` | 历史会话列表 | `list_sessions()` |
| `POST /api/sessions` `{request?}` | 新建会话（可顺带跑首轮） | `Session()` |
| `GET /api/sessions/{tid}` | 会话快照：轮次/计划/图状态/最近报告/**挂起中的审批** | `graph.get_state()`；`tasks[].interrupts` 非空 → 重建审批卡（服务重启后 SSE 缓冲丢失的兜底） |
| `POST /api/sessions/{tid}/messages` `{text}` | 发一轮需求 | `_run_graph_core(..., drain=web_drain)`，忙时 409 |
| `POST /api/sessions/{tid}/approvals` `{approval_id, action, indices?, note?}` | 审批决策 | 唤醒 web_drain 的等待队列 |
| `GET /api/sessions/{tid}/changes/{i}` | 单条改动全文 + diff hunks（「详情」懒加载） | `difflib.unified_diff` 服务端算好结构化返回 |
| `POST /api/sessions/{tid}/undo` | 回退最近一轮文件改动 | `_undo_latest(tid)` |
| `POST /api/sessions/{tid}/retry` | 断点续跑 | `resume_pending(graph, sess, drain=web_drain)` |
| `GET /api/sessions/{tid}/events` | SSE 事件流 | 见 4.2 |
| `GET /api/audit?limit=50` | 审计尾部（M3） | 读 `audit.jsonl` 尾 N 行 |

### 4.2 SSE 事件（`text/event-stream`，带 `id:`，断线用 Last-Event-ID 从环形缓冲重放）

事件类型刻意收敛为 6 种，语义细节由 `node` 事件的 `node` 字段区分：

```
round_start   {round, request, thread_id}
node          {round, node, data}          # planner/agent/worker/guard/verifier/reviewer/report
                                            # data = 节点输出增量，大字段裁剪（见下）
approval_required {round, approval_id, changes: [{index, action, category, permission,
                   path?, command?, preview: {...}, bytes_hint}]}
round_end     {round, usage, session_total, verdict}     # usage 来自 _token_usage_snapshot()
error         {round, phase, message, recoverable, hint} # recoverable=true 时提示 /retry
info          {message}                     # auto_allow 放行、熔断提示等杂项
```

`node.data` 裁剪规则：`pending_changes` 里每条只留 `_shown_change` 摘要（content/old/new/code → 长度）；`feedback` 截前 500 字符 + 总长。**全文只经两处出**：审批卡（预览规则同 CLI）与懒加载详情端点——与 CLI「防长内容刷屏」同一哲学，浏览器端同理。

```json
// 审批卡示例（preview 规则与 cli._print_change_approval 逐条对齐）
{"event": "approval_required", "id": 42, "data": {
  "round": 2, "approval_id": "blue-20260207-abc123-1",
  "changes": [
    {"index": 0, "action": "plan_patch", "category": "write", "permission": "ask",
     "path": "tools.py", "bytes_hint": 312,
     "preview": {"old_head": ["def grep(pattern, path=\".\", glob=None):", ...],
                 "new_head": ["def grep(pattern, path=\".\", glob=None, max_results=200):", ...],
                 "rule": "patch_3lines"}},
    {"index": 1, "action": "plan_run_command", "category": "command", "permission": "ask",
     "command": "pytest tools.py -x", "preview": {"rule": "command_full"}}
  ]}}
```

决策回传即 guard 的 resume 协议原样：`{"action": "approve", "indices": [0]}` / `{"action": "reject", "note": "..."}` / `{"action": "modify", "note": "..."}`。

---

## 5. 审批闭环（本方案的心脏）

### 5.1 web_drain 伪代码

```python
def web_drain(graph, config, sess):          # 签名与 _drain / _auto_drain 一致
    while True:
        cur = graph.get_state(config)
        if not cur.next: break
        tasks = [t for t in cur.tasks if t.interrupts]
        if not tasks: break                   # 断在非审批点：留给 /retry（语义同 _drain）
        payload = tasks[0].interrupts[0].value
        aid = f"{sess.thread_id}:{cur.next[0]}:{len(...)}"      # 稳定审批 id
        bus.publish("approval_required", build_card(payload))    # 含 CLI 同款预览
        decision = bus.wait_decision(aid)      # queue.get() 阻塞，REST 侧投递
        # 阻塞期间进程崩溃 → checkpoint 仍在 interrupt 态，重启后 GET session 重建卡片，无丢失
        _audit_log(sess.thread_id, {**decision, "source": "web"}, payload["changes"])
        for chunk in graph.stream(Command(resume=decision), config=config, stream_mode="updates"):
            for node_name, output in chunk.items():
                _emit_step(node_name, output)   # 与 _drain 同路：正常节点事件回流 SSE
```

### 5.2 CLI ↔ Web 审批语义对齐表（安全底线不降级）

| CLI（`_drain`） | Web 审批卡 | 后端动作 |
|---|---|---|
| `[y] 全批` | 「全部批准」主按钮 | `{"action":"approve"}` |
| `[n] 全拒` + 原因 | 「全部拒绝」+ note 输入（默认"用户拒绝"） | `{"action":"reject","note":...}` |
| `[m] 意见` | 「退回修改」+ 意见必填 | `{"action":"modify","note":...}` |
| `[d] 详情` | 每条改动「展开全文」懒加载（diff hunks / 语法块） | GET changes/{i} |
| `[1,3] 选批` | 每条复选框 + 「批准勾选的 N 项」 | `{"action":"approve","indices":[0,2]}` |
| EOFError → 默认拒 | SSE 断开/服务停止 | **不自动批**——interrupt 态天然挂起，重启后原样重现（fail-closed 优于 CLI） |
| （guard 内部）整批 allow → auto_allow | 不出审批卡，`info` 事件提示「配置放行」 | guard 原逻辑，零改动 |

### 5.3 一轮的生命周期状态机（前端按事件推导）

```
idle → running(planner) → running(agent | workers×N) → awaiting_approval ─┐
                                    ▲                                     │ 决策
                                    └── modify ←──────────────────────────┤
                                                                          ↓
              running(verifier → reviewer) ─revise→ running(agent)（≤3 圈）
                                          └─pass→ running(report) → idle
```

---

## 6. 前端页面设计（零构建单页）

### 6.1 布局

```
┌────────────┬───────────────────────────────────────┬─────────────────┐
│ 会话侧栏     │  对话流（主体，滚动）                    │ 状态面板          │
│            │                                       │ （可折叠）        │
│ ＋新会话    │  ┌ 你：给 grep 加 max_results 参数 ┐    │                 │
│            │  └────────────────────────────────┘    │  [图拓扑小地图]   │
│ ● 当前会话  │  ┌ 计划：3 步 ▸ 并行×2 ────────────┐   │   planner ✓     │
│  轮次 3     │  └────────────────────────────────┘    │   worker×2 ●    │
│  ⏸ 等审批   │  ┌ 过程：worker 暂存 2 项改动 ──────┐   │   guard  ⏸     │
│            │  └────────────────────────────────┘    │   verifier      │
│ ○ 会话 B   │  ┌ ⏸ 审批请求（2 项）────────────── ┐   │   reviewer      │
│   2 小时前  │  │ ☑ patch  tools.py   [write·ask]  │  │   report        │
│ ○ 会话 C   │  │ ☐ run    pytest …   [cmd·ask]  ⚠ │  │                 │
│            │  │ [展开全文/diff]                    │  │  模型 glm-4.7   │
│ ─────────  │  │ 意见：________                     │  │  窗口 128k      │
│ ↩ 撤销本轮  │  │ [全部批准] [批准勾选1项]            │  │  ──────────    │
│ 🔁 断点续跑 │  │ [退回修改] [全部拒绝]               │  │  本轮 token     │
│ 📋 审计尾部 │  └────────────────────────────────┘   │   12.3k / 34 次  │
└────────────┴───────────────────────────────────────┴─────────────────┘
   输入框：[ 描述需求，/ 唤起命令面板 ]                    [发送]（运行中禁用）
```

### 6.2 卡片类型（对话流里的全部消息形态）

| 卡片 | 数据源（事件） | 呈现要点 |
|---|---|---|
| 请求 | 本地输入 | 右侧气泡；轮次号 |
| 计划 | `node(planner)` | 步骤 checklist；`parallel_tasks≥2` 显示并行徽章 + N 个 worker 占位 chip |
| 过程 | `node(agent/worker)` | 折叠条：暂存改动摘要（`_shown_change` 规则）、worker notes；只读工具调用静默（细节在文件日志） |
| **审批卡** | `approval_required` | 见 6.3，本方案核心 |
| 执行结果 | `node(guard, verdict=approved/rejected)` | 已执行摘要 + 改动文件清单；rejected 显眼标注 |
| 验证 | `node(verifier)` | `【自动验证结果】`段按行着色：✓ 绿 ✗ 红（对齐 `_print_node`） |
| 评审 | `node(reviewer)` | `pass/revise` 徽章 + feedback；revise 时标注第几圈（≤3） |
| 报告 | `node(report)` | Markdown 渲染（vendored marked + DOMPurify 消毒） |
| 用量 | `round_end` | 本轮 token / 成本估算 / 上下文占用百分比 |
| 错误 | `error` | 红卡 +「可 /retry 续跑」按钮（recoverable 时） |

### 6.3 审批卡详设

- **逐条列表**：序号 + 类型图标（📝write / 🔧patch / ⚠️command / 🐍python）+ 目标（路径或命令全文）+ 权限徽章（`allow/ask/deny`，色分）+ 尺寸提示（`312 B` / `1.2 KB`）。
  命令与 Python 条目整体加⚠️底色——对应 CLI 里这两类的高风险认知。
- **预览**：与 CLI 同规则内联展示（patch 的 old/new 前三行对、write_file 前五行、python 前十行、命令全文）；「展开全文」懒加载：
  - `plan_patch` → 服务端 `difflib.unified_diff`（与 `_print_change_rich` 同算法）生成结构化 hunks，前端红绿渲染，行号对齐；
  - `plan_write_file` / `plan_run_python` → 带行号代码块，>300 行虚拟滚动或分页。
- **决策区**：复选框（默认全选）+ 四个动作按钮（§5.2 对齐表）+ 意见输入（选「退回修改」时必填）。
  决策后卡片冻结为结果态（已批准 1 项/跳过 1 项，写入审计 ✓）。
- **键盘**：`Y` 全批 / `N` 全拒 / `1-9` 切勾选 / `⌘/Ctrl+Enter` 提交——CLI 肌肉记忆的网页版。

### 6.4 图拓扑小地图

手绘静态 SVG（拓扑固定：`planner → {agent | workers×N} → guard → verifier → reviewer ⇄ agent → report`），
当前节点高亮呼吸，已过节点打勾；worker chip 数量由 planner 事件的 `parallel_tasks` 驱动；
`awaiting_approval` 时 guard 节点闪烁 + 侧栏会话标 ⏸。revise 回边画淡色箭头，激活时闪一下。
（比通用图谱库零依赖、零运行时开销，且拓扑变更时手动同步——图拓扑本就稳定。）

### 6.5 命令面板与杂项

- 输入框敲 `/` 弹面板：`/undo` `/retry` `/history` `/graph`（全屏拓扑模态）`/model`（M4）——与 CLI 斜杠命令一一对应。
- 会话切换即 `GET /api/sessions/{tid}` 重建对话流（从 checkpoint values 摘要 + 环形缓冲回放），挂起审批自动重现。
- 深色主题默认（开发者工具审美），无框架 CSS 变量主题化。

---

## 7. 安全设计

1. **默认只绑 `127.0.0.1`**；`--host 0.0.0.0` 时**强制要求 token**（启动生成随机 token 打印到控制台，Jupyter 模式；`BLUE_WEB_TOKEN` 可固定），不提供 token 拒绝启动——fail-closed。所有非静态 API 校验 `Authorization: Bearer`（localhost 模式豁免）。
2. **审批永不自动**：SSE 断开、前端崩溃、服务重启，都只是让 interrupt 继续挂起（checkpoint 语义天然支持），不存在超时自动批。Web 层不提供任何 auto-approve 端点（benchmark 的 `--auto-approve` 仍只属于 CLI）。
3. **密钥零暴露**：`/api/health` 只报「key 已配置与否」与 base_url 域名；任何端点不回传 `.env` 内容、模型 key、`.blue.toml` 原文（只报生效三元组）。
4. **XSS**：report/feedback 是模型生成的 Markdown/文本 → 渲染一律 DOMPurify 消毒 + 代码块纯文本转义；文件内容展示同理（改动的文件里完全可能藏 `<script>`）。
5. **CSRF/同源**：不设 CORS 头（仅同源），写操作一律 `application/json`（跨站表单打不进来）。
6. **审计一致性**：Web 审批与 CLI 同走 `_audit_log`，decision 附 `"source": "web"`；`auto_allow`（配置放行）与人工 `approve` 的区分语义原样保留——审计数据不被 UI 污染。
7. **依赖供应链**：vendor 库（marked/purify）入库前固定版本、本地托管，**不走 CDN**（本地工具必须离线可用，也避免供应链注入）。

---

## 8. 并发与一致性

| 场景 | 策略 |
|---|---|
| 同一会话二连发 | 409 `round_running`；前端发送按钮运行中禁用 |
| 不同会话并行 | 默认全局串行队列（执行器互斥）；`--concurrency N` 放开（benchmark 多进程已证 sqlite WAL 可承受，谨慎起步） |
| Web 与 CLI 同会话 | **v0.8.x 已落地**：① 跨进程执行锁（`exec_locks` 表，holder/pid/心跳，90s 过期接管）——直连双写者互斥，冲突 409 `session_busy` 报持有方；② CLI 客户端模式（`webclient.py`）——CLI 探测到 Web 引擎后自动变客户端（`--local` 强制直连），执行权归一 Web 引擎，CLI 经 REST/SSE 成为第二视图，单写者强一致（§8 旧「约定禁止」升级为此机制） |
| SSE 断线 | 浏览器 EventSource 自动重连 + Last-Event-ID；服务端每会话环形缓冲（最近 500 事件）重放 |
| 服务重启 | 会话/checkpoint/审计全在 `~/.blue` 落盘，无损；挂起中的审批由 `GET session` 的 interrupts 重建；内存中的 `Session.token_usage` 清零（与 CLI 重启同语义） |
| web_drain 等待中进程被杀 | 同上——interrupt 态持久，无丢失；at-least-once 语义与 CLI `/retry` 完全一致 |

---

## 9. 技术选型与依赖

| 层 | 选型 | 理由 |
|---|---|---|
| 后端 | **FastAPI + uvicorn**（SSE 用原生 `StreamingResponse`，不引 sse-starlette） | 能 `import agent` 是硬要求（D1）；async + pydantic 定协议 schema；静态托管内置 |
| 推送 | **SSE** 而非 WebSocket | 单向推送足够（决策走 REST）；EventSource 原生断线重连 + Last-Event-ID；少一个协议状态机 |
| 前端 | **零构建单页**：原生 ES modules + 手写组件（≤1500 行 JS） | 用户既定选型；贴合项目零依赖哲学；`blue web` 一条命令即用，无 Node 工具链 |
| Markdown | vendored `marked.min.js` + `dompurify.min.js`（`static/vendor/`，固定版本本地托管） | report 是 Markdown；消毒不可省（§7.4） |
| diff | 服务端 `difflib`（标准库）生成 hunks JSON | 与 CLI rich 渲染同算法同结果；前端只做着色 |
| 新依赖 | 全部进 extras：`web = ["fastapi>=0.115", "uvicorn>=0.30"]` | 核心依赖零变化；`pipx install bluecode[web]` 可选装配 |

打包（pyproject 增量）：

```toml
[project.optional-dependencies]
web = ["fastapi>=0.115", "uvicorn>=0.30"]

[tool.setuptools]
py-modules = ["agent", "tools", "prompts", "session", "cli", "doctor", "models"]
packages = ["web"]                       # 新增：web 是带静态资源的包

[tool.setuptools.package-data]
web = ["static/**/*"]                    # index.html/js/css/vendor 随包分发
```

启动方式（与 `init`/`doctor` 子命令先例一致）：`blue web [--host --port --concurrency --no-browser]`，
缺 web 依赖时给一行安装提示（fastapi ImportError 捕获后 fail-fast，退出码 1）。开发态 `python web/server.py` 直跑。

---

## 10. 目录结构

```
web/
├── __init__.py
├── server.py       # FastAPI app、REST 路由、静态托管、启动入口（blue web 委托到此）
├── executor.py     # ExecutorRegistry：会话工作线程、全局串行队列、web_drain 审批桥
├── events.py       # SSE 事件构造（大字段裁剪规则）、环形缓冲、asyncio 桥
├── approve.py      # 审批卡数据：预览生成（复用 cli._preview_lines 规则）、diff hunks、决策校验
└── static/
    ├── index.html
    ├── app.js      # 事件路由 + 状态机 + 组件渲染
    ├── app.css
    └── vendor/     # marked.min.js、purify.min.js（本地固定版本）
```

---

## 11. 核心改动清单（实现时的全部侵入点）

1. `agent.py:1354`：`resume_pending(graph, sess, drain=_drain)` 加第三个带默认值参数——Web `/retry` 注入 web_drain；CLI 调用点零改动。
2. `pyproject.toml`：extras + packages + package-data（§9）。
3. `agent.py main()`：加 `web` 子命令分支（委托 `web.server:main`，ImportError 时友好提示装 `bluecode[web]`）。

其余（step 回调注册、banner 打印、`_finish_round_usage` 播报）全部原样复用——`_run_graph_core` 的 print 在服务端控制台输出，恰好充当服务日志镜像；guard 的 auto_allow 分支打印同理。**图节点、状态、reducer、prompt 一行不动。**

---

## 12. 里程碑与验收

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M0 骨架**（~1 天） | server 起服务、SSE 通、step 事件上屏、fake model 冒烟跑一轮 | 浏览器看到 planner→report 事件序列；`validate_graph.py` 仍全绿 |
| **M1 审批闭环** | approval_required 卡片、决策 REST、web_drain、选批/详情/diff、审计落盘 | fake model 走「暂存→审批→approve/reject/modify/indices」四路径全通过；audit.jsonl 出现 `source: web` 记录 |
| **M2 会话管理** | 会话列表/切换/快照重建、undo、retry、SSE 重连重放 | 服务重启后挂起审批原样重现；undo/retry 与 CLI 行为一致 |
| **M3 观测面板** | 图小地图、token/上下文表、health/权限展示、审计尾部 | 长会话（≥5 轮）token 表与 CLI `/history` 数值一致 |
| **M4 可选** | 模型切换器、benchmark 结果查看器（读 `results/run-*.json`）、局域网 token 按需启用 | 按需评估 |

**测试策略**（沿用离线验证哲学，不需 API key）：`web/tests/` 用 httpx `AsyncClient` 打 app + fake model
（`patch("agent._make_model")` 与 `_make_plain_model` 指向同一实例、patch `should_skip_planner`——与
`validate_graph.py` 完全相同的坑位清单）；MemorySaver 替身；`agent.DB_PATH/AUDIT_LOG/BACKUP_ROOT` 指临时目录三件套照搬。
断言 SSE 事件序列、决策→resume 数据流、indices 语义、审计与快照副作用。改动核心（`resume_pending` 签名）后必跑 `python validate_graph.py` 回归。

---

## 13. 风险与开放问题

| # | 风险/问题 | 缓解 |
|---|---|---|
| 1 | **审批挂起期间浏览器关了** | 非问题反而更优：interrupt 态持久，重开页面即重现（§8）；可选加桌面通知（M4，Notification API） |
| 2 | 节点级粒度无打字机效果 | 已列入非目标；`astream_events` 落地后协议加 `delta` 事件即可平滑升级（SSE 天然增量） |
| 3 | 超大 diff / 超长报告卡顿 | 预览+懒加载（§4.2 裁剪规则）+ 虚拟滚动；本地单用户，实际上限可控 |
| 4 | `Session.token_usage` 内存态、重启清零 | 与 CLI 同语义，文档标注；不为此改存储 |
| 5 | pipx 安装后静态资源漏打包 | M0 就用 `pipx install .` 验 package-data，不留到最后 |
| 6 | CLI 与 Web 同会话并发操作 | v0.8.x 已解决：跨进程执行锁互斥直连双写者 + CLI 客户端模式归一单写者（§8）；锁崩溃残留靠 90s 心跳过期接管 |
| 7 | `_run_graph_core`/guard 的 print 混入服务日志 | v1 接受（当镜像日志看）；嘈杂再加大字段静音开关 |
| 8 | Safari 对 SSE 的连接数限制（6 个/域） | 单会话单连接 + 页面唯一，不触界；文档备注 |

---

## 14. 远期方向（不在本方案范围）

- 局域网/移动端审批模式（token 已预留，PWA + Web Push 通知）；
- benchmark 结果可视化（数据已在 `results/run-*.json`，只缺读它的页面）；
- astream_events 落地后的 token 级流式与中断按钮（协议预留 `delta` 事件类型）；
- LangGraph Studio 集成评估（roadmap 原定 v0.8 后再议，本方案图小地图先行满足 80% 需求）。

---

## 15. 实现偏差记录（v0.8 落地时）

M0–M2 实现与本文的偏离，全部记录在案（实现方与协议方共同维护）：

| 本文 § | 设计 | 实现 | 影响 |
|---|---|---|---|
| §7.4 / §9 Markdown | vendored marked + DOMPurify 渲染报告 | **一律 `textContent`，绝不 innerHTML**（零第三方 JS、零供应链） | XSS 面更小；报告暂以纯文本展示、不渲染 Markdown（升级点：引入 sanitizer 即可，协议不变） |
| §5 审计 | 调 `agent._audit_log` 记 source | web 层 `_audit_log_web` 独立写同一 jsonl、逐字段镜像 | 核心 `_audit_log` 只抄 action/indices/note 三键、source 会被静默丢弃（读码确认）；独立写保住 `source: web` 区分 |
| §7.1 token | 非 loopback 无 token 拒绝启动 | **缺省自动生成随机 token 打印**（`--token` / `BLUE_WEB_TOKEN` 可固定） | 更便捷且仍 fail-closed（token 强制校验所有 /api） |
| §4.2 SSE 鉴权 | 未规定 | token 模式走 `?token=` 查询参数 | EventSource 不能带 header 的妥协；URL 有日志暴露面，默认 loopback 模式无此问题 |
| §10 模块 | `approve.py` 独立 | 并入 `events.py`（卡构造/预览/diff/决策校验同文件） | 无功能差异 |
| §6.4 / §6.5 | 图小地图 + 命令面板 | 未做（归 M3/M4，v0.10） | M0–M2 验收不含此项 |
| §8 并发 | 全局串行 + 409 | 另加：**step 回调按执行线程过滤**（`executor._step_bridge`） | 防会话 A 执行中、会话 B 排队时 A 的节点事件混入 B 的 SSE（评审发现的 P2，已修 + 回归测试 `test_no_cross_session_event_leak`） |
| §9 前端 | 组件化示例结构 | 单文件 `app.js`（879 行）无框架、无构建 | 符合零构建选型；M3 观测面板在此文件上增量 |
| §12 测试 | httpx 归口 extras | `web` extras = fastapi+uvicorn；`test` extras = httpx（requirements.txt 同步补齐） | 依赖声明闭环，`pip install bluecode[web]` / 冒烟测试均可直接装 |
