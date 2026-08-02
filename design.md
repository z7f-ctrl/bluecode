# 个人版 Coding Agent 设计文档

> 一个精简、有趣、能真正跑起来的个人编程助手，用 LangGraph 的核心能力实现。
> 文档状态：**v0.1 设计稿（历史文档）**。实现已演进超出本文档——当前状态见 README.md。
> 主要偏差：reviewer 已去毒舌人设（v0.2.5）；图新增 verifier 自动验证节点；工具集扩至 8 个（新增 final_answer、plan_run_python）；agent 节点为手写工具循环而非 create_react_agent（§6.2 的备选方案）。

---

## 1. 一句话定位

**「小蓝」（代号 Blue）**：一个跑在你自己电脑上的个人 coding agent——你说需求，它列计划、动手改代码、跑测试，还会像毒舌同事一样**自审挑刺**，直到自己满意才交付；涉及写文件 / 执行命令等危险动作时，它会停下来**等你确认**。

## 2. 设计原则

| 原则 | 说明 |
|---|---|
| 精简 | 核心逻辑一个 `agent.py`（约 300~400 行）就能跑，不引入框架、不用 FastAPI |
| 有趣 | 自审环节是一个带人格的「毒舌评审」，输出有梗有表情，让循环过程可读、好玩 |
| 可运行 | 兼容任意 OpenAI 兼容接口（含你 test2.py 里用的本地服务），配置走环境变量 |
| 安全 | 默认只读 + 写文件/跑命令前人工审批，工作目录沙箱化 |
| 可恢复 | 检查点持久化，中断后可以断点续跑、随时 fork 分支 |

## 3. 使用方式（CLI）

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=<你的 OpenAI 兼容端点>
export MODEL_NAME=<你的模型名>

python agent.py "给 hello.py 加上错误处理，并写一个 pytest 测试"
```

交互流程（伪终端）：

```
[蓝] 收到！让我先列个计划：
     1. 读 hello.py，评估现状
     2. 设计错误处理方案
     3. 修改代码 + 新增测试
     4. 跑 pytest 验证
[蓝] 执行第 1 步：读文件 hello.py
[蓝] 准备写入 hello.py（新增 try/except）... ⏸ 等你审批
     [y] 允许   [n] 拒绝   [m] 修改意见
[你] y
[蓝] 已写入。跑 pytest ...
[毒舌评审] 嗯，except 捕获太宽了，吞掉 KeyboardInterrupt 会挨骂的。回去改。🔪
[蓝] 收到批评，修改中...
[毒舌评审] 这版还行。放行。✅
[蓝] 完成 ✅ 改动清单：hello.py(修改)、test_hello.py(新增)
```

## 4. 整体架构

```
                    ┌──────────────────────────────────────┐
   用户请求 ────────▶│  planner（计划拆解）                    │
                    └───────────────┬──────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────┐
              ┌────▶│  agent（LLM + 工具循环）               │
              │     │  - 读文件 / 写文件 / 跑命令 / 搜索      │
              │     └───────────────┬──────────────────────┘
              │                     ▼
              │     ┌──────────────────────────────────────┐
              │     │  guard（安全闸门）                      │
              │     │  是否涉及写文件 / 跑命令？              │
              │     │  是 → interrupt() 等人审批 → resume    │
              │     └───────────────┬──────────────────────┘
              │                     ▼
              │     ┌──────────────────────────────────────┐
              │     │  reviewer（毒舌评审）                   │
              │     │  verdict = pass / revise（带修改意见） │
              │     └──────┬───────────────┬──────────────┘
              │            │ pass          │ revise
              │            ▼               │
              │         [END]              │
              └────────────────────────────┘  (回到 agent 继续改)
```

- **主循环**：`planner → agent → guard → reviewer`，评审不通过就弹回 `agent`（循环次数上限 3，防止死循环）。
- **条件边**：`reviewer` 用一条条件边根据 `verdict` 决定回炉还是收工——这是 LangGraph 最典型的用法。
- **人工介入**：`guard` 节点用 `interrupt()` 冻结图，等用户 `Command(resume=...)` 后从原处继续，不重跑前面的节点。

## 5. 状态设计

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 完整对话历史（含工具结果）
    request: str                              # 原始需求（invoke 时注入）
    plan: list[str]                           # planner 拆出的步骤
    current_step: int                         # 当前执行到第几步
    pending_changes: list[dict]               # 待审批的改动 [{path, action, content, cmd}]
    review_rounds: int                        # 自审轮数（上限 3）
    verdict: str                              # reviewer 结论：pass / revise
    feedback: str                             # reviewer 的修改意见
```

字段说明：

| 字段 | 类型 | 理由 |
|---|---|---|
| `messages` | `Annotated[list, add_messages]` | 复用 LangChain 消息归并，工具结果自动追加，模型天然有上下文 |
| `plan` / `current_step` | list / int | 让 agent 有「计划感」，避免东一榔头西一棒子 |
| `pending_changes` | list[dict] | guard 节点把「要写的东西」攒起来一次性给人看，审批体验好 |
| `review_rounds` / `verdict` / `feedback` | 标量 | 控制自审循环终止条件，三字段一起决定边的走向 |

> 设计取舍：没有用复杂的嵌套状态，`messages` 承担模型上下文，其余都是轻量控制字段——符合「精简」原则。

## 6. 节点设计

### 6.1 `planner` — 计划拆解
- 输入：`request` + 当前仓库文件清单
- 输出：`plan`（3~6 步的中文步骤列表）
- 实现：一次 LLM 调用，要求输出 JSON 数组；解析失败则退回「单步计划」兜底

### 6.2 `agent` — 执行者（工具循环）
- 复用 `create_agent`（`langgraph.prebuilt`）或手写 `ToolNode + tools_condition`
- 工具集（见 §7）只读为主；写文件/跑命令的工具**只生成 `pending_changes`，不真正执行**
- 每完成一个工具调用，更新 `current_step` 并汇报进度

### 6.3 `guard` — 安全闸门（人机协作核心）
```python
def guard(state: AgentState) -> AgentState:
    if not state["pending_changes"]:
        return {"verdict": "proceed"}            # 纯读操作，直接放行
    answer = interrupt({                          # ⏸ 冻结图，等人回复
        "changes": state["pending_changes"],
        "question": "以上改动/命令是否允许执行？",
    })
    if answer.get("action") == "reject":
        return {"verdict": "rejected", "feedback": answer.get("note", "")}
    if answer.get("action") == "modify":
        return {"verdict": "revise", "feedback": answer.get("note", "")}
    return {"verdict": "approved"}                # 执行写操作，清空 pending_changes
```

### 6.4 `reviewer` — 毒舌评审（有趣的灵魂）
- 系统提示词注入人设：*「你是资深但刻薄的主程，看完 diff 会先毒舌再给真意见，但绝不冤枉好人」*
- 输入：`request`、`pending_changes`（或 diff）、测试结果
- 输出：`verdict = pass / revise` + 一句毒舌总结 + 一条具体修改建议
- 彩蛋：评审意见里会随机插入 emoji（🔪 / 🍵 / 🚨 / 😤），`review_rounds == 2` 时文案会自动「收敛」不再毒舌

### 6.5 `report` — 收尾
- 汇总改动清单、测试结果、评审轮数，用简洁的 Markdown 汇报

## 7. 工具集

| 工具 | 读/写 | 是否需审批 | 说明 |
|---|---|---|---|
| `list_files` | 读 | 否 | 列目录（限定工作目录内） |
| `read_file` | 读 | 否 | 带行号读取 |
| `grep` | 读 | 否 | 关键词搜索（rg 实现） |
| `plan_write_file` | 写（暂存） | **是** | 只把改动放进 `pending_changes` |
| `plan_patch` | 写（暂存） | **是** | 生成 unified diff 供展示 |
| `plan_run_command` | 执行（暂存） | **是** | 只记录命令，由 guard 批准后才真正 subprocess 执行 |

安全边界：
- 所有路径都做 `resolve()` 校验，越界直接报错
- 命令执行默认 `timeout=60`，禁止 shell 管道类高危组合（`rm -rf /` 等关键词拦截）
- 工作目录固定为当前项目目录，不提供「任意路径」能力

## 8. 用到的 LangGraph 能力（与代码对应）

| 能力 | 用法 | 在本项目中的作用 |
|---|---|---|
| `StateGraph + TypedDict` | 状态 schema | 全局数据流，节点只返回增量字段 |
| `add_messages` reducer | `Annotated[list, add_messages]` | 消息自动追加，避免手工管理历史 |
| `add_conditional_edges` | reviewer 出边 | 根据 `verdict` 决定 `revise → agent` 还是 `→ END`，形成自审循环 |
| `interrupt()` + `Command(resume=...)` | guard 节点 | 人机审批：图冻在 guard，用户回复后原地继续 |
| `ToolNode + tools_condition` | agent 节点 | 工具调用的标准循环，省去手写 while |
| `MemorySaver / SqliteSaver` | `compile(checkpointer=...)` | 会话记忆；重启后可 `get_state` / `update_state` 续跑 |
| `astream_events` | CLI 输出层 | 流式打印 token 与节点进度，有「在干活」的实感 |
| `Send`（可选扩展） | 多文件并行 | 一次评审不通过时，把多个文件的修改意见并行下发 |
| 子图 | reviewer 独立成子图 | 评审逻辑可单独测试、复用 |

图的核心接线（约 20 行，即「精简」的证明）：

```python
builder = StateGraph(AgentState)
builder.add_node("planner", planner)
builder.add_node("agent", agent_node)          # create_agent(tools=...)
builder.add_node("guard", guard)
builder.add_node("reviewer", reviewer)
builder.add_node("report", report)

builder.add_edge(START, "planner")
builder.add_edge("planner", "agent")
builder.add_edge("agent", "guard")
builder.add_edge("guard", "reviewer")
builder.add_conditional_edges(
    "reviewer",
    route_by_verdict,                           # pass→report / revise→agent
    {"pass": "report", "revise": "agent"},
)
builder.add_edge("report", END)

graph = builder.compile(checkpointer=MemorySaver())
```

## 9. 文件结构（下一步实现用）

```
langgraph/
├── design.md          # 本文档
├── agent.py           # 核心：状态、节点、建图、CLI（300~400 行）
├── tools.py           # 7 个工具 + 安全校验
├── prompts.py         # planner / 毒舌评审 的系统提示词（文案集中管理）
├── requirements.txt   # langgraph, langchain-openai, langchain-core
└── hello.py 等        # 现有示例文件，作为演示改动目标
```

## 10. 运行与验证方式

- 初始化：`pip install -r requirements.txt`
- 试玩：`python agent.py "用一句话说说 hello.py 的问题并修复它"`
- 验证：图编译后先用 `graph.get_graph().draw_ascii()` 打印拓扑确认边正确；
  再跑一轮「只读任务」（如 `"统计仓库文件行数"`）确认不走审批分支；
  最后跑「写文件任务」验证 interrupt/resume 全链路
- 持久化：`SqliteSaver` 路径默认 `~/.blue/checkpoints.sqlite`，`python agent.py --resume` 恢复上次会话

## 11. Roadmap（按优先级）

> 截至 v0.4 的实际进度，见 README.md 的 Roadmap 表。

1. **v0.1**：单文件跑通 plan → act → guard → review 全流程 + CLI 流式输出 ✅
2. **v0.2**：SqliteSaver 持久化 + `--resume` / 会话列表 ✅（含多轮会话 + 斜杠命令）
3. **v0.3**：多文件并行修改（`Send`），评审意见按文件分发 —— 未实现；实际 v0.3 做了 smolagents 四件套（final_answer / 命令白名单 / plan_run_python / step 回调）
4. **v0.4**：接入 LangGraph Studio 可视化调试 —— 未实现；实际 v0.4 做了 QuixBugs benchmark
5. **v0.5**：评测集（10 个标准任务），记录每次的 review 轮数、成功率 —— 已超额实现（QuixBugs 31 题）

## 12. 风险与对策

| 风险 | 对策 |
|---|---|
| 自审循环死循环 | `review_rounds` 上限 3，超限强制 pass 并提示人工 |
| 模型输出非法 JSON | planner 失败降级为单步计划；工具参数走 pydantic 校验 |
| 危险命令 | guard 审批 + 关键词拦截 + timeout + 工作目录沙箱 |
| 上下文过长 | agent 节点对 `messages` 做截断/摘要后传给模型 |
| API key 泄露 | 全部走环境变量，README 里明确提示（你 test2.py 里硬编码的 key 建议轮换） |

---

*文档完。下一步：按 §9 实现 v0.1。*
