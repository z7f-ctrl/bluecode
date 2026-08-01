# Bluecode（小蓝）

> 一个跑在你自己电脑上的个人 coding agent：你说需求，它列计划、动手改代码、跑测试，还会像毒舌同事一样自审挑刺，直到自己满意才交付；涉及写文件 / 执行命令等危险动作时，它会停下来等你确认。

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 实现，核心逻辑全部手写、无框架依赖，兼容任意 OpenAI 兼容接口。

## 快速开始

```bash
# 1. 进入虚拟环境（如有）
source <你的 venv 路径>/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置模型（任意 OpenAI 兼容端点）
cp .env.example .env   # 然后编辑 .env，填入真实值（.env 已被 .gitignore 排除，不入库）
# 或直接走环境变量：
# export OPENAI_API_KEY=sk-xxx
# export OPENAI_BASE_URL=<你的 OpenAI 兼容端点>
# export MODEL_NAME=<你的模型名>
# 两种方式等效，agent.py 会自动加载 .env；显式环境变量优先于 .env。

# 4. 跑一个需求
python agent.py "给 hello.py 加上错误处理，并写一个 pytest 测试"
```

### 基本命令

| 命令 | 作用 |
|---|---|
| `python agent.py --show-graph` | 打印图拓扑（`grandalf` 提供 ASCII 渲染） |
| `python validate_graph.py` | 离线功能验证，不需要 API key |
| `python agent.py "需求"` | 跑一个需求（交互式，需 API key） |
| `python agent.py --resume` | v0.2 占位，暂未实现 |

## 它是怎么工作的

```
用户请求 ──▶ planner（拆计划）──▶ agent（LLM+工具循环）──▶ guard（安全闸门）
                                              ▲            │
                                              │            ▼
                                              │         reviewer（毒舌评审）
                                              │            │
                                              └──revise───  │ pass
                                                            ▼
                                                         report（交付）
```

五个节点（`agent.py`）：

| 节点 | 职责 |
|---|---|
| `planner` | 把需求拆成 3~6 步中文计划（JSON 解析失败自动降级为单步） |
| `agent` | 手写工具循环；只读工具直接执行，写/执行工具只**暂存**到 `pending_changes`，不真动手 |
| `guard` | 有待审批改动时 `interrupt()` 冻结图，等你 `y`/`n`/`m`；通过才真正写文件/跑命令 |
| `reviewer` | 「毒舌老蓝」自审：`verdict: pass / revise`，revise 弹回 agent 重改（上限 3 轮） |
| `report` | 汇总改动清单 + 评审轮数 + 执行结果，输出 Markdown 交付报告 |

**安全模型**：工作目录沙箱（路径越界直接拒）+ 命令危险关键词启发式拦截（`rm -rf`/管道/sudo/reboot 等）+ 关键的最后一道人工审批。两条防线叠加，缺一不可——命令校验同时挂在「agent 暂存时」和「guard 审批执行前」两处。

## 状态设计

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 完整对话历史（含工具结果）
    request: str                              # 原始需求
    plan: list[str]                           # planner 拆出的步骤
    current_step: int                         # 当前执行步
    pending_changes: list[dict]               # 待审批改动 [{action, ...}]
    review_rounds: int                        # 自审轮数（上限 3）
    verdict: str                              # pass / revise / proceed / rejected / approved
    feedback: str                             # 评审意见 / 执行结果
```

## 工具集（`tools.py`）

| 工具 | 类型 | 审批 | 说明 |
|---|---|---|---|
| `list_files` | 只读 | 否 | 列目录（沙箱内） |
| `read_file` | 只读 | 否 | 带行号读取 |
| `grep` | 只读 | 否 | 正则搜索（最多 50 条） |
| `plan_write_file` | 暂存 | **是** | 完整覆盖写入 |
| `plan_patch` | 暂存 | **是** | 唯一 old→new 文本替换 |
| `plan_run_command` | 暂存 | **是** | shell 命令（timeout 60s + 危险词拦截） |

所有工具路径都经 `resolve()` 校验，越界即报错；工作目录固定为当前项目目录。

## 项目结构

```
langgraph/
├── agent.py            # 状态、节点、建图、CLI（~300 行）
├── tools.py            # 6 工具 + 安全校验（check_command_safety）
├── prompts.py          # planner / agent / 毒舌评审 / report 提示词
├── validate_graph.py   # 离线功能验证（fake model，不需 API key）
├── requirements.txt
├── .env.example     # 配置模板（入库），复制为 .env 后填真实值
├── .env             # 本地配置（不入库，含密钥）
└── design.md        # 设计稿（§5~§8 是实现的直接来源）
```

> AGENTS.md（含本地环境细节）已被 .gitignore 排除，不入库。

## 验证

```bash
# 离线六路验证：只读 / 写+审批通过 / 拒绝 / revise 回边 / cwd 越界双路径拦截
python validate_graph.py
# 期望输出：ALL OFFLINE TESTS PASSED ✅
```

真机验证需要 API key（见「快速开始」）。真实模型会触发 revise 回边和 interrupt 审批——这是设计意图，不是 bug。

## Roadmap

| 版本 | 内容 | 状态 |
|---|---|---|
| v0.1 | 单文件跑通 plan→act→guard→review 全流程 + CLI 交互 | ✅ 已实现 |
| v0.2 | `SqliteSaver` 持久化 + `--resume` / 会话列表 | 未实现 |
| v0.3 | 多文件并行修改（`Send`），评审意见按文件分发 | 未实现 |
| v0.4 | LangGraph Studio 可视化调试 | 未实现 |
| v0.5 | 评测集（10 个标准任务） | 未实现 |

## 安全说明（重要）

- **API key 只放在本地 `.env`（已被 .gitignore 排除）或环境变量**，绝不写进代码或已提交的文件。不要把 `.env` 同步到网盘/云存储。
- `test2.py` 硬编码过一个真实 key，**该文件已被 `.gitignore` 排除，不进入版本库**；但它仍在你的本地磁盘上。建议尽快轮换该 key（design §12 已要求），轮换后如确认无敏感信息可移出 .gitignore。
- 危险命令拦截是启发式的，不是完备沙箱——**真正的安全门槛是 guard 人工审批**，不要让它在无人值守模式下自动运行。
- 默认只读优先：模型被引导先读后写，写操作必须过审批。

## 已知限制（v0.1）

- `--resume` 尚未实现（v0.2）。
- `astream_events` token 级流式输出未做，当前是节点级播报。
- reviewer 的 `feedback` 会携带原始 `verdict:` 行一起打印，略显粗糙。
- 毒舌评审每次任务要多轮 LLM 调用（通常 4~7 次），API 限流时体验会慢。
