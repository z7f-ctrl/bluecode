# Bluecode（小蓝）

> 一个跑在你自己电脑上的个人 coding agent：你说需求，它列计划、动手改代码、自动验证、自审挑刺，直到自己满意才交付；涉及写文件 / 执行命令等危险动作时，它会停下来等你确认。

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
| `python validate_graph.py` | 离线功能验证（17 项，不需要 API key） |
| `python agent.py "需求"` | 跑一个需求（交互式，需 API key） |
| `python agent.py "需求" --auto-approve` | benchmark 模式：guard 自动审批，不中断（CI/评测用） |
| `python agent.py --resume` | 列出历史会话并恢复 |

## 它是怎么工作的

```
                 ┌─ 独立子任务≥2 → Send×N worker（并行，只读+暂存）─┐
用户请求 ──▶ planner（拆计划）┤                                    ├──▶ guard（安全闸门）
                 └─ 否则 → agent（LLM+工具循环）──────────────────┘        │
                                              ▲                            ▼
                                              │                        verifier（自动验证）
                                              │                            │
                                              │                        reviewer（评审）
                                              │                            │
                                              └──────────revise────────    │ pass
                                                                           ▼
                                                                        report（交付）
```

七个节点（`agent.py`）：

| 节点 | 职责 |
|---|---|
| `planner` | 把需求拆成 3~6 步中文计划；简单需求（短、无多步骤词）跳过模型调用直接单步；输出 `{steps, parallel_tasks}`，完全独立的子任务 ≥2 时走并行（上限 4 个） |
| `agent` | 手写工具循环；只读工具直接执行，写/执行工具只**暂存**到 `pending_changes`，不真动手；`final_answer` 显式终止 |
| `worker` | 并行 worker（v0.5）：处理派发的一个独立子任务，只读+暂存不执行，产出经 reducer 聚合到 guard 一次审批 |
| `guard` | 有待审批改动时 `interrupt()` 冻结图，等你 `y`/`n`/`m`；通过才真正执行；`--auto-approve` 模式下自动放行 |
| `verifier` | 审批通过后自动验证：`py_compile` 语法检查改动文件 + 自动发现并跑 pytest（最多 3 个文件，30s 超时），结果追加到 feedback |
| `reviewer` | 自审：`verdict: pass / revise`，revise 弹回 agent 重改（上限 3 轮）；判定有 verifier 的客观验证结果做依据 |
| `report` | 汇总改动清单 + 评审轮数 + 验证/执行结果，输出 Markdown 交付报告 |

**安全模型**：工作目录沙箱（路径越界直接拒）+ 命令黑名单关键词拦截（含管道 `|`、复合命令 `&&`/`;`、命令替换 `$()`/反引号——防"第二段藏刀"）+ 可选命令白名单（`BLUE_COMMAND_WHITELIST`）+ 关键的最后一道人工审批。命令校验同时挂在「agent 暂存时」和「guard 审批执行前」两处（双路径）。

## 状态设计

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 完整对话历史（含工具结果）
    request: str                              # 原始需求
    plan: list[str]                           # planner 拆出的步骤
    current_step: int                         # 当前执行步
    pending_changes: list[dict]               # 待审批改动 [{action, ...}]（reducer 聚合：空=清空，非空=追加）
    review_rounds: int                        # 自审轮数（上限 3）
    verdict: str                              # pass / revise / proceed / rejected / approved
    feedback: str                             # 评审意见 / 执行结果 / 自动验证结果
    parallel_tasks: list[str]                 # planner 拆出的可并行子任务（空 = 走串行）
    current_subtask: str                      # Send 注入给单个 worker 的子任务
    worker_notes: list[str]                   # 各并行 worker 的一句话总结（reducer 聚合）
    executed_changes: list[dict]                # guard 执行通过后留存的完整改动（reviewer 看 diff、report 列清单）
    changed_files: list[str]                    # guard 写入的改动文件列表（verifier 读）
```

## 工具集（`tools.py`）

只读工具（免审批）：

| 工具 | 说明 |
|---|---|
| `list_files` | 列目录（沙箱内） |
| `read_file` | 带行号读取（单次最多 500 行） |
| `grep` | 正则搜索（最多 50 条，单行截断 200 字符防 minified 长行） |
| `final_answer` | 任务完成时显式终止 agent 循环，给出最终答复摘要 |

暂存工具（需审批，只攒 `pending_changes` 不真执行）：

| 工具 | 说明 |
|---|---|
| `plan_write_file` | 完整覆盖写入 |
| `plan_patch` | old→new 文本替换：默认要求唯一匹配（歧义即拒防误改）；多处时传 `occurrence=N` 指定替换第 N 处，比重写整个文件省 token |
| `plan_run_command` | shell 命令（timeout 60s + 黑名单拦截 + 可选白名单） |
| `plan_run_python` | 受限 Python 沙箱执行：ast 静态检查（import 白名单——不含 subprocess，防绕过命令校验 + 禁 dunder 属性 + 节点数上限）+ 受限 builtins（剔 open/eval/exec，getattr/setattr 包装拦 dunder 字符串参数）+ 30s 超时。适合一次组合多操作，比多次 plan_run_command 省轮次 |

所有路径经 `_resolve()` 校验，越界即报错；工作目录固定为当前项目目录。

### 命令白名单（可选）

```bash
# .env 或环境变量：设置后命令头必须在白名单内，否则拦截
BLUE_COMMAND_WHITELIST=python3,pytest,ls,cat
```

未设置时保持黑名单现状（拦 `rm -rf`/管道/sudo 等）；设置后升级为「没列的都拦」，误伤更少、语义更直白。

## 项目结构

```
bluecode/
├── agent.py            # 状态、节点、建图、CLI、step 回调机制（~950 行）
├── tools.py            # 8 工具 + 安全校验（命令/Python 双路径）
├── prompts.py          # planner / agent / worker / reviewer / report 提示词（文案集中）
├── validate_graph.py   # 离线功能验证（fake model，10 项，不需 API key）
├── requirements.txt
├── .env.example        # 配置模板（入库），复制为 .env 后填真实值
├── .env                # 本地配置（不入库，含密钥）
├── design.md           # v0.1 设计稿（历史文档，实现已演进超出）
└── benchmarks/
    └── quixbugs/       # QuixBugs 修 bug 基准（40 题 = 31 简单 + 9 图算法，见 README）
        ├── prepare.py        # 题库提取
        ├── run_benchmark.py  # 基准 runner
        └── README.md
```

> AGENTS.md（含本地环境细节）已被 .gitignore 排除，不入库。

## 验证

```bash
# 离线 17 项验证：只读 / 写+审批 / 拒绝 / revise 回边 / cwd 越界双路径 /
# 多轮会话 / 会话元信息持久化 / revise 消息压缩 / planner 跳过 / 并行 worker 扇出 /
# 安全加固（命令复合符 / subprocess / getattr dunder）/ 工具易用性（grep 截断 + patch occurrence）/ 滑动窗口 / report 模板化
python validate_graph.py
# 期望输出：ALL OFFLINE TESTS PASSED ✅
```

真机验证需要 API key（见「快速开始」）。真实模型会触发 revise 回边和 interrupt 审批——这是设计意图，不是 bug。

## Benchmark（QuixBugs）

内置 [QuixBugs](https://github.com/jkoppel/QuixBugs) 40 个算法修 bug 题作为能力评测（31 个简单题 + 9 个依赖 `node.py` 的图/链表题）：

```bash
git clone --depth 1 https://github.com/jkoppel/QuixBugs.git /tmp/quixbugs-src
python3 benchmarks/quixbugs/prepare.py                 # 生成题库
python3 benchmarks/quixbugs/run_benchmark.py           # 全量 40 题
python3 benchmarks/quixbugs/run_benchmark.py --algo gcd  # 单题冒烟
python3 benchmarks/quixbugs/run_benchmark.py --workers 4  # 并行跑题
```

详见 `benchmarks/quixbugs/README.md`。

## Roadmap

| 版本 | 内容 | 状态 |
|---|---|---|
| v0.1 | 单文件跑通 plan→act→guard→review 全流程 + CLI 交互 | ✅ 已实现 |
| v0.2 | `SqliteSaver` 持久化 + 多轮会话 + 斜杠命令 + `--resume` | ✅ 已实现 |
| v0.2.5 | reviewer 风格收敛（去毒舌人设）+ planner 条件化 + verifier 自动验证节点 | ✅ 已实现 |
| v0.3 | 借鉴 smolagents：`final_answer` 显式终止 + 命令白名单 + `plan_run_python` 受限沙箱 + step 回调注册机制 | ✅ 已实现 |
| v0.4 | QuixBugs benchmark（31 题修 bug 判分）+ `--auto-approve` 模式 | ✅ 已实现 |
| v0.4.5 | reviewer 判定 fail-closed + 判分管线修复 + benchmark 扩至 40 题（+9 图算法）+ runner 并行 | ✅ 已实现 |
| v0.5 | 多文件并行修改：`Send` 扇出 worker + reducer 聚合 + 一次性审批 | ✅ 已实现 |
| v0.5.x | 日志两层 + token 追踪 + 滑动窗口 + 安全加固 + CLI 颜色 + executed_changes + 审批预览/详情 + 工具易用性 | ✅ 已实现 |
| v0.6 | 审批与信任：diff 渲染（rich/pygments）、逐条审批（序号选择批准）、`/undo` 快照回退、审计日志 `audit.jsonl`、CI 退出码、成本播报 | 未实现 |
| v0.7 | 产品化分发：`pyproject.toml` + `blue` 命令（pipx 安装）、`blue init` 引导 / `blue doctor` 自检、权限分级 `.blue.toml`（allow/ask/deny）、`/retry` 失败恢复 | 未实现 |
| v0.8 | 基准扩展：FAIL_TO_PASS/PASS_TO_PASS 双判据、BugsInPy（图算法 9 题已在 v0.4.5 完成） | 未实现 |
| backlog | LangGraph Studio（等图复杂度上来再做）、token 级流式输出 + Ctrl-C 打断、prompts 中英双语化 | 暂缓 |

## 安全说明（重要）

- **API key 只放在本地 `.env`（已被 .gitignore 排除）或环境变量**，绝不写进代码或已提交的文件。不要把 `.env` 同步到网盘/云存储。
- `test2.py` 硬编码过一个真实 key，**该文件已被 `.gitignore` 排除，不进入版本库**；但它仍在你的本地磁盘上。建议尽快轮换该 key。
- 危险命令拦截是启发式的，不是完备沙箱——**真正的安全门槛是 guard 人工审批**。`--auto-approve` 仅限 benchmark/CI 场景，不要在生产代码库上无人值守运行。
- `plan_run_python` 的沙箱（ast 检查 + 受限 builtins）是纵深防御的一层，不是完备隔离——同样依赖审批兜底。
- 默认只读优先：模型被引导先读后写，写操作必须过审批。

## 已知限制

- `astream_events` token 级流式输出未做，当前是节点级播报（已通过 step 回调机制解耦，可挂自定义 UI）。
- agent/worker 工具循环内有滑动窗口（`AGENT_MSG_WINDOW=20`）：较早的工具交互会被压缩成摘要喂给后续调用，极端长任务里模型可能"忘记"早期细节（摘要保留了文件/改动/执行结果的要点）。
- 只读/拒绝场景的交付报告是模板直出（不调 LLM），文风朴素；涉及实际改动的报告仍由 LLM 组织语言。
- reviewer 每次任务要多轮 LLM 调用（通常 4~7 次），API 限流时体验会慢。
- 会话元信息（`sessions` 表）与 LangGraph checkpoint 分开存储，极端情况下可能不一致（如手动删 checkpoint 文件）。
- `plan_run_python` 无 CPU/内存硬限制（有 30s 超时和 ast 节点数上限），恶意代码仍可能耗资源——审批时留意。

## 多轮会话与斜杠命令

直接运行 `python agent.py`（不带参数）即进入多轮模式：

```
> 给 hello.py 加错误处理
...（图执行 + 审批）...
> /history       # 查看当前会话状态
> /clear         # 开启新 thread，清空上下文
> /quit          # 退出
```

可用命令：`/help` `/quit` `/exit` `/clear` `/new` `/history` `/graph` `/resume`。

交互界面带颜色区分：用户输入提示符 `>`（亮青）、[蓝] 播报（蓝）、[蓝·worker]（青）、放行/成功（绿）、待审批/打回（黄）、验证失败行（红）。仅真实终端启用——管道/重定向（benchmark 子进程）自动无色；设 `NO_COLOR=1` 可强制关闭（遵循 no-color.org 惯例）。`input()` 提示符的 ANSI 码用 `\001\002` 包围（readline 光标安全），并自动启用 readline 行编辑/历史。

每轮需求结束自动播报本轮 token 消耗（prompt + completion + 调用次数）与会话累计；`/history` 可查看会话累计。单次调用明细（usage/耗时/全文）见 `BLUE_LOG_LLM=1` 的 LLM 日志。

会话状态持久化到 `~/.blue/checkpoints.sqlite`；`--resume` 或交互中的 `/resume` 可恢复历史 thread。

## 扩展：step 回调

节点输出通过回调链处理（借鉴 smolagents），默认回调是 CLI 打印。挂自定义 UI/日志：

```python
import agent
agent.register_step_callback(lambda node, output: my_ui.render(node, output))
```

回调异常不阻断主流程；`agent.clear_step_callbacks()` 可清空。

## 日志

两层文件日志，都在 `~/.blue/logs/`（本地，不入库）：

| 日志 | 开启方式 | 内容 |
|---|---|---|
| `blue-<date>.log` | 默认开启（CLI 启动时挂载） | 节点事件摘要：verdict、暂存清单、评审轮数、异常堆栈 |
| `blue-llm-<date>.log` | `BLUE_LOG_LLM=1`（环境变量或 .env） | 每次 LLM 调用全文：caller、请求摘要、响应全文、finish_reason、token usage、耗时 |

benchmark 的完整输出另存 `benchmarks/quixbugs/results/run-<ts>/<algo>.agent.log|.pytest.log`（json 里只存尾部摘要+路径）。
