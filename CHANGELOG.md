# Changelog（Bluecode 小蓝）

已完成版本的变更记录（原 README Roadmap 表整理而来，版本倒序，最新在前）。
未实现的计划与 backlog 见 [TODO.md](TODO.md)。

## v0.8.4 — blue web 终端 REPL 模式

- `blue web` 的终端不再是服务端日志镜像，而是与 Web 页面**平级的交互控制台**：uvicorn 转后台线程（log 静音），主线程跑客户端 REPL（`webclient.run_client`），与页面同引擎单写者、REST/SSE 实时同步——都能发需求、都能看节点播报、都能审批。
- `/quit` 或 Ctrl-D 退出并关停服务；`--no-repl` 或 stdin 非 TTY 回退纯服务模式；交互循环补 Ctrl-C 优雅退出。
- `agent.QUIET_CONSOLE` 静音服务端镜像打印（banner/token 播报/auto_allow/中断提示），防与 REPL 双份输出。
- Web 冒烟增至 16 场景（`test_web_repl_mode`）。

## v0.8.3 — Web 交付报告 Markdown 渲染

- 报告从纯文本升级为轻量 Markdown：**手写 DOM 渲染器**（零第三方 JS，只 createElement + textContent，绝不 innerHTML 模型内容——XSS 安全），支持标题/粗体/斜体/行内代码/围栏代码块/列表/引用/表格/分隔线/链接（http/https 白名单，`javascript:` 降级纯文本）。

## v0.8.2 — Web 观测增强（M3+M4）

- 右侧观测面板（可折叠）：手绘 SVG 图小地图（节点状态机：frontier active/已过 done/guard 待批闪烁/revise 回边 flash/parallel_tasks 驱动 worker chip）+ token/上下文表 + 环境面板（模型/窗口/key/权限）+ 审计尾部模态（`/api/audit`）。
- M4 模型切换器：`GET/POST /api/models`，薄包 models 注册表，语义与 CLI `/model` 一致。

## v0.8.1 — BugsInPy pilot

- 真实仓库 bug 基准 pilot（httpie/2 自验+E2E）：每题独立 venv（pyenv 按 python_version 解析旧解释器，缓存 `~/.blue/bip-envs`）+ 坏钉放宽（arm64 macOS 编译失败）+ **精确 F2P/P2P**（buggy 失败 ∧ fixed 通过，防元数据漂移误收）+ `--self-test` 加题门槛。

## v0.8.x — CLI/Web 双向同步

- **M1 跨进程执行锁**（`session.py`）：CLI 与 Web 两进程直连同一 checkpoints.sqlite 的并发写竞争，用 `exec_locks` 表串行化（thread_id 主键 + holder/pid/heartbeat_at，90s 心跳过期自动接管崩溃残留，fail-closed 抛 `SessionBusyError`）；server 四端点（messages/approvals/undo/retry）先 peek 再执行，被占时 409 `session_busy` 报持有方。
- **M2 CLI 客户端模式**（`webclient.py`）：交互式 `blue` 探测到本机 Web 引擎自动转客户端——执行权归一 Web 引擎进程（单写者强一致），CLI 只做输入（POST）/事件（SSE 行协议，断线重连）/审批桥（y/n/m/d → POST approvals）；命令 `/sessions` `/use N` `/new` `/history` `/context` `/undo` `/retry` `/status`。

## v0.8 — Web 控制台核心（M0–M2）

- `blue web` 子命令 + FastAPI/SSE 骨架：6 种事件（round_start/node/approval_required/round_end/error/info）、环形缓冲（Last-Event-ID 重放）、web_drain 审批桥、会话管理 REST（快照/undo/retry/重启重建审批卡/context）。
- 安全（fail-closed）：默认仅本机、非 loopback 强制 token（缺省自动生成随机 token 打印，Jupyter 式）、无 auto-approve 端点、未知 approval_id 404、无决策默认 reject、POST 仅收 JSON（否则 415）、不设 CORS。

## v0.7.3 — benchmark 双判据

- FAIL_TO_PASS / PASS_TO_PASS 拆分（报告区分「未修复」与「修坏回归」）；correct/buggy 基线自验；runner 续航加固（`--retry-failed` 重跑失败题、`--ensure-baseline` 基线自愈、agent 超时分 partial）。

## v0.7.2 — 多模型管理

- `~/.blue/models.toml` 注册表 + `/model` 会话内切换（清缓存即时生效）+ 上下文占用百分比（轮末 token 播报与 `/history` 同源同公式）。

## v0.7.1 — 模块拆分

- `agent.py` 拆为 facade + session/cli/doctor 三个子模块（显式重导出，测试 patch 目标零改动）；`_safe_os` 沙箱收口（拦 system/popen/exec*/environ/chdir）；guard 异常兜底。

## v0.7 阶段二 — 产品化分发

- `pyproject.toml` + console_scripts `blue`（pipx 安装，`'.[web]'` 含 Web 依赖）+ `blue init` 交互引导（key getpass 不回显、覆盖前备份）+ `blue doctor` 六项自检（Python/依赖/配置/目录/API+模型/tool calling，退出码 0/1）+ 配置三层（环境变量 > 项目 .env > 全局 ~/.blue/.env）。

## v0.7 阶段一 — 断点续跑与权限分级

- `/retry` 断点续跑：进程内崩溃 / 进程重启后（`--resume`）/ 死在审批点三场景统一；上一轮已正常结束则空操作。
- LLM 瞬时错误自动退避重试（429/5xx/408/409 等，2s→8s→20s+抖动，非瞬时错误直接抛）。
- `.blue.toml` 权限分级：两层 TOML 逐键合并（项目覆盖全局），write/command/python 三键 allow/ask/deny 三档，缺省 ask、非法值回落 ask（fail-closed）。

## v0.6 — 审批与信任

- diff 渲染（rich/pygments）、逐条审批（`1,3` 序号选批）、`/undo` 文件快照回退、审计日志 `audit.jsonl`、CI 退出码、成本播报。

## v0.5.x — 多文件并行与加固

- **v0.5**：多文件并行修改（`Send` 扇出 worker + `_resettable_add` reducer 聚合 + 一次性审批）。
- **v0.5.x**：日志两层（节点日志 + LLM 全文日志）、token 追踪、消息滑动窗口压缩、安全加固（subprocess 移出 import 白名单、getattr/setattr dunder 拦截、cwd 越界双路径）、CLI 颜色、`executed_changes` 留存、审批预览/详情、工具易用性。

## v0.4.5 — reviewer fail-closed 与判分管线

- reviewer 判定 fail-closed（模型不按格式输出时默认 revise，曾 fail-open 导致 bug 未修就交付）+ 判分管线修复 + benchmark 扩至 40 题（+9 图/链表题）+ runner 并行。

## v0.4 — QuixBugs benchmark

- QuixBugs 修 bug 判分基准（31 题）+ `--auto-approve` 无人值守模式（guard 自动审批）。

## v0.3 — smolagents 借鉴

- `final_answer` 显式终止 + 命令白名单 + `plan_run_python` 受限沙箱（ast 检查 + 受限 builtins）+ step 回调注册机制。

## v0.2.5 — verifier 与 reviewer 收敛

- reviewer 风格收敛（去毒舌人设，直接专业给结论）+ planner 条件化（短需求跳过模型调用）+ verifier 自动验证节点（py_compile + 自动发现跑 pytest）。

## v0.2 — 持久化与多轮会话

- `SqliteSaver` 持久化（`~/.blue/checkpoints.sqlite`）+ 多轮会话 + 斜杠命令 + `--resume`。

## v0.1 — 首个全流程

- 单文件跑通 plan→act→guard→review 全流程 + CLI 交互。
