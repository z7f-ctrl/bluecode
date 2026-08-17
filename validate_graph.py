"""离线功能验证：用 fake model 跑完整图，验证 interrupt/resume 与条件路由。

注意：离线测试使用 MemorySaver + 临时 sqlite 文件，避免污染真实 ~/.blue/checkpoints.sqlite。
"""
import json
import os
import tempfile
from unittest.mock import patch

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

# 在 import agent 之前把 DB_PATH 指到临时文件，避免测试写入用户真实数据库
_TMP_DB = tempfile.NamedTemporaryFile(prefix="blue-test-", suffix=".sqlite", delete=False)
_TMP_DB.close()
os.environ.setdefault("BLUE_TEST_DB", _TMP_DB.name)
_TMP_BACKUPS = tempfile.mkdtemp(prefix="blue-test-backups-")

import agent
import session as _session_mod
agent.DB_PATH = os.environ["BLUE_TEST_DB"]
# _get_conn/_save_session_meta 读 session 模块自己的全局 DB_PATH，
# agent.DB_PATH 只改重导出的引用——不 patch session 侧会写真实 ~/.blue
# （普通终端下可写没暴露，受限环境直接 OperationalError）
_session_mod.DB_PATH = os.environ["BLUE_TEST_DB"]
agent.BACKUP_ROOT = _TMP_BACKUPS  # 快照/undo 同样隔离，不污染真实 ~/.blue/backups
# run_round 异常路径直接调 _node_logger()（不经回调注册），节点日志目录同样隔离
agent.LOG_DIR = os.path.join(_TMP_BACKUPS, "logs")


class FakeModel:
    """按调用次序返回脚本化响应。sequence: list[str]，每次 invoke 弹一个。"""
    def __init__(self, sequence):
        self.seq = list(sequence)
        self.calls = 0

    def invoke(self, messages, *a, **k):
        item = self.seq.pop(0) if self.seq else "（耗尽）fallback"
        self.calls += 1
        if isinstance(item, Exception):
            raise item  # 脚本化异常：测自动重试与 /retry 断点续跑
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=item)


class RecordingFake(FakeModel):
    """FakeModel + 记录每次 invoke 收到的全部消息文本（断言 prompt 内容用）。"""
    def __init__(self, sequence):
        super().__init__(sequence)
        self.inputs: list[str] = []

    def invoke(self, messages, *a, **k):
        self.inputs.append("\n".join(str(m.content) for m in messages))
        return super().invoke(messages, *a, **k)


def run_on(model_fake, request, resume_action=None, thread_id="test-1"):
    """跑一个 thread，遇 interrupt 用 resume_action 续命。返回 (node顺序, 最终state)。

    注意：patch 掉 should_skip_planner，保证 planner 总是消耗一次模型调用，
    使 fake model 的脚本化响应序列与各节点一一对应。skip 路径由专门测试覆盖。
    """
    graph = agent.build_graph(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = agent.initial_state(request)
    state["thread_id"] = thread_id
    order = []
    with patch("agent._make_model", lambda: model_fake), \
         patch("agent._make_plain_model", lambda: model_fake), \
         patch("agent.should_skip_planner", lambda r: False):
        # 首轮
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for n, _o in chunk.items():
                order.append(n)
        # 有 interrupt 就 resume
        while True:
            cur = graph.get_state(config)
            if not cur.next:
                break
            for task in cur.tasks:
                if task.interrupts:
                    val = resume_action or {"action": "approve"}
                    for chunk in graph.stream(agent.Command(resume=val), config=config, stream_mode="updates"):
                        for n, _o in chunk.items():
                            order.append(n)
    return order, graph.get_state(config).values


def test_readonly_no_interrupt():
    # 只读：grep 一次即给最终答复 → 不应出现 guard 审批（pending 为空）
    calls = [
        AIMessage(content='["读文件", "报告"]'),                      # planner
        AIMessage(content="先用 grep 查一下", tool_calls=[
            {"name": "grep", "args": {"pattern": "你好"}, "id": "t1"}]),
        AIMessage(content="统计完成（2 处匹配）。最终答复：完成。"),      # agent 收尾
        AIMessage(content="verdict: pass\nfeedback: 文件很小，改起来没危险。"),  # reviewer
        AIMessage(content="# 报告\n只读任务已统计。"),                   # report
    ]
    order, state = run_on(FakeModel(calls), "统计 hello.py 里的中文词")
    print("read-only node order:", "→".join(order))
    assert "guard" in order, "guard 应经过"
    assert state["pending_changes"] == [], "只读任务不应有 pending_changes"
    assert state["verdict"] == "pass", f"应为 pass，实际 {state['verdict']}"
    print("PASS read-only\n")


def test_write_requires_resume():
    # 写文件：plan_write_file → interrupt → 用户 approve 才真执行
    # 注意：guard 审批通过会真的写文件，所以目标固定为一次性 scratch 文件并在最后清理。
    req = "把一段话写入 scratch.txt"
    scratch = "__scratch_write_target__.txt"
    try:
        calls = [
            AIMessage(content='["写 scratch.txt"]'),
            AIMessage(content="计划。", tool_calls=[
                {"name": "plan_write_file", "args": {"path": scratch, "content": "小蓝写的测试内容\n"}, "id": "w1"}]),
            AIMessage(content="已按审批写入，答复：完成。"),
            AIMessage(content="verdict: pass\nfeedback: 简单改动，放行。"),
            AIMessage(content="# 报告\n改动完成。"),
        ]
        order, state = run_on(FakeModel(calls), req, resume_action={"action": "approve"})
        print("write node order:", "→".join(order))
        assert state["verdict"] == "pass"
        assert state["pending_changes"] == [], "审批后应清空 pending_changes"
        assert open(scratch, encoding="utf-8").read() == "小蓝写的测试内容\n", "审批通过后应真的写入该文件"
        print("PASS write+resume：审批通过后才真正落盘 ✔")
    finally:
        import os
        if os.path.exists(scratch):
            os.remove(scratch)
    print()


def test_reject_no_write():
    # 拒绝：interrupt 后 reject → 不应写文件，reviewer 应 pass（尊重用户）
    req = "往新建 bye.py 写内容"
    calls = [
        AIMessage(content='["写 bye.py"]'),
        AIMessage(content="计划。", tool_calls=[
            {"name": "plan_write_file", "args": {"path": "bye.py", "content": "x"}, "id": "w1"}]),
        AIMessage(content="用户拒绝？回复。"),   # 实际 rejected 后 reviewer pass，不回 agent
        AIMessage(content="verdict: pass\nfeedback: 用户拒绝，不改。"),
        AIMessage(content="# 报告\n未改动。"),
    ]
    order, state = run_on(FakeModel(calls), req, resume_action={"action": "reject", "note": "不需要"})
    print("reject node order:", "→".join(order))
    assert state["verdict"] == "pass"
    assert state["pending_changes"] == []
    print("PASS reject (未写文件)\n")


def test_revise_loops_back():
    # 评审打回 → 条件边 revise→agent → 模型重改 → 二评 pass → report
    req = "把 hello.py 第 3 行的变量改名"
    scratch = "__scratch_rv__.txt"
    try:
        calls = [
            AIMessage(content='["改 hello.py 变量名"]'),
            AIMessage(content="先读。", tool_calls=[
                {"name": "read_file", "args": {"path": "hello.py"}, "id": "r1"}]),
            AIMessage(content="准备写。", tool_calls=[
                {"name": "plan_write_file", "args": {"path": scratch, "content": "v1\n"}, "id": "w1"}]),
            # review → revise（打回）
            AIMessage(content="verdict: revise\nfeedback: 🚨 变量名用了个魔鬼数字，回去改。"),
            AIMessage(content="已按意见重写。答复：完成。"),
            # 二评 → pass
            AIMessage(content="verdict: pass\nfeedback: 这版行了。"),
            AIMessage(content="# 报告\n两次评审后完成。"),
        ]
        order, state = run_on(FakeModel(calls), req, resume_action={"action": "approve"})
        print("revise node order:", "→".join(order))
        assert "reviewer" in order
        assert order.count("agent") >= 2, "应出现两次 agent（第一次 + 打回后重来）"
        assert state["verdict"] == "pass"
        assert state["review_rounds"] >= 2
        print("PASS revise→agent 回边 ✔  (评审轮数=%d)" % state["review_rounds"])
    finally:
        import os
        if os.path.exists(scratch):
            os.remove(scratch)
    print()


def test_run_command_cwd_sandbox():
    # 方案B：cwd 必须受沙箱约束。两条路径都要拦截越界 cwd。
    from tools import execute_change

    # ① 最后防线：execute_change 直接拒绝越界 cwd（不得真执行）
    res = execute_change({"action": "plan_run_command", "command": "pwd", "cwd": "../../../../etc"})
    print("execute_change cwd 越界结果:", res)
    assert "执行失败" in res and "越界" in res, f"越界 cwd 应被拒，实际：{res}"
    print("PASS cwd sandbox（execute_change 最后防线）✔\n")

    # ② 提前拦截：agent 暂存阶段就把越界 cwd 挡下，不进 pending_changes
    bad_calls = [
        AIMessage(content='["跑命令"]'),
        AIMessage(content="跑一下。", tool_calls=[
            {"name": "plan_run_command",
             "args": {"command": "ls", "cwd": "../../../../etc"}, "id": "c1"}]),
        AIMessage(content="命令被拦截了？给用户答复。"),   # 不再产出新工具调用
    ]
    order, state = run_on(FakeModel(bad_calls), "在 /etc 下跑 ls", resume_action={"action": "approve"})
    print("cwd-blocked node order:", "→".join(order))
    assert state["pending_changes"] == [], "越界 cwd 不应进入待审批列表"
    assert state["verdict"] != "approved"
    print("PASS cwd sandbox（agent 暂存提前拦截）✔\n")


def test_multi_round_same_thread():
    """多轮会话：同一 thread_id 跨两次 build_graph（模拟进程重启），checkpoint 仍在。"""
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    thread = "multi-round-test"
    db = os.environ["BLUE_TEST_DB"]

    def fresh_graph():
        conn = sqlite3.connect(db, check_same_thread=False)
        return agent.build_graph(checkpointer=SqliteSaver(conn))

    def run_with(graph, model_fake, request, resume_action=None):
        config: RunnableConfig = {"configurable": {"thread_id": thread}}
        state = agent.initial_state(request)
        order = []
        with patch("agent._make_model", lambda: model_fake), \
             patch("agent._make_plain_model", lambda: model_fake), \
             patch("agent.should_skip_planner", lambda r: False):
            for chunk in graph.stream(state, config=config, stream_mode="updates"):
                for n, _o in chunk.items():
                    order.append(n)
            while True:
                cur = graph.get_state(config)
                if not cur.next:
                    break
                for task in cur.tasks:
                    if task.interrupts:
                        val = resume_action or {"action": "approve"}
                        for chunk in graph.stream(agent.Command(resume=val), config=config, stream_mode="updates"):
                            for n, _o in chunk.items():
                                order.append(n)
        return order, graph.get_state(config).values

    # 第一轮："进程 1"：只读任务
    calls1 = [
        AIMessage(content='["读文件"]'),
        AIMessage(content="用 grep 查一下", tool_calls=[
            {"name": "grep", "args": {"pattern": "hello"}, "id": "g1"}]),
        AIMessage(content="第一轮完成。"),
        AIMessage(content="verdict: pass\nfeedback: 只读无风险。"),
        AIMessage(content="# 报告\n第一轮结束。"),
    ]
    g1 = fresh_graph()
    order1, state1 = run_with(g1, FakeModel(calls1), "第一轮：统计 hello")
    print("round1 order:", "→".join(order1))
    assert state1["verdict"] == "pass"

    # 第二轮："进程 2"：新建 graph（模拟重启），同 thread 继续
    calls2 = [
        AIMessage(content='["写文件"]'),
        AIMessage(content="计划写文件。", tool_calls=[
            {"name": "plan_write_file", "args": {"path": "__mr__.txt", "content": "第二轮\n"}, "id": "w1"}]),
        AIMessage(content="已写入。"),
        AIMessage(content="verdict: pass\nfeedback: 简单。"),
        AIMessage(content="# 报告\n第二轮结束。"),
    ]
    g2 = fresh_graph()
    try:
        order2, state2 = run_with(g2, FakeModel(calls2), "第二轮：写文件", resume_action={"action": "approve"})
        print("round2 order:", "→".join(order2))
        assert state2["verdict"] == "pass"
        assert state2["pending_changes"] == []
        assert os.path.exists("__mr__.txt"), "审批后应真的写入文件"
    finally:
        if os.path.exists("__mr__.txt"):
            os.remove("__mr__.txt")

    # sqlite 里应真有该 thread 的 checkpoint 记录
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread,)
        ).fetchone()
        assert row and row[0] > 0, "sqlite 中应存在该 thread 的 checkpoint"
    finally:
        conn.close()
    print("PASS multi-round same thread（跨 graph 实例，sqlite 持久化生效）✔\n")


def test_session_meta_persistence():
    """会话元信息应写入 sessions 辅助表。"""
    sess = agent.Session()
    sess.round = 3
    agent._save_session_meta(sess)
    sessions = agent.list_sessions()
    found = [s for s in sessions if s["thread_id"] == sess.thread_id]
    assert found, f"sessions 表应包含 {sess.thread_id}"
    assert found[0]["rounds"] == 3
    print("PASS session meta persistence ✔\n")


def test_revise_compresses_messages():
    """revise 回环时，reviewer 应把 messages 压缩成单条摘要。"""
    req = "改 hello.py 变量名"
    scratch = "__scratch_rv_compress__.txt"
    calls = [
        AIMessage(content='["改 hello.py"]'),
        AIMessage(content="先读。", tool_calls=[
            {"name": "read_file", "args": {"path": "hello.py"}, "id": "r1"}]),
        AIMessage(content="准备写。", tool_calls=[
            {"name": "plan_write_file", "args": {"path": scratch, "content": "v1\n"}, "id": "w1"}]),
        AIMessage(content="verdict: revise\nfeedback: 变量名太烂，回去改。"),
        AIMessage(content="已按意见重写。"),
        AIMessage(content="verdict: pass\nfeedback: 这版行了。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    try:
        order, state = run_on(FakeModel(calls), req, resume_action={"action": "approve"})
        print("revise-compress order:", "→".join(order))
        assert state["verdict"] == "pass"
        # revise 回环后 messages 应被压缩成 1 条摘要
        msgs = state.get("messages", [])
        assert len(msgs) <= 2, f"revise 回环后 messages 应被压缩，实际 {len(msgs)} 条"
        if msgs:
            first = msgs[0].content if hasattr(msgs[0], "content") else str(msgs[0])
            assert "【上一轮执行摘要】" in first or "摘要" in first, f"首条消息应为压缩摘要，实际：{first[:100]}"
        print("PASS revise message compression ✔\n")
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def test_skip_planner_for_simple_request():
    """简单需求（短、无多步骤词）应跳过 planner 的模型调用，直接单步计划。"""
    assert agent.should_skip_planner("统计行数") is True
    assert agent.should_skip_planner("读一下 hello.py") is True
    assert agent.should_skip_planner("修复 bug 并添加测试") is False  # 含"并"
    assert agent.should_skip_planner("这是一个非常长的需求描述，超过三十个字 therefore 需要 planner 拆解") is False

    # skip 时 planner 不消耗模型调用：fake model 第一个响应应被 agent 拿到
    calls = [
        AIMessage(content="直接用 grep 查", tool_calls=[
            {"name": "grep", "args": {"pattern": "hello"}, "id": "g1"}]),
        AIMessage(content="统计完成。"),
        AIMessage(content="verdict: pass\nfeedback: 只读。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    graph = agent.build_graph(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "skip-test"}}
    state = agent.initial_state("统计 hello 出现次数")
    order = []
    fake = FakeModel(calls)  # 两个 patch 必须共享同一实例，否则各持 seq 副本导致序列错位
    with patch("agent._make_model", lambda: fake), \
         patch("agent._make_plain_model", lambda: fake):
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for n, _o in chunk.items():
                order.append(n)
        while True:
            cur = graph.get_state(config)
            if not cur.next:
                break
            for task in cur.tasks:
                if task.interrupts:
                    for chunk in graph.stream(agent.Command(resume={"action": "approve"}), config=config, stream_mode="updates"):
                        for n, _o in chunk.items():
                            order.append(n)
    final = graph.get_state(config).values
    print("skip-planner order:", "→".join(order))
    assert final["plan"] == ["统计 hello 出现次数"], "skip 时应退化为单步计划"
    assert final["verdict"] == "pass"
    print("PASS skip planner for simple request ✔\n")


def test_parallel_workers_fanout():
    """planner 产出 parallel_tasks ≥2 → Send 扇出并行 worker，改动经 reducer 聚合到 guard 一次审批。"""
    alpha, beta = "__scratch_alpha__.txt", "__scratch_beta__.txt"

    class FakeByContent:
        """并行 worker 的模型调用次序不确定，按消息内容路由脚本化响应。"""
        def invoke(self, messages, *a, **k):
            sys_prompt = str(messages[0].content) if messages else ""
            if "计划节点" in sys_prompt:
                return AIMessage(content=json.dumps({
                    "steps": ["并行写两个文件"],
                    "parallel_tasks": [f"写入 {alpha}", f"写入 {beta}"],
                }))
            if "评审节点" in sys_prompt:
                return AIMessage(content="verdict: pass\nfeedback: 并行改动没问题。")
            if "收尾节点" in sys_prompt:
                return AIMessage(content="# 报告\n两个文件并行完成。")
            joined = "\n".join(str(m.content) for m in messages)
            # 必须按"你负责的子任务："行路由——总需求里同时提及两个文件，直接全文匹配会撞车
            sub_line = next((l for l in joined.split("\n") if "你负责的子任务：" in l), "")
            if alpha in sub_line:
                return AIMessage(content="暂存 alpha", tool_calls=[
                    {"name": "plan_write_file", "args": {"path": alpha, "content": "alpha 内容\n"}, "id": "wa"}])
            if beta in sub_line:
                return AIMessage(content="暂存 beta", tool_calls=[
                    {"name": "plan_write_file", "args": {"path": beta, "content": "beta 内容\n"}, "id": "wb"}])
            raise AssertionError("FakeByContent 未匹配到任何分支")

    try:
        order, state = run_on(FakeByContent(), f"并行写入 {alpha} 和 {beta}",
                              resume_action={"action": "approve"})
        print("parallel node order:", "→".join(order))
        assert "worker" in order, "应扇出到并行 worker"
        assert "agent" not in order, "并行路径不应走串行 agent"
        assert state["verdict"] == "pass"
        assert state["pending_changes"] == [], "审批后应清空 pending_changes"
        assert len(state["worker_notes"]) == 2, f"两个 worker 的 notes 应聚合，实际 {state['worker_notes']}"
        assert open(alpha, encoding="utf-8").read() == "alpha 内容\n", "worker A 的改动应落盘"
        assert open(beta, encoding="utf-8").read() == "beta 内容\n", "worker B 的改动应落盘"
        print("PASS parallel workers fanout（Send 扇出 + reducer 聚合 + 一次审批）✔\n")
    finally:
        for f in (alpha, beta):
            if os.path.exists(f):
                os.remove(f)


def test_security_hardening():
    """纵深防御加固：命令复合符 / subprocess import / getattr dunder 逃逸。"""
    from tools import _execute_python, check_command_safety, check_python_safety

    # 1. 复合命令/命令替换全拦（此前 ls && rm x 能过：黑名单只查整串关键词）
    for bad in ["ls && rm important.py", "cat a.txt; rm b.py", "echo $(whoami)", "cat `id`"]:
        try:
            check_command_safety(bad)
            raise AssertionError(f"应拦截但未拦：{bad}")
        except ValueError:
            pass
    check_command_safety("python3 -m pytest test_x.py -v")  # 正常单命令不受影响

    # 2. subprocess 移出 import 白名单（此前 subprocess.run 绕过命令校验）
    try:
        check_python_safety("import subprocess")
        raise AssertionError("应拦截 import subprocess")
    except ValueError:
        pass

    # 3. getattr/setattr 的 dunder 字符串参数运行时被拦（ast 只看属性节点，看不到字符串）
    r = _execute_python("print(getattr(str, '__class__'))")
    assert "执行失败" in r and "dunder" in r, f"getattr dunder 应被运行时拦截，实际：{r}"
    r = _execute_python("setattr(str, '__x__', 1)")
    assert "执行失败" in r and "dunder" in r, f"setattr dunder 应被运行时拦截，实际：{r}"
    # 正常 getattr 不受影响
    r = _execute_python("print(getattr('abc', 'upper')())")
    assert "ABC" in r, f"正常 getattr 应可用，实际：{r}"

    # 4. 沙箱 os 代理收口：system/popen/environ/chdir 均不可达（chdir 会把相对路径
    #    操作偏出 WORKDIR 沙箱）；os.path 等安全子集不受影响
    r = _execute_python("import os\nos.system('echo x')")
    assert "执行失败" in r, f"os.system 应被代理拦截，实际：{r}"
    r = _execute_python("import os\nos.chdir('/tmp')")
    assert "执行失败" in r, f"os.chdir 应被代理拦截，实际：{r}"
    r = _execute_python("import os\nprint(type(os.environ))")
    assert "执行失败" in r, f"os.environ 应被代理拦截，实际：{r}"
    r = _execute_python("import os\nprint(os.path.join('a', 'b'))")
    assert "a/b" in r or "a\\b" in r, f"os.path 应可用，实际：{r}"
    print("PASS security hardening（&& ; $() 反引号 / subprocess / getattr+setattr dunder / os 代理收口）✔\n")


def test_tool_usability():
    """grep 长行截断 + plan_patch occurrence 指定第 N 处替换。"""
    from tools import execute_change, grep

    scratch = "__scratch_usability__.txt"
    try:
        # 1. grep：单行截断到 200 字符 + …（防 minified 长行撑爆上下文）
        with open(scratch, "w", encoding="utf-8") as f:
            f.write("short match\n" + "x" * 500 + " match tail\n")
        out = grep.invoke({"pattern": "match", "path": scratch})
        for line in out.split("\n"):
            content = line.split(":", 2)[-1]
            assert len(content) <= 201, f"grep 结果行应 ≤201 字符（200+…），实际 {len(content)}"
        assert "…" in out, "长行应有 … 后缀"

        # 2. plan_patch occurrence=2：只替换第 2 处
        with open(scratch, "w", encoding="utf-8") as f:
            f.write("foo A\nfoo B\nfoo C\n")
        r = execute_change({"action": "plan_patch", "path": scratch,
                            "old": "foo", "new": "bar", "occurrence": 2})
        assert "已补丁" in r and "第 2/3 处" in r, f"occurrence=2 应成功：{r}"
        with open(scratch, encoding="utf-8") as f:
            assert f.read() == "foo A\nbar B\nfoo C\n", "应只替换第 2 处"

        # 3. occurrence 超界报错
        r = execute_change({"action": "plan_patch", "path": scratch,
                            "old": "foo", "new": "bar", "occurrence": 5})
        assert "失败" in r and "只出现 2 次" in r, f"occurrence=5 超界应失败：{r}"

        # 4. 默认（occurrence=0）多处仍拒绝，且报错文案指引 occurrence 用法
        r = execute_change({"action": "plan_patch", "path": scratch, "old": "foo", "new": "bar"})
        assert "失败" in r and "occurrence=N" in r, f"默认多处匹配应拒绝并指引：{r}"
        print("PASS tool usability（grep 长行截断 + plan_patch occurrence）✔\n")
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def test_agent_sliding_window():
    """agent 循环内滑动窗口：长工具序列后，invoke 看到的历史被压缩且带摘要。"""
    class RecordingFake:
        """记录每次 invoke 的消息数和是否含【早前操作摘要】。"""
        def __init__(self, seq):
            self.seq = list(seq)
            self.seen: list[tuple[int, bool]] = []

        def invoke(self, messages, *a, **k):
            self.seen.append((
                len(messages),
                any(isinstance(m, HumanMessage) and "【早前操作摘要】" in str(m.content) for m in messages),
            ))
            item = self.seq.pop(0) if self.seq else AIMessage(content="done")
            return item if isinstance(item, AIMessage) else AIMessage(content=item)

    calls = [AIMessage(content='["查"]')]  # planner
    for i in range(13):  # 13 次 grep：head=3、每轮 +2，第 12 次 invoke 前 len=25 > 3+20 触发窗口
        calls.append(AIMessage(content=f"查{i}", tool_calls=[
            {"name": "grep", "args": {"pattern": f"p{i}"}, "id": f"g{i}"}]))
    calls.append(AIMessage(content="查完了。"))                       # agent 收尾
    calls.append(AIMessage(content="verdict: pass\nfeedback: 只读 ok。"))  # reviewer
    fake = RecordingFake(calls)
    order, state = run_on(fake, "反复查很多东西")
    agent_seen = fake.seen[1:15]  # agent 的 14 次调用（13 grep + 1 收尾）
    lengths = [n for n, _ in agent_seen]
    has_summary = any(s for _, s in agent_seen[1:])  # 第 2 次调用起某轮应带摘要
    print(f"sliding-window：agent invoke 消息数 {lengths}")
    assert has_summary, "窗口压缩应产生【早前操作摘要】"
    # 稳态：head(3) + 摘要(1) + 窗口(20) ≈ 24，留余量断言上限
    assert max(lengths) <= agent.AGENT_MSG_WINDOW + 6, \
        f"滑动窗口应压住消息数（≈{agent.AGENT_MSG_WINDOW}+6），实际 {max(lengths)}"
    assert state["verdict"] == "pass"
    print("PASS agent sliding window（循环内压缩 + 摘要衔接）✔\n")


def test_report_template_saves_llm():
    """只读/拒绝场景 report 走模板，不调 LLM（省 ~2000 token/次）。

    只读路径的模型调用：planner(1) + agent(N)。reviewer 对 proceed 短路不调 LLM，
    report 模板化后也不调——整个只读任务比此前省 report 那一次调用。"""
    calls = [
        AIMessage(content='["读"]'),                                    # planner
        AIMessage(content="查一下", tool_calls=[
            {"name": "grep", "args": {"pattern": "hello"}, "id": "g1"}]),
        AIMessage(content="找到 3 处。"),                                # agent 收尾（last_ai）
        AIMessage(content="verdict: pass\nfeedback: 不应被消耗"),        # reviewer proceed 短路
        AIMessage(content="这条也不应被消耗（report 已模板化）"),
    ]
    fake = FakeModel(calls)
    order, state = run_on(fake, "统计 hello 出现次数")
    assert fake.calls == 3, f"只读任务应只调 3 次模型（planner+agent×2），实际 {fake.calls}"
    assert "交付报告" in state["feedback"], "report 应输出模板报告"
    assert "找到 3 处" in state["feedback"], "模板报告应带上 last_ai 答复"
    print("PASS report template（只读场景零 LLM 调用 + last_ai 答复保留）✔\n")


def test_worker_fault_tolerance():
    """并行 worker 单点失败不拖垮整图：失败降级为 note，兄弟 worker 成果保留。"""
    alpha, beta = "__scratch_ft_alpha__.txt", "__scratch_ft_beta__.txt"

    class FakeFlaky:
        """alpha 子任务模拟 API 异常；beta 正常。"""
        def invoke(self, messages, *a, **k):
            sys_prompt = str(messages[0].content) if messages else ""
            if "计划节点" in sys_prompt:
                return AIMessage(content=json.dumps({
                    "steps": ["并行"], "parallel_tasks": [f"写 {alpha}", f"写 {beta}"]}))
            if "评审节点" in sys_prompt:
                return AIMessage(content="verdict: pass\nfeedback: 部分完成可接受。")
            if "收尾节点" in sys_prompt:
                return AIMessage(content="# 报告\n部分完成。")
            joined = "\n".join(str(m.content) for m in messages)
            sub_line = next((l for l in joined.split("\n") if "你负责的子任务：" in l), "")
            if alpha in sub_line:
                raise ConnectionError("模拟 API 超时/限流")
            if beta in sub_line:
                return AIMessage(content="写 beta", tool_calls=[
                    {"name": "plan_write_file", "args": {"path": beta, "content": "beta ok\n"}, "id": "wb"}])
            raise AssertionError("FakeFlaky 未匹配到任何分支")

    try:
        order, state = run_on(FakeFlaky(), f"并行写 {alpha} 和 {beta}",
                              resume_action={"action": "approve"})
        assert state["verdict"] == "pass", "单 worker 失败不应拖垮整图"
        with open(beta, encoding="utf-8") as f:
            assert f.read() == "beta ok\n", "兄弟 worker 的改动应保留（不被失败方的空 pending 清空）"
        assert not os.path.exists(alpha), "失败 worker 不应有产出"
        notes = " ".join(state["worker_notes"])
        assert "子任务失败" in notes, f"失败 worker 应有失败 note：{notes}"
        print("PASS worker fault tolerance（单点失败降级 + 兄弟成果保留）✔\n")
    finally:
        for f in (alpha, beta):
            if os.path.exists(f):
                os.remove(f)


def test_token_usage_tracking():
    """token 追踪：_logged_invoke 无条件累加，run_round 轮末汇总进 Session。"""
    # _extract_usage 双来源：LangChain 标准 usage_metadata / OpenAI 原生 response_metadata
    m1 = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    m2 = AIMessage(content="x", response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert agent._extract_usage(m1) == (10, 5), f"usage_metadata 来源提取错误：{agent._extract_usage(m1)}"
    assert agent._extract_usage(m2) == (7, 3), f"response_metadata 来源提取错误：{agent._extract_usage(m2)}"

    # 只读一轮（planner + agent×2 = 3 次调用），fake 响应带 usage
    def _u(p, c):
        return {"token_usage": {"prompt_tokens": p, "completion_tokens": c}}
    calls = [
        AIMessage(content='["读"]', response_metadata=_u(100, 10)),
        AIMessage(content="查", tool_calls=[{"name": "grep", "args": {"pattern": "x"}, "id": "g1"}],
                  response_metadata=_u(200, 20)),
        AIMessage(content="查完了", response_metadata=_u(300, 30)),
    ]
    fake = FakeModel(calls)
    graph = agent.build_graph(checkpointer=MemorySaver())
    sess = agent.Session(thread_id="token-test")
    with patch("agent._make_model", lambda: fake), \
         patch("agent._make_plain_model", lambda: fake), \
         patch("agent.should_skip_planner", lambda r: False):
        agent.run_round(graph, sess, "只读需求")
    t = sess.token_usage
    assert t["calls"] == 3, f"应记 3 次调用，实际 {t}"
    assert t["prompt"] == 600 and t["completion"] == 60, f"累计错误：{t}"
    # 第二轮同一 Session：继续累加（会话级累计）。skip planner 后 agent 直接吃第一条响应
    fake2 = FakeModel([
        AIMessage(content="再查", tool_calls=[{"name": "grep", "args": {"pattern": "y"}, "id": "g2"}],
                  response_metadata=_u(50, 5)),
        AIMessage(content="收尾", response_metadata=_u(50, 5)),
    ])
    with patch("agent._make_model", lambda: fake2), \
         patch("agent._make_plain_model", lambda: fake2), \
         patch("agent.should_skip_planner", lambda r: True):  # 跳过 planner：agent×2
        agent.run_round(graph, sess, "再来一个只读")
    t = sess.token_usage
    assert t["calls"] == 5 and t["prompt"] == 700 and t["completion"] == 70, f"会话累计错误：{t}"
    print("PASS token usage tracking（双来源提取 + 轮末汇总 + Session 累计）✔\n")


def test_report_gets_executed_changes():
    """P1 修复验证：report 的改动清单与 reviewer 的 diff 来自 executed_changes。

    此前 report 传 state['pending_changes']（guard 审批后必为 []），改动清单永远为空；
    reviewer 也只看执行结果文本看不到 diff。test_report_template 只覆盖只读分支，没抓到这个。"""
    scratch = "__scratch_report_changes__.txt"
    try:
        calls = [
            AIMessage(content='["写文件"]'),                                    # planner
            AIMessage(content="写", tool_calls=[                               # agent 第 1 次：暂存后 break（只调 1 次）
                {"name": "plan_write_file", "args": {"path": scratch, "content": "独特标记 CONTENT_MARK_123\n"}, "id": "w1"}]),
            AIMessage(content="verdict: pass\nfeedback: ok。"),                  # reviewer
            AIMessage(content="# 报告\n完成。"),                                  # report
        ]
        fake = RecordingFake(calls)
        order, state = run_on(fake, f"写入 {scratch}", resume_action={"action": "approve"})
        reviewer_input = next((t for t in fake.inputs if "评审轮数" in t), "")
        report_input = next((t for t in fake.inputs if "最终改动清单" in t), "")
        assert "CONTENT_MARK_123" in reviewer_input, "reviewer 应看到实际改动内容（diff 可见性）"
        assert "CONTENT_MARK_123" in report_input, "report 的改动清单不应为空（此前永远 []）"
        assert state["executed_changes"], "executed_changes 应留存完整改动"
        assert state["changed_files"] == [scratch], f"changed_files 字段错误：{state['changed_files']}"
        print("PASS report/reviewer executed_changes（清单非空 + diff 可见 + changed_files 字段）✔\n")
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def test_audit_log():
    """审批决定写 audit.jsonl：approve 一条记录，thread/动作/改动摘要/时间齐全。"""
    scratch = "__scratch_audit__.txt"
    with tempfile.TemporaryDirectory() as td:
        audit_path = os.path.join(td, "audit.jsonl")
        calls = [
            AIMessage(content='["写"]'),
            AIMessage(content="写", tool_calls=[
                {"name": "plan_write_file", "args": {"path": scratch, "content": "audit 内容\n"}, "id": "w1"}]),
            AIMessage(content="verdict: pass\nfeedback: ok。"),
            AIMessage(content="# 报告\n完成。"),
        ]
        fake = FakeModel(calls)
        graph = agent.build_graph(checkpointer=MemorySaver())
        sess = agent.Session(thread_id="audit-test")
        try:
            with patch("agent.AUDIT_LOG", audit_path), \
                 patch("agent._make_model", lambda: fake), \
                 patch("agent._make_plain_model", lambda: fake), \
                 patch("agent.should_skip_planner", lambda r: False), \
                 patch("builtins.input", lambda *a, **k: "y"):  # 模拟审批按 y
                agent.run_round(graph, sess, f"写入 {scratch}")
            with open(audit_path, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f]
            assert len(recs) == 1, f"应恰好一条审计记录，实际 {len(recs)}"
            rec = recs[0]
            assert rec["action"] == "approve" and rec["thread"] == "audit-test", f"审计字段错误：{rec}"
            assert rec["changes"] and rec["changes"][0].get("path") == scratch, f"改动摘要错误：{rec}"
            assert rec["changes"][0].get("content_len") == len("audit 内容\n"), "大字段应转长度（jsonl 不膨胀）"
            assert rec["ts"], "应有时间戳"
            print("PASS audit log（approve 记录 + 字段齐全 + 大字段转长度）✔\n")
        finally:
            if os.path.exists(scratch):
                os.remove(scratch)


def test_selective_approval():
    """逐条审批：3 条改动选批 1,3 → 只执行 2 条，跳过记录进 reviewer 视野。"""
    a, b, c = "__scratch_sa_a__.txt", "__scratch_sa_b__.txt", "__scratch_sa_c__.txt"
    calls = [
        AIMessage(content='["写三文件"]'),
        AIMessage(content="写三个", tool_calls=[
            {"name": "plan_write_file", "args": {"path": a, "content": "A\n"}, "id": "w1"},
            {"name": "plan_write_file", "args": {"path": b, "content": "B\n"}, "id": "w2"},
            {"name": "plan_write_file", "args": {"path": c, "content": "C\n"}, "id": "w3"},
        ]),
        AIMessage(content="verdict: pass\nfeedback: 部分批准。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    fake = RecordingFake(calls)
    graph = agent.build_graph(checkpointer=MemorySaver())
    sess = agent.Session(thread_id="sa-test")
    try:
        with patch("agent._make_model", lambda: fake), \
             patch("agent._make_plain_model", lambda: fake), \
             patch("agent.should_skip_planner", lambda r: False), \
             patch("builtins.input", lambda *a, **k: "1,3"):  # 选批第 1、3 条
            agent.run_round(graph, sess, "写三个文件")
        assert os.path.exists(a) and os.path.exists(c), "选批的 1,3 应执行"
        assert not os.path.exists(b), "未选的 2 不应执行"
        reviewer_input = next((t for t in fake.inputs if "评审轮数" in t), "")
        assert "【跳过】" in reviewer_input, f"reviewer 应看到跳过记录：{reviewer_input[-200:]}"
        print("PASS selective approval（选批 1,3 + 未选不执行 + 跳过记录进 reviewer）✔\n")
    finally:
        for f in (a, b, c):
            if os.path.exists(f):
                os.remove(f)


def test_undo_snapshot_restore():
    """/undo：guard 执行前自动快照，patch 回退到改动前、新建文件被删除、单轮指针。"""
    scratch = "__scratch_undo__.txt"
    new_file = "__scratch_undo_new__.txt"
    with open(scratch, "w", encoding="utf-8") as f:
        f.write("原始内容 v1\n")
    calls = [
        AIMessage(content='["改文件"]'),
        AIMessage(content="改", tool_calls=[
            {"name": "plan_patch", "args": {"path": scratch, "old": "v1", "new": "v2"}, "id": "w1"},
            {"name": "plan_write_file", "args": {"path": new_file, "content": "新建\n"}, "id": "w2"},
        ]),
        AIMessage(content="verdict: pass\nfeedback: ok。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    fake = FakeModel(calls)
    graph = agent.build_graph(checkpointer=MemorySaver())
    sess = agent.Session(thread_id="undo-test")
    try:
        with patch("agent._make_model", lambda: fake), \
             patch("agent._make_plain_model", lambda: fake), \
             patch("agent.should_skip_planner", lambda r: False), \
             patch("builtins.input", lambda *a, **k: "y"):
            agent.run_round(graph, sess, "改两个文件")
        # 改动已生效
        with open(scratch, encoding="utf-8") as f:
            assert f.read() == "原始内容 v2\n", "patch 应已生效"
        assert os.path.exists(new_file), "新文件应已创建"
        # undo：patch 回退 + 新建删除
        report = agent._undo_latest(sess.thread_id)
        with open(scratch, encoding="utf-8") as f:
            assert f.read() == "原始内容 v1\n", f"patch 应被回退：{report}"
        assert not os.path.exists(new_file), f"新建文件应被删除：{report}"
        # 再 undo 一次：单轮指针已消费
        assert "没有可回退" in agent._undo_latest(sess.thread_id), "latest 指针应只撤一轮"
        print("PASS undo（快照回退 patch + 删除新建 + 单轮指针）✔\n")
    finally:
        for f in (scratch, new_file):
            if os.path.exists(f):
                os.remove(f)


def test_llm_auto_retry():
    """_logged_invoke 自动重试：瞬时错误退避重试 2 次后成功；非瞬时错误直接抛。"""
    class RateLimitError(Exception):  # 类型名命中 _TRANSIENT_EXC_NAMES 兜底匹配
        pass

    flaky = FakeModel([RateLimitError("限流"), RateLimitError("再限"), "最终成功"])
    with patch("agent.time.sleep", lambda s: None), \
         patch("agent.random.uniform", lambda a, b: 0.0):  # 不等真退避
        resp = agent._logged_invoke(flaky, [], "test")
    assert resp.content == "最终成功", f"重试后应成功，实际 {resp.content!r}"
    assert flaky.calls == 3, f"应调用 3 次（2 败 1 成），实际 {flaky.calls}"

    hard = FakeModel([ValueError("bad request")])  # 非瞬时：不重试直接抛
    with patch("agent.time.sleep", lambda s: None):
        try:
            agent._logged_invoke(hard, [], "test")
            raise AssertionError("非瞬时错误应直接抛出")
        except ValueError:
            pass
    assert hard.calls == 1, f"非瞬时错误不应重试，实际调用 {hard.calls} 次"
    print("PASS llm auto retry（瞬时重试 2 次成功 + 非瞬时直接抛）✔\n")


def test_resume_pending_after_crash():
    """/retry 断点续跑：planner 崩溃 → run_round 中断不写文件 → resume_pending 续跑完成；
    已正常结束的轮 resume_pending 空操作返回 False。"""
    scratch = "__scratch_retry__.txt"
    calls = [
        RuntimeError("boom"),                      # planner 首调即崩（非瞬时，不重试直接抛）
        AIMessage(content='["写 scratch"]'),        # 续跑后 planner 重跑
        AIMessage(content="写", tool_calls=[
            {"name": "plan_write_file", "args": {"path": scratch, "content": "断点续跑内容\n"}, "id": "w1"}]),
        # agent 暂存写改动后立即 break 进审批，无收尾调用；下一个是 reviewer
        AIMessage(content="verdict: pass\nfeedback: ok。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    fake = FakeModel(calls)
    graph = agent.build_graph(checkpointer=MemorySaver())
    sess = agent.Session(thread_id="retry-test")
    try:
        with patch("agent._make_model", lambda: fake), \
             patch("agent._make_plain_model", lambda: fake), \
             patch("agent.should_skip_planner", lambda r: False), \
             patch("builtins.input", lambda *a, **k: "y"):
            agent.run_round(graph, sess, f"写入 {scratch}")  # planner 崩溃，本轮中断
            assert not os.path.exists(scratch), "崩溃中断不应写文件"
            cur = graph.get_state(sess.config)
            assert cur.next, "崩溃后 checkpoint 应有待执行节点"
            assert agent.resume_pending(graph, sess) is True, "有断点应续跑"
        with open(scratch, encoding="utf-8") as f:
            assert f.read() == "断点续跑内容\n", "续跑后应完成写入"
        assert agent.resume_pending(graph, sess) is False, "已正常结束应空操作（绝不默默重跑）"
        print("PASS /retry resume（崩溃续跑完成 + 已结束空操作）✔\n")
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def test_permission_allow_skips_interrupt():
    """.blue.toml write=allow：纯写批次跳过 interrupt 直接执行 + 审计记 auto_allow；
    混合批次（write=allow + command=ask）仍走人工审批。"""
    import tools
    scratch = "__scratch_allow__.txt"
    cmd_scratch = "__scratch_allow_cmd__.txt"
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "config.toml")
        audit_path = os.path.join(td, "audit.jsonl")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('[permissions]\nwrite = "allow"\n')
        interrupt_calls = []
        real_interrupt = agent.interrupt
        def spy_interrupt(payload):
            interrupt_calls.append(payload)
            return real_interrupt(payload)
        # run_on 内部已 patch 双模型 + should_skip_planner，这里只补配置/审计/intercept 间谍
        def cfg_patches():
            return (
                patch("tools.GLOBAL_CONFIG_PATH", cfg),
                patch("tools.PROJECT_CONFIG_NAME", ".blue-test-nonexistent.toml"),
                patch("agent.AUDIT_LOG", audit_path),
                patch("agent.interrupt", spy_interrupt),
            )
        try:
            # ① 纯写批次：不 interrupt，直接执行 + 审计 auto_allow
            # 注意：agent 暂存写改动后立即 break 进审批（agent.py 的 pending break），
            # 不会再调一次模型要收尾文本——序列里没有 agent 收尾项
            calls = [
                AIMessage(content='["写"]'),
                AIMessage(content="写", tool_calls=[
                    {"name": "plan_write_file", "args": {"path": scratch, "content": "放行内容\n"}, "id": "w1"}]),
                AIMessage(content="verdict: pass\nfeedback: ok。"),
                AIMessage(content="# 报告\n完成。"),
            ]
            p = cfg_patches()
            with p[0], p[1], p[2], p[3]:
                order, state = run_on(FakeModel(calls), f"写入 {scratch}", thread_id="allow-test")
            assert not interrupt_calls, "纯 allow 批次不应触发 interrupt"
            assert state["verdict"] == "pass", f"应一轮通过：{state['verdict']}"
            with open(scratch, encoding="utf-8") as f:
                assert f.read() == "放行内容\n", "配置放行应直接执行"
            with open(audit_path, encoding="utf-8") as f:
                recs = [json.loads(line) for line in f]
            assert len(recs) == 1 and recs[0]["action"] == "auto_allow", f"审计应记 auto_allow：{recs}"

            # ② 混合批次：command=ask（缺省）→ 仍 interrupt
            interrupt_calls.clear()
            calls2 = [
                AIMessage(content='["写+跑"]'),
                AIMessage(content="写跑", tool_calls=[
                    {"name": "plan_write_file", "args": {"path": cmd_scratch, "content": "x\n"}, "id": "w1"},
                    {"name": "plan_run_command", "args": {"command": "echo mixed"}, "id": "c1"}]),
                AIMessage(content="verdict: pass\nfeedback: ok。"),
                AIMessage(content="# 报告\n完成。"),
            ]
            p = cfg_patches()
            with p[0], p[1], p[2], p[3]:
                run_on(FakeModel(calls2), "写文件再跑命令", thread_id="allow-mixed")
            # interrupt() 被调 2 次 = 1 次人工审批：resume 时 guard 节点整体重跑，
            # 第一次调用抛出中断、第二次（重跑后）返回 resume 值
            assert len(interrupt_calls) == 2, f"混合批次（含 ask 类别）应整批走人工审批：{len(interrupt_calls)}"
            print("PASS permission allow（纯写直批 + auto_allow 审计 + 混合批次仍审批）✔\n")
        finally:
            for f in (scratch, cmd_scratch):
                if os.path.exists(f):
                    os.remove(f)


def test_permission_deny_and_fallback():
    """.blue.toml command=deny：暂存即拒（反馈模型）+ execute_change 最后防线；
    配置缺失/非法回落 ask（fail-closed）；只读工具恒 allow。"""
    import tools
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "config.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('[permissions]\ncommand = "deny"\n')
        with patch("tools.GLOBAL_CONFIG_PATH", cfg), \
             patch("tools.PROJECT_CONFIG_NAME", ".blue-test-nonexistent.toml"):
            # execute_change 最后防线
            out = tools.execute_change({"action": "plan_run_command", "command": "echo hi"})
            assert "执行失败" in out and "deny" in out, f"deny 应拦执行：{out}"
            assert tools.permission_for_action("grep") == "allow", "只读工具恒 allow"
            assert tools.permission_for_action("plan_write_file") == "ask", "未配的类别回落 ask"
            # 暂存即拒：agent 收到「被拦截」反馈，不产生 pending_changes
            # deny 拦截后 pending 为空 → 不 break，agent 继续调模型直到无 tool_calls 收尾；
            # 之后 guard proceed → reviewer 短路 pass → report 模板，均不消耗模型调用
            calls = [
                AIMessage(content='["跑命令"]'),
                AIMessage(content="跑", tool_calls=[
                    {"name": "plan_run_command", "args": {"command": "echo staged"}, "id": "c1"}]),
                AIMessage(content="被配置禁止，最终答复：无法执行。"),
            ]
            fake = RecordingFake(calls)
            with patch("agent._make_model", lambda: fake), \
                 patch("agent._make_plain_model", lambda: fake), \
                 patch("agent.should_skip_planner", lambda r: False):
                order, state = run_on(fake, "跑个命令", thread_id="deny-test")
            assert state["pending_changes"] == [], "deny 暂存即拒，不应有 pending"
            agent_round2 = next((t for t in fake.inputs if "被拦截" in t), "")
            assert "被 .blue.toml 配置禁止" in agent_round2, f"模型应收到 deny 反馈：{agent_round2[-200:]}"
        # 非法配置回落 ask
        bad = os.path.join(td, "bad.toml")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("[permissions]\nwrite = \"sometimes\"\n")
        with patch("tools.GLOBAL_CONFIG_PATH", bad), \
             patch("tools.PROJECT_CONFIG_NAME", ".blue-test-nonexistent.toml"):
            assert tools.load_permissions() == {"write": "ask", "command": "ask", "python": "ask"}, \
                "非法值应回落 ask（fail-closed）"
    print("PASS permission deny（暂存拒 + 执行防线 + 非法回落 ask + 只读恒 allow）✔\n")


def test_execution_order_files_first():
    """guard 固定执行顺序：幂等的写文件类在前，命令/Python 在后，同组内保持稳定。"""
    mixed = [
        {"action": "plan_run_command", "command": "make"},
        {"action": "plan_write_file", "path": "a"},
        {"action": "plan_run_python", "code": "x=1"},
        {"action": "plan_patch", "path": "b"},
    ]
    ordered = agent._execution_order(mixed)
    assert [c["action"] for c in ordered] == [
        "plan_write_file", "plan_patch", "plan_run_command", "plan_run_python"], \
        f"写文件类应先执行：{[c['action'] for c in ordered]}"
    print("PASS execution order（写文件在前 + 组内稳定）✔\n")


def test_cross_round_context():
    """多轮连贯：每轮需求注入 messages → 后续轮 planner/agent 能看到历史需求；
    历史超 HISTORY_MAX_CHARS 时 planner 入口压成【历史会话摘要】（含历史需求文本）。"""
    req1 = "统计 hello.py 的行数"
    def round_calls(tag):
        return [
            AIMessage(content='["查一下"]'),
            AIMessage(content="grep 查", tool_calls=[
                {"name": "grep", "args": {"pattern": "hello"}, "id": f"g{tag}"}]),
            AIMessage(content=f"第{tag}轮答复：完成。"),
            AIMessage(content="verdict: pass\nfeedback: 只读。"),
            AIMessage(content="# 报告\n完成。"),
        ]
    fake = RecordingFake(round_calls("一") + round_calls("二") + round_calls("三"))
    graph = agent.build_graph(checkpointer=MemorySaver())
    sess = agent.Session(thread_id="xround-test")
    with patch("agent._make_model", lambda: fake), \
         patch("agent._make_plain_model", lambda: fake), \
         patch("agent.should_skip_planner", lambda r: False):
        agent.run_round(graph, sess, req1)
        agent.run_round(graph, sess, "它有多少个函数")
        # 只读轮每轮消耗 3 次模型调用（planner、agent×2；reviewer 短路 pass、
        # report 模板化均不调模型）：第二轮 planner = inputs[3]
        planner2 = fake.inputs[3]
        assert req1 in planner2, f"第二轮 planner 应带历史需求：{planner2[:200]}"
        assert "【历史对话】" in planner2, "有历史时 planner prompt 应有历史段"
        # 第二轮 agent（inputs[4]）：state messages 里应含第一轮需求
        assert req1 in fake.inputs[4], "第二轮 agent 应看到第一轮需求"
        # 第三轮：阈值调到 10，强制跨轮压缩
        with patch("agent.HISTORY_MAX_CHARS", 10):
            agent.run_round(graph, sess, "再数一遍")
        planner3 = fake.inputs[6]
        assert "【历史会话摘要】" in planner3, f"超阈值应压缩历史：{planner3[:200]}"
        assert req1 in planner3, "摘要应保留历史需求文本（指代连贯性）"
        final = graph.get_state(sess.config).values
        contents = [str(m.content) for m in final["messages"]]
        assert any("【历史会话摘要】" in c for c in contents), f"state 中应有压缩摘要：{contents}"
        assert not any("第一轮答复" in c for c in contents), \
            f"第一轮的旧消息应被 RemoveMessage 清除：{contents}"
    print("PASS cross-round context（历史需求可见 + 超阈值压缩摘要 + 摘要保留需求）✔\n")


def test_web_search_readonly():
    """web_search：只读免审批直接执行；结果整形（编号 + 摘要截 200）；无 key 返回引导文本不抛异常。"""
    import tools
    # 无 key → 引导文本（不抛异常，防模型编造结果）
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TAVILY_API_KEY", None)
        out = tools.web_search.invoke({"query": "x"})
        assert "TAVILY_API_KEY" in out and "不要编造" in out, f"无 key 应返回引导文本：{out}"
    # 有 key + fake 后端 → 走通图内只读直执行
    fake_results = [
        {"title": "结果一", "url": "https://a.example", "content": "甲" * 300},
        {"title": "结果二", "url": "https://b.example", "content": "短摘要"},
    ]
    calls = [
        AIMessage(content='["联网查"]'),
        AIMessage(content="查一下", tool_calls=[
            {"name": "web_search", "args": {"query": "LangGraph 最新版本"}, "id": "s1"}]),
        AIMessage(content="查完，最终答复：完成。"),
    ]
    fake = RecordingFake(calls)
    with patch("tools._tavily_search", lambda q, n: fake_results), \
         patch.dict(os.environ, {"TAVILY_API_KEY": "dummy"}):
        order, state = run_on(fake, "查一下 LangGraph 最新版本", thread_id="ws-test")
    assert state["verdict"] == "pass"
    assert state["pending_changes"] == [], "只读搜索不应有 pending"
    agent_round2 = fake.inputs[2]  # agent 第二次调用应带着搜索结果
    assert "结果一" in agent_round2 and "https://a.example" in agent_round2
    assert "甲" * 200 + "…" in agent_round2, "摘要应截断到 200 字符"
    assert "甲" * 201 not in agent_round2
    print("PASS web_search（只读直执行 + 摘要截断 + 无 key 引导文本）✔\n")


def test_web_fetch():
    """web_fetch：HTML 转纯文本（去 script/style）、边界标记、截断；私网/非法协议拒绝。"""
    import tools
    assert "拒绝" in tools.web_fetch.invoke({"url": "http://127.0.0.1/admin"})
    assert "拒绝" in tools.web_fetch.invoke({"url": "http://192.168.1.1/"})
    # IPv6 变体：无括号 loopback / 全展开 loopback / IPv4-mapped IPv6（私网映射同样拒绝）
    assert "拒绝" in tools.web_fetch.invoke({"url": "http://[::1]/admin"})
    assert "拒绝" in tools.web_fetch.invoke({"url": "http://[0:0:0:0:0:0:0:1]/admin"})
    assert "拒绝" in tools.web_fetch.invoke({"url": "http://[::ffff:127.0.0.1]/admin"})
    assert "拒绝" in tools.web_fetch.invoke({"url": "http://[::ffff:10.0.0.1]/"})
    assert "仅支持" in tools.web_fetch.invoke({"url": "file:///etc/passwd"})

    html = (b"<html><head><style>body{color:red}</style></head><body>"
            b"<h1>\xe6\xa0\x87\xe9\xa2\x98</h1><p>\xe6\xad\xa3\xe6\x96\x87</p>"
            b"<script>alert(1)</script></body></html>")

    class FakeResp:
        headers = {"Content-Type": "text/html; charset=utf-8"}
        payload = html
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1): return self.payload

    with patch("urllib.request.urlopen", lambda *a, **k: FakeResp()):
        out = tools.web_fetch.invoke({"url": "https://example.com/page"})
    assert "【网页内容开始" in out and "【网页内容结束】" in out, "应有注入防护边界标记"
    assert "标题" in out and "正文" in out
    assert "alert" not in out and "color" not in out, "script/style 应被剔除"

    class BigResp(FakeResp):
        payload = b"<body><p>" + b"x" * 5000 + b"</p></body>"

    with patch("urllib.request.urlopen", lambda *a, **k: BigResp()):
        out = tools.web_fetch.invoke({"url": "https://example.com/big", "max_chars": 100})
    assert "已截断" in out, "超长正文应截断"
    print("PASS web_fetch（HTML 转文本 + 边界标记 + 截断 + 私网拒绝）✔\n")


def test_doctor_checks():
    """doctor 各自检项判定逻辑（网络 patch 在 _fetch_model_ids / _make_plain_model 边界）。"""
    import urllib.error
    # 配置缺失 / 齐全
    # 注册表（~/.blue/models.toml）也能提供配置，patch 掉以纯测 env 缺失场景
    with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_BASE_URL": "", "MODEL_NAME": ""}), \
            patch("agent.load_models", lambda: {}):
        ok, msg = agent._check_config()
        assert not ok and "配置缺失" in msg
    with patch.dict(os.environ, {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://x/v1",
                                 "MODEL_NAME": "m1"}):
        ok, _ = agent._check_config()
        assert ok
        with patch("agent._fetch_model_ids", lambda: ["m1", "m2"]):
            ok, msg = agent._check_api_and_model()
            assert ok and "m1" in msg
        with patch("agent._fetch_model_ids", lambda: ["m2", "m1-pro"]):
            ok, msg = agent._check_api_and_model()
            assert not ok and "相近" in msg and "m1-pro" in msg, "应给出相近模型提示"
        err401 = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with patch("agent._fetch_model_ids", side_effect=err401):
            ok, msg = agent._check_api_and_model()
            assert not ok and "认证失败" in msg
        err404 = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        with patch("agent._fetch_model_ids", side_effect=err404):
            ok, msg = agent._check_api_and_model()
            assert not ok and "OPENAI_BASE_URL" in msg, "404 应指向 base_url 路径 typo"
        with patch("agent._fetch_model_ids", side_effect=ConnectionError("refused")):
            ok, msg = agent._check_api_and_model()
            assert not ok and "不可达" in msg
        # tool calling 探测
        class TCFake:
            def bind_tools(self, ts): return self
            def invoke(self, msgs):
                return AIMessage(content="", tool_calls=[
                    {"name": "_ping", "args": {"x": "ok"}, "id": "p1"}])
        class NoTCFake(TCFake):
            def invoke(self, msgs): return AIMessage(content="我不会调工具")
        with patch("agent._make_plain_model", lambda: TCFake()):
            ok, _ = agent._check_tool_calling()
            assert ok
        with patch("agent._make_plain_model", lambda: NoTCFake()):
            ok, msg = agent._check_tool_calling()
            assert not ok and "tool_calls" in msg
    print("PASS doctor checks（配置/API/模型/tool calling 判定 + 401/404/不可达分流）✔\n")


def test_init_writes_env():
    """blue init：getpass 不回显收集 → 写全局 .env（权限 600 + 覆盖前备份）→ 接 doctor；
    已有文件时拒绝覆盖则不动。"""
    import stat
    with tempfile.TemporaryDirectory() as td:
        env_path = os.path.join(td, ".env")
        with patch("agent.ENV_GLOBAL_PATH", env_path), \
             patch("builtins.input", lambda *a, **k: next(inputs)), \
             patch("getpass.getpass", lambda *a, **k: next(secrets_)), \
             patch("agent.cmd_doctor", lambda: 0):
            # ① 全新写入
            inputs = iter(["https://api.example/v1", "test-model"])
            secrets_ = iter(["sk-test", "tv-test"])
            assert agent.cmd_init() == 0
            with open(env_path, encoding="utf-8") as f:
                body = f.read()
            assert "OPENAI_BASE_URL=https://api.example/v1" in body
            assert "MODEL_NAME=test-model" in body and "OPENAI_API_KEY=sk-test" in body
            mode = stat.S_IMODE(os.stat(env_path).st_mode)
            assert mode == 0o600, f".env 权限应为 600，实际 {oct(mode)}"
            # ② 已有文件 + 拒绝覆盖 → 不动
            inputs = iter(["n"])
            assert agent.cmd_init() == 1
            with open(env_path, encoding="utf-8") as f:
                assert f.read() == body, "拒绝覆盖后文件不应变化"
            # ③ 已有文件 + 确认覆盖 → 备份 .bak
            inputs = iter(["y", "https://api2.example/v1", "m2"])
            secrets_ = iter(["sk-2", ""])
            assert agent.cmd_init() == 0
            assert os.path.exists(env_path + ".bak"), "覆盖前应备份 .bak"
            with open(env_path, encoding="utf-8") as f:
                assert "MODEL_NAME=m2" in f.read()
            with open(env_path + ".bak", encoding="utf-8") as f:
                assert "MODEL_NAME=test-model" in f.read(), "备份应保留旧值"
    print("PASS blue init（写全局 env + 600 权限 + 覆盖备份 + 拒绝不动）✔\n")


def test_parallel_readonly_tool_calls_no_duplication():
    """一条 AI 消息带 N 个并行只读 tool_calls：ToolNode 只执行一次，
    每个 tool_call_id 恰好一条 ToolMessage（N² 重复会被网关按非法请求 400 拒，glm 实测）。"""
    calls = [
        AIMessage(content='["并行查"]'),
        AIMessage(content="并行查三个", tool_calls=[
            {"name": "grep", "args": {"pattern": "hello"}, "id": "r1"},
            {"name": "grep", "args": {"pattern": "agent"}, "id": "r2"},
            {"name": "list_files", "args": {"dir": "."}, "id": "r3"},
        ]),
        AIMessage(content="查完，最终答复：完成。"),
    ]
    order, state = run_on(FakeModel(calls), "并行查三个东西", thread_id="par-ro-test")
    assert state["verdict"] == "pass"
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    ids = [tm.tool_call_id for tm in tool_msgs]
    assert sorted(ids) == ["r1", "r2", "r3"], \
        f"每个 tool_call_id 应恰好一条 ToolMessage，实际 {ids}（重复=ToolNode 被调了多次）"
    print("PASS parallel read-only calls（N 个并行调用无 N² ToolMessage 重复）✔\n")


def test_permission_batch_read_once():
    """guard 批次权限判定入口读一次配置、批内复用（不逐条重读 TOML）；
    permission_for_action 传 perms 参数时零磁盘 IO。"""
    import tools
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "config.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('[permissions]\nwrite = "allow"\n')
        with patch("tools.GLOBAL_CONFIG_PATH", cfg), \
             patch("tools.PROJECT_CONFIG_NAME", ".blue-test-nonexistent.toml"):
            # ① perms 参数生效且不重读：批内判定走传入的合并结果
            perms = tools.load_permissions()
            calls = {"n": 0}
            real_load = tools.load_permissions
            def counting_load():
                calls["n"] += 1
                return real_load()
            # permission_for_action 不传 perms → 现读；传 perms → 零调用
            assert tools.permission_for_action("plan_write_file") == "allow"
            before = calls["n"]
            with patch("tools.load_permissions", counting_load):
                assert tools.permission_for_action("plan_write_file", perms) == "allow"
                assert tools.permission_for_action("plan_patch", perms) == "allow"
                assert tools.permission_for_action("plan_run_command", perms) == "ask"
                assert tools.permission_for_action("grep", perms) == "allow"  # 只读恒 allow
            assert calls["n"] == before, "传 perms 后不应再调 load_permissions"

            # ② guard 纯 allow 批次：整个 guard 只读一次配置（入口），不随条数放大
            import agent
            with patch("agent.load_permissions", counting_load):
                st = agent.initial_state("写两文件")
                st["thread_id"] = "batch-read-once"
                st["pending_changes"] = [
                    {"action": "plan_write_file", "path": "__scratch_br1__.txt", "content": "1\n"},
                    {"action": "plan_write_file", "path": "__scratch_br2__.txt", "content": "2\n"},
                ]
                try:
                    before = calls["n"]
                    out = agent.guard(st)
                    assert calls["n"] - before == 1, f"guard 批次应只读一次配置，实际 {calls['n'] - before}"
                    assert out["verdict"] == "approved", f"纯 allow 批次应直批：{out['verdict']}"
                finally:
                    for f in ("__scratch_br1__.txt", "__scratch_br2__.txt"):
                        if os.path.exists(f):
                            os.remove(f)
    print("PASS permission batch read once（perms 复用零 IO + guard 批次只读一次）✔\n")


def test_grep_large_file_skip():
    """grep：超过大小上限的文件跳过并提示，不读进内存；正常文件不受影响。"""
    import tools
    scratch_big = "__scratch_grep_big__.log"
    scratch_small = "__scratch_grep_small__.txt"
    try:
        # 用小上限模拟大文件（避免真的造 10MB 文件拖慢测试）
        with open(scratch_big, "w", encoding="utf-8") as f:
            f.write("match-big\n" + "y" * 2000 + "\n")
        with open(scratch_small, "w", encoding="utf-8") as f:
            f.write("match-small\n")
        with patch("tools._GREP_MAX_FILE_BYTES", 100):
            out = tools.grep.invoke({"pattern": "match-", "path": "."})
        assert "match-small" in out, f"小文件应正常命中：{out}"
        assert "match-big" not in out, f"超上限文件应跳过：{out}"
        assert "跳过" in out and "大文件" in out, f"应有跳过提示：{out}"
    finally:
        for f in (scratch_big, scratch_small):
            if os.path.exists(f):
                os.remove(f)
    print("PASS grep large file skip（超上限跳过 + 提示 + 小文件不受影响）✔\n")


def test_resume_checkpoint_guard():
    """resume 的 checkpoint 可用性探测：_resume_picker 与 /resume 选中 thread 后
    先 get_state 探测，损坏/丢失的 checkpoint 提示并回落，不带进 run_round。"""
    import cli
    # ① _resume_picker：graph.get_state 抛异常（库损坏）→ 返回 None（回落新会话）
    class BrokenGraph:
        def get_state(self, config):
            raise RuntimeError("sqlite disk image is malformed")
    with patch("cli.list_sessions", lambda: [
        {"thread_id": "blue-dead", "rounds": 1, "last_active": "x"}]), \
         patch("builtins.input", lambda *a: "1"):
        assert cli._resume_picker(BrokenGraph()) is None, "损坏 checkpoint 应回落 None"

    # ② _resume_picker：get_state 返回空 state（从未写入）→ 同样回落
    class EmptyGraph:
        def get_state(self, config):
            class S: values, next = {}, ()
            return S()
    with patch("cli.list_sessions", lambda: [
        {"thread_id": "blue-empty", "rounds": 1, "last_active": "x"}]), \
         patch("builtins.input", lambda *a: "1"):
        assert cli._resume_picker(EmptyGraph()) is None, "空 checkpoint 应回落 None"

    # ③ 正常 checkpoint → 返回 tid（探测通过）
    class GoodGraph:
        def get_state(self, config):
            class S:
                values = {"request": "x"}
                next = ()
            return S()
    with patch("cli.list_sessions", lambda: [
        {"thread_id": "blue-live", "rounds": 2, "last_active": "x"}]), \
         patch("builtins.input", lambda *a: "1"):
        assert cli._resume_picker(GoodGraph()) == "blue-live", "正常 checkpoint 应返回 tid"

    # ④ /resume 路径：损坏 checkpoint 提示且不换 session（走 handle_slash 分支）
    sess = agent.Session.__new__(agent.Session)
    sess.thread_id = "blue-current"
    with patch("cli.list_sessions", lambda: [
        {"thread_id": "blue-dead2", "rounds": 1, "last_active": "x"}]), \
         patch("builtins.input", lambda *a: "1"):
        cont, new_sess = cli.handle_slash("/resume", sess, BrokenGraph())
    assert cont and new_sess is None, "损坏 checkpoint 不应切换 session"
    print("PASS resume checkpoint guard（损坏/空 checkpoint 探测 + 正常放行 + /resume 不切换）✔\n")


def test_command_head_precedence():
    """_command_head 边界：单 token 含 = 或单独 env 时不得 IndexError（or 两侧曾被
    and 优先级吞掉 i 越界守卫）；VAR=/env 前缀正确跳过，flag 形态 token 不跳过。"""
    import tools
    assert tools._command_head("VAR=val") == ""           # 曾 IndexError
    assert tools._command_head("env") == ""               # 曾 IndexError
    assert tools._command_head("env VAR=1 python3 x.py") == "python3"
    assert tools._command_head("FOO=bar ls -l") == "ls"
    assert tools._command_head("FOO=bar env pytest") == "pytest"
    assert tools._command_head("ls -la") == "ls"
    assert tools._command_head("-x=1 cmd") == "-x=1"      # flag 形态不跳过
    assert tools._command_head("") == ""
    print("PASS command head（越界不崩 + VAR=/env 前缀跳过 + flag 不跳过）✔\n")


def test_model_registry_switch():
    """多模型注册表（v0.8）：TOML 解析、激活优先级、/model 切换清缓存、
    kwargs 注册表命中/env 回落、context_window 配置读取。注册表路径与运行时
    override 均 patch 隔离，不碰真实 ~/.blue。"""
    import models
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, "models.toml")
        with open(reg, "w", encoding="utf-8") as f:
            f.write(
                '[models.gpt4o]\nmodel = "gpt-4o"\n'
                'base_url = "https://api.openai.com/v1"\ncontext_window = 128000\n\n'
                '[models.glm]\nmodel = "glm-4.6"\n'
                'base_url = "https://open.bigmodel.cn/api/paas/v4"\ncontext_window = 200000\n\n'
                '[active]\nname = "glm"\n'
            )
        with patch("models.MODELS_PATH", reg), \
             patch.dict(os.environ, {"MODEL_NAME": "", "OPENAI_BASE_URL": "https://env.example/v1",
                                     "OPENAI_API_KEY": "sk-env"}):
            models.clear_active_override()
            # ① 注册表解析
            reg_models = models.load_models()
            assert set(reg_models) == {"gpt4o", "glm"}, f"注册表解析错误：{reg_models}"
            assert reg_models["glm"]["model"] == "glm-4.6"
            assert reg_models["glm"]["context_window"] == 200000
            # ② 激活优先级：无 env 无 override → [active].name
            assert models.active_model_name() == "glm"
            kw = models.model_kwargs()
            assert kw["model"] == "glm-4.6" and kw["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
            assert kw["api_key"] == "sk-env", "条目未配 api_key 应回落 env"
            # ③ env 命中注册表 → 用注册表配置
            with patch.dict(os.environ, {"MODEL_NAME": "gpt4o"}):
                assert models.active_model_name() == "gpt4o"
                kw = models.model_kwargs()
                assert kw["model"] == "gpt-4o" and kw["base_url"] == "https://api.openai.com/v1"
                assert models.context_window() == 128000
            # ④ env 未注册 → 裸 env 回落（旧行为）
            with patch.dict(os.environ, {"MODEL_NAME": "unknown-model"}):
                kw = models.model_kwargs()
                assert kw["model"] == "unknown-model" and kw["base_url"] == "https://env.example/v1"
                assert models.context_window() == models.DEFAULT_CONTEXT_WINDOW
            # ⑤ set_active_model + agent 缓存失效
            ok, msg = models.set_active_model("gpt4o")
            assert ok and models.active_model_name() == "gpt4o"
            assert agent.set_active_model("glm").startswith("已切换")
            assert agent._model_cache is None and agent._plain_model_cache is None, "切换必须清模型缓存"
            assert models.active_model_name() == "glm"
            assert models.context_window() == 200000
            # ⑥ 非法切换 fail-closed
            ok, msg = models.set_active_model("nope")
            assert not ok and "不在注册表" in msg, f"非法切换应失败：{msg}"
            # ⑦ list_models：注册表 + env 未注册补尾
            with patch.dict(os.environ, {"MODEL_NAME": ""}):
                names = [m["name"] for m in models.list_models()]
                assert names == ["gpt4o", "glm"], f"列表错误：{names}"
            with patch.dict(os.environ, {"MODEL_NAME": "env-extra"}):
                names = [m["name"] for m in models.list_models()]
                assert names == ["gpt4o", "glm", "env-extra"], f"env 未注册应补尾：{names}"
        models.clear_active_override()
    # ⑧ 无注册表文件 → 全回落 env（旧行为）
    with patch("models.MODELS_PATH", "/nonexistent/models.toml"), \
         patch.dict(os.environ, {"MODEL_NAME": "env-only"}):
        models.clear_active_override()
        assert models.load_models() == {}
        assert models.active_model_name() == "env-only"
        kw = models.model_kwargs()
        assert kw["model"] == "env-only"
    models.clear_active_override()
    print("PASS model registry（TOML 解析 + 激活优先级 + 切换清缓存 + env 回落）✔\n")


def test_context_usage_display():
    """context 占用追踪（v0.8）：单次调用 prompt=上下文占用取峰值；
    Session.context 跨轮取 max 不累加；百分比串格式；无调用不显示。"""
    # ① _record_usage 峰值语义
    agent._reset_token_usage()
    m1 = AIMessage(content="x", response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 5}})
    m2 = AIMessage(content="y", response_metadata={"token_usage": {"prompt_tokens": 700, "completion_tokens": 5}})
    m3 = AIMessage(content="z", response_metadata={"token_usage": {"prompt_tokens": 300, "completion_tokens": 5}})
    agent._record_usage(m1)
    agent._record_usage(m2)
    agent._record_usage(m3)
    snap = agent._token_usage_snapshot()
    assert snap["prompt"] == 1100 and snap["calls"] == 3
    assert snap["context"] == 700, f"context 应取峰值，实际 {snap['context']}"
    # ② 百分比串
    with patch("agent.active_context_window", lambda: 1000):
        s = agent._context_usage_str(700)
        assert "700" in s and "70.0%" in s, f"百分比串错误：{s}"
        assert agent._context_usage_str(0) == "", "无调用应返回空串"
    # ③ Session 累计语义：context 取 max，prompt/completion 累加
    sess = agent.Session(thread_id="ctx-test")
    with patch.object(agent, "_token_usage_snapshot",
                      lambda: {"prompt": 100, "completion": 10, "calls": 2, "context": 500}):
        agent._finish_round_usage(sess)
    with patch.object(agent, "_token_usage_snapshot",
                      lambda: {"prompt": 50, "completion": 5, "calls": 1, "context": 300}):
        agent._finish_round_usage(sess)  # 回落轮：context 取 max 保留 500
    t = sess.token_usage
    assert t["prompt"] == 150 and t["completion"] == 15 and t["calls"] == 3, f"累计错误：{t}"
    assert t["context"] == 500, f"context 跨轮应取 max（不是累加 800），实际 {t['context']}"
    agent._reset_token_usage()
    print("PASS context usage（峰值采集 + max 累计 + 百分比串 + 空值静默）✔\n")


def test_model_slash_command():
    """/model 命令：无参列出+交互选择切换；/model <名> 直接切；序号切换；
    无注册表时提示不可切。路径 patch 隔离。"""
    import cli
    import models
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, "models.toml")
        with open(reg, "w", encoding="utf-8") as f:
            f.write('[models.a1]\nmodel = "model-a"\nbase_url = "https://a/v1"\n'
                    '[models.b2]\nmodel = "model-b"\nbase_url = "https://b/v1"\n')
        sess = agent.Session(thread_id="model-cmd")
        with patch("models.MODELS_PATH", reg), \
             patch.dict(os.environ, {"MODEL_NAME": ""}):
            models.clear_active_override()
            # ① 无参 + 选择序号 2 → 切换 b2
            with patch("builtins.input", lambda *a: "2"):
                cli.handle_slash("/model", sess, None)
            assert models.active_model_name() == "b2"
            assert agent._model_cache is None, "切换后应清缓存"
            # ② 带名直接切
            cli.handle_slash("/model a1", sess, None)
            assert models.active_model_name() == "a1"
            # ③ 非法名：提示失败、不切换
            cli.handle_slash("/model nope", sess, None)
            assert models.active_model_name() == "a1"
            # ④ 回车取消：不切换
            with patch("builtins.input", lambda *a: ""):
                cli.handle_slash("/model", sess, None)
            assert models.active_model_name() == "a1"
        models.clear_active_override()
    # ⑤ 无注册表：列出提示不可切
    with patch("models.MODELS_PATH", "/nonexistent/models.toml"):
        models.clear_active_override()
        with patch.dict(os.environ, {"MODEL_NAME": "env-model"}):
            cli.handle_slash("/model", agent.Session(thread_id="x"), None)
    models.clear_active_override()
    print("PASS /model command（列表选择 + 直接切换 + 非法拒绝 + 无注册表提示）✔\n")


def test_run_round_auto():
    """回归：--auto-approve 路径自动审批、文件真落盘、返回最终 state。
    曾用错误 config 取 state，--auto-approve 丢 interrupt 后数据。"""
    scratch = os.path.abspath("scratch-auto-test.txt")
    audit_path = os.path.join(_TMP_BACKUPS, "audit-auto.jsonl")
    calls = [
        AIMessage(content='["写 scratch 文件", "报告"]'),
        AIMessage(content="写入文件", tool_calls=[
            {"name": "plan_write_file", "args": {"path": scratch, "content": "auto ok"}, "id": "t1"}]),
        AIMessage(content="verdict: pass\nfeedback: 写入正确。"),
        AIMessage(content="# 报告\n完成。"),
    ]
    fake = FakeModel(calls)
    graph = agent.build_graph(checkpointer=MemorySaver())
    sess = agent.Session(thread_id="auto-test")
    try:
        with patch("agent.AUDIT_LOG", audit_path), \
             patch("agent._make_model", lambda: fake), \
             patch("agent._make_plain_model", lambda: fake), \
             patch("agent.should_skip_planner", lambda r: False):
            state = agent.run_round_auto(graph, sess, f"写入 {scratch}")
        assert os.path.exists(scratch), "auto 审批后文件应真落盘"
        assert state.get("verdict") == "pass", f"应为 pass，实际 {state.get('verdict')}"
        assert state.get("pending_changes") == [], "审批后 pending 应为空"
        print("PASS run_round_auto（自动审批 + 落盘 + 最终 state 返回）✔\n")
    finally:
        if os.path.exists(scratch):
            os.unlink(scratch)


if __name__ == "__main__":
    try:
        test_readonly_no_interrupt()
        test_write_requires_resume()
        test_reject_no_write()
        test_revise_loops_back()
        test_run_command_cwd_sandbox()
        test_multi_round_same_thread()
        test_session_meta_persistence()
        test_revise_compresses_messages()
        test_skip_planner_for_simple_request()
        test_parallel_workers_fanout()
        test_security_hardening()
        test_tool_usability()
        test_agent_sliding_window()
        test_report_template_saves_llm()
        test_worker_fault_tolerance()
        test_token_usage_tracking()
        test_report_gets_executed_changes()
        test_audit_log()
        test_selective_approval()
        test_undo_snapshot_restore()
        test_llm_auto_retry()
        test_resume_pending_after_crash()
        test_permission_allow_skips_interrupt()
        test_permission_deny_and_fallback()
        test_execution_order_files_first()
        test_cross_round_context()
        test_web_search_readonly()
        test_web_fetch()
        test_doctor_checks()
        test_init_writes_env()
        test_parallel_readonly_tool_calls_no_duplication()
        test_permission_batch_read_once()
        test_grep_large_file_skip()
        test_resume_checkpoint_guard()
        test_command_head_precedence()
        test_model_registry_switch()
        test_context_usage_display()
        test_model_slash_command()
        test_run_round_auto()
        print("ALL OFFLINE TESTS PASSED ✅")
    finally:
        # 清理临时数据库文件
        if os.path.exists(_TMP_DB.name):
            os.unlink(_TMP_DB.name)
        wal = _TMP_DB.name + "-wal"
        if os.path.exists(wal):
            os.unlink(wal)
        shm = _TMP_DB.name + "-shm"
        if os.path.exists(shm):
            os.unlink(shm)
        import shutil as _sh
        _sh.rmtree(_TMP_BACKUPS, ignore_errors=True)