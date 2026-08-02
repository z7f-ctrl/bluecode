"""离线功能验证：用 fake model 跑完整图，验证 interrupt/resume 与条件路由。

注意：离线测试使用 MemorySaver + 临时 sqlite 文件，避免污染真实 ~/.blue/checkpoints.sqlite。
"""
import json
import os
import tempfile
from unittest.mock import patch

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

# 在 import agent 之前把 DB_PATH 指到临时文件，避免测试写入用户真实数据库
_TMP_DB = tempfile.NamedTemporaryFile(prefix="blue-test-", suffix=".sqlite", delete=False)
_TMP_DB.close()
os.environ.setdefault("BLUE_TEST_DB", _TMP_DB.name)

import agent
agent.DB_PATH = os.environ["BLUE_TEST_DB"]


class FakeModel:
    """按调用次序返回脚本化响应。sequence: list[str]，每次 invoke 弹一个。"""
    def __init__(self, sequence):
        self.seq = list(sequence)
        self.calls = 0

    def invoke(self, messages, *a, **k):
        item = self.seq.pop(0) if self.seq else "（耗尽）fallback"
        self.calls += 1
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=item)


def run_on(model_fake, request, resume_action=None, thread_id="test-1"):
    """跑一个 thread，遇 interrupt 用 resume_action 续命。返回 (node顺序, 最终state)。

    注意：patch 掉 should_skip_planner，保证 planner 总是消耗一次模型调用，
    使 fake model 的脚本化响应序列与各节点一一对应。skip 路径由专门测试覆盖。
    """
    graph = agent.build_graph(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = agent.initial_state(request)
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
    calls = [
        AIMessage(content='["改 hello.py"]'),
        AIMessage(content="先读。", tool_calls=[
            {"name": "read_file", "args": {"path": "hello.py"}, "id": "r1"}]),
        AIMessage(content="准备写。", tool_calls=[
            {"name": "plan_write_file", "args": {"path": "__rv__.txt", "content": "v1\n"}, "id": "w1"}]),
        AIMessage(content="verdict: revise\nfeedback: 变量名太烂，回去改。"),
        AIMessage(content="已按意见重写。"),
        AIMessage(content="verdict: pass\nfeedback: 这版行了。"),
        AIMessage(content="# 报告\n完成。"),
    ]
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
    print("PASS security hardening（&& ; $() 反引号 / subprocess / getattr+setattr dunder）✔\n")


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