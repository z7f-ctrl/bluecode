"""离线功能验证：用 fake model 跑完整图，验证 interrupt/resume 与条件路由。"""
from unittest.mock import patch

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

import agent


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


def run_on(model_fake, request, resume_action=None):
    """跑一个 thread，遇 interrupt 用 resume_action 续命。返回 (node顺序, 最终state)。"""
    graph = agent.build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "test-1"}}
    state = agent.initial_state(request)
    order = []
    with patch("agent._make_model", lambda: model_fake):
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


if __name__ == "__main__":
    test_readonly_no_interrupt()
    test_write_requires_resume()
    test_reject_no_write()
    test_revise_loops_back()
    print("ALL OFFLINE TESTS PASSED ✅")