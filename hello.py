from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# 1. 定义状态：messages 字段用 Annotated 指定一个「归并函数」
#    这里 operator.add 表示新值会追加到旧列表，而不是覆盖
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]


# 2. 定义节点：每个节点接收 state，返回对 state 的更新
def greet(state: State):
    return {"messages": ["你好，我是节点 A"]}

def respond(state: State):
    return {"messages": ["你好，我是节点 B，收到：" + state["messages"][-1]]}


# 3. 建图
builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("respond", respond)

# 4. 连边：START -> greet -> respond -> END
builder.add_edge(START, "greet")
builder.add_edge("greet", "respond")
builder.add_edge("respond", END)

# 5. 编译并运行
graph = builder.compile()
result = graph.invoke({"messages": []})

print(result["messages"])
# ['你好，我是节点 A', '你好，我是节点 B，收到：你好，我是节点 A']