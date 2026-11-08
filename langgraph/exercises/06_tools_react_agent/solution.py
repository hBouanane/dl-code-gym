"""
Exercise 06 - Tools + ToolNode + tools_condition: the ReAct loop  [agents]

The agent loop: agent -> (tool calls?) -> tools -> agent -> ... -> END
1. Two tools with @tool (the docstring is the description the LLM sees!):
     multiply(a: int, b: int) -> int ; get_weather(city: str) -> str  (return f"Sunny in {city}")
2. fake_llm node (stands in for llm.bind_tools(tools)):
     - if the last message is a ToolMessage -> AIMessage(f"The answer is {last.content}.")
     - elif "times" in the question -> AIMessage(content="", tool_calls=[{"name": "multiply",
           "args": {"a": .., "b": ..}, "id": "call_1"}])   (numbers parsed with re.findall)
     - elif "weather in" -> tool call get_weather with the city (last word)
     - else -> AIMessage("I can only multiply or check the weather.")
3. Write your own router should_continue(state): "tools" if last message has tool_calls else END
   (this is what langgraph.prebuilt.tools_condition does)
4. Graph: START -> agent ; add_conditional_edges("agent", should_continue, ["tools", END]) ;
          add_node("tools", ToolNode(tools)) ; "tools" -> "agent"
5. ask(q) returns the final message content. "what is 6 times 7" -> "The answer is 42."
Self-check expects these names: multiply, graph, ask
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


tools = [multiply, get_weather]


def agent(state: MessagesState) -> dict:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(f"The answer is {last.content}.")]}
    q = last.content.lower()
    if "times" in q:
        a, b = map(int, re.findall(r"\d+", q)[:2])
        call = {"name": "multiply", "args": {"a": a, "b": b}, "id": "call_1"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}
    if "weather in" in q:
        city = last.content.rstrip("?").split()[-1]
        call = {"name": "get_weather", "args": {"city": city}, "id": "call_2"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}
    return {"messages": [AIMessage("I can only multiply or check the weather.")]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()


def ask(q):
    return graph.invoke({"messages": [HumanMessage(q)]})["messages"][-1].content


print(ask("what is 6 times 7"))
print(ask("what is the weather in Rabat?"))
print(ask("tell me a joke"))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert multiply.invoke({"a": 3, "b": 4}) == 12
    assert ask("what is 6 times 7") == "The answer is 42."
    assert "Rabat" in ask("weather in Rabat")
    out = graph.invoke({"messages": [HumanMessage("what is 2 times 5")]})["messages"]
    assert [m.type for m in out] == ["human", "ai", "tool", "ai"]
    print("All checks passed!")
