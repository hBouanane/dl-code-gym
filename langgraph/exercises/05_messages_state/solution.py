"""
Exercise 05 - Chat state: MessagesState, add_messages, message types  [chat]

1. Use the prebuilt MessagesState (its `messages` key uses the add_messages reducer).
2. Write fake_llm(messages) -> AIMessage: if the last message contains "hello" reply
   "Hi! How can I help?" else reply "You said: <content>". (Stand-in for a real model, see below.)
3. Node chatbot(state) -> {"messages": [fake_llm(state["messages"])]}
4. START -> chatbot -> END. result = graph.invoke({"messages": [HumanMessage("hello")]})
   -> result["messages"] has 2 messages (human + ai)
5. add_messages detail: call add_messages([HumanMessage("a", id="1")], [HumanMessage("b", id="1")])
   -> the message with the same id is REPLACED, not appended. Store it in `merged`.
Real model (later): from langchain.chat_models import init_chat_model
                    llm = init_chat_model("anthropic:claude-sonnet-5-5"); llm.invoke(state["messages"])
Self-check expects these names: graph, result, merged
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages


def fake_llm(messages) -> AIMessage:
    last = messages[-1].content
    if "hello" in last.lower():
        return AIMessage("Hi! How can I help?")
    return AIMessage(f"You said: {last}")


def chatbot(state: MessagesState) -> dict:
    return {"messages": [fake_llm(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("hello")]})
for m in result["messages"]:
    m.pretty_print()

merged = add_messages([HumanMessage("a", id="1")], [HumanMessage("b", id="1")])
print(merged)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert len(result["messages"]) == 2
    assert result["messages"][-1].type == "ai" and "Hi" in result["messages"][-1].content
    assert graph.invoke({"messages": [HumanMessage("test")]})["messages"][-1].content == "You said: test"
    assert len(merged) == 1 and merged[0].content == "b"
    print("All checks passed!")
