"""
Exercise 07 - Memory with a checkpointer and thread_id  [chat]

1. Chat graph on MessagesState with a fake bot node:
     "my name is X"     -> "Nice to meet you, X!"
     "what is my name"  -> look back through ALL previous human messages for "my name is X"
                           -> "Your name is X." (or "I don't know your name.")
     otherwise          -> "OK."
2. graph = builder.compile(checkpointer=InMemorySaver())
3. config = {"configurable": {"thread_id": "user-1"}} ; send "my name is Haitam" then
   "what is my name" as two SEPARATE invokes (only the new message each time!)
   -> the bot remembers because the state is saved per thread.
4. A different thread_id "user-2" does not know the name.
5. snapshot = graph.get_state(config) -> snapshot.values["messages"] has 4 messages
Self-check expects these names: graph, chat(text, thread_id) -> reply string
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


def bot(state: MessagesState) -> dict:
    last = state["messages"][-1].content.lower()
    if last.startswith("my name is"):
        reply = f"Nice to meet you, {state['messages'][-1].content.split()[-1]}!"
    elif "what is my name" in last:
        name = None
        for m in state["messages"]:
            if m.type == "human" and m.content.lower().startswith("my name is"):
                name = m.content.split()[-1]
        reply = f"Your name is {name}." if name else "I don't know your name."
    else:
        reply = "OK."
    return {"messages": [AIMessage(reply)]}


builder = StateGraph(MessagesState)
builder.add_node("bot", bot)
builder.add_edge(START, "bot")
builder.add_edge("bot", END)
graph = builder.compile(checkpointer=InMemorySaver())


def chat(text, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": [HumanMessage(text)]}, config)["messages"][-1].content


print(chat("my name is Haitam", "user-1"))
print(chat("what is my name", "user-1"))
print(chat("what is my name", "user-2"))
snapshot = graph.get_state({"configurable": {"thread_id": "user-1"}})
print(len(snapshot.values["messages"]), "messages in thread user-1")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert chat("my name is Sara", "t-a") == "Nice to meet you, Sara!"
    assert chat("what is my name", "t-a") == "Your name is Sara."
    assert chat("what is my name", "t-b") == "I don't know your name."
    assert len(graph.get_state({"configurable": {"thread_id": "t-a"}}).values["messages"]) == 4
    print("All checks passed!")
