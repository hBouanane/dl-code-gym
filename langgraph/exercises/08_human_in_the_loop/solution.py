"""
Exercise 08 - Human-in-the-loop: interrupt() and Command(resume=...)  [agents]

An email agent that must get approval before sending.
1. State: request: str ; draft: str ; status: str
2. Node write_draft: draft = f"Dear client, {request}. Best regards."
3. Node human_review: decision = interrupt({"draft": state["draft"]})   <- pauses the graph here
     if decision["approved"]: status = "sent"
     else: status = "cancelled"
4. Edges START -> write_draft -> human_review -> END. Compile WITH a checkpointer (required).
5. first = graph.invoke({"request": "your order shipped"}, config)
     -> first contains "__interrupt__"; graph.get_state(config).next == ("human_review",)
6. Resume: final = graph.invoke(Command(resume={"approved": True}), config) -> status "sent"
Self-check expects these names: graph
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    request: str
    draft: str
    status: str


def write_draft(state: State) -> dict:
    return {"draft": f"Dear client, {state['request']}. Best regards."}


def human_review(state: State) -> dict:
    decision = interrupt({"draft": state["draft"]})
    return {"status": "sent" if decision["approved"] else "cancelled"}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("human_review", human_review)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "human_review")
builder.add_edge("human_review", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "email-1"}}
first = graph.invoke({"request": "your order shipped"}, config)
print("paused with:", first["__interrupt__"][0].value)
print("next node:", graph.get_state(config).next)
final = graph.invoke(Command(resume={"approved": True}), config)
print(final["status"])


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": "check-1"}}
    r1 = graph.invoke({"request": "hello"}, cfg)
    assert "__interrupt__" in r1 and graph.get_state(cfg).next == ("human_review",)
    assert graph.invoke(Command(resume={"approved": True}), cfg)["status"] == "sent"
    cfg2 = {"configurable": {"thread_id": "check-2"}}
    graph.invoke({"request": "hello"}, cfg2)
    assert graph.invoke(Command(resume={"approved": False}), cfg2)["status"] == "cancelled"
    print("All checks passed!")
