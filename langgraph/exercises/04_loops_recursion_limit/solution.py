"""
Exercise 04 - Loops (generate -> evaluate -> retry) and recursion_limit  [routing]

1. State: draft: str ; attempts: int ; approved: bool
2. Node generate: draft += " +improved", attempts += 1
   Node evaluate: approved = attempts >= 3
3. Conditional edge after evaluate: END if approved else back to "generate"  (a LOOP)
4. result = graph.invoke({"draft": "v0", "attempts": 0, "approved": False}) -> attempts == 3
5. Build a second graph `bad_graph` where evaluate never approves, and invoke it with
   config={"recursion_limit": 6}. Catch langgraph.errors.GraphRecursionError -> set hit_limit = True
Self-check expects these names: result, hit_limit
"""

from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    draft: str
    attempts: int
    approved: bool


def generate(state: State) -> dict:
    return {"draft": state["draft"] + " +improved", "attempts": state["attempts"] + 1}


def build(evaluate):
    builder = StateGraph(State)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges("evaluate", lambda s: END if s["approved"] else "generate")
    return builder.compile()


graph = build(lambda s: {"approved": s["attempts"] >= 3})
result = graph.invoke({"draft": "v0", "attempts": 0, "approved": False})
print(result)

bad_graph = build(lambda s: {"approved": False})
hit_limit = False
try:
    bad_graph.invoke({"draft": "v0", "attempts": 0, "approved": False}, config={"recursion_limit": 6})
except GraphRecursionError:
    hit_limit = True
print("hit recursion limit:", hit_limit)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert result["attempts"] == 3 and result["approved"] is True
    assert result["draft"].count("+improved") == 3
    assert hit_limit
    print("All checks passed!")
