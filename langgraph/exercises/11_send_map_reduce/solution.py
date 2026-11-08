"""
Exercise 11 - Map-reduce with Send (dynamic number of parallel branches)  [composition]

1. OverallState: subjects: list[str] ; jokes: Annotated[list[str], operator.add] ; best: str
   JokeState:   subject: str           (the private input of each branch)
2. Router continue_to_jokes(state) -> [Send("make_joke", {"subject": s}) for s in state["subjects"]]
   builder.add_conditional_edges(START, continue_to_jokes, ["make_joke"])
3. make_joke(state: JokeState) -> {"jokes": [f"Why did the {subject} cross the road?"]}
4. make_joke -> pick_best (best = longest joke) -> END
5. Invoke with subjects ["cat", "elephant", "dog"] -> 3 jokes, best mentions elephant.
The number of branches is decided at RUNTIME, which plain edges can't do.
Self-check expects these names: graph
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best: str


class JokeState(TypedDict):
    subject: str


def continue_to_jokes(state: OverallState):
    return [Send("make_joke", {"subject": s}) for s in state["subjects"]]


def make_joke(state: JokeState) -> dict:
    return {"jokes": [f"Why did the {state['subject']} cross the road?"]}


def pick_best(state: OverallState) -> dict:
    return {"best": max(state["jokes"], key=len)}


builder = StateGraph(OverallState)
builder.add_node("make_joke", make_joke)
builder.add_node("pick_best", pick_best)
builder.add_conditional_edges(START, continue_to_jokes, ["make_joke"])
builder.add_edge("make_joke", "pick_best")
builder.add_edge("pick_best", END)
graph = builder.compile()

print(graph.invoke({"subjects": ["cat", "elephant", "dog"], "jokes": []}))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    out = graph.invoke({"subjects": ["cat", "elephant", "dog"], "jokes": []})
    assert len(out["jokes"]) == 3 and "elephant" in out["best"]
    assert len(graph.invoke({"subjects": list("abcde"), "jokes": []})["jokes"]) == 5
    print("All checks passed!")
