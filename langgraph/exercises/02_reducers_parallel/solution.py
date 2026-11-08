"""
Exercise 02 - Reducers (Annotated + operator.add) and parallel fan-out / fan-in  [basics]

1. class State(TypedDict): topic: str ; notes: Annotated[list[str], operator.add] ; summary: str
   The reducer operator.add means: updates to `notes` are APPENDED instead of overwriting.
2. Three nodes research_a / research_b / research_c, each returns {"notes": [f"<name>: fact about {topic}"]}
3. Node combine: summary = " | ".join(sorted(state["notes"]))
4. Edges: START -> each of the 3 research nodes (they run IN PARALLEL in the same step),
   then builder.add_edge(["research_a", "research_b", "research_c"], "combine")  (waits for all 3)
   combine -> END
5. result = graph.invoke({"topic": "transformers", "notes": []})
Question: what happens without the reducer when 3 parallel nodes write `notes`? (InvalidUpdateError)
Self-check expects these names: graph, result
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    summary: str


def make_researcher(name):
    def research(state: State) -> dict:
        return {"notes": [f"{name}: fact about {state['topic']}"]}
    return research


def combine(state: State) -> dict:
    return {"summary": " | ".join(sorted(state["notes"]))}


builder = StateGraph(State)
for name in ["research_a", "research_b", "research_c"]:
    builder.add_node(name, make_researcher(name))
    builder.add_edge(START, name)
builder.add_node("combine", combine)
builder.add_edge(["research_a", "research_b", "research_c"], "combine")
builder.add_edge("combine", END)
graph = builder.compile()

result = graph.invoke({"topic": "transformers", "notes": []})
print(result["summary"])


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert len(result["notes"]) == 3
    assert result["summary"].startswith("research_a: fact about transformers")
    print("All checks passed!")
