"""
Exercise 12 - Multi-agent supervisor with Command(goto=..., update=...)  [agents]

Nodes can route themselves by returning Command instead of a dict (no edges needed between them).
1. State: task: str ; research: str ; draft: str ; steps: Annotated[list[str], operator.add]
2. supervisor(state) -> Command[Literal["researcher", "writer", "__end__"]]:
     no research yet -> Command(goto="researcher", update={"steps": ["supervisor->researcher"]})
     no draft yet    -> Command(goto="writer",     update={"steps": ["supervisor->writer"]})
     else            -> Command(goto=END,          update={"steps": ["supervisor->end"]})
3. researcher -> Command(goto="supervisor", update={"research": f"facts about {task}", "steps": ["researcher"]})
   writer     -> Command(goto="supervisor", update={"draft": f"Article using {research}", "steps": ["writer"]})
4. Only one edge: START -> supervisor. Add the 3 nodes.
Self-check expects these names: graph
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    task: str
    research: str
    draft: str
    steps: Annotated[list[str], operator.add]


def supervisor(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    if not state.get("research"):
        return Command(goto="researcher", update={"steps": ["supervisor->researcher"]})
    if not state.get("draft"):
        return Command(goto="writer", update={"steps": ["supervisor->writer"]})
    return Command(goto=END, update={"steps": ["supervisor->end"]})


def researcher(state: State) -> Command[Literal["supervisor"]]:
    return Command(goto="supervisor", update={"research": f"facts about {state['task']}", "steps": ["researcher"]})


def writer(state: State) -> Command[Literal["supervisor"]]:
    return Command(goto="supervisor", update={"draft": f"Article using {state['research']}", "steps": ["writer"]})


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_edge(START, "supervisor")
graph = builder.compile()

out = graph.invoke({"task": "LangGraph", "steps": []})
print(out["steps"], out["draft"], sep="\n")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    out = graph.invoke({"task": "Docker", "steps": []})
    assert out["steps"] == ["supervisor->researcher", "researcher", "supervisor->writer", "writer", "supervisor->end"]
    assert out["draft"] == "Article using facts about Docker"
    print("All checks passed!")
