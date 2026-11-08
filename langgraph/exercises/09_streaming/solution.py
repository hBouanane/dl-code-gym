"""
Exercise 09 - Streaming: updates, values and custom events  [basics]

1. Linear graph step_1 -> step_2 -> step_3, State: log: Annotated[list[str], operator.add]
   each step appends its own name.
2. stream_mode="updates": for chunk in graph.stream(inp, stream_mode="updates") -> chunk is
   {node_name: update}. Collect the node names in order -> `order`
3. stream_mode="values": collect the full state after each step -> `values`
4. Custom events: inside step_2 call writer = get_stream_writer(); writer({"progress": 50})
   and collect them with stream_mode="custom" -> `custom`
(Streaming is how a chat UI shows progress token-by-token / step-by-step.)
Self-check expects these names: order, values, custom
"""

import operator
from typing import Annotated, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    log: Annotated[list[str], operator.add]


def step_1(state: State) -> dict:
    return {"log": ["step_1"]}


def step_2(state: State) -> dict:
    writer = get_stream_writer()
    writer({"progress": 50})
    return {"log": ["step_2"]}


def step_3(state: State) -> dict:
    return {"log": ["step_3"]}


builder = StateGraph(State)
for fn in [step_1, step_2, step_3]:
    builder.add_node(fn.__name__, fn)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)
graph = builder.compile()

inp = {"log": []}
order = [name for chunk in graph.stream(inp, stream_mode="updates") for name in chunk]
values = [state for state in graph.stream(inp, stream_mode="values")]
custom = [event for event in graph.stream(inp, stream_mode="custom")]
print(order, values[-1], custom, sep="\n")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert order == ["step_1", "step_2", "step_3"]
    assert values[-1]["log"] == ["step_1", "step_2", "step_3"]
    assert custom == [{"progress": 50}]
    print("All checks passed!")
