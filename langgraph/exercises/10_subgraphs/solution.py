"""
Exercise 10 - Subgraphs: a compiled graph used as a node  [composition]

1. Subgraph with State {text: str}: nodes lower (text.lower()) -> strip_punct (remove .,!?)
   compile it -> preprocess_graph
2. Parent graph with State {text: str ; n_words: int}:
     add_node("preprocess", preprocess_graph)   <- shared key `text` flows in and out
     add_node("analyze", ...) sets n_words
     START -> preprocess -> analyze -> END
3. Different schema case: a subgraph with State {query: str ; result: str} called INSIDE a node
   function of the parent: out = sub.invoke({"query": state["text"]}); return {"text": out["result"]}
   (you transform the state in and out yourself)
Self-check expects these names: preprocess_graph, graph
"""

import string
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SubState(TypedDict):
    text: str


def lower(state: SubState) -> dict:
    return {"text": state["text"].lower()}


def strip_punct(state: SubState) -> dict:
    return {"text": state["text"].translate(str.maketrans("", "", ".,!?"))}


sub = StateGraph(SubState)
sub.add_node("lower", lower)
sub.add_node("strip_punct", strip_punct)
sub.add_edge(START, "lower")
sub.add_edge("lower", "strip_punct")
sub.add_edge("strip_punct", END)
preprocess_graph = sub.compile()


class ParentState(TypedDict):
    text: str
    n_words: int


class OtherState(TypedDict):
    query: str
    result: str


other = StateGraph(OtherState)
other.add_node("reverse", lambda s: {"result": " ".join(reversed(s["query"].split()))})
other.add_edge(START, "reverse")
other.add_edge("reverse", END)
other_graph = other.compile()


def reverse_words(state: ParentState) -> dict:
    out = other_graph.invoke({"query": state["text"]})
    return {"text": out["result"]}


parent = StateGraph(ParentState)
parent.add_node("preprocess", preprocess_graph)
parent.add_node("reverse", reverse_words)
parent.add_node("analyze", lambda s: {"n_words": len(s["text"].split())})
parent.add_edge(START, "preprocess")
parent.add_edge("preprocess", "reverse")
parent.add_edge("reverse", "analyze")
parent.add_edge("analyze", END)
graph = parent.compile()

print(graph.invoke({"text": "Hello, World! LangGraph is FUN."}))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert preprocess_graph.invoke({"text": "Hi, There!"})["text"] == "hi there"
    out = graph.invoke({"text": "Hello, World! LangGraph is FUN."})
    assert out["n_words"] == 5 and out["text"].startswith("fun is")
    print("All checks passed!")
