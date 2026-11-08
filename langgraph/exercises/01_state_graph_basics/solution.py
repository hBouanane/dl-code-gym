"""
Exercise 01 - StateGraph basics: state, nodes, edges, compile, invoke  [basics]

The 6 lines behind every LangGraph app.
1. class State(TypedDict): text: str ; word_count: int ; shouted: str
2. Node count_words(state) -> returns {"word_count": number of words in state["text"]}
   Node shout(state)       -> returns {"shouted": state["text"].upper() + "!"}
   (a node returns ONLY the keys it updates, not the whole state)
3. builder = StateGraph(State); add both nodes; edges START -> count_words -> shout -> END
4. graph = builder.compile(); result = graph.invoke({"text": "hello lang graph"})
5. Print graph.get_graph().draw_mermaid() and paste it into mermaid.live to see the graph.
Self-check expects these names: graph, result
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    text: str
    word_count: int
    shouted: str


def count_words(state: State) -> dict:
    return {"word_count": len(state["text"].split())}


def shout(state: State) -> dict:
    return {"shouted": state["text"].upper() + "!"}


builder = StateGraph(State)
builder.add_node("count_words", count_words)
builder.add_node("shout", shout)
builder.add_edge(START, "count_words")
builder.add_edge("count_words", "shout")
builder.add_edge("shout", END)
graph = builder.compile()

result = graph.invoke({"text": "hello lang graph"})
print(result)
print(graph.get_graph().draw_mermaid())


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert result["word_count"] == 3
    assert result["shouted"] == "HELLO LANG GRAPH!"
    assert graph.invoke({"text": "a b"})["word_count"] == 2
    print("All checks passed!")
