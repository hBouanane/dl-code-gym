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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert result["word_count"] == 3
    assert result["shouted"] == "HELLO LANG GRAPH!"
    assert graph.invoke({"text": "a b"})["word_count"] == 2
    print("All checks passed!")
