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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert preprocess_graph.invoke({"text": "Hi, There!"})["text"] == "hi there"
    out = graph.invoke({"text": "Hello, World! LangGraph is FUN."})
    assert out["n_words"] == 5 and out["text"].startswith("fun is")
    print("All checks passed!")
