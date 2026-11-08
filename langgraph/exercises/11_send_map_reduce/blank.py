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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    out = graph.invoke({"subjects": ["cat", "elephant", "dog"], "jokes": []})
    assert len(out["jokes"]) == 3 and "elephant" in out["best"]
    assert len(graph.invoke({"subjects": list("abcde"), "jokes": []})["jokes"]) == 5
    print("All checks passed!")
