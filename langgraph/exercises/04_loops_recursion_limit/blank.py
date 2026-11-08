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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert result["attempts"] == 3 and result["approved"] is True
    assert result["draft"].count("+improved") == 3
    assert hit_limit
    print("All checks passed!")
