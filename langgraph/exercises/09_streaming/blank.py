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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert order == ["step_1", "step_2", "step_3"]
    assert values[-1]["log"] == ["step_1", "step_2", "step_3"]
    assert custom == [{"progress": 50}]
    print("All checks passed!")
