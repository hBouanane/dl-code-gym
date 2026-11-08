"""
Exercise 08 - Human-in-the-loop: interrupt() and Command(resume=...)  [agents]

An email agent that must get approval before sending.
1. State: request: str ; draft: str ; status: str
2. Node write_draft: draft = f"Dear client, {request}. Best regards."
3. Node human_review: decision = interrupt({"draft": state["draft"]})   <- pauses the graph here
     if decision["approved"]: status = "sent"
     else: status = "cancelled"
4. Edges START -> write_draft -> human_review -> END. Compile WITH a checkpointer (required).
5. first = graph.invoke({"request": "your order shipped"}, config)
     -> first contains "__interrupt__"; graph.get_state(config).next == ("human_review",)
6. Resume: final = graph.invoke(Command(resume={"approved": True}), config) -> status "sent"
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
    from langgraph.types import Command
    cfg = {"configurable": {"thread_id": "check-1"}}
    r1 = graph.invoke({"request": "hello"}, cfg)
    assert "__interrupt__" in r1 and graph.get_state(cfg).next == ("human_review",)
    assert graph.invoke(Command(resume={"approved": True}), cfg)["status"] == "sent"
    cfg2 = {"configurable": {"thread_id": "check-2"}}
    graph.invoke({"request": "hello"}, cfg2)
    assert graph.invoke(Command(resume={"approved": False}), cfg2)["status"] == "cancelled"
    print("All checks passed!")
