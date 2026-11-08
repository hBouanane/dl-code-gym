"""
Exercise 03 - Routing with add_conditional_edges  [routing]

A support-ticket router.
1. State: question: str ; category: str ; answer: str
2. Node classify: category = "billing" if "refund" or "invoice" in the question (lowercase),
   "tech" if "error" or "bug" in it, otherwise "general"
3. Nodes billing_agent / tech_agent / general_agent each set answer = "[<category>] handled: <question>"
4. Router function route(state) -> returns state["category"]
   builder.add_conditional_edges("classify", route,
        {"billing": "billing_agent", "tech": "tech_agent", "general": "general_agent"})
5. Every agent -> END. Write ask(q) that invokes the graph and returns the answer.
Self-check expects these names: graph, ask

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert ask("Where is my INVOICE?").startswith("[billing]")
    assert ask("there is a bug in the app").startswith("[tech]")
    assert ask("hello").startswith("[general]")
    print("All checks passed!")
