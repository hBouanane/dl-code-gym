"""
Exercise 07 - Memory with a checkpointer and thread_id  [chat]

1. Chat graph on MessagesState with a fake bot node:
     "my name is X"     -> "Nice to meet you, X!"
     "what is my name"  -> look back through ALL previous human messages for "my name is X"
                           -> "Your name is X." (or "I don't know your name.")
     otherwise          -> "OK."
2. graph = builder.compile(checkpointer=InMemorySaver())
3. config = {"configurable": {"thread_id": "user-1"}} ; send "my name is Haitam" then
   "what is my name" as two SEPARATE invokes (only the new message each time!)
   -> the bot remembers because the state is saved per thread.
4. A different thread_id "user-2" does not know the name.
5. snapshot = graph.get_state(config) -> snapshot.values["messages"] has 4 messages
Self-check expects these names: graph, chat(text, thread_id) -> reply string

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert chat("my name is Sara", "t-a") == "Nice to meet you, Sara!"
    assert chat("what is my name", "t-a") == "Your name is Sara."
    assert chat("what is my name", "t-b") == "I don't know your name."
    assert len(graph.get_state({"configurable": {"thread_id": "t-a"}}).values["messages"]) == 4
    print("All checks passed!")
