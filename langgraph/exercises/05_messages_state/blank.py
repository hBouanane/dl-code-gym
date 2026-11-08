"""
Exercise 05 - Chat state: MessagesState, add_messages, message types  [chat]

1. Use the prebuilt MessagesState (its `messages` key uses the add_messages reducer).
2. Write fake_llm(messages) -> AIMessage: if the last message contains "hello" reply
   "Hi! How can I help?" else reply "You said: <content>". (Stand-in for a real model, see below.)
3. Node chatbot(state) -> {"messages": [fake_llm(state["messages"])]}
4. START -> chatbot -> END. result = graph.invoke({"messages": [HumanMessage("hello")]})
   -> result["messages"] has 2 messages (human + ai)
5. add_messages detail: call add_messages([HumanMessage("a", id="1")], [HumanMessage("b", id="1")])
   -> the message with the same id is REPLACED, not appended. Store it in `merged`.
Real model (later): from langchain.chat_models import init_chat_model
                    llm = init_chat_model("anthropic:claude-sonnet-5-5"); llm.invoke(state["messages"])
Self-check expects these names: graph, result, merged

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert len(result["messages"]) == 2
    assert result["messages"][-1].type == "ai" and "Hi" in result["messages"][-1].content
    assert graph.invoke({"messages": [HumanMessage("test")]})["messages"][-1].content == "You said: test"
    assert len(merged) == 1 and merged[0].content == "b"
    print("All checks passed!")
