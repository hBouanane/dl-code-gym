"""
Exercise 06 - Tools + ToolNode + tools_condition: the ReAct loop  [agents]

The agent loop: agent -> (tool calls?) -> tools -> agent -> ... -> END
1. Two tools with @tool (the docstring is the description the LLM sees!):
     multiply(a: int, b: int) -> int ; get_weather(city: str) -> str  (return f"Sunny in {city}")
2. fake_llm node (stands in for llm.bind_tools(tools)):
     - if the last message is a ToolMessage -> AIMessage(f"The answer is {last.content}.")
     - elif "times" in the question -> AIMessage(content="", tool_calls=[{"name": "multiply",
           "args": {"a": .., "b": ..}, "id": "call_1"}])   (numbers parsed with re.findall)
     - elif "weather in" -> tool call get_weather with the city (last word)
     - else -> AIMessage("I can only multiply or check the weather.")
3. Write your own router should_continue(state): "tools" if last message has tool_calls else END
   (this is what langgraph.prebuilt.tools_condition does)
4. Graph: START -> agent ; add_conditional_edges("agent", should_continue, ["tools", END]) ;
          add_node("tools", ToolNode(tools)) ; "tools" -> "agent"
5. ask(q) returns the final message content. "what is 6 times 7" -> "The answer is 42."
Self-check expects these names: multiply, graph, ask

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert multiply.invoke({"a": 3, "b": 4}) == 12
    assert ask("what is 6 times 7") == "The answer is 42."
    assert "Rabat" in ask("weather in Rabat")
    out = graph.invoke({"messages": [HumanMessage("what is 2 times 5")]})["messages"]
    assert [m.type for m in out] == ["human", "ai", "tool", "ai"]
    print("All checks passed!")
