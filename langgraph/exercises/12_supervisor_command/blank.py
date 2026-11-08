"""
Exercise 12 - Multi-agent supervisor with Command(goto=..., update=...)  [agents]

Nodes can route themselves by returning Command instead of a dict (no edges needed between them).
1. State: task: str ; research: str ; draft: str ; steps: Annotated[list[str], operator.add]
2. supervisor(state) -> Command[Literal["researcher", "writer", "__end__"]]:
     no research yet -> Command(goto="researcher", update={"steps": ["supervisor->researcher"]})
     no draft yet    -> Command(goto="writer",     update={"steps": ["supervisor->writer"]})
     else            -> Command(goto=END,          update={"steps": ["supervisor->end"]})
3. researcher -> Command(goto="supervisor", update={"research": f"facts about {task}", "steps": ["researcher"]})
   writer     -> Command(goto="supervisor", update={"draft": f"Article using {research}", "steps": ["writer"]})
4. Only one edge: START -> supervisor. Add the 3 nodes.
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
    out = graph.invoke({"task": "Docker", "steps": []})
    assert out["steps"] == ["supervisor->researcher", "researcher", "supervisor->writer", "writer", "supervisor->end"]
    assert out["draft"] == "Article using facts about Docker"
    print("All checks passed!")
