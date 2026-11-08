"""
Exercise 02 - Reducers (Annotated + operator.add) and parallel fan-out / fan-in  [basics]

1. class State(TypedDict): topic: str ; notes: Annotated[list[str], operator.add] ; summary: str
   The reducer operator.add means: updates to `notes` are APPENDED instead of overwriting.
2. Three nodes research_a / research_b / research_c, each returns {"notes": [f"<name>: fact about {topic}"]}
3. Node combine: summary = " | ".join(sorted(state["notes"]))
4. Edges: START -> each of the 3 research nodes (they run IN PARALLEL in the same step),
   then builder.add_edge(["research_a", "research_b", "research_c"], "combine")  (waits for all 3)
   combine -> END
5. result = graph.invoke({"topic": "transformers", "notes": []})
Question: what happens without the reducer when 3 parallel nodes write `notes`? (InvalidUpdateError)
Self-check expects these names: graph, result

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert len(result["notes"]) == 3
    assert result["summary"].startswith("research_a: fact about transformers")
    print("All checks passed!")
