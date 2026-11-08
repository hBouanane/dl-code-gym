"""
Exercise 13 - A RAG graph: retrieve -> grade -> (rewrite query) -> generate  [agents]

A corrective-RAG skeleton with a toy keyword retriever (swap in a vector store later).
DOCS = ["LangGraph builds stateful agents as graphs.",
        "FastAPI is a Python web framework for APIs.",
        "Docker packages apps into containers."]
SYNONYMS = {"container": "docker", "containers": "docker", "api": "fastapi", "agent": "langgraph"}
1. State: question: str ; query: str ; docs: list[str] ; rewrites: int ; answer: str
2. retrieve: score each doc = number of lowercase words of `query` present in the doc's words
   (strip "." and "?", drop stopwords like "a", "is", "the", "what"), keep docs with score > 0, sorted by score desc, top 2.
3. Router after retrieve: docs found -> "generate"; no docs and rewrites < 1 -> "rewrite"; else "generate"
4. rewrite: query = words replaced via SYNONYMS ; rewrites += 1 ; then back to retrieve
5. generate: answer = "Based on: " + docs[0] if docs else "I don't know."
6. ask(question) invokes with query = question, rewrites = 0, returns the answer.
Self-check expects these names: ask

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert ask("what does Docker do?") == "Based on: Docker packages apps into containers."
    assert ask("container") == "Based on: Docker packages apps into containers."
    assert ask("pizza") == "I don't know."
    print("All checks passed!")
