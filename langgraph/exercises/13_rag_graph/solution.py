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
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

DOCS = ["LangGraph builds stateful agents as graphs.",
        "FastAPI is a Python web framework for APIs.",
        "Docker packages apps into containers."]
SYNONYMS = {"container": "docker", "containers": "docker", "api": "fastapi", "agent": "langgraph"}


class State(TypedDict):
    question: str
    query: str
    docs: list[str]
    rewrites: int
    answer: str


STOPWORDS = {"a", "an", "the", "is", "to", "of", "for", "in", "what", "how", "does", "do", "into", "as"}


def words(text):
    return set(text.lower().replace(".", "").replace("?", "").split()) - STOPWORDS


def retrieve(state: State) -> dict:
    q = words(state["query"])
    scored = [(len(q & words(d)), d) for d in DOCS]
    hits = [d for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0]
    return {"docs": hits[:2]}


def route(state: State) -> str:
    if state["docs"]:
        return "generate"
    return "rewrite" if state["rewrites"] < 1 else "generate"


def rewrite(state: State) -> dict:
    new = " ".join(SYNONYMS.get(w, w) for w in state["query"].lower().replace("?", "").split())
    return {"query": new, "rewrites": state["rewrites"] + 1}


def generate(state: State) -> dict:
    return {"answer": "Based on: " + state["docs"][0] if state["docs"] else "I don't know."}


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route, ["generate", "rewrite"])
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)
graph = builder.compile()


def ask(question):
    return graph.invoke({"question": question, "query": question, "rewrites": 0})["answer"]


print(ask("What is FastAPI?"))
print(ask("how to ship a container"))
print(ask("best pizza in town"))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert ask("what does Docker do?") == "Based on: Docker packages apps into containers."
    assert ask("container") == "Based on: Docker packages apps into containers."
    assert ask("pizza") == "I don't know."
    print("All checks passed!")
