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
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    category: str
    answer: str


def classify(state: State) -> dict:
    q = state["question"].lower()
    if "refund" in q or "invoice" in q:
        category = "billing"
    elif "error" in q or "bug" in q:
        category = "tech"
    else:
        category = "general"
    return {"category": category}


def make_agent(category):
    def agent(state: State) -> dict:
        return {"answer": f"[{category}] handled: {state['question']}"}
    return agent


def route(state: State) -> str:
    return state["category"]


builder = StateGraph(State)
builder.add_node("classify", classify)
for c in ["billing", "tech", "general"]:
    builder.add_node(f"{c}_agent", make_agent(c))
    builder.add_edge(f"{c}_agent", END)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route,
                              {"billing": "billing_agent", "tech": "tech_agent", "general": "general_agent"})
graph = builder.compile()


def ask(q):
    return graph.invoke({"question": q})["answer"]


print(ask("I want a refund"), ask("I get an error 500"), ask("What are your hours?"), sep="\n")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    assert ask("Where is my INVOICE?").startswith("[billing]")
    assert ask("there is a bug in the app").startswith("[tech]")
    assert ask("hello").startswith("[general]")
    print("All checks passed!")
