"""
Exercise 01 - Routes, path params, query params  [basics]

app = FastAPI(title="Practice API")
  GET  /                      -> {"message": "hello"}
  GET  /items/{item_id}       -> item_id: int (path param, FastAPI validates it: "abc" -> 422)
                                 returns {"item_id": item_id}
  GET  /search?q=...&limit=10 -> q: str (required), limit: int = 10 (optional with default)
                                 returns {"q": q, "limit": limit}
  GET  /users/{user_id}/orders/{order_id} -> two path params
  GET  /files?tags=a&tags=b   -> tags: list[str] = Query(default=[]) -> {"tags": tags}
Run it for real:  uvicorn solution:app --reload  then open http://127.0.0.1:8000/docs
Self-check expects these names: app

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    c = TestClient(app)
    assert c.get("/").json() == {"message": "hello"}
    assert c.get("/items/5").json() == {"item_id": 5}
    assert c.get("/items/abc").status_code == 422
    assert c.get("/search", params={"q": "x"}).json() == {"q": "x", "limit": 10}
    assert c.get("/search").status_code == 422
    assert c.get("/users/1/orders/2").json() == {"user_id": 1, "order_id": 2}
    assert c.get("/files?tags=a&tags=b").json() == {"tags": ["a", "b"]}
    print("All checks passed!")
