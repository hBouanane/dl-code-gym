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
"""

from fastapi import FastAPI, Query

app = FastAPI(title="Practice API")


@app.get("/")
def root():
    return {"message": "hello"}


@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}


@app.get("/search")
def search(q: str, limit: int = 10):
    return {"q": q, "limit": limit}


@app.get("/users/{user_id}/orders/{order_id}")
def get_order(user_id: int, order_id: int):
    return {"user_id": user_id, "order_id": order_id}


@app.get("/files")
def files(tags: list[str] = Query(default=[])):
    return {"tags": tags}


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
