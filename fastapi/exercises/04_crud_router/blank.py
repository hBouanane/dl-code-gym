"""
Exercise 04 - Full CRUD with APIRouter, prefix and tags  [basics]

1. Models: TodoCreate(title: str, done: bool = False) ; TodoUpdate(title: str | None = None,
   done: bool | None = None) ; Todo(TodoCreate) + id: int
2. router = APIRouter(prefix="/todos", tags=["todos"]) with an in-memory dict and a counter:
     POST   /todos          -> 201, returns Todo
     GET    /todos          -> list[Todo], optional query done: bool | None to filter
     GET    /todos/{id}     -> Todo or 404
     PATCH  /todos/{id}     -> partial update: data = body.model_dump(exclude_unset=True)
     DELETE /todos/{id}     -> 204 or 404
3. app = FastAPI(); app.include_router(router)
Key idea: exclude_unset=True -> only fields the client actually sent are updated.
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
    r = c.post("/todos", json={"title": "learn fastapi"})
    assert r.status_code == 201
    tid = r.json()["id"]
    c.post("/todos", json={"title": "learn docker", "done": True})
    assert len(c.get("/todos").json()) >= 2
    assert all(t["done"] for t in c.get("/todos", params={"done": True}).json())
    r = c.patch(f"/todos/{tid}", json={"done": True})
    assert r.json() == {"id": tid, "title": "learn fastapi", "done": True}
    assert c.delete(f"/todos/{tid}").status_code == 204
    assert c.get(f"/todos/{tid}").status_code == 404
    print("All checks passed!")
