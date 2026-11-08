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
"""

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel


class TodoCreate(BaseModel):
    title: str
    done: bool = False


class TodoUpdate(BaseModel):
    title: str | None = None
    done: bool | None = None


class Todo(TodoCreate):
    id: int


router = APIRouter(prefix="/todos", tags=["todos"])
TODOS: dict[int, Todo] = {}
_next_id = 1


def get_or_404(todo_id: int) -> Todo:
    if todo_id not in TODOS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Todo not found")
    return TODOS[todo_id]


@router.post("", response_model=Todo, status_code=status.HTTP_201_CREATED)
def create(body: TodoCreate):
    global _next_id
    todo = Todo(id=_next_id, **body.model_dump())
    TODOS[todo.id] = todo
    _next_id += 1
    return todo


@router.get("", response_model=list[Todo])
def list_todos(done: bool | None = None):
    return [t for t in TODOS.values() if done is None or t.done == done]


@router.get("/{todo_id}", response_model=Todo)
def read(todo_id: int):
    return get_or_404(todo_id)


@router.patch("/{todo_id}", response_model=Todo)
def update(todo_id: int, body: TodoUpdate):
    todo = get_or_404(todo_id)
    updated = todo.model_copy(update=body.model_dump(exclude_unset=True))
    TODOS[todo_id] = updated
    return updated


@router.delete("/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(todo_id: int):
    get_or_404(todo_id)
    del TODOS[todo_id]


app = FastAPI()
app.include_router(router)


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
