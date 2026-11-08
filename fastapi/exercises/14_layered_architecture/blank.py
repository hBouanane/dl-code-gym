"""
Exercise 14 - Layered architecture: router -> service -> repository  [architecture]

The first step away from "everything in the route function". Write 3 layers (one file here;
in a project they would be api/, services/, repositories/):
  REPOSITORY (data access only, no business rules):
     class UserRepository: add(user: dict) -> dict ; get_by_email(email) -> dict | None ; list() -> list
  SERVICE (business rules only, no HTTP, no SQL):
     class UserService(repo): register(name, email) -> dict
        - email is lowercased
        - raises EmailAlreadyUsed (a plain Python exception) if the email exists
  API (HTTP only: parse request, call the service, map errors to status codes):
     get_user_service() dependency returning UserService(repo) with a module-level repo
     POST /users -> 201 ; EmailAlreadyUsed -> 409
     GET /users -> list
Rule to feel: the service can be tested with NO FastAPI at all.
Self-check expects these names: app, UserRepository, UserService, EmailAlreadyUsed, get_user_service

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
    svc = UserService(UserRepository())
    assert svc.register("A", "A@X.com")["email"] == "a@x.com"
    try:
        svc.register("B", "a@x.com")
        raise AssertionError("should raise")
    except EmailAlreadyUsed:
        pass
    fresh = UserService(UserRepository())
    app.dependency_overrides[get_user_service] = lambda: fresh
    c = TestClient(app)
    assert c.post("/users", json={"name": "H", "email": "h@x.com"}).status_code == 201
    assert c.post("/users", json={"name": "H2", "email": "H@x.com"}).status_code == 409
    assert len(c.get("/users").json()) == 1
    app.dependency_overrides.clear()
    print("All checks passed!")
